from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from time import perf_counter

from .models import (
    AgentConfig,
    AgentForecast,
    Article,
    Fixture,
    MatchIntelligence,
)
from .providers import LLMProvider


SYSTEM_PROMPT = """你是世界杯比赛预测集群中的一名专业分析 Agent。
只能使用提供的新闻证据和赛程，必须区分事实与推断。
禁止虚构伤病、赛果、赛程、引语或 URL，只返回一个 JSON 对象。
胜、平、负概率必须是 0 到 1 之间的小数，且三项之和必须等于 1。
只预测提供的下一场比赛，champion_predictions 必须返回空数组。
key_findings、risks、rationale 等所有分析文字必须使用简体中文。
球队名称可以保留赛程中提供的原文。引用的 URL 必须来自新闻证据。
"""


def _build_prompt(
    competition_name: str,
    agent: AgentConfig,
    articles: list[Article],
    fixtures: list[Fixture],
    intelligence: MatchIntelligence | None,
) -> str:
    evidence = [
        {
            "source": article.source,
            "title": article.title,
            "url": str(article.url),
            "published_at": (
                article.published_at.isoformat() if article.published_at else None
            ),
            "summary": article.summary,
        }
        for article in articles
    ]
    fixture_data = [fixture.model_dump(mode="json") for fixture in fixtures]
    schema = AgentForecast.model_json_schema()
    return f"""赛事：{competition_name}
专业角色：{agent.role}
Agent ID：{agent.id}
当前 UTC 时间：{datetime.now(UTC).isoformat()}

需要预测的下一场比赛：
{json.dumps(fixture_data, ensure_ascii=False)}

新闻证据：
{json.dumps(evidence, ensure_ascii=False)}

结构化比赛数据：
{json.dumps(
    intelligence.model_dump(mode="json") if intelligence else None,
    ensure_ascii=False,
)}

返回符合以下 schema 的 JSON：
{json.dumps(schema, ensure_ascii=False)}
agent_id 必须设置为 {json.dumps(agent.id)}。
只输出这一场比赛的胜平负概率、最可能比分和中文分析。
champion_predictions 返回 []。
分析时必须明确综合近期赛果、排名、进失球、xG、射门、阵容、伤停、
主客场表现、历史交锋、赔率变化、天气、场地和旅行因素。
结构化数据中缺失的因素必须在 risks 中说明，禁止自行补充。
"""


async def run_agent(
    competition_name: str,
    agent: AgentConfig,
    provider: LLMProvider,
    articles: list[Article],
    fixtures: list[Fixture],
    intelligence: MatchIntelligence | None,
) -> AgentForecast:
    if provider.is_mock:
        print(
            f"[MOCK] Agent={agent.id} model={provider.config.model}: "
            "未调用真实大模型 API",
            flush=True,
        )
    else:
        print(
            f"[LLM REQUEST] Agent={agent.id} provider={provider.config.id} "
            f"model={provider.config.model}",
            flush=True,
        )
        print(f"[LLM URL] {provider.endpoint}", flush=True)

    started_at = perf_counter()
    try:
        result = await provider.generate_json(
            SYSTEM_PROMPT,
            _build_prompt(
                competition_name,
                agent,
                articles,
                fixtures,
                intelligence,
            ),
        )
    except Exception as error:
        elapsed = perf_counter() - started_at
        print(
            f"[LLM FAILED] Agent={agent.id} elapsed={elapsed:.2f}s "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        raise

    elapsed = perf_counter() - started_at
    if provider.is_mock:
        print(f"[MOCK DONE] Agent={agent.id} elapsed={elapsed:.2f}s", flush=True)
    else:
        print(
            f"[LLM SUCCESS] Agent={agent.id} provider={provider.config.id} "
            f"model={provider.config.model} elapsed={elapsed:.2f}s",
            flush=True,
        )
    result["agent_id"] = agent.id
    result["provider_id"] = provider.config.id
    result.setdefault("generated_at", datetime.now(UTC).isoformat())
    return AgentForecast.model_validate(result)


async def run_cluster(
    competition_name: str,
    agents: list[AgentConfig],
    providers: dict[str, LLMProvider],
    articles: list[Article],
    fixtures: list[Fixture],
    max_articles_per_agent: int,
    intelligence: MatchIntelligence | None,
) -> tuple[list[AgentForecast], list[str]]:
    tasks = [
        run_agent(
            competition_name,
            agent,
            providers[agent.provider],
            articles[:max_articles_per_agent],
            fixtures,
            intelligence,
        )
        for agent in agents
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    forecasts: list[AgentForecast] = []
    warnings: list[str] = []
    for agent, result in zip(agents, results, strict=True):
        if isinstance(result, BaseException):
            warnings.append(f"Agent {agent.id} 调用失败：{result}")
        else:
            forecasts.append(result)
    return forecasts, warnings
