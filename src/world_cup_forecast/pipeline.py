from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter

from .agents import run_cluster
from .config import load_config
from .consensus import build_consensus
from .models import Fixture
from .news import collect_news
from .providers import build_provider
from .report import write_reports
from .structured_data import collect_structured_data, data_completeness


def select_next_fixture(
    fixtures: list[Fixture], now: datetime | None = None
) -> Fixture:
    current_time = now or datetime.now(UTC)
    upcoming = [
        fixture
        for fixture in fixtures
        if fixture.status == "scheduled"
        and fixture.kickoff.astimezone(UTC) >= current_time.astimezone(UTC)
    ]
    if not upcoming:
        raise RuntimeError("没有找到尚未开始的下一场比赛，请更新赛程文件")
    return min(upcoming, key=lambda fixture: fixture.kickoff.astimezone(UTC))


async def run_daily(config_path: str) -> tuple[Path, Path]:
    path = Path(config_path).resolve()
    config = load_config(path)
    fixtures_path = Path(config.competition.fixtures_file)
    if not fixtures_path.is_absolute():
        fixtures_path = path.parent / fixtures_path
    fixtures = TypeAdapter(list[Fixture]).validate_python(
        json.loads(fixtures_path.read_text(encoding="utf-8"))
    )
    next_fixture = select_next_fixture(fixtures)
    fixtures = [next_fixture]
    local_kickoff = next_fixture.kickoff.astimezone(
        ZoneInfo(config.competition.timezone)
    )
    print(
        f"[下一场比赛] {next_fixture.home_team} vs "
        f"{next_fixture.away_team}，开球时间 {local_kickoff:%Y-%m-%d %H:%M %Z}",
        flush=True,
    )

    articles_task = collect_news(
        config.news_sources,
        config.run.news_lookback_hours,
        config.run.max_articles,
        config.run.request_timeout_seconds,
    )
    data_task = collect_structured_data(
        config.data_sources,
        next_fixture,
        path.parent,
        config.run.request_timeout_seconds,
    )
    (articles, news_warnings), (
        intelligence,
        data_warnings,
    ) = await asyncio.gather(articles_task, data_task)
    completeness, missing_factors = data_completeness(intelligence)
    print(
        f"[结构化数据] 完整度 {completeness:.0%}"
        + (
            f"，缺失：{'、'.join(missing_factors)}"
            if missing_factors
            else "，关键因素齐全"
        ),
        flush=True,
    )
    providers = {
        item.id: build_provider(
            item, config.run.request_timeout_seconds, fixtures
        )
        for item in config.providers
    }
    forecasts, agent_warnings = await run_cluster(
        config.competition.name,
        config.agents,
        providers,
        articles,
        fixtures,
        config.run.max_articles_per_agent,
        intelligence,
    )
    consensus = build_consensus(
        forecasts,
        config.agents,
        config.providers,
        news_warnings + data_warnings + agent_warnings,
    )
    output_dir = Path(config.run.output_dir)
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    return write_reports(
        output_dir,
        config.competition.name,
        config.competition.timezone,
        consensus,
        forecasts,
        articles,
        next_fixture,
        intelligence,
    )
