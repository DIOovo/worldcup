from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import (
    AgentForecast,
    Article,
    ConsensusForecast,
    Fixture,
    MatchIntelligence,
)
from .structured_data import data_completeness


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "缺失"


def _lineup_summary(data: MatchIntelligence) -> str:
    home = (
        data.home_availability.confirmed_lineup
        or data.home_availability.expected_lineup
    )
    away = (
        data.away_availability.confirmed_lineup
        or data.away_availability.expected_lineup
    )
    return f"主队 {len(home)} 人，客队 {len(away)} 人"


def _availability_summary(data: MatchIntelligence) -> str:
    affected = [
        item
        for item in (
            data.home_availability.players + data.away_availability.players
        )
        if item.status in {"doubtful", "injured", "suspended"}
    ]
    return (
        "；".join(f"{item.player}({item.status})" for item in affected)
        if affected
        else "无已知缺阵"
    )


def _odds_summary(data: MatchIntelligence) -> str:
    if not data.odds:
        return "缺失"
    first, latest = data.odds[0], data.odds[-1]
    return (
        f"{first.bookmaker} "
        f"{first.home_decimal:.2f}/{first.draw_decimal:.2f}/{first.away_decimal:.2f}"
        f" -> {latest.home_decimal:.2f}/{latest.draw_decimal:.2f}/"
        f"{latest.away_decimal:.2f}"
    )


def _weather_summary(data: MatchIntelligence) -> str:
    if not data.weather:
        return "缺失"
    weather = data.weather
    return (
        f"{weather.condition or '未知'}，{weather.temperature_c} C，"
        f"降水 {weather.precipitation_mm} mm，风速 {weather.wind_kph} km/h"
    )


def _venue_summary(data: MatchIntelligence) -> str:
    if not data.venue:
        return "缺失"
    venue = data.venue
    return (
        f"{venue.stadium or '未知球场'}，{venue.city or '未知城市'}，"
        f"{venue.surface or '未知草皮'}，海拔 {venue.altitude_m} m"
    )


def _travel_summary(data: MatchIntelligence) -> str:
    if not data.travel:
        return "缺失"
    travel = data.travel
    return (
        f"主队 {travel.home_distance_km} km/"
        f"{travel.home_timezone_shift_hours} 小时时差，"
        f"客队 {travel.away_distance_km} km/"
        f"{travel.away_timezone_shift_hours} 小时时差"
    )


def _outcome(prediction) -> str:
    probabilities = {
        prediction.home_team: prediction.home_win_probability,
        "平局": prediction.draw_probability,
        prediction.away_team: prediction.away_win_probability,
    }
    return max(probabilities, key=probabilities.get)


def render_markdown(
    competition_name: str,
    timezone: str,
    consensus: ConsensusForecast,
    forecasts: list[AgentForecast],
    articles: list[Article],
    fixture: Fixture,
    intelligence: MatchIntelligence | None,
) -> str:
    local_time = consensus.generated_at.astimezone(ZoneInfo(timezone))
    kickoff = fixture.kickoff.astimezone(ZoneInfo(timezone))
    if not consensus.match_predictions:
        raise RuntimeError("模型没有返回下一场比赛的预测结果")
    prediction = consensus.match_predictions[0]
    outcome = _outcome(prediction)
    outcome_probability = max(
        prediction.home_win_probability,
        prediction.draw_probability,
        prediction.away_win_probability,
    )
    result_label = "预测结果" if outcome == "平局" else "预测胜方"
    completeness, missing_factors = data_completeness(intelligence)
    lines = [
        f"# {competition_name} 下一场比赛预测报告",
        "",
        f"- 报告生成时间：{local_time:%Y-%m-%d %H:%M %Z}",
        f"- 比赛：**{fixture.home_team} vs {fixture.away_team}**",
        f"- 阶段：{fixture.stage}",
        f"- 开球时间：{kickoff:%Y-%m-%d %H:%M %Z}",
        f"- {result_label}：**{outcome}**",
        f"- 该结果综合概率：**{outcome_probability:.1%}**",
        f"- 预测比分：**{prediction.home_team} "
        f"{prediction.home_goals}-{prediction.away_goals} "
        f"{prediction.away_team}**",
        f"- Agent 完成数：{consensus.agent_count}",
        f"- Agent 结论一致率：{consensus.agreement_score:.1%}",
        f"- 使用新闻数：{len(articles)}",
        f"- 结构化数据完整度：{completeness:.0%}",
        "",
        "> 本报告是概率预测，不代表确定赛果，也不构成投注建议。",
        "",
        "## 胜平负概率",
        "",
        "| 主队胜 | 平局 | 客队胜 |",
        "|---:|---:|---:|",
        f"| {prediction.home_win_probability:.1%} "
        f"| {prediction.draw_probability:.1%} "
        f"| {prediction.away_win_probability:.1%} |",
    ]

    lines.extend(["", "## 结构化判断因素", ""])
    if intelligence:
        home = intelligence.home_metrics
        away = intelligence.away_metrics
        lines.extend(
            [
                "| 指标 | 主队 | 客队 |",
                "|---|---:|---:|",
                f"| FIFA 排名 | {home.fifa_rank or '缺失'} "
                f"| {away.fifa_rank or '缺失'} |",
                f"| Elo | {home.elo_rating or '缺失'} "
                f"| {away.elo_rating or '缺失'} |",
                f"| 近期场均进球 | "
                f"{home.recent_goals_for_per_game or '缺失'} "
                f"| {away.recent_goals_for_per_game or '缺失'} |",
                f"| 近期场均失球 | "
                f"{home.recent_goals_against_per_game or '缺失'} "
                f"| {away.recent_goals_against_per_game or '缺失'} |",
                f"| 近期场均 xG | {home.recent_xg_for_per_game or '缺失'} "
                f"| {away.recent_xg_for_per_game or '缺失'} |",
                f"| 近期场均 xGA | {home.recent_xg_against_per_game or '缺失'} "
                f"| {away.recent_xg_against_per_game or '缺失'} |",
                f"| 对应主/客场胜率 | "
                f"{_percent(home.home_or_away_win_rate)} "
                f"| {_percent(away.home_or_away_win_rate)} |",
                f"| 休息天数 | "
                f"{home.rest_days if home.rest_days is not None else '缺失'} "
                f"| {away.rest_days if away.rest_days is not None else '缺失'} |",
                "",
                f"- 近期比赛样本：主队 {len(home.recent_matches)} 场，"
                f"客队 {len(away.recent_matches)} 场",
                f"- 历史交锋：{len(intelligence.head_to_head)} 场",
                f"- 阵容：{_lineup_summary(intelligence)}",
                f"- 伤停与停赛：{_availability_summary(intelligence)}",
                f"- 赔率变化：{_odds_summary(intelligence)}",
                f"- 天气：{_weather_summary(intelligence)}",
                f"- 场地：{_venue_summary(intelligence)}",
                f"- 旅行：{_travel_summary(intelligence)}",
                f"- 数据来源：{'、'.join(intelligence.sources) or '未标明'}",
                f"- 数据更新时间：{intelligence.updated_at:%Y-%m-%d %H:%M %Z}",
            ]
        )
        if intelligence.notes:
            lines.append(f"- 数据备注：{'；'.join(intelligence.notes)}")
    else:
        lines.append("- 未获取到结构化足球数据。")
    if missing_factors:
        lines.append(f"- 缺失因素：{'、'.join(missing_factors)}")

    lines.extend(["", "## Agent 分析摘要", ""])
    for forecast in forecasts:
        findings = "；".join(forecast.key_findings[:3]) or "未返回分析摘要。"
        lines.append(
            f"- **{forecast.agent_id}** ({forecast.provider_id}, "
            f"置信度 {forecast.confidence:.0%})：{findings}"
        )

    if consensus.warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in consensus.warnings)

    lines.extend(["", "## 新闻证据", ""])
    for article in articles[:30]:
        published = (
            article.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if article.published_at
            else "发布时间未知"
        )
        lines.append(
            f"- [{article.title}]({article.url})，来源：{article.source}，{published}"
        )
    if not articles:
        lines.append("- 本次未获取到可用新闻，预测可靠性会受到影响。")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    output_dir: str | Path,
    competition_name: str,
    timezone: str,
    consensus: ConsensusForecast,
    forecasts: list[AgentForecast],
    articles: list[Article],
    fixture: Fixture,
    intelligence: MatchIntelligence | None,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    date = consensus.generated_at.astimezone(ZoneInfo(timezone)).date().isoformat()
    markdown_path = directory / f"{date}.md"
    json_path = directory / f"{date}.json"
    markdown_path.write_text(
        render_markdown(
            competition_name,
            timezone,
            consensus,
            forecasts,
            articles,
            fixture,
            intelligence,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "fixture": fixture.model_dump(mode="json"),
                "intelligence": (
                    intelligence.model_dump(mode="json")
                    if intelligence
                    else None
                ),
                "consensus": consensus.model_dump(mode="json"),
                "agents": [item.model_dump(mode="json") for item in forecasts],
                "articles": [item.model_dump(mode="json") for item in articles],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_path, json_path
