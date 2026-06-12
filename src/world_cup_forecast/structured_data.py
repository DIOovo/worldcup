from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from .models import DataSourceConfig, Fixture, MatchIntelligence


def _pick_match(payload: object, match_id: str) -> MatchIntelligence | None:
    if isinstance(payload, dict) and "matches" in payload:
        payload = payload["matches"]
    if isinstance(payload, dict):
        payload = [payload]
    records = TypeAdapter(list[MatchIntelligence]).validate_python(payload)
    return next((item for item in records if item.match_id == match_id), None)


async def _load_source(
    source: DataSourceConfig,
    fixture: Fixture,
    config_dir: Path,
    timeout_seconds: int,
) -> MatchIntelligence | None:
    if source.type == "local_json":
        if not source.path:
            raise ValueError(f"数据源 {source.id} 缺少 path")
        path = Path(source.path)
        if not path.is_absolute():
            path = config_dir / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _pick_match(payload, fixture.match_id)

    if not source.url:
        raise ValueError(f"数据源 {source.id} 缺少 url")
    headers: dict[str, str] = {}
    if source.api_key_env:
        key = os.getenv(source.api_key_env)
        if not key:
            raise RuntimeError(f"缺少环境变量 {source.api_key_env}")
        prefix = f"{source.api_key_prefix} " if source.api_key_prefix else ""
        headers[source.api_key_header] = f"{prefix}{key}"
    params = {
        "match_id": fixture.match_id,
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "kickoff": fixture.kickoff.isoformat(),
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(source.url, params=params, headers=headers)
        response.raise_for_status()
    return _pick_match(response.json(), fixture.match_id)


def _merge(
    records: list[MatchIntelligence],
) -> MatchIntelligence:
    result = records[0]
    for record in records[1:]:
        base = result.model_dump()
        incoming = record.model_dump(exclude_none=True)
        for field in (
            "home_metrics",
            "away_metrics",
            "home_availability",
            "away_availability",
            "weather",
            "venue",
            "travel",
        ):
            if field in incoming:
                if isinstance(incoming[field], dict) and isinstance(base.get(field), dict):
                    base[field].update(incoming[field])
                else:
                    base[field] = incoming[field]
        for field in ("head_to_head", "odds", "notes", "sources"):
            base[field] = [*base.get(field, []), *incoming.get(field, [])]
        base["updated_at"] = max(result.updated_at, record.updated_at)
        result = MatchIntelligence.model_validate(base)
    result.sources = list(dict.fromkeys(result.sources))
    return result


async def collect_structured_data(
    sources: list[DataSourceConfig],
    fixture: Fixture,
    config_dir: Path,
    timeout_seconds: int,
) -> tuple[MatchIntelligence | None, list[str]]:
    enabled = [source for source in sources if source.enabled]
    if not enabled:
        return None, ["未配置结构化足球数据源"]
    results = await asyncio.gather(
        *[
            _load_source(source, fixture, config_dir, timeout_seconds)
            for source in enabled
        ],
        return_exceptions=True,
    )
    records: list[MatchIntelligence] = []
    warnings: list[str] = []
    for source, result in zip(enabled, results, strict=True):
        if isinstance(result, BaseException):
            warnings.append(f"结构化数据源 {source.id} 获取失败：{result}")
        elif result is None:
            warnings.append(
                f"结构化数据源 {source.id} 没有 match_id={fixture.match_id} 的数据"
            )
        else:
            records.append(result)
    return (_merge(records) if records else None), warnings


def data_completeness(data: MatchIntelligence | None) -> tuple[float, list[str]]:
    if data is None:
        return 0.0, [
            "近期赛果",
            "FIFA/Elo 排名",
            "进球/失球/xG/射门",
            "首发阵容",
            "伤停与停赛",
            "主客场表现",
            "历史交锋",
            "赔率变化",
            "天气",
            "场地",
            "旅行距离与时差",
        ]
    checks = {
        "近期赛果": bool(data.home_metrics.recent_matches)
        and bool(data.away_metrics.recent_matches),
        "FIFA/Elo 排名": any(
            value is not None
            for value in (
                data.home_metrics.fifa_rank,
                data.home_metrics.elo_rating,
                data.away_metrics.fifa_rank,
                data.away_metrics.elo_rating,
            )
        ),
        "进球/失球/xG/射门": all(
            value is not None
            for value in (
                data.home_metrics.recent_goals_for_per_game,
                data.home_metrics.recent_goals_against_per_game,
                data.home_metrics.recent_xg_for_per_game,
                data.home_metrics.recent_xg_against_per_game,
                data.away_metrics.recent_goals_for_per_game,
                data.away_metrics.recent_goals_against_per_game,
                data.away_metrics.recent_xg_for_per_game,
                data.away_metrics.recent_xg_against_per_game,
            )
        )
        and all(
            match.shots_for is not None and match.shots_against is not None
            for match in (
                data.home_metrics.recent_matches
                + data.away_metrics.recent_matches
            )
        ),
        "首发阵容": bool(
            data.home_availability.confirmed_lineup
            or data.home_availability.expected_lineup
        )
        and bool(
            data.away_availability.confirmed_lineup
            or data.away_availability.expected_lineup
        ),
        "伤停与停赛": bool(data.home_availability.players)
        or bool(data.away_availability.players),
        "主客场表现": data.home_metrics.home_or_away_win_rate is not None
        and data.away_metrics.home_or_away_win_rate is not None,
        "历史交锋": bool(data.head_to_head),
        "赔率变化": len(data.odds) >= 2,
        "天气": data.weather is not None,
        "场地": data.venue is not None,
        "旅行距离与时差": data.travel is not None,
    }
    missing = [name for name, present in checks.items() if not present]
    return (len(checks) - len(missing)) / len(checks), missing
