from datetime import UTC, datetime, timedelta

from world_cup_forecast.models import (
    ConsensusForecast,
    Fixture,
    MatchPrediction,
)
from world_cup_forecast.report import render_markdown


def test_report_is_chinese_and_only_describes_next_match():
    fixture = Fixture(
        match_id="m1",
        kickoff=datetime.now(UTC) + timedelta(hours=2),
        stage="小组赛",
        home_team="墨西哥",
        away_team="南非",
    )
    consensus = ConsensusForecast(
        generated_at=datetime.now(UTC),
        champion_predictions=[],
        match_predictions=[
            MatchPrediction(
                match_id="m1",
                home_team="墨西哥",
                away_team="南非",
                home_goals=2,
                away_goals=1,
                home_win_probability=0.55,
                draw_probability=0.25,
                away_win_probability=0.2,
            )
        ],
        agreement_score=0.8,
        agent_count=4,
    )

    report = render_markdown(
        "世界杯",
        "Asia/Shanghai",
        consensus,
        [],
        [],
        fixture,
        None,
    )

    assert "下一场比赛预测报告" in report
    assert "预测胜方：**墨西哥**" in report
    assert "该结果综合概率：**55.0%**" in report
    assert "Agent 结论一致率：80.0%" in report
    assert "墨西哥 2-1 南非" in report
    assert "冠军" not in report
