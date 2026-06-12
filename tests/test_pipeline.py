from datetime import UTC, datetime, timedelta

import pytest

from world_cup_forecast.models import Fixture
from world_cup_forecast.pipeline import select_next_fixture


def _fixture(match_id: str, kickoff: datetime, status: str = "scheduled") -> Fixture:
    return Fixture(
        match_id=match_id,
        kickoff=kickoff,
        stage="Group stage",
        home_team=f"{match_id}-home",
        away_team=f"{match_id}-away",
        status=status,
    )


def test_select_next_fixture_uses_earliest_future_scheduled_match():
    now = datetime(2026, 6, 12, 0, 0, tzinfo=UTC)
    fixtures = [
        _fixture("later", now + timedelta(hours=8)),
        _fixture("past", now - timedelta(hours=1)),
        _fixture("next", now + timedelta(hours=2)),
        _fixture("finished", now + timedelta(hours=1), status="finished"),
    ]

    assert select_next_fixture(fixtures, now).match_id == "next"


def test_select_next_fixture_fails_when_schedule_is_stale():
    now = datetime(2026, 6, 12, 0, 0, tzinfo=UTC)

    with pytest.raises(RuntimeError, match="没有找到"):
        select_next_fixture([_fixture("past", now - timedelta(hours=1))], now)
