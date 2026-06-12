import asyncio
from pathlib import Path

from world_cup_forecast.models import DataSourceConfig, Fixture
from world_cup_forecast.structured_data import (
    collect_structured_data,
    data_completeness,
)


def test_local_structured_data_has_all_factor_groups():
    root = Path(__file__).resolve().parents[1]
    fixture = Fixture.model_validate(
        {
            "match_id": "example-001",
            "kickoff": "2026-06-12T20:00:00-05:00",
            "stage": "Group stage",
            "home_team": "Mexico",
            "away_team": "South Africa",
        }
    )
    data, warnings = asyncio.run(
        collect_structured_data(
            [
                DataSourceConfig(
                    id="local",
                    type="local_json",
                    path="data/match_intelligence.example.json",
                )
            ],
            fixture,
            root,
            10,
        )
    )

    completeness, missing = data_completeness(data)
    assert warnings == []
    assert data is not None
    assert completeness == 1
    assert missing == []
