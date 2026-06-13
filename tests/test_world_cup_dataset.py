from __future__ import annotations

import pandas as pd

from match_feature_engineering import (
    filter_world_cup_training_matches,
    split_dataset_by_date,
)
from world_cup_teams import (
    is_world_cup_2026_team,
    normalize_team_name,
    validate_world_cup_team_count,
)


def test_world_cup_team_aliases_and_count() -> None:
    validate_world_cup_team_count()
    assert normalize_team_name(" czechia ") == "Czech Republic"
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("curacao") == "Curaçao"
    assert is_world_cup_2026_team("Korea Republic")
    assert not is_world_cup_2026_team("Indonesia")


def test_world_cup_filter_does_not_modify_input() -> None:
    data = pd.DataFrame(
        [
            {
                "date": "2023-01-01",
                "home_team": " USA ",
                "away_team": "Mexico",
                "tournament": "FIFA World Cup",
            },
            {
                "date": "2023-01-02",
                "home_team": "Brazil",
                "away_team": "Morocco",
                "tournament": "Friendly",
            },
            {
                "date": "2023-01-03",
                "home_team": "Indonesia",
                "away_team": "Mexico",
                "tournament": "FIFA World Cup qualification",
            },
        ]
    )
    original = data.copy(deep=True)

    result = filter_world_cup_training_matches(data)

    pd.testing.assert_frame_equal(data, original)
    assert len(result) == 1
    assert result.iloc[0]["home_team"] == "United States"


def test_dataset_split_is_chronological() -> None:
    data = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2023-12-31",
                "2024-01-01",
            ],
            "home_team": ["Brazil"] * 3,
            "away_team": ["Argentina"] * 3,
            "tournament": ["FIFA World Cup"] * 3,
        }
    )

    train, validation, test = split_dataset_by_date(
        data,
        "2024-01-01",
        "2025-01-01",
    )

    assert train["date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-12-31"]
    assert validation["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01"]
    assert test["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01"]
