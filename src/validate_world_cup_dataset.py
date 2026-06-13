"""Validate the generated 2026 World Cup-only training dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_cup_teams import (
    WORLD_CUP_2026_TEAMS,
    normalize_team_name,
    validate_world_cup_team_count,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "world_cup_2026_features_train.csv"
)

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "home_elo",
    "away_elo",
    "target",
    "result",
}


def _expected_result(
    home_score: int,
    away_score: int,
) -> tuple[int, str]:
    """Return the expected numeric and text labels for a scoreline."""

    if home_score > away_score:
        return 2, "HOME_WIN"
    if home_score < away_score:
        return 0, "AWAY_WIN"
    return 1, "DRAW"


def validate_dataset(
    dataset_file: Path = DATASET_FILE,
) -> pd.DataFrame:
    """Load and validate the World Cup-only feature dataset."""

    validate_world_cup_team_count()
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"没有找到世界杯训练数据：{dataset_file}"
        )

    data = pd.read_csv(dataset_file)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            "世界杯训练数据缺少字段："
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("世界杯训练数据为空")

    validated = data.copy()
    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="coerce",
    )
    validated["home_team"] = (
        validated["home_team"]
        .astype(str)
        .map(normalize_team_name)
    )
    validated["away_team"] = (
        validated["away_team"]
        .astype(str)
        .map(normalize_team_name)
    )

    if validated[list(REQUIRED_COLUMNS)].isna().any().any():
        null_counts = (
            validated[list(REQUIRED_COLUMNS)]
            .isna()
            .sum()
        )
        raise ValueError(
            "关键字段存在空值："
            f"{null_counts[null_counts > 0].to_dict()}"
        )

    teams = (
        set(validated["home_team"])
        | set(validated["away_team"])
    )
    unexpected = sorted(teams - WORLD_CUP_2026_TEAMS)
    if unexpected:
        raise ValueError(
            "发现非 2026 世界杯球队："
            f"{unexpected}"
        )

    friendly_mask = (
        validated["tournament"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == "friendly".casefold()
    )
    if friendly_mask.any():
        raise ValueError(
            "世界杯训练数据仍包含 Friendly，"
            f"数量：{int(friendly_mask.sum())}"
        )

    if not validated["date"].is_monotonic_increasing:
        raise ValueError("世界杯训练数据没有按日期升序排列")

    for column in ("home_score", "away_score", "target"):
        numeric = pd.to_numeric(
            validated[column],
            errors="coerce",
        )
        if numeric.isna().any():
            raise ValueError(f"{column} 存在非法数字")
        if not (numeric == numeric.astype(int)).all():
            raise ValueError(f"{column} 必须为整数")
        validated[column] = numeric.astype(int)

    if (
        (validated["home_score"] < 0)
        | (validated["away_score"] < 0)
    ).any():
        raise ValueError("比分不能为负数")

    invalid_labels: list[str] = []
    for index, row in validated.iterrows():
        expected_target, expected_result = _expected_result(
            int(row["home_score"]),
            int(row["away_score"]),
        )
        if (
            int(row["target"]) != expected_target
            or str(row["result"]) != expected_result
        ):
            invalid_labels.append(
                f"row={index}, "
                f"{row['home_team']} vs {row['away_team']}, "
                f"score={row['home_score']}-{row['away_score']}, "
                f"target={row['target']}, result={row['result']}"
            )
    if invalid_labels:
        raise ValueError(
            "target/result 与真实比分不一致："
            + " | ".join(invalid_labels[:10])
        )

    return validated


def main() -> None:
    """Validate the dataset and print tournament and team distributions."""

    data = validate_dataset()
    print(f"验证文件：{DATASET_FILE}")
    print(f"比赛数量：{len(data)}")
    print(f"球队数量：{len(set(data['home_team']) | set(data['away_team']))}")
    print("非世界杯球队：无")
    print("Friendly：无")
    print("\n赛事分布：")
    print(data["tournament"].value_counts().to_string())

    team_counts = pd.concat(
        [
            data["home_team"],
            data["away_team"],
        ],
        ignore_index=True,
    ).value_counts()
    print("\n球队比赛数量排名：")
    print(team_counts.to_string())
    print("\n世界杯专用数据集验证通过")


if __name__ == "__main__":
    main()
