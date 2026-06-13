"""
国家队比赛特征工程。

功能：
1. 读取国际比赛历史数据；
2. 清理未来赛程和无比分比赛；
3. 按时间顺序逐场处理；
4. 为每场比赛生成赛前特征；
5. 计算球队近期状态、Elo、休息天数和历史交锋；
6. 生成可用于机器学习训练的 CSV 文件。

注意：
每场比赛的特征只能使用该场比赛之前的数据，
避免发生未来数据泄漏。
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque

import numpy as np
import pandas as pd

from world_cup_teams import (
    WORLD_CUP_2026_TEAMS,
    is_world_cup_2026_team,
    normalize_team_name,
    validate_world_cup_team_count,
)

# ============================================================
# 一、路径配置
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "international_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FULL_FEATURE_FILE = (
    OUTPUT_DIR
    / "match_features_full.csv"
)

TRAIN_FEATURE_FILE = (
    OUTPUT_DIR
    / "match_features_train.csv"
)

WORLD_CUP_FEATURE_FILE = (
    OUTPUT_DIR
    / "world_cup_2026_features_train.csv"
)

WORLD_CUP_TRAIN_FILE = (
    OUTPUT_DIR
    / "world_cup_2026_train.csv"
)

WORLD_CUP_VALIDATION_FILE = (
    OUTPUT_DIR
    / "world_cup_2026_validation.csv"
)

WORLD_CUP_TEST_FILE = (
    OUTPUT_DIR
    / "world_cup_2026_test.csv"
)


# ============================================================
# 二、全局参数
# ============================================================

DEFAULT_ELO = 1500.0

MAX_HISTORY_SIZE = 20

MAX_H2H_HISTORY_SIZE = 10

MIN_HISTORY_MATCHES = 5

TRAIN_END_DATE = "2024-01-01"

VALIDATION_END_DATE = "2025-01-01"

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.85,
    "UEFA Euro": 0.90,
    "Copa América": 0.90,
    "AFC Asian Cup": 0.85,
    "African Cup of Nations": 0.85,
    "CONCACAF Gold Cup": 0.80,
    "UEFA Nations League": 0.75,
    "Friendly": 0.35,
}


# ============================================================
# 三、读取和清洗数据
# ============================================================

def load_matches() -> pd.DataFrame:
    """
    读取国家队历史比赛数据。
    """

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"没有找到历史比赛数据：{RAW_DATA_FILE}"
        )

    matches = pd.read_csv(RAW_DATA_FILE)

    return matches


def normalize_boolean(value: Any) -> bool:
    """
    将不同格式的布尔值转换为 Python bool。

    支持：
    True
    False
    "TRUE"
    "FALSE"
    1
    0
    """

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value_text = str(value).strip().lower()

    return value_text in {
        "true",
        "1",
        "yes",
        "y"
    }


def clean_matches(
    matches: pd.DataFrame
) -> pd.DataFrame:
    """
    清理历史比赛数据。

    删除：
    1. 日期为空的数据；
    2. 球队名称为空的数据；
    3. 比分为空的未来比赛；
    4. 无法转换为数字的比分。
    """

    cleaned = matches.copy()

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce"
    )

    cleaned["home_score"] = pd.to_numeric(
        cleaned["home_score"],
        errors="coerce"
    )

    cleaned["away_score"] = pd.to_numeric(
        cleaned["away_score"],
        errors="coerce"
    )

    cleaned = cleaned.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score"
        ]
    ).copy()

    cleaned["home_score"] = (
        cleaned["home_score"].astype(int)
    )

    cleaned["away_score"] = (
        cleaned["away_score"].astype(int)
    )

    cleaned["home_team"] = (
        cleaned["home_team"]
        .astype(str)
        .map(normalize_team_name)
    )

    cleaned["away_team"] = (
        cleaned["away_team"]
        .astype(str)
        .map(normalize_team_name)
    )

    cleaned["neutral"] = (
        cleaned["neutral"]
        .apply(normalize_boolean)
    )

    cleaned = cleaned.sort_values(
        by="date",
        ascending=True
    ).reset_index(drop=True)

    return cleaned


# ============================================================
# 四、比赛结果相关函数
# ============================================================

def get_result_label(
    home_score: int,
    away_score: int
) -> int:
    """
    将比赛结果转换为机器学习标签。

    2：主队胜
    1：平局
    0：客队胜
    """

    if home_score > away_score:
        return 2

    if home_score < away_score:
        return 0

    return 1


def get_result_text(
    home_score: int,
    away_score: int
) -> str:
    """
    获取可读的比赛结果名称。
    """

    if home_score > away_score:
        return "HOME_WIN"

    if home_score < away_score:
        return "AWAY_WIN"

    return "DRAW"


def get_team_points(
    goals_for: int,
    goals_against: int
) -> int:
    """
    根据一支球队视角计算积分。

    胜：3分
    平：1分
    负：0分
    """

    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


# ============================================================
# 五、球队近期状态特征
# ============================================================

def safe_mean(values: list[float]) -> float:
    """
    安全计算平均值。

    没有历史数据时返回 0。
    """

    if not values:
        return 0.0

    return float(np.mean(values))


def calculate_recent_features(
    history: Deque[dict[str, Any]],
    window: int
) -> dict[str, float]:
    """
    计算球队最近若干场比赛的特征。

    history 中每条记录都是球队视角：

    {
        "date": 比赛时间,
        "goals_for": 本队进球,
        "goals_against": 本队失球,
        "points": 本队积分,
        "win": 是否获胜,
        "draw": 是否平局,
        "loss": 是否失败
    }
    """

    recent_matches = list(history)[-window:]

    match_count = len(recent_matches)

    if match_count == 0:
        return {
            "matches": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "avg_goal_difference": 0.0,
            "avg_points": 0.0,
            "clean_sheet_rate": 0.0,
            "scoring_rate": 0.0
        }

    wins = sum(
        item["win"]
        for item in recent_matches
    )

    draws = sum(
        item["draw"]
        for item in recent_matches
    )

    losses = sum(
        item["loss"]
        for item in recent_matches
    )

    goals_for = [
        item["goals_for"]
        for item in recent_matches
    ]

    goals_against = [
        item["goals_against"]
        for item in recent_matches
    ]

    points = [
        item["points"]
        for item in recent_matches
    ]

    goal_differences = [
        item["goals_for"]
        - item["goals_against"]
        for item in recent_matches
    ]

    clean_sheets = sum(
        item["goals_against"] == 0
        for item in recent_matches
    )

    scoring_matches = sum(
        item["goals_for"] > 0
        for item in recent_matches
    )

    return {
        "matches": float(match_count),

        "win_rate": round(
            wins / match_count,
            6
        ),

        "draw_rate": round(
            draws / match_count,
            6
        ),

        "loss_rate": round(
            losses / match_count,
            6
        ),

        "avg_goals_for": round(
            safe_mean(goals_for),
            6
        ),

        "avg_goals_against": round(
            safe_mean(goals_against),
            6
        ),

        "avg_goal_difference": round(
            safe_mean(goal_differences),
            6
        ),

        "avg_points": round(
            safe_mean(points),
            6
        ),

        "clean_sheet_rate": round(
            clean_sheets / match_count,
            6
        ),

        "scoring_rate": round(
            scoring_matches / match_count,
            6
        )
    }


def add_recent_features(
    feature_row: dict[str, Any],
    prefix: str,
    history: Deque[dict[str, Any]]
) -> None:
    """
    把最近5场、10场特征添加到比赛特征中。

    prefix：
    home 或 away。
    """

    for window in (5, 10):
        recent = calculate_recent_features(
            history=history,
            window=window
        )

        for feature_name, feature_value in recent.items():
            column_name = (
                f"{prefix}_{feature_name}_{window}"
            )

            feature_row[column_name] = feature_value


# ============================================================
# 六、Elo 评分
# ============================================================

def calculate_expected_score(
    rating_a: float,
    rating_b: float
) -> float:
    """
    计算 Elo 体系中球队 A 的预期得分。
    """

    return 1.0 / (
        1.0
        + 10.0 ** (
            (rating_b - rating_a) / 400.0
        )
    )


def get_elo_actual_score(
    home_score: int,
    away_score: int
) -> float:
    """
    将实际比赛结果转换为 Elo 得分。

    胜：1
    平：0.5
    负：0
    """

    if home_score > away_score:
        return 1.0

    if home_score < away_score:
        return 0.0

    return 0.5


def get_elo_k_factor(
    tournament: str
) -> float:
    """
    根据比赛类型设置 Elo 更新幅度。

    比赛越重要，K 值越大。
    """

    tournament_text = (
        str(tournament).lower()
    )

    if "world cup" in tournament_text:
        return 40.0

    if "qualification" in tournament_text:
        return 30.0

    if "qualifier" in tournament_text:
        return 30.0

    if "nations league" in tournament_text:
        return 25.0

    if "friendly" in tournament_text:
        return 15.0

    return 20.0


def get_tournament_weight(
    tournament: str,
) -> float:
    """
    Return the configured importance weight for a tournament.

    The first version only exposes the value for future feature experiments.
    Existing Elo calculations deliberately remain unchanged.
    """

    normalized = str(tournament).strip().casefold()
    for name, weight in TOURNAMENT_WEIGHTS.items():
        if normalized == name.casefold():
            return weight
    return 0.60


def update_elo_ratings(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool
) -> tuple[float, float]:
    """
    根据比赛结果更新双方 Elo。

    非中立场时，为主队增加约 80 点主场优势。
    """

    home_advantage = (
        0.0 if neutral else 80.0
    )

    adjusted_home_elo = (
        home_elo + home_advantage
    )

    expected_home = calculate_expected_score(
        adjusted_home_elo,
        away_elo
    )

    expected_away = 1.0 - expected_home

    actual_home = get_elo_actual_score(
        home_score,
        away_score
    )

    actual_away = 1.0 - actual_home

    k_factor = get_elo_k_factor(
        tournament
    )

    new_home_elo = (
        home_elo
        + k_factor
        * (actual_home - expected_home)
    )

    new_away_elo = (
        away_elo
        + k_factor
        * (actual_away - expected_away)
    )

    return (
        round(new_home_elo, 6),
        round(new_away_elo, 6)
    )


# ============================================================
# 七、休息时间
# ============================================================

def calculate_rest_days(
    match_date: pd.Timestamp,
    last_match_date: pd.Timestamp | None
) -> int:
    """
    计算球队距离上一场比赛的休息天数。
    """

    if last_match_date is None:
        return 30

    rest_days = (
        match_date - last_match_date
    ).days

    # 限制极端值，避免长期未比赛产生几百天
    return max(
        0,
        min(rest_days, 60)
    )


# ============================================================
# 八、历史交锋特征
# ============================================================

def get_team_pair_key(
    team_a: str,
    team_b: str
) -> tuple[str, str]:
    """
    给两支球队生成固定顺序的键。
    """

    return tuple(
        sorted([team_a, team_b])
    )


def calculate_h2h_features(
    h2h_history: Deque[dict[str, Any]],
    home_team: str,
    away_team: str
) -> dict[str, float]:
    """
    计算双方历史交锋特征。

    所有数据均从当前主队视角计算。
    """

    records = list(h2h_history)[-5:]

    match_count = len(records)

    if match_count == 0:
        return {
            "h2h_matches": 0.0,
            "h2h_home_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_home_avg_goals": 0.0,
            "h2h_away_avg_goals": 0.0
        }

    home_wins = 0
    away_wins = 0
    draws = 0

    home_goals: list[int] = []
    away_goals: list[int] = []

    for record in records:
        if record["team_a"] == home_team:
            current_home_score = (
                record["team_a_score"]
            )
            current_away_score = (
                record["team_b_score"]
            )
        else:
            current_home_score = (
                record["team_b_score"]
            )
            current_away_score = (
                record["team_a_score"]
            )

        home_goals.append(
            current_home_score
        )

        away_goals.append(
            current_away_score
        )

        if current_home_score > current_away_score:
            home_wins += 1
        elif current_home_score < current_away_score:
            away_wins += 1
        else:
            draws += 1

    return {
        "h2h_matches": float(match_count),

        "h2h_home_win_rate": round(
            home_wins / match_count,
            6
        ),

        "h2h_draw_rate": round(
            draws / match_count,
            6
        ),

        "h2h_away_win_rate": round(
            away_wins / match_count,
            6
        ),

        "h2h_home_avg_goals": round(
            safe_mean(home_goals),
            6
        ),

        "h2h_away_avg_goals": round(
            safe_mean(away_goals),
            6
        )
    }


# ============================================================
# 九、赛事类型特征
# ============================================================

def get_tournament_features(
    tournament: str
) -> dict[str, int]:
    """
    将赛事名称转换为简单的数值特征。
    """

    tournament_text = (
        str(tournament).lower()
    )

    return {
        "is_world_cup": int(
            "fifa world cup" in tournament_text
            and "qualification" not in tournament_text
        ),

        "is_world_cup_qualifier": int(
            "world cup qualification"
            in tournament_text
        ),

        "is_friendly": int(
            "friendly" in tournament_text
        ),

        "is_nations_league": int(
            "nations league"
            in tournament_text
        ),

        "is_confederation_tournament": int(
            any(
                keyword in tournament_text
                for keyword in [
                    "uefa euro",
                    "copa américa",
                    "african cup",
                    "asian cup",
                    "gold cup"
                ]
            )
        )
    }


# ============================================================
# 十、生成训练特征
# ============================================================

def build_match_features(
    matches: pd.DataFrame
) -> pd.DataFrame:
    """
    按时间顺序生成每场比赛的赛前特征。

    关键逻辑：

    1. 先读取球队在比赛前的历史数据；
    2. 生成当前比赛特征；
    3. 再用当前比赛结果更新历史数据。

    这样不会把当前比赛结果泄漏到当前特征中。
    """

    team_histories: dict[
        str,
        Deque[dict[str, Any]]
    ] = defaultdict(
        lambda: deque(
            maxlen=MAX_HISTORY_SIZE
        )
    )

    team_elos: dict[str, float] = defaultdict(
        lambda: DEFAULT_ELO
    )

    last_match_dates: dict[
        str,
        pd.Timestamp
    ] = {}

    h2h_histories: dict[
        tuple[str, str],
        Deque[dict[str, Any]]
    ] = defaultdict(
        lambda: deque(
            maxlen=MAX_H2H_HISTORY_SIZE
        )
    )

    feature_rows: list[dict[str, Any]] = []

    for _, match in matches.iterrows():
        match_date = match["date"]

        home_team = str(
            match["home_team"]
        )

        away_team = str(
            match["away_team"]
        )

        home_score = int(
            match["home_score"]
        )

        away_score = int(
            match["away_score"]
        )

        tournament = str(
            match["tournament"]
        )

        neutral = bool(
            match["neutral"]
        )

        # ----------------------------------------------------
        # 第一步：读取比赛前状态
        # ----------------------------------------------------

        home_history = team_histories[
            home_team
        ]

        away_history = team_histories[
            away_team
        ]

        home_elo = team_elos[
            home_team
        ]

        away_elo = team_elos[
            away_team
        ]

        home_rest_days = calculate_rest_days(
            match_date,
            last_match_dates.get(home_team)
        )

        away_rest_days = calculate_rest_days(
            match_date,
            last_match_dates.get(away_team)
        )

        pair_key = get_team_pair_key(
            home_team,
            away_team
        )

        h2h_features = calculate_h2h_features(
            h2h_history=h2h_histories[pair_key],
            home_team=home_team,
            away_team=away_team
        )

        # ----------------------------------------------------
        # 第二步：构建当前比赛特征
        # ----------------------------------------------------

        feature_row: dict[str, Any] = {
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "city": match.get("city"),
            "country": match.get("country"),

            "neutral": int(neutral),

            "home_elo": round(
                home_elo,
                6
            ),

            "away_elo": round(
                away_elo,
                6
            ),

            "elo_difference": round(
                home_elo - away_elo,
                6
            ),

            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,

            "rest_days_difference": (
                home_rest_days
                - away_rest_days
            ),

            "home_score": home_score,
            "away_score": away_score,

            "target": get_result_label(
                home_score,
                away_score
            ),

            "result": get_result_text(
                home_score,
                away_score
            )
        }

        add_recent_features(
            feature_row=feature_row,
            prefix="home",
            history=home_history
        )

        add_recent_features(
            feature_row=feature_row,
            prefix="away",
            history=away_history
        )

        feature_row.update(
            h2h_features
        )

        feature_row.update(
            get_tournament_features(
                tournament
            )
        )

        feature_rows.append(
            feature_row
        )

        # ----------------------------------------------------
        # 第三步：当前比赛结束后，更新历史状态
        # ----------------------------------------------------

        home_points = get_team_points(
            home_score,
            away_score
        )

        away_points = get_team_points(
            away_score,
            home_score
        )

        home_history.append({
            "date": match_date,
            "opponent": away_team,
            "goals_for": home_score,
            "goals_against": away_score,
            "points": home_points,
            "win": int(home_points == 3),
            "draw": int(home_points == 1),
            "loss": int(home_points == 0)
        })

        away_history.append({
            "date": match_date,
            "opponent": home_team,
            "goals_for": away_score,
            "goals_against": home_score,
            "points": away_points,
            "win": int(away_points == 3),
            "draw": int(away_points == 1),
            "loss": int(away_points == 0)
        })

        new_home_elo, new_away_elo = (
            update_elo_ratings(
                home_elo=home_elo,
                away_elo=away_elo,
                home_score=home_score,
                away_score=away_score,
                tournament=tournament,
                neutral=neutral
            )
        )

        team_elos[home_team] = new_home_elo
        team_elos[away_team] = new_away_elo

        last_match_dates[home_team] = match_date
        last_match_dates[away_team] = match_date

        sorted_team_a, sorted_team_b = pair_key

        if home_team == sorted_team_a:
            team_a_score = home_score
            team_b_score = away_score
        else:
            team_a_score = away_score
            team_b_score = home_score

        h2h_histories[pair_key].append({
            "date": match_date,
            "team_a": sorted_team_a,
            "team_b": sorted_team_b,
            "team_a_score": team_a_score,
            "team_b_score": team_b_score
        })

    return pd.DataFrame(feature_rows)


# ============================================================
# 十一、筛选可训练数据
# ============================================================

def filter_training_features(
    features: pd.DataFrame,
    min_history_matches: int = MIN_HISTORY_MATCHES
) -> pd.DataFrame:
    """
    过滤历史数据不足的比赛。

    例如：
    主队和客队在比赛前至少都已经有5场历史比赛。
    """

    filtered = features[
        (
            features["home_matches_5"]
            >= min_history_matches
        )
        &
        (
            features["away_matches_5"]
            >= min_history_matches
        )
    ].copy()

    return filtered.reset_index(drop=True)


def filter_world_cup_training_matches(
    data: pd.DataFrame,
    exclude_friendlies: bool = True,
) -> pd.DataFrame:
    """
    Return matches played by two configured 2026 World Cup teams.

    Team names are normalized after copying the input, so the caller's
    DataFrame is never modified. Friendly target matches can be excluded
    while still remaining available during the earlier state calculation.
    """

    required_columns = {
        "home_team",
        "away_team",
        "tournament",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "世界杯筛选缺少字段："
            f"{sorted(missing_columns)}"
        )

    filtered = data.copy()
    filtered["home_team"] = (
        filtered["home_team"]
        .astype(str)
        .map(normalize_team_name)
    )
    filtered["away_team"] = (
        filtered["away_team"]
        .astype(str)
        .map(normalize_team_name)
    )

    team_mask = (
        filtered["home_team"].map(is_world_cup_2026_team)
        & filtered["away_team"].map(is_world_cup_2026_team)
    )
    filtered = filtered[team_mask].copy()

    if exclude_friendlies:
        tournament_names = (
            filtered["tournament"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        filtered = filtered[
            tournament_names != "friendly".casefold()
        ].copy()

    return filtered.reset_index(drop=True)


def split_dataset_by_date(
    data: pd.DataFrame,
    train_end_date: str,
    validation_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a football dataset chronologically without random shuffling.

    Training rows are before ``train_end_date``; validation rows begin at
    that boundary and end before ``validation_end_date``; test rows begin
    at the validation boundary.
    """

    if "date" not in data.columns:
        raise ValueError("时间拆分缺少 date 字段")

    dated = data.copy()
    dated["date"] = pd.to_datetime(
        dated["date"],
        errors="coerce",
    )
    if dated["date"].isna().any():
        raise ValueError("时间拆分发现无法解析的 date")

    train_boundary = pd.Timestamp(train_end_date)
    validation_boundary = pd.Timestamp(validation_end_date)
    if train_boundary >= validation_boundary:
        raise ValueError(
            "train_end_date 必须早于 validation_end_date"
        )

    dated = dated.sort_values("date").reset_index(drop=True)
    train = dated[
        dated["date"] < train_boundary
    ].copy()
    validation = dated[
        (dated["date"] >= train_boundary)
        & (dated["date"] < validation_boundary)
    ].copy()
    test = dated[
        dated["date"] >= validation_boundary
    ].copy()

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _validate_world_cup_rows(data: pd.DataFrame) -> None:
    """Raise if a supposedly World Cup-only dataset contains another team."""

    teams = set(data["home_team"]) | set(data["away_team"])
    unexpected = sorted(teams - WORLD_CUP_2026_TEAMS)
    if unexpected:
        raise ValueError(
            "世界杯专用数据中发现非参赛球队："
            f"{unexpected}"
        )


def _complete_training_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with sufficient history and complete model fields."""

    history_filtered = filter_training_features(data)
    optional_columns = {"city", "country"}
    required_columns = [
        column
        for column in history_filtered.columns
        if column not in optional_columns
    ]
    return (
        history_filtered
        .dropna(subset=required_columns)
        .reset_index(drop=True)
    )


def _print_dataset_summary(
    name: str,
    data: pd.DataFrame,
) -> None:
    """Print chronological split size, date range, labels, and team count."""

    print(f"\n{name}：")
    print(f"比赛数量：{len(data)}")
    if data.empty:
        print("最早日期：无")
        print("最晚日期：无")
    else:
        print(f"最早日期：{data['date'].min().date()}")
        print(f"最晚日期：{data['date'].max().date()}")
    counts = data["target"].value_counts().to_dict()
    print(f"主胜数量：{int(counts.get(2, 0))}")
    print(f"平局数量：{int(counts.get(1, 0))}")
    print(f"客胜数量：{int(counts.get(0, 0))}")
    teams = set(data["home_team"]) | set(data["away_team"])
    print(f"球队数量：{len(teams)}")


# ============================================================
# 十二、保存结果
# ============================================================

def save_features(
    full_features: pd.DataFrame,
    train_features: pd.DataFrame
) -> None:
    """
    保存完整特征和可训练特征。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    full_features.to_csv(
        FULL_FEATURE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    train_features.to_csv(
        TRAIN_FEATURE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"完整特征已保存：{FULL_FEATURE_FILE}"
    )

    print(
        f"训练特征已保存：{TRAIN_FEATURE_FILE}"
    )


def save_world_cup_features(
    world_cup_features: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Save the World Cup-only full training pool and chronological splits."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        WORLD_CUP_FEATURE_FILE: world_cup_features,
        WORLD_CUP_TRAIN_FILE: train,
        WORLD_CUP_VALIDATION_FILE: validation,
        WORLD_CUP_TEST_FILE: test,
    }
    for path, frame in outputs.items():
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"世界杯数据已保存：{path}")


# ============================================================
# 十三、主程序
# ============================================================

def main() -> None:
    """
    特征工程主流程。
    """

    validate_world_cup_team_count()
    print("开始读取历史比赛数据……")

    raw_matches = load_matches()

    print(
        f"原始比赛数量：{len(raw_matches)}"
    )

    print("开始清洗比赛数据……")

    clean_data = clean_matches(
        raw_matches
    )

    print(
        f"已结束比赛数量：{len(clean_data)}"
    )

    print("开始生成赛前特征……")

    full_features = build_match_features(
        clean_data
    )

    print(
        f"完整特征数量：{len(full_features)}"
    )

    train_features = filter_training_features(
        full_features
    )

    print(
        f"可训练特征数量：{len(train_features)}"
    )

    save_features(
        full_features=full_features,
        train_features=train_features
    )

    world_cup_all = filter_world_cup_training_matches(
        full_features,
        exclude_friendlies=False,
    )
    print(
        "世界杯球队之间的比赛数量："
        f"{len(world_cup_all)}"
    )

    world_cup_non_friendly = filter_world_cup_training_matches(
        full_features,
        exclude_friendlies=True,
    )
    print(
        "排除友谊赛后的数量："
        f"{len(world_cup_non_friendly)}"
    )

    world_cup_features = _complete_training_rows(
        world_cup_non_friendly
    )
    _validate_world_cup_rows(world_cup_features)
    print(
        "世界杯专用训练样本数量："
        f"{len(world_cup_features)}"
    )
    teams = (
        set(world_cup_features["home_team"])
        | set(world_cup_features["away_team"])
    )
    print(f"涉及的球队数量：{len(teams)}")
    print(
        "标签分布："
        f"{world_cup_features['target'].value_counts().sort_index().to_dict()}"
    )

    train, validation, test = split_dataset_by_date(
        world_cup_features,
        train_end_date=TRAIN_END_DATE,
        validation_end_date=VALIDATION_END_DATE,
    )
    _print_dataset_summary("训练集", train)
    _print_dataset_summary("验证集", validation)
    _print_dataset_summary("测试集", test)
    save_world_cup_features(
        world_cup_features,
        train,
        validation,
        test,
    )

    print("\n输出文件路径：")
    for output_path in (
        FULL_FEATURE_FILE,
        TRAIN_FEATURE_FILE,
        WORLD_CUP_FEATURE_FILE,
        WORLD_CUP_TRAIN_FILE,
        WORLD_CUP_VALIDATION_FILE,
        WORLD_CUP_TEST_FILE,
    ):
        print(output_path)

    print("\n世界杯专用训练标签分布：")

    print(
        world_cup_features["result"]
        .value_counts()
    )

    print("\n最近5条世界杯专用训练数据：")

    preview_columns = [
        "date",
        "home_team",
        "away_team",
        "home_elo",
        "away_elo",
        "elo_difference",
        "home_win_rate_5",
        "away_win_rate_5",
        "home_avg_goals_for_5",
        "away_avg_goals_for_5",
        "neutral",
        "result",
        "target"
    ]

    print(
        world_cup_features[
            preview_columns
        ]
        .tail(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
