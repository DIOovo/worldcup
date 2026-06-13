
"""
大模型 + 泊松模型足球预测历史回测。

功能：
1. 读取无未来数据泄漏的历史比赛特征；
2. 只保留2026世界杯参赛球队之间的非友谊赛；
3. 选择指定日期范围或最近若干场比赛；
4. 将每场比赛的赛前特征发送给预测器；
5. 保存主胜、平局、客胜概率；
6. 保存预期进球、最可能比分和Top-N比分；
7. 对比真实比赛结果和真实比分；
8. 计算Accuracy、Log Loss、Brier Score；
9. 计算精确比分命中率、Top-3/Top-5比分命中率；
10. 计算进球MAE、大小球和双方进球指标；
11. 支持中断后继续执行；
12. 防止不同字段结构写入同一CSV。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from llm_match_predictor import LLMMatchPredictor
from world_cup_teams import (
    is_world_cup_2026_team,
    normalize_team_name,
)


# ============================================================
# 一、文件路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "world_cup_2026_features_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "backtest"
)

# 使用全新的版本名，避免读取之前列结构不一致的CSV
BACKTEST_VERSION = "world_cup_hybrid_poisson_v3"

BACKTEST_RESULT_FILE = (
    OUTPUT_DIR
    / f"llm_backtest_results_{BACKTEST_VERSION}.csv"
)

BACKTEST_SUMMARY_FILE = (
    OUTPUT_DIR
    / f"llm_backtest_summary_{BACKTEST_VERSION}.json"
)


# ============================================================
# 二、回测参数
# ============================================================

DEFAULT_MATCH_LIMIT = 100

REQUEST_INTERVAL_SECONDS = 1.0

MAX_RETRIES = 3

RETRY_WAIT_SECONDS = 5

MIN_PROBABILITY = 1e-15

ONLY_WORLD_CUP_TEAMS = True

EXCLUDE_FRIENDLIES = True


# ============================================================
# 三、回测特征必需字段
# ============================================================

BACKTEST_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
    "target",
    "result",

    "home_elo",
    "away_elo",
    "elo_difference",

    "home_matches_5",
    "away_matches_5",
    "home_matches_10",
    "away_matches_10",

    "home_win_rate_5",
    "away_win_rate_5",
    "home_win_rate_10",
    "away_win_rate_10",

    "home_draw_rate_5",
    "away_draw_rate_5",
    "home_draw_rate_10",
    "away_draw_rate_10",

    "home_loss_rate_5",
    "away_loss_rate_5",
    "home_loss_rate_10",
    "away_loss_rate_10",

    "home_avg_goals_for_5",
    "away_avg_goals_for_5",
    "home_avg_goals_for_10",
    "away_avg_goals_for_10",

    "home_avg_goals_against_5",
    "away_avg_goals_against_5",
    "home_avg_goals_against_10",
    "away_avg_goals_against_10",

    "home_avg_goal_difference_5",
    "away_avg_goal_difference_5",
    "home_avg_goal_difference_10",
    "away_avg_goal_difference_10",

    "home_avg_points_5",
    "away_avg_points_5",
    "home_avg_points_10",
    "away_avg_points_10",

    "home_clean_sheet_rate_5",
    "away_clean_sheet_rate_5",
    "home_clean_sheet_rate_10",
    "away_clean_sheet_rate_10",

    "home_scoring_rate_5",
    "away_scoring_rate_5",
    "home_scoring_rate_10",
    "away_scoring_rate_10",

    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",

    "h2h_matches",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
}


# ============================================================
# 四、统一回测结果列
# ============================================================

RESULT_COLUMNS = [
    # 数据集和版本信息
    "backtest_version",
    "only_world_cup_teams",
    "exclude_friendlies",
    "dataset_file",

    # 比赛基本信息
    "date",
    "home_team",
    "away_team",
    "tournament",
    "neutral",

    # 胜平负
    "actual_result",
    "predicted_result",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "correct",

    # 真实比分
    "actual_home_score",
    "actual_away_score",
    "actual_scoreline",

    # 最可能比分
    "predicted_home_score",
    "predicted_away_score",
    "predicted_scoreline",
    "scoreline_probability",
    "exact_score_correct",

    # Top-N比分
    "alternate_scorelines",
    "top3_score_correct",
    "top5_score_correct",

    # 预期进球
    "base_home_expected_goals",
    "base_away_expected_goals",
    "home_goal_adjustment",
    "away_goal_adjustment",
    "expected_home_goals",
    "expected_away_goals",
    "expected_total_goals",

    # 附加概率
    "over_2_5_probability",
    "under_2_5_probability",
    "both_teams_score_probability",

    # 真实附加标签
    "actual_over_2_5",
    "actual_both_teams_score",

    # 附加判断
    "predicted_over_2_5",
    "predicted_both_teams_score",
    "over_2_5_correct",
    "both_teams_score_correct",

    # 模型说明
    "confidence",
    "adjustment_reasons",
    "analysis",
    "key_factors",
    "model",

    # 状态
    "status",
    "error",
]


# ============================================================
# 五、通用工具函数
# ============================================================

def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    将值转换为float。

    转换失败时返回default。
    """

    if value is None:
        return default

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    """
    将值转换为int。

    转换失败时返回default。
    """

    if value is None:
        return default

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def normalize_result_record(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    根据RESULT_COLUMNS统一结果字段和字段顺序。

    缺少的字段自动填写None。
    多余字段不会写入CSV。
    """

    return {
        column: result.get(column)
        for column in RESULT_COLUMNS
    }


def parse_alternate_scorelines(
    value: Any,
) -> list[dict[str, Any]]:
    """
    将备选比分转换为列表。

    兼容：
    1. Python list；
    2. JSON字符串；
    3. 空值。
    """

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError:
            return []

        if isinstance(parsed, list):
            return parsed

    return []


def get_scoreline_set(
    alternate_scorelines: Any,
    limit: int,
) -> set[str]:
    """
    从备选比分中提取前limit个比分字符串。
    """

    items = parse_alternate_scorelines(
        alternate_scorelines
    )

    scorelines: set[str] = set()

    for item in items[:limit]:
        if not isinstance(item, dict):
            continue

        scoreline = str(
            item.get(
                "scoreline",
                "",
            )
        ).strip()

        if scoreline:
            scorelines.add(scoreline)

    return scorelines


# ============================================================
# 六、读取历史赛前特征
# ============================================================

def load_historical_features() -> pd.DataFrame:
    """
    读取世界杯专用比赛赛前特征。

    筛选规则：
    1. 双方均为2026世界杯参赛球队；
    2. 排除Friendly；
    3. 必需字段完整；
    4. target和result必须与比分一致。
    """

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"没有找到历史特征文件：{FEATURE_FILE}"
        )

    data = pd.read_csv(
        FEATURE_FILE
    )

    original_count = len(data)

    print(
        f"世界杯特征文件原始数量："
        f"{original_count}"
    )

    missing_columns = (
        BACKTEST_REQUIRED_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "回测特征文件缺少必要字段："
            f"{sorted(missing_columns)}"
        )

    data = data.copy()

    data["home_team"] = (
        data["home_team"]
        .astype(str)
        .map(normalize_team_name)
    )

    data["away_team"] = (
        data["away_team"]
        .astype(str)
        .map(normalize_team_name)
    )

    if ONLY_WORLD_CUP_TEAMS:
        world_cup_mask = (
            data["home_team"].map(
                is_world_cup_2026_team
            )
            &
            data["away_team"].map(
                is_world_cup_2026_team
            )
        )

        data = data[
            world_cup_mask
        ].copy()

    if EXCLUDE_FRIENDLIES:
        tournament_text = (
            data["tournament"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )

        data = data[
            tournament_text
            != "friendly".casefold()
        ].copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=sorted(
            BACKTEST_REQUIRED_COLUMNS
        )
    ).copy()

    data["home_score"] = pd.to_numeric(
        data["home_score"],
        errors="raise",
    ).astype(int)

    data["away_score"] = pd.to_numeric(
        data["away_score"],
        errors="raise",
    ).astype(int)

    data["target"] = pd.to_numeric(
        data["target"],
        errors="raise",
    ).astype(int)

    expected_target = data.apply(
        lambda row: (
            2
            if row["home_score"]
            > row["away_score"]
            else 0
            if row["home_score"]
            < row["away_score"]
            else 1
        ),
        axis=1,
    ).astype(int)

    target_mismatch = (
        data["target"]
        != expected_target
    )

    if target_mismatch.any():
        invalid_rows = data.loc[
            target_mismatch,
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "target",
            ],
        ]

        raise ValueError(
            "回测数据target与真实比分不一致：\n"
            f"{invalid_rows.head(10)}"
        )

    expected_result = expected_target.map({
        2: "HOME_WIN",
        1: "DRAW",
        0: "AWAY_WIN",
    })

    actual_result_text = (
        data["result"]
        .astype(str)
        .str.strip()
    )

    result_mismatch = (
        actual_result_text
        != expected_result
    )

    if result_mismatch.any():
        invalid_rows = data.loc[
            result_mismatch,
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "result",
            ],
        ]

        raise ValueError(
            "回测数据result与真实比分不一致：\n"
            f"{invalid_rows.head(10)}"
        )

    print(
        "世界杯球队、非友谊赛和完整字段"
        f"筛选后数量：{len(data)}"
    )

    return data.sort_values(
        "date"
    ).reset_index(drop=True)


# ============================================================
# 七、选择回测比赛
# ============================================================

def select_backtest_matches(
    data: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    match_limit: int | None = DEFAULT_MATCH_LIMIT,
) -> pd.DataFrame:
    """
    根据日期范围和数量选择比赛。

    start_date包含该日期。
    end_date不包含该日期。
    match_limit表示最多选择最后多少场。
    """

    selected = data.copy()

    if start_date is not None:
        selected = selected[
            selected["date"]
            >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        selected = selected[
            selected["date"]
            < pd.Timestamp(end_date)
        ]

    selected = selected.sort_values(
        [
            "date",
            "home_team",
            "away_team",
        ]
    )

    print(
        "日期范围筛选后比赛数量：",
        len(selected),
    )

    if match_limit is not None:
        selected = selected.tail(
            match_limit
        )

    return selected.reset_index(
        drop=True
    )


# ============================================================
# 八、构造预测输入
# ============================================================

def extract_team_features(
    row: pd.Series,
    prefix: str,
) -> dict[str, Any]:
    """
    提取主队或客队的赛前特征。

    prefix必须是home或away。
    """

    if prefix not in {
        "home",
        "away",
    }:
        raise ValueError(
            f"未知球队前缀：{prefix}"
        )

    return {
        "team": row[f"{prefix}_team"],

        "elo": float(
            row[f"{prefix}_elo"]
        ),

        "matches_5": int(
            row[f"{prefix}_matches_5"]
        ),

        "win_rate_5": float(
            row[f"{prefix}_win_rate_5"]
        ),

        "draw_rate_5": float(
            row[f"{prefix}_draw_rate_5"]
        ),

        "loss_rate_5": float(
            row[f"{prefix}_loss_rate_5"]
        ),

        "avg_goals_for_5": float(
            row[
                f"{prefix}_avg_goals_for_5"
            ]
        ),

        "avg_goals_against_5": float(
            row[
                f"{prefix}_avg_goals_against_5"
            ]
        ),

        "avg_goal_difference_5": float(
            row[
                f"{prefix}_avg_goal_difference_5"
            ]
        ),

        "avg_points_5": float(
            row[f"{prefix}_avg_points_5"]
        ),

        "clean_sheet_rate_5": float(
            row[
                f"{prefix}_clean_sheet_rate_5"
            ]
        ),

        "scoring_rate_5": float(
            row[
                f"{prefix}_scoring_rate_5"
            ]
        ),

        "matches_10": int(
            row[f"{prefix}_matches_10"]
        ),

        "win_rate_10": float(
            row[f"{prefix}_win_rate_10"]
        ),

        "draw_rate_10": float(
            row[f"{prefix}_draw_rate_10"]
        ),

        "loss_rate_10": float(
            row[f"{prefix}_loss_rate_10"]
        ),

        "avg_goals_for_10": float(
            row[
                f"{prefix}_avg_goals_for_10"
            ]
        ),

        "avg_goals_against_10": float(
            row[
                f"{prefix}_avg_goals_against_10"
            ]
        ),

        "avg_goal_difference_10": float(
            row[
                f"{prefix}_avg_goal_difference_10"
            ]
        ),

        "avg_points_10": float(
            row[f"{prefix}_avg_points_10"]
        ),

        "clean_sheet_rate_10": float(
            row[
                f"{prefix}_clean_sheet_rate_10"
            ]
        ),

        "scoring_rate_10": float(
            row[
                f"{prefix}_scoring_rate_10"
            ]
        ),

        "rest_days": int(
            row[f"{prefix}_rest_days"]
        ),
    }


def build_historical_prediction_features(
    row: pd.Series,
) -> dict[str, Any]:
    """
    将历史比赛行转换成预测输入。

    不向模型发送真实比分、result和target。
    """

    home_features = extract_team_features(
        row=row,
        prefix="home",
    )

    away_features = extract_team_features(
        row=row,
        prefix="away",
    )

    return {
        "prediction_date": (
            row["date"].strftime(
                "%Y-%m-%d"
            )
        ),

        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "tournament": row["tournament"],
        "neutral": bool(
            row["neutral"]
        ),

        "home_team_features": (
            home_features
        ),

        "away_team_features": (
            away_features
        ),

        "derived_features": {
            "elo_difference": float(
                row["elo_difference"]
            ),

            "win_rate_5_difference": round(
                home_features["win_rate_5"]
                - away_features["win_rate_5"],
                6,
            ),

            "win_rate_10_difference": round(
                home_features["win_rate_10"]
                - away_features["win_rate_10"],
                6,
            ),

            "avg_points_5_difference": round(
                home_features["avg_points_5"]
                - away_features["avg_points_5"],
                6,
            ),

            "avg_goals_for_5_difference": round(
                home_features[
                    "avg_goals_for_5"
                ]
                - away_features[
                    "avg_goals_for_5"
                ],
                6,
            ),

            "avg_goals_against_5_difference": round(
                home_features[
                    "avg_goals_against_5"
                ]
                - away_features[
                    "avg_goals_against_5"
                ],
                6,
            ),

            "rest_days_difference": int(
                row[
                    "rest_days_difference"
                ]
            ),

            "h2h_matches": float(
                row["h2h_matches"]
            ),

            "h2h_home_win_rate": float(
                row["h2h_home_win_rate"]
            ),

            "h2h_draw_rate": float(
                row["h2h_draw_rate"]
            ),

            "h2h_away_win_rate": float(
                row["h2h_away_win_rate"]
            ),
        },

        "weather": None,
    }


# ============================================================
# 九、标签转换
# ============================================================

def target_to_result(
    target: int,
) -> str:
    """
    将数字标签转换为文本标签。
    """

    mapping = {
        2: "HOME_WIN",
        1: "DRAW",
        0: "AWAY_WIN",
    }

    if target not in mapping:
        raise ValueError(
            f"未知比赛标签：{target}"
        )

    return mapping[target]


# ============================================================
# 十、读取已有回测结果
# ============================================================

def load_existing_results() -> pd.DataFrame:
    """
    读取已有结果。

    CSV损坏时给出明确错误。
    """

    if not BACKTEST_RESULT_FILE.exists():
        return pd.DataFrame(
            columns=RESULT_COLUMNS
        )

    try:
        data = pd.read_csv(
            BACKTEST_RESULT_FILE
        )

    except pd.errors.ParserError as error:
        raise RuntimeError(
            "现有回测CSV已经损坏或包含不同列结构。\n"
            f"文件：{BACKTEST_RESULT_FILE}\n"
            "请删除该文件，或者修改BACKTEST_VERSION。"
        ) from error

    missing_columns = [
        column
        for column in RESULT_COLUMNS
        if column not in data.columns
    ]

    extra_columns = [
        column
        for column in data.columns
        if column not in RESULT_COLUMNS
    ]

    if missing_columns or extra_columns:
        raise RuntimeError(
            "现有回测CSV字段与当前代码不一致。\n"
            f"文件：{BACKTEST_RESULT_FILE}\n"
            f"缺少字段：{missing_columns}\n"
            f"多余字段：{extra_columns}\n"
            "请删除该结果文件，或更换BACKTEST_VERSION。"
        )

    return data[
        RESULT_COLUMNS
    ].copy()


def build_match_key(
    date_text: str,
    home_team: str,
    away_team: str,
) -> str:
    """
    生成比赛唯一键。
    """

    return (
        f"{str(date_text).strip()}|"
        f"{str(home_team).strip()}|"
        f"{str(away_team).strip()}"
    )


def get_completed_match_keys(
    existing_results: pd.DataFrame,
) -> set[str]:
    """
    获取已经成功完成的比赛键。
    """

    if existing_results.empty:
        return set()

    successful = existing_results[
        existing_results["status"]
        == "SUCCESS"
    ]

    return {
        build_match_key(
            date_text=row["date"],
            home_team=row["home_team"],
            away_team=row["away_team"],
        )
        for _, row in successful.iterrows()
    }


# ============================================================
# 十一、调用预测器并重试
# ============================================================

def predict_with_retry(
    predictor: LLMMatchPredictor,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    调用预测器，失败后重试。
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            return predictor.predict(
                features
            )

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            print(
                f"第 {attempt}/{MAX_RETRIES} "
                f"次请求失败：{error}"
            )

            if attempt < MAX_RETRIES:
                print(
                    f"{RETRY_WAIT_SECONDS} "
                    "秒后重试……"
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

    raise RuntimeError(
        f"达到最大重试次数："
        f"{last_error}"
    )


# ============================================================
# 十二、构造成功结果
# ============================================================

def build_success_result(
    row: pd.Series,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """
    整理一场成功预测的结果。
    """

    actual_home_score = int(
        row["home_score"]
    )

    actual_away_score = int(
        row["away_score"]
    )

    actual_scoreline = (
        f"{actual_home_score}-"
        f"{actual_away_score}"
    )

    predicted_home_score = safe_int(
        prediction.get(
            "predicted_home_score"
        ),
        default=0,
    )

    predicted_away_score = safe_int(
        prediction.get(
            "predicted_away_score"
        ),
        default=0,
    )

    if predicted_home_score is None:
        predicted_home_score = 0

    if predicted_away_score is None:
        predicted_away_score = 0

    predicted_scoreline = (
        f"{predicted_home_score}-"
        f"{predicted_away_score}"
    )

    actual_result = target_to_result(
        int(row["target"])
    )

    predicted_result = str(
        prediction.get(
            "predicted_result",
            "",
        )
    ).strip()

    alternate_scorelines = (
        prediction.get(
            "alternate_scorelines",
            [],
        )
    )

    top3_scores = get_scoreline_set(
        alternate_scorelines,
        limit=3,
    )

    top5_scores = get_scoreline_set(
        alternate_scorelines,
        limit=5,
    )

    actual_total_goals = (
        actual_home_score
        + actual_away_score
    )

    actual_over_2_5 = int(
        actual_total_goals >= 3
    )

    actual_both_teams_score = int(
        actual_home_score > 0
        and actual_away_score > 0
    )

    over_2_5_probability = safe_float(
        prediction.get(
            "over_2_5_probability"
        ),
        default=0.0,
    )

    both_teams_score_probability = safe_float(
        prediction.get(
            "both_teams_score_probability"
        ),
        default=0.0,
    )

    if over_2_5_probability is None:
        over_2_5_probability = 0.0

    if both_teams_score_probability is None:
        both_teams_score_probability = 0.0

    predicted_over_2_5 = int(
        over_2_5_probability >= 0.5
    )

    predicted_both_teams_score = int(
        both_teams_score_probability >= 0.5
    )

    result = {
        "backtest_version": (
            BACKTEST_VERSION
        ),

        "only_world_cup_teams": (
            ONLY_WORLD_CUP_TEAMS
        ),

        "exclude_friendlies": (
            EXCLUDE_FRIENDLIES
        ),

        "dataset_file": str(
            FEATURE_FILE
        ),

        "date": row[
            "date"
        ].strftime("%Y-%m-%d"),

        "home_team": row[
            "home_team"
        ],

        "away_team": row[
            "away_team"
        ],

        "tournament": row[
            "tournament"
        ],

        "neutral": int(
            row["neutral"]
        ),

        "actual_result": actual_result,

        "predicted_result": (
            predicted_result
        ),

        "home_win_probability": safe_float(
            prediction.get(
                "home_win_probability"
            ),
            default=0.0,
        ),

        "draw_probability": safe_float(
            prediction.get(
                "draw_probability"
            ),
            default=0.0,
        ),

        "away_win_probability": safe_float(
            prediction.get(
                "away_win_probability"
            ),
            default=0.0,
        ),

        "correct": int(
            actual_result
            == predicted_result
        ),

        "actual_home_score": (
            actual_home_score
        ),

        "actual_away_score": (
            actual_away_score
        ),

        "actual_scoreline": (
            actual_scoreline
        ),

        "predicted_home_score": (
            predicted_home_score
        ),

        "predicted_away_score": (
            predicted_away_score
        ),

        "predicted_scoreline": (
            predicted_scoreline
        ),

        "scoreline_probability": safe_float(
            prediction.get(
                "scoreline_probability"
            ),
            default=0.0,
        ),

        "exact_score_correct": int(
            actual_scoreline
            == predicted_scoreline
        ),

        "alternate_scorelines": json.dumps(
            alternate_scorelines,
            ensure_ascii=False,
        ),

        "top3_score_correct": int(
            actual_scoreline
            in top3_scores
        ),

        "top5_score_correct": int(
            actual_scoreline
            in top5_scores
        ),

        "base_home_expected_goals": safe_float(
            prediction.get(
                "base_home_expected_goals"
            )
        ),

        "base_away_expected_goals": safe_float(
            prediction.get(
                "base_away_expected_goals"
            )
        ),

        "home_goal_adjustment": safe_float(
            prediction.get(
                "home_goal_adjustment"
            )
        ),

        "away_goal_adjustment": safe_float(
            prediction.get(
                "away_goal_adjustment"
            )
        ),

        "expected_home_goals": safe_float(
            prediction.get(
                "expected_home_goals"
            ),
            default=0.0,
        ),

        "expected_away_goals": safe_float(
            prediction.get(
                "expected_away_goals"
            ),
            default=0.0,
        ),

        "expected_total_goals": safe_float(
            prediction.get(
                "expected_total_goals"
            ),
            default=0.0,
        ),

        "over_2_5_probability": (
            over_2_5_probability
        ),

        "under_2_5_probability": safe_float(
            prediction.get(
                "under_2_5_probability"
            ),
            default=0.0,
        ),

        "both_teams_score_probability": (
            both_teams_score_probability
        ),

        "actual_over_2_5": (
            actual_over_2_5
        ),

        "actual_both_teams_score": (
            actual_both_teams_score
        ),

        "predicted_over_2_5": (
            predicted_over_2_5
        ),

        "predicted_both_teams_score": (
            predicted_both_teams_score
        ),

        "over_2_5_correct": int(
            actual_over_2_5
            == predicted_over_2_5
        ),

        "both_teams_score_correct": int(
            actual_both_teams_score
            == predicted_both_teams_score
        ),

        "confidence": safe_float(
            prediction.get(
                "confidence"
            )
        ),

        "adjustment_reasons": json.dumps(
            prediction.get(
                "adjustment_reasons",
                [],
            ),
            ensure_ascii=False,
        ),

        "analysis": str(
            prediction.get(
                "analysis",
                "",
            )
        ),

        "key_factors": json.dumps(
            prediction.get(
                "key_factors",
                [],
            ),
            ensure_ascii=False,
        ),

        "model": prediction.get(
            "model"
        ),

        "status": "SUCCESS",

        "error": "",
    }

    return normalize_result_record(
        result
    )


# ============================================================
# 十三、构造失败结果
# ============================================================

def build_failure_result(
    row: pd.Series,
    error: Exception,
) -> dict[str, Any]:
    """
    构造失败记录。

    与成功记录使用完全相同的列结构。
    """

    actual_home_score = int(
        row["home_score"]
    )

    actual_away_score = int(
        row["away_score"]
    )

    actual_total_goals = (
        actual_home_score
        + actual_away_score
    )

    result = {
        "backtest_version": (
            BACKTEST_VERSION
        ),

        "only_world_cup_teams": (
            ONLY_WORLD_CUP_TEAMS
        ),

        "exclude_friendlies": (
            EXCLUDE_FRIENDLIES
        ),

        "dataset_file": str(
            FEATURE_FILE
        ),

        "date": row[
            "date"
        ].strftime("%Y-%m-%d"),

        "home_team": row[
            "home_team"
        ],

        "away_team": row[
            "away_team"
        ],

        "tournament": row[
            "tournament"
        ],

        "neutral": int(
            row["neutral"]
        ),

        "actual_result": target_to_result(
            int(row["target"])
        ),

        "predicted_result": "",

        "home_win_probability": None,
        "draw_probability": None,
        "away_win_probability": None,

        "correct": 0,

        "actual_home_score": (
            actual_home_score
        ),

        "actual_away_score": (
            actual_away_score
        ),

        "actual_scoreline": (
            f"{actual_home_score}-"
            f"{actual_away_score}"
        ),

        "predicted_home_score": None,
        "predicted_away_score": None,
        "predicted_scoreline": None,
        "scoreline_probability": None,
        "exact_score_correct": 0,

        "alternate_scorelines": "[]",
        "top3_score_correct": 0,
        "top5_score_correct": 0,

        "base_home_expected_goals": None,
        "base_away_expected_goals": None,
        "home_goal_adjustment": None,
        "away_goal_adjustment": None,

        "expected_home_goals": None,
        "expected_away_goals": None,
        "expected_total_goals": None,

        "over_2_5_probability": None,
        "under_2_5_probability": None,
        "both_teams_score_probability": None,

        "actual_over_2_5": int(
            actual_total_goals >= 3
        ),

        "actual_both_teams_score": int(
            actual_home_score > 0
            and actual_away_score > 0
        ),

        "predicted_over_2_5": None,
        "predicted_both_teams_score": None,
        "over_2_5_correct": 0,
        "both_teams_score_correct": 0,

        "confidence": None,
        "adjustment_reasons": "[]",
        "analysis": "",
        "key_factors": "[]",
        "model": None,

        "status": "FAILED",

        "error": str(error),
    }

    return normalize_result_record(
        result
    )


# ============================================================
# 十四、保存结果
# ============================================================

def append_result(
    result: dict[str, Any],
) -> None:
    """
    将单场结果安全追加到CSV。

    所有记录都强制使用RESULT_COLUMNS顺序。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_result = (
        normalize_result_record(
            result
        )
    )

    result_frame = pd.DataFrame(
        [normalized_result],
        columns=RESULT_COLUMNS,
    )

    if BACKTEST_RESULT_FILE.exists():
        try:
            existing_header = list(
                pd.read_csv(
                    BACKTEST_RESULT_FILE,
                    nrows=0,
                ).columns
            )

        except pd.errors.ParserError as error:
            raise RuntimeError(
                "现有回测CSV已经损坏：\n"
                f"{BACKTEST_RESULT_FILE}\n"
                "请删除该文件或更换BACKTEST_VERSION。"
            ) from error

        if existing_header != RESULT_COLUMNS:
            raise RuntimeError(
                "现有回测CSV列结构与当前代码不一致。\n"
                f"文件：{BACKTEST_RESULT_FILE}\n"
                f"已有列数：{len(existing_header)}\n"
                f"当前列数：{len(RESULT_COLUMNS)}\n"
                "请删除旧文件或更换BACKTEST_VERSION。"
            )

    write_header = (
        not BACKTEST_RESULT_FILE.exists()
    )

    result_frame.to_csv(
        BACKTEST_RESULT_FILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 十五、概率评估指标
# ============================================================

def calculate_log_loss(
    results: pd.DataFrame,
) -> float:
    """
    计算三分类Log Loss。
    """

    losses: list[float] = []

    for _, row in results.iterrows():
        actual_result = row[
            "actual_result"
        ]

        if actual_result == "HOME_WIN":
            probability = row[
                "home_win_probability"
            ]

        elif actual_result == "DRAW":
            probability = row[
                "draw_probability"
            ]

        else:
            probability = row[
                "away_win_probability"
            ]

        probability = max(
            MIN_PROBABILITY,
            min(
                1.0
                - MIN_PROBABILITY,
                float(probability),
            ),
        )

        losses.append(
            -math.log(probability)
        )

    if not losses:
        return 0.0

    return float(
        sum(losses)
        / len(losses)
    )


def calculate_brier_score(
    results: pd.DataFrame,
) -> float:
    """
    计算三分类Brier Score。
    """

    scores: list[float] = []

    for _, row in results.iterrows():
        actual_vector = {
            "HOME_WIN": [
                1.0,
                0.0,
                0.0,
            ],

            "DRAW": [
                0.0,
                1.0,
                0.0,
            ],

            "AWAY_WIN": [
                0.0,
                0.0,
                1.0,
            ],
        }[
            row["actual_result"]
        ]

        predicted_vector = [
            float(
                row[
                    "home_win_probability"
                ]
            ),

            float(
                row[
                    "draw_probability"
                ]
            ),

            float(
                row[
                    "away_win_probability"
                ]
            ),
        ]

        score = sum(
            (
                predicted_value
                - actual_value
            ) ** 2
            for predicted_value, actual_value
            in zip(
                predicted_vector,
                actual_vector,
            )
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(
        sum(scores)
        / len(scores)
    )


# ============================================================
# 十六、分类指标
# ============================================================

def calculate_class_metrics(
    results: pd.DataFrame,
    class_name: str,
) -> dict[str, float | int]:
    """
    计算Precision、Recall和F1。
    """

    actual_positive = (
        results["actual_result"]
        == class_name
    )

    predicted_positive = (
        results["predicted_result"]
        == class_name
    )

    true_positive = int(
        (
            actual_positive
            & predicted_positive
        ).sum()
    )

    false_positive = int(
        (
            ~actual_positive
            & predicted_positive
        ).sum()
    )

    false_negative = int(
        (
            actual_positive
            & ~predicted_positive
        ).sum()
    )

    precision_denominator = (
        true_positive
        + false_positive
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    precision = (
        true_positive
        / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        true_positive
        / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    return {
        "support": int(
            actual_positive.sum()
        ),

        "predicted": int(
            predicted_positive.sum()
        ),

        "precision": round(
            precision,
            6,
        ),

        "recall": round(
            recall,
            6,
        ),

        "f1": round(
            f1,
            6,
        ),
    }


# ============================================================
# 十七、比分评估指标
# ============================================================

def calculate_score_metrics(
    successful: pd.DataFrame,
) -> dict[str, float]:
    """
    计算比分预测指标。
    """

    actual_total_goals = (
        successful[
            "actual_home_score"
        ]
        + successful[
            "actual_away_score"
        ]
    )

    exact_score_accuracy = float(
        successful[
            "exact_score_correct"
        ].mean()
    )

    top3_score_accuracy = float(
        successful[
            "top3_score_correct"
        ].mean()
    )

    top5_score_accuracy = float(
        successful[
            "top5_score_correct"
        ].mean()
    )

    home_goal_mae = float(
        (
            successful[
                "expected_home_goals"
            ]
            - successful[
                "actual_home_score"
            ]
        )
        .abs()
        .mean()
    )

    away_goal_mae = float(
        (
            successful[
                "expected_away_goals"
            ]
            - successful[
                "actual_away_score"
            ]
        )
        .abs()
        .mean()
    )

    total_goal_mae = float(
        (
            successful[
                "expected_total_goals"
            ]
            - actual_total_goals
        )
        .abs()
        .mean()
    )

    integer_score_home_mae = float(
        (
            successful[
                "predicted_home_score"
            ]
            - successful[
                "actual_home_score"
            ]
        )
        .abs()
        .mean()
    )

    integer_score_away_mae = float(
        (
            successful[
                "predicted_away_score"
            ]
            - successful[
                "actual_away_score"
            ]
        )
        .abs()
        .mean()
    )

    over_2_5_accuracy = float(
        successful[
            "over_2_5_correct"
        ].mean()
    )

    both_teams_score_accuracy = float(
        successful[
            "both_teams_score_correct"
        ].mean()
    )

    return {
        "exact_score_accuracy": round(
            exact_score_accuracy,
            6,
        ),

        "top3_score_accuracy": round(
            top3_score_accuracy,
            6,
        ),

        "top5_score_accuracy": round(
            top5_score_accuracy,
            6,
        ),

        "home_goal_mae": round(
            home_goal_mae,
            6,
        ),

        "away_goal_mae": round(
            away_goal_mae,
            6,
        ),

        "total_goal_mae": round(
            total_goal_mae,
            6,
        ),

        "integer_score_home_mae": round(
            integer_score_home_mae,
            6,
        ),

        "integer_score_away_mae": round(
            integer_score_away_mae,
            6,
        ),

        "over_2_5_accuracy": round(
            over_2_5_accuracy,
            6,
        ),

        "both_teams_score_accuracy": round(
            both_teams_score_accuracy,
            6,
        ),
    }


# ============================================================
# 十八、汇总指标
# ============================================================

def evaluate_results(
    results: pd.DataFrame,
) -> dict[str, Any]:
    """
    计算全部回测指标。
    """

    successful = results[
        results["status"]
        == "SUCCESS"
    ].copy()

    if successful.empty:
        raise ValueError(
            "没有成功的回测结果"
        )

    numeric_columns = [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "actual_home_score",
        "actual_away_score",
        "predicted_home_score",
        "predicted_away_score",
        "expected_home_goals",
        "expected_away_goals",
        "expected_total_goals",
        "exact_score_correct",
        "top3_score_correct",
        "top5_score_correct",
        "over_2_5_correct",
        "both_teams_score_correct",
        "correct",
    ]

    for column in numeric_columns:
        successful[column] = pd.to_numeric(
            successful[column],
            errors="raise",
        )

    accuracy = float(
        successful[
            "correct"
        ].mean()
    )

    actual_distribution = (
        successful[
            "actual_result"
        ]
        .value_counts()
        .to_dict()
    )

    predicted_distribution = (
        successful[
            "predicted_result"
        ]
        .value_counts()
        .to_dict()
    )

    score_metrics = (
        calculate_score_metrics(
            successful
        )
    )

    return {
        "backtest_version": (
            BACKTEST_VERSION
        ),

        "only_world_cup_teams": (
            ONLY_WORLD_CUP_TEAMS
        ),

        "exclude_friendlies": (
            EXCLUDE_FRIENDLIES
        ),

        "dataset_file": str(
            FEATURE_FILE
        ),

        "total_records": int(
            len(results)
        ),

        "successful_records": int(
            len(successful)
        ),

        "failed_records": int(
            (
                results["status"]
                == "FAILED"
            ).sum()
        ),

        "accuracy": round(
            accuracy,
            6,
        ),

        "log_loss": round(
            calculate_log_loss(
                successful
            ),
            6,
        ),

        "brier_score": round(
            calculate_brier_score(
                successful
            ),
            6,
        ),

        "actual_distribution": (
            actual_distribution
        ),

        "predicted_distribution": (
            predicted_distribution
        ),

        "class_metrics": {
            class_name: (
                calculate_class_metrics(
                    successful,
                    class_name,
                )
            )
            for class_name in [
                "HOME_WIN",
                "DRAW",
                "AWAY_WIN",
            ]
        },

        **score_metrics,
    }


# ============================================================
# 十九、保存与打印汇总
# ============================================================

def save_summary(
    summary: dict[str, Any],
) -> None:
    """
    保存汇总JSON。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BACKTEST_SUMMARY_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_summary(
    summary: dict[str, Any],
) -> None:
    """
    打印回测结果。
    """

    print("\n" + "=" * 72)
    print("世界杯球队足球预测历史回测结果")
    print("=" * 72)

    print(
        "回测版本：",
        summary[
            "backtest_version"
        ],
    )

    print(
        "总记录数：",
        summary[
            "total_records"
        ],
    )

    print(
        "成功预测：",
        summary[
            "successful_records"
        ],
    )

    print(
        "失败预测：",
        summary[
            "failed_records"
        ],
    )

    print("\n胜平负指标：")

    print(
        "准确率：",
        (
            f"{summary['accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "Log Loss：",
        summary[
            "log_loss"
        ],
    )

    print(
        "Brier Score：",
        summary[
            "brier_score"
        ],
    )

    print("\n比分预测指标：")

    print(
        "Top-1精确比分命中率：",
        (
            f"{summary['exact_score_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "Top-3比分命中率：",
        (
            f"{summary['top3_score_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "Top-5比分命中率：",
        (
            f"{summary['top5_score_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "主队预期进球MAE：",
        summary[
            "home_goal_mae"
        ],
    )

    print(
        "客队预期进球MAE：",
        summary[
            "away_goal_mae"
        ],
    )

    print(
        "总进球MAE：",
        summary[
            "total_goal_mae"
        ],
    )

    print(
        "整数主队比分MAE：",
        summary[
            "integer_score_home_mae"
        ],
    )

    print(
        "整数客队比分MAE：",
        summary[
            "integer_score_away_mae"
        ],
    )

    print(
        "大于2.5球判断准确率：",
        (
            f"{summary['over_2_5_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "双方都进球判断准确率：",
        (
            f"{summary['both_teams_score_accuracy'] * 100:.2f}%"
        ),
    )

    print("\n实际结果分布：")

    print(
        summary[
            "actual_distribution"
        ]
    )

    print("\n模型预测分布：")

    print(
        summary[
            "predicted_distribution"
        ]
    )

    print("\n各类别指标：")

    for class_name, metrics in (
        summary[
            "class_metrics"
        ].items()
    ):
        print(
            class_name,
            metrics,
        )

    print("=" * 72)


# ============================================================
# 二十、回测主流程
# ============================================================

def run_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    match_limit: int | None = DEFAULT_MATCH_LIMIT,
) -> None:
    """
    执行完整回测。
    """

    print(
        "正在读取历史赛前特征……"
    )

    feature_data = (
        load_historical_features()
    )

    print(
        "历史可训练比赛数量："
        f"{len(feature_data)}"
    )

    matches = select_backtest_matches(
        data=feature_data,
        start_date=start_date,
        end_date=end_date,
        match_limit=match_limit,
    )

    print(
        "本次计划回测比赛数量："
        f"{len(matches)}"
    )

    if matches.empty:
        raise ValueError(
            "没有找到符合条件的回测比赛"
        )

    print("\n本次回测比赛列表：")

    for _, match in matches.iterrows():
        print(
            f"{match['date']:%Y-%m-%d} "
            f"{match['home_team']} vs "
            f"{match['away_team']} "
            f"({match['tournament']})"
        )

    existing_results = (
        load_existing_results()
    )

    completed_keys = (
        get_completed_match_keys(
            existing_results
        )
    )

    predictor = (
        LLMMatchPredictor()
    )

    for index, row in matches.iterrows():
        date_text = row[
            "date"
        ].strftime("%Y-%m-%d")

        match_key = build_match_key(
            date_text=date_text,
            home_team=row["home_team"],
            away_team=row["away_team"],
        )

        if match_key in completed_keys:
            print(
                f"[{index + 1}/{len(matches)}] "
                "跳过已完成："
                f"{row['home_team']} "
                f"vs {row['away_team']}"
            )

            continue

        print(
            f"\n[{index + 1}/{len(matches)}] "
            f"{date_text} "
            f"{row['home_team']} "
            f"vs {row['away_team']}"
        )

        features = (
            build_historical_prediction_features(
                row
            )
        )

        try:
            prediction = (
                predict_with_retry(
                    predictor=predictor,
                    features=features,
                )
            )

            result = (
                build_success_result(
                    row=row,
                    prediction=prediction,
                )
            )

            print(
                "真实结果：",
                result[
                    "actual_result"
                ],
            )

            print(
                "预测结果：",
                result[
                    "predicted_result"
                ],
            )

            print(
                "真实比分：",
                result[
                    "actual_scoreline"
                ],
            )

            print(
                "最可能比分：",
                result[
                    "predicted_scoreline"
                ],
            )

            print(
                "该比分概率：",
                result[
                    "scoreline_probability"
                ],
            )

            print(
                "Top-1精确命中：",
                bool(
                    result[
                        "exact_score_correct"
                    ]
                ),
            )

            print(
                "Top-3命中：",
                bool(
                    result[
                        "top3_score_correct"
                    ]
                ),
            )

            print(
                "Top-5命中：",
                bool(
                    result[
                        "top5_score_correct"
                    ]
                ),
            )

            print(
                "预期进球：",
                {
                    "主队": result[
                        "expected_home_goals"
                    ],
                    "客队": result[
                        "expected_away_goals"
                    ],
                    "总进球": result[
                        "expected_total_goals"
                    ],
                },
            )

            print(
                "胜平负是否正确：",
                bool(
                    result[
                        "correct"
                    ]
                ),
            )

            print(
                "概率：",
                {
                    "主胜": result[
                        "home_win_probability"
                    ],
                    "平局": result[
                        "draw_probability"
                    ],
                    "客胜": result[
                        "away_win_probability"
                    ],
                },
            )

        except Exception as error:
            print(
                "本场回测失败：",
                error,
            )

            result = (
                build_failure_result(
                    row=row,
                    error=error,
                )
            )

        append_result(
            result
        )

        time.sleep(
            REQUEST_INTERVAL_SECONDS
        )

    print(
        "\n开始计算回测指标……"
    )

    all_results = (
        load_existing_results()
    )

    selected_keys = {
        build_match_key(
            date_text=(
                row["date"].strftime(
                    "%Y-%m-%d"
                )
            ),
            home_team=row[
                "home_team"
            ],
            away_team=row[
                "away_team"
            ],
        )
        for _, row in matches.iterrows()
    }

    current_results = all_results[
        all_results.apply(
            lambda result_row: (
                build_match_key(
                    date_text=result_row[
                        "date"
                    ],
                    home_team=result_row[
                        "home_team"
                    ],
                    away_team=result_row[
                        "away_team"
                    ],
                )
                in selected_keys
            ),
            axis=1,
        )
    ].copy()

    summary = evaluate_results(
        current_results
    )

    save_summary(
        summary
    )

    print_summary(
        summary
    )

    print(
        "\n逐场结果文件：",
        BACKTEST_RESULT_FILE,
    )

    print(
        "汇总结果文件：",
        BACKTEST_SUMMARY_FILE,
    )


# ============================================================
# 二十一、程序入口
# ============================================================

def main() -> None:
    """
    回测当前已结束的2026世界杯比赛。

    日期范围：
    2026-06-09 <= date < 2026-06-13
    """

    run_backtest(
        start_date="2026-06-09",
        end_date="2026-06-13",
        match_limit=20,
    )


if __name__ == "__main__":
    main()
