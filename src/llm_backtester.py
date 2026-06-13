
"""
大模型足球预测历史回测。

功能：
1. 读取无未来数据泄漏的历史比赛特征；
2. 选择指定日期范围或最近若干场比赛；
3. 将每场比赛的赛前特征发送给大模型；
4. 保存主胜、平局、客胜概率；
5. 保存预期进球与预测比分；
6. 对比真实比赛结果和真实比分；
7. 计算 Accuracy、Log Loss、Brier Score；
8. 计算精确比分命中率和进球 MAE；
9. 支持中断后继续执行。

注意：
每场比赛都会调用一次远程大模型 API，
因此会产生时间和费用消耗。
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


# ============================================================
# 一、文件路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "match_features_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "backtest"
)

# 使用新版本名称，避免覆盖原来的 v3 回测结果
BACKTEST_VERSION = "hybrid_poisson_v1_20"

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

# 默认最多回测多少场
DEFAULT_MATCH_LIMIT = 100

# 两次 API 请求之间的暂停时间
REQUEST_INTERVAL_SECONDS = 1.0

# 单场比赛最多重试次数
MAX_RETRIES = 3

# 重试间隔
RETRY_WAIT_SECONDS = 5

# 计算 Log Loss 时防止 log(0)
MIN_PROBABILITY = 1e-15


# ============================================================
# 三、读取历史赛前特征
# ============================================================

def load_historical_features() -> pd.DataFrame:
    """
    读取历史比赛赛前特征。

    match_features_train.csv 中的每一行，
    都是在对应比赛开始之前生成的特征。
    """

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"没有找到历史特征文件：{FEATURE_FILE}"
        )

    data = pd.read_csv(FEATURE_FILE)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "target",
        ]
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

    return data.sort_values(
        "date"
    ).reset_index(drop=True)


# ============================================================
# 四、选择回测比赛
# ============================================================

def select_backtest_matches(
    data: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    match_limit: int | None = DEFAULT_MATCH_LIMIT,
) -> pd.DataFrame:
    """
    根据日期范围和数量选择回测比赛。

    start_date:
        包含该日期，即 date >= start_date。

    end_date:
        不包含该日期，即 date < end_date。

    match_limit:
        最多选择最后多少场。
        设置为 None 表示选择日期范围内全部比赛。
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
        "date"
    )

    print(
        "日期范围筛选后比赛数量：",
        len(selected),
    )

    if match_limit is not None:
        selected = selected.tail(
            match_limit
        )

    return selected.reset_index(drop=True)


# ============================================================
# 五、构造大模型输入特征
# ============================================================

def extract_team_features(
    row: pd.Series,
    prefix: str,
) -> dict[str, Any]:
    """
    从历史比赛的一行中提取一支球队的赛前特征。

    prefix:
        home 或 away。
    """

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
            row[f"{prefix}_avg_goals_for_5"]
        ),

        "avg_goals_against_5": float(
            row[f"{prefix}_avg_goals_against_5"]
        ),

        "avg_goal_difference_5": float(
            row[f"{prefix}_avg_goal_difference_5"]
        ),

        "avg_points_5": float(
            row[f"{prefix}_avg_points_5"]
        ),

        "clean_sheet_rate_5": float(
            row[f"{prefix}_clean_sheet_rate_5"]
        ),

        "scoring_rate_5": float(
            row[f"{prefix}_scoring_rate_5"]
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
            row[f"{prefix}_avg_goals_for_10"]
        ),

        "avg_goals_against_10": float(
            row[f"{prefix}_avg_goals_against_10"]
        ),

        "avg_goal_difference_10": float(
            row[f"{prefix}_avg_goal_difference_10"]
        ),

        "avg_points_10": float(
            row[f"{prefix}_avg_points_10"]
        ),

        "clean_sheet_rate_10": float(
            row[f"{prefix}_clean_sheet_rate_10"]
        ),

        "scoring_rate_10": float(
            row[f"{prefix}_scoring_rate_10"]
        ),

        "rest_days": int(
            row[f"{prefix}_rest_days"]
        ),
    }


def build_historical_prediction_features(
    row: pd.Series,
) -> dict[str, Any]:
    """
    将一场历史比赛转换为大模型预测输入。

    不向模型发送：
    - home_score；
    - away_score；
    - result；
    - target。

    因此大模型无法直接看到真实答案。
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
            row["date"].strftime("%Y-%m-%d")
        ),

        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "tournament": row["tournament"],
        "neutral": bool(row["neutral"]),

        "home_team_features": home_features,
        "away_team_features": away_features,

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
                home_features["avg_goals_for_5"]
                - away_features["avg_goals_for_5"],
                6,
            ),

            "avg_goals_against_5_difference": round(
                home_features["avg_goals_against_5"]
                - away_features["avg_goals_against_5"],
                6,
            ),

            "rest_days_difference": int(
                row["rest_days_difference"]
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

        # 当前历史回测暂时不添加历史天气
        "weather": None,
    }


# ============================================================
# 六、标签转换
# ============================================================

def target_to_result(
    target: int,
) -> str:
    """
    将数字标签转换为文本。

    2：主胜
    1：平局
    0：客胜
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
# 七、断点续跑
# ============================================================

def load_existing_results() -> pd.DataFrame:
    """
    读取已经保存的回测结果。

    如果结果文件不存在，返回空 DataFrame。
    """

    if not BACKTEST_RESULT_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        BACKTEST_RESULT_FILE
    )


def build_match_key(
    date_text: str,
    home_team: str,
    away_team: str,
) -> str:
    """
    生成比赛唯一标识。
    """

    return (
        f"{date_text}|"
        f"{home_team}|"
        f"{away_team}"
    )


def get_completed_match_keys(
    existing_results: pd.DataFrame,
) -> set[str]:
    """
    获取已经成功完成的比赛键。
    """

    if existing_results.empty:
        return set()

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "status",
    }

    if not required_columns.issubset(
        existing_results.columns
    ):
        return set()

    successful = existing_results[
        existing_results["status"]
        == "SUCCESS"
    ]

    return {
        build_match_key(
            date_text=str(row["date"]),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
        )
        for _, row in successful.iterrows()
    }


# ============================================================
# 八、调用大模型并重试
# ============================================================

def predict_with_retry(
    predictor: LLMMatchPredictor,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    调用大模型。

    网络异常、JSON 异常或供应商异常时自动重试。
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
                    f"{RETRY_WAIT_SECONDS} 秒后重试……"
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

    raise RuntimeError(
        f"达到最大重试次数：{last_error}"
    )


# ============================================================
# 九、单场结果整理
# ============================================================

def build_success_result(
    row: pd.Series,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """
    整理一场成功预测的回测结果。
    """

    actual_home_score = int(
        row["home_score"]
    )

    actual_away_score = int(
        row["away_score"]
    )

    predicted_home_score = int(
        prediction.get(
            "predicted_home_score",
            0,
        )
    )

    predicted_away_score = int(
        prediction.get(
            "predicted_away_score",
            0,
        )
    )

    actual_target = int(
        row["target"]
    )

    actual_result = target_to_result(
        actual_target
    )

    predicted_result = str(
        prediction["predicted_result"]
    )

    return {
        "date": (
            row["date"].strftime("%Y-%m-%d")
        ),

        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "tournament": row["tournament"],
        "neutral": int(row["neutral"]),

        # 胜平负结果
        "actual_result": actual_result,
        "predicted_result": predicted_result,

        "home_win_probability": float(
            prediction[
                "home_win_probability"
            ]
        ),

        "draw_probability": float(
            prediction[
                "draw_probability"
            ]
        ),

        "away_win_probability": float(
            prediction[
                "away_win_probability"
            ]
        ),

        "correct": int(
            actual_result
            == predicted_result
        ),

        # 真实比分
        "actual_home_score": actual_home_score,
        "actual_away_score": actual_away_score,

        "actual_scoreline": (
            f"{actual_home_score}-"
            f"{actual_away_score}"
        ),

        # 预测比分
        "predicted_home_score": (
            predicted_home_score
        ),

        "predicted_away_score": (
            predicted_away_score
        ),

        "predicted_scoreline": (
            f"{predicted_home_score}-"
            f"{predicted_away_score}"
        ),

        # 大模型给出的预期进球
        "expected_home_goals": float(
            prediction.get(
                "expected_home_goals",
                0.0,
            )
        ),

        "expected_away_goals": float(
            prediction.get(
                "expected_away_goals",
                0.0,
            )
        ),

        "expected_total_goals": float(
            prediction.get(
                "expected_total_goals",
                0.0,
            )
        ),

        # 是否精确命中比分
        "exact_score_correct": int(
            actual_home_score
            == predicted_home_score
            and
            actual_away_score
            == predicted_away_score
        ),

        "confidence": prediction.get(
            "confidence"
        ),

        "analysis": prediction.get(
            "analysis",
            "",
        ),

        "key_factors": json.dumps(
            prediction.get(
                "key_factors",
                [],
            ),
            ensure_ascii=False,
        ),

        "alternate_scorelines": json.dumps(
            prediction.get(
                "alternate_scorelines",
                [],
            ),
            ensure_ascii=False,
        ),

        "status": "SUCCESS",
        "error": "",
    }


def build_failure_result(
    row: pd.Series,
    error: Exception,
) -> dict[str, Any]:
    """
    保存预测失败的比赛记录。

    字段与成功记录保持一致，
    防止 CSV 列结构不统一。
    """

    actual_home_score = int(
        row["home_score"]
    )

    actual_away_score = int(
        row["away_score"]
    )

    return {
        "date": (
            row["date"].strftime("%Y-%m-%d")
        ),

        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "tournament": row["tournament"],
        "neutral": int(row["neutral"]),

        "actual_result": target_to_result(
            int(row["target"])
        ),

        "predicted_result": "",

        "home_win_probability": None,
        "draw_probability": None,
        "away_win_probability": None,

        "correct": 0,

        "actual_home_score": actual_home_score,
        "actual_away_score": actual_away_score,

        "actual_scoreline": (
            f"{actual_home_score}-"
            f"{actual_away_score}"
        ),

        "predicted_home_score": None,
        "predicted_away_score": None,
        "predicted_scoreline": "",

        "expected_home_goals": None,
        "expected_away_goals": None,
        "expected_total_goals": None,

        "exact_score_correct": 0,

        "confidence": None,
        "analysis": "",
        "key_factors": "[]",
        "alternate_scorelines": "[]",

        "status": "FAILED",
        "error": str(error),
    }


# ============================================================
# 十、保存回测进度
# ============================================================

def append_result(
    result: dict[str, Any],
) -> None:
    """
    每完成一场比赛，立即将结果追加到 CSV。

    这样程序中途停止后不会丢失之前结果。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_frame = pd.DataFrame(
        [result]
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
# 十一、概率评估指标
# ============================================================

def calculate_log_loss(
    results: pd.DataFrame,
) -> float:
    """
    计算三分类 Log Loss。

    越低越好。
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
                1.0 - MIN_PROBABILITY,
                float(probability),
            ),
        )

        losses.append(
            -math.log(probability)
        )

    if not losses:
        return 0.0

    return float(
        sum(losses) / len(losses)
    )


def calculate_brier_score(
    results: pd.DataFrame,
) -> float:
    """
    计算三分类 Brier Score。

    越低越好。
    """

    scores: list[float] = []

    for _, row in results.iterrows():
        actual_result = row[
            "actual_result"
        ]

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
        }[actual_result]

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
                predicted_probability
                - actual_value
            ) ** 2
            for predicted_probability, actual_value
            in zip(
                predicted_vector,
                actual_vector,
            )
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(
        sum(scores) / len(scores)
    )


# ============================================================
# 十二、分类指标
# ============================================================

def calculate_class_metrics(
    results: pd.DataFrame,
    class_name: str,
) -> dict[str, float | int]:
    """
    计算某个类别的 Precision、Recall 和 F1。
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
        2 * precision * recall
        / (precision + recall)
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
# 十三、比分评估指标
# ============================================================

def calculate_score_metrics(
    successful: pd.DataFrame,
) -> dict[str, float]:
    """
    计算比分预测指标。

    exact_score_accuracy:
        精确比分命中率。

    home_goal_mae:
        主队预期进球与真实进球的平均绝对误差。

    away_goal_mae:
        客队预期进球与真实进球的平均绝对误差。

    total_goal_mae:
        预期总进球与真实总进球的平均绝对误差。

    integer_score_home_mae:
        预测整数主队比分的平均绝对误差。

    integer_score_away_mae:
        预测整数客队比分的平均绝对误差。
    """

    exact_score_accuracy = float(
        successful[
            "exact_score_correct"
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

    actual_total_goals = (
        successful[
            "actual_home_score"
        ]
        + successful[
            "actual_away_score"
        ]
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

    return {
        "exact_score_accuracy": round(
            exact_score_accuracy,
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
    }


# ============================================================
# 十四、汇总评估
# ============================================================

def evaluate_results(
    results: pd.DataFrame,
) -> dict[str, Any]:
    """
    计算完整回测指标。
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
        "correct",
    ]

    for column in numeric_columns:
        successful[column] = pd.to_numeric(
            successful[column],
            errors="raise",
        )

    accuracy = float(
        successful["correct"].mean()
    )

    actual_distribution = (
        successful["actual_result"]
        .value_counts()
        .to_dict()
    )

    predicted_distribution = (
        successful["predicted_result"]
        .value_counts()
        .to_dict()
    )

    score_metrics = calculate_score_metrics(
        successful
    )

    summary = {
        "backtest_version": (
            BACKTEST_VERSION
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

        # 胜平负指标
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

        # 比分指标
        **score_metrics,
    }

    return summary


# ============================================================
# 十五、保存和打印汇总
# ============================================================

def save_summary(
    summary: dict[str, Any],
) -> None:
    """
    保存回测汇总结果。
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
    打印完整回测指标。
    """

    print("\n" + "=" * 70)
    print("大模型历史回测结果")
    print("=" * 70)

    print(
        "回测版本：",
        summary["backtest_version"],
    )

    print(
        "总记录数：",
        summary["total_records"],
    )

    print(
        "成功预测：",
        summary["successful_records"],
    )

    print(
        "失败预测：",
        summary["failed_records"],
    )

    print("\n胜平负指标：")

    print(
        "准确率：",
        f"{summary['accuracy'] * 100:.2f}%",
    )

    print(
        "Log Loss：",
        summary["log_loss"],
    )

    print(
        "Brier Score：",
        summary["brier_score"],
    )

    print("\n比分预测指标：")

    print(
        "精确比分命中率：",
        (
            f"{summary['exact_score_accuracy'] * 100:.2f}%"
        ),
    )

    print(
        "主队预期进球 MAE：",
        summary["home_goal_mae"],
    )

    print(
        "客队预期进球 MAE：",
        summary["away_goal_mae"],
    )

    print(
        "总进球 MAE：",
        summary["total_goal_mae"],
    )

    print(
        "整数主队比分 MAE：",
        summary["integer_score_home_mae"],
    )

    print(
        "整数客队比分 MAE：",
        summary["integer_score_away_mae"],
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
        summary["class_metrics"].items()
    ):
        print(
            class_name,
            metrics,
        )

    print("=" * 70)


# ============================================================
# 十六、回测主流程
# ============================================================

def run_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    match_limit: int | None = DEFAULT_MATCH_LIMIT,
) -> None:
    """
    执行完整大模型历史回测。
    """

    print("正在读取历史赛前特征……")

    feature_data = (
        load_historical_features()
    )

    print(
        f"历史可训练比赛数量："
        f"{len(feature_data)}"
    )

    matches = select_backtest_matches(
        data=feature_data,
        start_date=start_date,
        end_date=end_date,
        match_limit=match_limit,
    )

    print(
        f"本次计划回测比赛数量："
        f"{len(matches)}"
    )

    if matches.empty:
        raise ValueError(
            "没有找到符合条件的回测比赛"
        )

    existing_results = (
        load_existing_results()
    )

    completed_keys = (
        get_completed_match_keys(
            existing_results
        )
    )

    predictor = LLMMatchPredictor()

    for index, row in matches.iterrows():
        date_text = row[
            "date"
        ].strftime("%Y-%m-%d")

        match_key = build_match_key(
            date_text=date_text,
            home_team=str(
                row["home_team"]
            ),
            away_team=str(
                row["away_team"]
            ),
        )

        if match_key in completed_keys:
            print(
                f"[{index + 1}/{len(matches)}] "
                f"跳过已完成："
                f"{row['home_team']} "
                f"vs {row['away_team']}"
            )

            continue

        print(
            f"\n[{index + 1}/{len(matches)}] "
            f"{date_text} "
            f"{row['home_team']} "
            f"vs "
            f"{row['away_team']}"
        )

        features = (
            build_historical_prediction_features(
                row
            )
        )

        try:
            prediction = predict_with_retry(
                predictor=predictor,
                features=features,
            )

            result = build_success_result(
                row=row,
                prediction=prediction,
            )

            print(
                "真实结果：",
                result["actual_result"],
            )

            print(
                "预测结果：",
                result["predicted_result"],
            )

            print(
                "真实比分：",
                result["actual_scoreline"],
            )

            print(
                "预测比分：",
                result["predicted_scoreline"],
            )

            print(
                "精确比分命中：",
                bool(
                    result[
                        "exact_score_correct"
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
                bool(result["correct"]),
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

            result = build_failure_result(
                row=row,
                error=error,
            )

        append_result(result)

        time.sleep(
            REQUEST_INTERVAL_SECONDS
        )

    print("\n开始计算回测指标……")

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
            home_team=str(
                row["home_team"]
            ),
            away_team=str(
                row["away_team"]
            ),
        )
        for _, row in matches.iterrows()
    }

    current_results = all_results[
        all_results.apply(
            lambda result_row: build_match_key(
                date_text=str(
                    result_row["date"]
                ),
                home_team=str(
                    result_row["home_team"]
                ),
                away_team=str(
                    result_row["away_team"]
                ),
            )
            in selected_keys,
            axis=1,
        )
    ].copy()

    summary = evaluate_results(
        current_results
    )

    save_summary(summary)

    print_summary(summary)

    print(
        "\n逐场结果文件：",
        BACKTEST_RESULT_FILE,
    )

    print(
        "汇总结果文件：",
        BACKTEST_SUMMARY_FILE,
    )


# ============================================================
# 十七、程序入口
# ============================================================

def main() -> None:
    """
    回测 2026-06-09 至 2026-06-12 的最后20场比赛。
    """

    run_backtest(
        start_date="2026-06-09",
        end_date="2026-06-13",
        match_limit=20,
    )


if __name__ == "__main__":
    main()

