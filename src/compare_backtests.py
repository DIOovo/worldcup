
"""
对比 v2 和 v3 两版大模型足球预测回测结果。

对比内容：
1. 检查两次回测是否使用完全相同的比赛；
2. 准确率；
3. Log Loss；
4. Brier Score；
5. 主胜、平局、客胜的 Precision、Recall、F1；
6. 真实结果与模型预测结果分布；
7. 哪些比赛由错误变正确；
8. 哪些比赛由正确变错误；
9. 判断 v3 是否值得扩大到 100 场回测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 一、路径配置
# ============================================================

# 当前文件路径，例如：
# /Users/gaoheyang/Documents/agent/src/compare_backtests.py
CURRENT_FILE = Path(__file__).resolve()

# 项目根目录：
# /Users/gaoheyang/Documents/agent
PROJECT_ROOT = CURRENT_FILE.parent.parent

# 回测结果目录
BACKTEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "backtest"
)


# ============================================================
# 二、v2 回测文件
# ============================================================

V2_RESULT_FILE = (
    BACKTEST_DIR
    / "llm_backtest_results_prompt_v2_20.csv"
)

V2_SUMMARY_FILE = (
    BACKTEST_DIR
    / "llm_backtest_summary_prompt_v2_20.json"
)


# ============================================================
# 三、v3 回测文件
# ============================================================

V3_RESULT_FILE = (
    BACKTEST_DIR
    / "llm_backtest_results_prompt_v3_20.csv"
)

V3_SUMMARY_FILE = (
    BACKTEST_DIR
    / "llm_backtest_summary_prompt_v3_20.json"
)


# ============================================================
# 四、读取文件
# ============================================================

def load_json(
    file_path: Path,
) -> dict[str, Any]:
    """
    读取 JSON 汇总文件。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"JSON 文件不存在：{file_path}"
        )

    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON 文件内容不是对象格式：{file_path}"
        )

    return data


def load_results(
    file_path: Path,
) -> pd.DataFrame:
    """
    读取逐场回测 CSV。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV 文件不存在：{file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "actual_result",
        "predicted_result",
        "correct",
        "status",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"回测文件缺少字段："
            f"{sorted(missing_columns)}\n"
            f"文件：{file_path}"
        )

    return data


# ============================================================
# 五、格式化工具
# ============================================================

def format_percentage(
    value: float,
    show_sign: bool = False,
) -> str:
    """
    将小数转换为百分比。

    例如：
    0.55 -> 55.00%
    0.05 -> +5.00%
    """

    if show_sign:
        return f"{value * 100:+.2f}%"

    return f"{value * 100:.2f}%"


def build_match_key(
    data: pd.DataFrame,
) -> pd.Series:
    """
    为每场比赛生成唯一标识。

    格式：
    日期|主队|客队
    """

    return (
        data["date"].astype(str).str.strip()
        + "|"
        + data["home_team"].astype(str).str.strip()
        + "|"
        + data["away_team"].astype(str).str.strip()
    )


# ============================================================
# 六、检查样本是否一致
# ============================================================

def validate_same_samples(
    v2_results: pd.DataFrame,
    v3_results: pd.DataFrame,
) -> None:
    """
    检查 v2 和 v3 是否使用完全相同的比赛。

    公平比较时，必须保证：
    1. 日期相同；
    2. 主队相同；
    3. 客队相同；
    4. 真实结果相同；
    5. 不存在重复比赛。
    """

    v2_data = v2_results.copy()
    v3_data = v3_results.copy()

    v2_data["match_key"] = build_match_key(
        v2_data
    )

    v3_data["match_key"] = build_match_key(
        v3_data
    )

    # 检查重复比赛
    v2_duplicate_count = int(
        v2_data["match_key"]
        .duplicated()
        .sum()
    )

    v3_duplicate_count = int(
        v3_data["match_key"]
        .duplicated()
        .sum()
    )

    # 获取比赛键集合
    v2_keys = set(
        v2_data["match_key"]
    )

    v3_keys = set(
        v3_data["match_key"]
    )

    only_in_v2 = (
        v2_keys - v3_keys
    )

    only_in_v3 = (
        v3_keys - v2_keys
    )

    print("=" * 80)
    print("样本一致性检查")
    print("=" * 80)

    print(
        "v2 回测记录数量：",
        len(v2_data),
    )

    print(
        "v3 回测记录数量：",
        len(v3_data),
    )

    print(
        "v2 唯一比赛数量：",
        len(v2_keys),
    )

    print(
        "v3 唯一比赛数量：",
        len(v3_keys),
    )

    print(
        "v2 重复比赛数量：",
        v2_duplicate_count,
    )

    print(
        "v3 重复比赛数量：",
        v3_duplicate_count,
    )

    print(
        "仅存在于 v2 的比赛：",
        len(only_in_v2),
    )

    print(
        "仅存在于 v3 的比赛：",
        len(only_in_v3),
    )

    if only_in_v2:
        print("\n仅存在于 v2：")

        for match_key in sorted(
            only_in_v2
        ):
            print("-", match_key)

    if only_in_v3:
        print("\n仅存在于 v3：")

        for match_key in sorted(
            only_in_v3
        ):
            print("-", match_key)

    if v2_duplicate_count > 0:
        raise ValueError(
            "v2 回测结果中存在重复比赛，"
            "不能进行公平对比。"
        )

    if v3_duplicate_count > 0:
        raise ValueError(
            "v3 回测结果中存在重复比赛，"
            "不能进行公平对比。"
        )

    if only_in_v2 or only_in_v3:
        raise ValueError(
            "v2 和 v3 使用的比赛样本不一致，"
            "不能直接比较。"
        )

    # 检查同一场比赛的真实结果是否一致
    merged = v2_data[
        [
            "match_key",
            "actual_result",
        ]
    ].merge(
        v3_data[
            [
                "match_key",
                "actual_result",
            ]
        ],
        on="match_key",
        suffixes=("_v2", "_v3"),
    )

    inconsistent_actual_results = merged[
        merged["actual_result_v2"]
        != merged["actual_result_v3"]
    ]

    if not inconsistent_actual_results.empty:
        print("\n真实结果不一致的比赛：")

        print(
            inconsistent_actual_results
            .to_string(index=False)
        )

        raise ValueError(
            "相同比赛在 v2 和 v3 中的真实结果不同。"
        )

    print("\n检查通过：v2 和 v3 使用完全相同的比赛。")


# ============================================================
# 七、主要指标对比
# ============================================================

def print_main_metrics(
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> None:
    """
    对比主要回测指标。
    """

    v2_accuracy = float(
        v2["accuracy"]
    )

    v3_accuracy = float(
        v3["accuracy"]
    )

    v2_log_loss = float(
        v2["log_loss"]
    )

    v3_log_loss = float(
        v3["log_loss"]
    )

    v2_brier = float(
        v2["brier_score"]
    )

    v3_brier = float(
        v3["brier_score"]
    )

    rows = [
        {
            "指标": "准确率",
            "v2": format_percentage(
                v2_accuracy
            ),
            "v3": format_percentage(
                v3_accuracy
            ),
            "v3-v2变化": format_percentage(
                v3_accuracy
                - v2_accuracy,
                show_sign=True,
            ),
            "越高/越低越好": "越高越好",
        },
        {
            "指标": "Log Loss",
            "v2": round(
                v2_log_loss,
                6,
            ),
            "v3": round(
                v3_log_loss,
                6,
            ),
            "v3-v2变化": round(
                v3_log_loss
                - v2_log_loss,
                6,
            ),
            "越高/越低越好": "越低越好",
        },
        {
            "指标": "Brier Score",
            "v2": round(
                v2_brier,
                6,
            ),
            "v3": round(
                v3_brier,
                6,
            ),
            "v3-v2变化": round(
                v3_brier
                - v2_brier,
                6,
            ),
            "越高/越低越好": "越低越好",
        },
    ]

    print("\n" + "=" * 80)
    print("提示词 v2 与 v3 主要指标对比")
    print("=" * 80)

    print(
        pd.DataFrame(rows)
        .to_string(index=False)
    )


# ============================================================
# 八、结果分布对比
# ============================================================

def print_distribution_comparison(
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> None:
    """
    对比真实结果分布和预测结果分布。
    """

    classes = [
        "HOME_WIN",
        "DRAW",
        "AWAY_WIN",
    ]

    rows = []

    for class_name in classes:
        rows.append({
            "类别": class_name,

            "v2真实数量": (
                v2[
                    "actual_distribution"
                ].get(
                    class_name,
                    0,
                )
            ),

            "v3真实数量": (
                v3[
                    "actual_distribution"
                ].get(
                    class_name,
                    0,
                )
            ),

            "v2预测数量": (
                v2[
                    "predicted_distribution"
                ].get(
                    class_name,
                    0,
                )
            ),

            "v3预测数量": (
                v3[
                    "predicted_distribution"
                ].get(
                    class_name,
                    0,
                )
            ),
        })

    print("\n" + "=" * 80)
    print("真实结果与模型预测分布对比")
    print("=" * 80)

    print(
        pd.DataFrame(rows)
        .to_string(index=False)
    )


# ============================================================
# 九、分类指标对比
# ============================================================

def print_class_metrics(
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> None:
    """
    对比主胜、平局、客胜的分类指标。
    """

    rows = []

    for class_name in [
        "HOME_WIN",
        "DRAW",
        "AWAY_WIN",
    ]:
        v2_metrics = (
            v2[
                "class_metrics"
            ][class_name]
        )

        v3_metrics = (
            v3[
                "class_metrics"
            ][class_name]
        )

        rows.append({
            "类别": class_name,

            "真实数量": v2_metrics[
                "support"
            ],

            "v2预测数量": v2_metrics[
                "predicted"
            ],

            "v3预测数量": v3_metrics[
                "predicted"
            ],

            "v2_precision": v2_metrics[
                "precision"
            ],

            "v3_precision": v3_metrics[
                "precision"
            ],

            "v2_recall": v2_metrics[
                "recall"
            ],

            "v3_recall": v3_metrics[
                "recall"
            ],

            "v2_f1": v2_metrics[
                "f1"
            ],

            "v3_f1": v3_metrics[
                "f1"
            ],
        })

    print("\n" + "=" * 80)
    print("各类别 Precision、Recall、F1 对比")
    print("=" * 80)

    print(
        pd.DataFrame(rows)
        .to_string(index=False)
    )


# ============================================================
# 十、逐场预测变化
# ============================================================

def compare_match_changes(
    v2_results: pd.DataFrame,
    v3_results: pd.DataFrame,
) -> None:
    """
    比较 v2 和 v3 的逐场预测变化。
    """

    v2_data = v2_results.copy()
    v3_data = v3_results.copy()

    v2_data["match_key"] = build_match_key(
        v2_data
    )

    v3_data["match_key"] = build_match_key(
        v3_data
    )

    merged = v2_data.merge(
        v3_data,
        on="match_key",
        suffixes=("_v2", "_v3"),
    )

    changed_predictions = merged[
        merged["predicted_result_v2"]
        != merged["predicted_result_v3"]
    ].copy()

    improved = merged[
        (merged["correct_v2"] == 0)
        & (merged["correct_v3"] == 1)
    ].copy()

    worsened = merged[
        (merged["correct_v2"] == 1)
        & (merged["correct_v3"] == 0)
    ].copy()

    unchanged_correct = merged[
        (merged["correct_v2"] == 1)
        & (merged["correct_v3"] == 1)
    ].copy()

    unchanged_wrong = merged[
        (merged["correct_v2"] == 0)
        & (merged["correct_v3"] == 0)
    ].copy()

    print("\n" + "=" * 80)
    print("逐场预测变化")
    print("=" * 80)

    print(
        "预测结果发生变化：",
        len(changed_predictions),
    )

    print(
        "由错误变正确：",
        len(improved),
    )

    print(
        "由正确变错误：",
        len(worsened),
    )

    print(
        "两版都正确：",
        len(unchanged_correct),
    )

    print(
        "两版都错误：",
        len(unchanged_wrong),
    )

    display_columns = [
        "date_v2",
        "home_team_v2",
        "away_team_v2",
        "actual_result_v2",
        "predicted_result_v2",
        "predicted_result_v3",
        "home_win_probability_v2",
        "draw_probability_v2",
        "away_win_probability_v2",
        "home_win_probability_v3",
        "draw_probability_v3",
        "away_win_probability_v3",
    ]

    if not changed_predictions.empty:
        print("\n所有预测发生变化的比赛：")

        print(
            changed_predictions[
                display_columns
            ].to_string(index=False)
        )

    if not improved.empty:
        print("\n由错误变正确的比赛：")

        print(
            improved[
                display_columns
            ].to_string(index=False)
        )

    if not worsened.empty:
        print("\n由正确变错误的比赛：")

        print(
            worsened[
                display_columns
            ].to_string(index=False)
        )


# ============================================================
# 十一、概率变化分析
# ============================================================

def print_probability_changes(
    v2_results: pd.DataFrame,
    v3_results: pd.DataFrame,
) -> None:
    """
    统计 v3 相比 v2 的平均概率变化。
    """

    v2_data = v2_results.copy()
    v3_data = v3_results.copy()

    v2_data["match_key"] = build_match_key(
        v2_data
    )

    v3_data["match_key"] = build_match_key(
        v3_data
    )

    merged = v2_data.merge(
        v3_data,
        on="match_key",
        suffixes=("_v2", "_v3"),
    )

    merged[
        "home_probability_change"
    ] = (
        merged[
            "home_win_probability_v3"
        ]
        - merged[
            "home_win_probability_v2"
        ]
    )

    merged[
        "draw_probability_change"
    ] = (
        merged[
            "draw_probability_v3"
        ]
        - merged[
            "draw_probability_v2"
        ]
    )

    merged[
        "away_probability_change"
    ] = (
        merged[
            "away_win_probability_v3"
        ]
        - merged[
            "away_win_probability_v2"
        ]
    )

    print("\n" + "=" * 80)
    print("v3 相对 v2 的平均概率变化")
    print("=" * 80)

    print(
        "主胜概率平均变化：",
        format_percentage(
            float(
                merged[
                    "home_probability_change"
                ].mean()
            ),
            show_sign=True,
        ),
    )

    print(
        "平局概率平均变化：",
        format_percentage(
            float(
                merged[
                    "draw_probability_change"
                ].mean()
            ),
            show_sign=True,
        ),
    )

    print(
        "客胜概率平均变化：",
        format_percentage(
            float(
                merged[
                    "away_probability_change"
                ].mean()
            ),
            show_sign=True,
        ),
    )


# ============================================================
# 十二、基线计算
# ============================================================

def calculate_majority_baseline(
    summary: dict[str, Any],
) -> tuple[str, float]:
    """
    计算始终预测最多类别时的基线准确率。

    例如真实分布：
    HOME_WIN = 10
    DRAW = 5
    AWAY_WIN = 5

    那么多数类别基线为：
    始终预测 HOME_WIN，准确率 50%。
    """

    distribution = summary[
        "actual_distribution"
    ]

    total = sum(
        int(value)
        for value in distribution.values()
    )

    if total == 0:
        return "UNKNOWN", 0.0

    majority_class = max(
        distribution,
        key=distribution.get,
    )

    majority_count = int(
        distribution[
            majority_class
        ]
    )

    accuracy = (
        majority_count / total
    )

    return (
        majority_class,
        accuracy,
    )


# ============================================================
# 十三、最终判断
# ============================================================

def print_final_judgement(
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> None:
    """
    根据主要指标判断是否值得扩大到100场。
    """

    v2_accuracy = float(
        v2["accuracy"]
    )

    v3_accuracy = float(
        v3["accuracy"]
    )

    v2_log_loss = float(
        v2["log_loss"]
    )

    v3_log_loss = float(
        v3["log_loss"]
    )

    v2_brier = float(
        v2["brier_score"]
    )

    v3_brier = float(
        v3["brier_score"]
    )

    v2_draw_recall = float(
        v2[
            "class_metrics"
        ]["DRAW"]["recall"]
    )

    v3_draw_recall = float(
        v3[
            "class_metrics"
        ]["DRAW"]["recall"]
    )

    v2_away_recall = float(
        v2[
            "class_metrics"
        ]["AWAY_WIN"]["recall"]
    )

    v3_away_recall = float(
        v3[
            "class_metrics"
        ]["AWAY_WIN"]["recall"]
    )

    majority_class, baseline_accuracy = (
        calculate_majority_baseline(
            v3
        )
    )

    accuracy_not_worse = (
        v3_accuracy >= v2_accuracy
    )

    accuracy_better = (
        v3_accuracy > v2_accuracy
    )

    log_loss_better = (
        v3_log_loss < v2_log_loss
    )

    brier_better = (
        v3_brier < v2_brier
    )

    draw_recall_better = (
        v3_draw_recall
        > v2_draw_recall
    )

    away_recall_better = (
        v3_away_recall
        > v2_away_recall
    )

    beats_majority_baseline = (
        v3_accuracy
        > baseline_accuracy
    )

    score = sum([
        accuracy_not_worse,
        log_loss_better,
        brier_better,
        draw_recall_better,
        away_recall_better,
        beats_majority_baseline,
    ])

    print("\n" + "=" * 80)
    print("综合判断")
    print("=" * 80)

    print(
        "多数类别基线：始终预测",
        majority_class,
    )

    print(
        "多数类别基线准确率：",
        format_percentage(
            baseline_accuracy
        ),
    )

    print(
        "v3 准确率：",
        format_percentage(
            v3_accuracy
        ),
    )

    print("\n指标改善情况：")

    print(
        "- 准确率不低于 v2：",
        accuracy_not_worse,
    )

    print(
        "- 准确率高于 v2：",
        accuracy_better,
    )

    print(
        "- Log Loss 更低：",
        log_loss_better,
    )

    print(
        "- Brier Score 更低：",
        brier_better,
    )

    print(
        "- 平局 Recall 提高：",
        draw_recall_better,
    )

    print(
        "- 客胜 Recall 提高：",
        away_recall_better,
    )

    print(
        "- 超过多数类别基线：",
        beats_majority_baseline,
    )

    print(
        "\n综合改善分数：",
        f"{score}/6",
    )

    if (
        score >= 4
        and accuracy_not_worse
        and (
            log_loss_better
            or brier_better
        )
    ):
        print(
            "\n结论：v3 整体优于 v2，"
            "可以扩大到 100 场回测。"
        )

    elif score >= 3:
        print(
            "\n结论：v3 有部分改善，"
            "但20场样本仍然太少。"
            "可以扩大到100场验证稳定性，"
            "但暂时不能认定模型已经有效。"
        )

    else:
        print(
            "\n结论：v3 没有表现出稳定改善，"
            "暂时不建议直接认定 v3 更好。"
            "应继续调整提示词或特征。"
        )


# ============================================================
# 十四、程序入口
# ============================================================

def main() -> None:
    """
    执行 v2 和 v3 完整对比。
    """

    print("正在读取 v2 汇总结果……")

    v2_summary = load_json(
        V2_SUMMARY_FILE
    )

    print("正在读取 v3 汇总结果……")

    v3_summary = load_json(
        V3_SUMMARY_FILE
    )

    print("正在读取 v2 逐场结果……")

    v2_results = load_results(
        V2_RESULT_FILE
    )

    print("正在读取 v3 逐场结果……")

    v3_results = load_results(
        V3_RESULT_FILE
    )

    # 必须先检查样本一致性
    validate_same_samples(
        v2_results=v2_results,
        v3_results=v3_results,
    )

    # 对比主要指标
    print_main_metrics(
        v2=v2_summary,
        v3=v3_summary,
    )

    # 对比结果分布
    print_distribution_comparison(
        v2=v2_summary,
        v3=v3_summary,
    )

    # 对比分类指标
    print_class_metrics(
        v2=v2_summary,
        v3=v3_summary,
    )

    # 对比逐场变化
    compare_match_changes(
        v2_results=v2_results,
        v3_results=v3_results,
    )

    # 对比平均概率变化
    print_probability_changes(
        v2_results=v2_results,
        v3_results=v3_results,
    )

    # 给出最终判断
    print_final_judgement(
        v2=v2_summary,
        v3=v3_summary,
    )


if __name__ == "__main__":
    main()

