
"""
足球比分概率计算器。

根据主队和客队的预期进球，使用独立泊松分布计算：

1. 每个比分的概率；
2. 最可能比分；
3. Top-N 候选比分；
4. 主胜、平局、客胜概率；
5. 大小球概率；
6. 双方是否进球概率。

说明：
泊松分布不是用于保证某个比分必然发生，
而是用于描述不同比分发生的概率。
"""

from __future__ import annotations

import math
from typing import Any


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    """
    计算进 goals 球的泊松概率。

    参数：
    goals:
        实际进球数量，例如 0、1、2。

    expected_goals:
        预期进球，也就是泊松分布的 lambda。
    """

    if goals < 0:
        return 0.0

    if expected_goals < 0:
        raise ValueError(
            "expected_goals 不能小于 0"
        )

    return (
        math.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )


def get_result_name(
    home_score: int,
    away_score: int,
) -> str:
    """
    根据比分返回胜平负结果。
    """

    if home_score > away_score:
        return "HOME_WIN"

    if home_score < away_score:
        return "AWAY_WIN"

    return "DRAW"


def calculate_scoreline_probabilities(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 8,
) -> list[dict[str, Any]]:
    """
    计算从 0-0 到 max_goals-max_goals 的比分概率。

    返回结果按概率从高到低排序。
    """

    if expected_home_goals < 0:
        raise ValueError(
            "主队预期进球不能小于 0"
        )

    if expected_away_goals < 0:
        raise ValueError(
            "客队预期进球不能小于 0"
        )

    if max_goals < 1:
        raise ValueError(
            "max_goals 必须至少为 1"
        )

    scorelines: list[dict[str, Any]] = []

    for home_score in range(
        max_goals + 1
    ):
        home_probability = poisson_probability(
            goals=home_score,
            expected_goals=expected_home_goals,
        )

        for away_score in range(
            max_goals + 1
        ):
            away_probability = poisson_probability(
                goals=away_score,
                expected_goals=expected_away_goals,
            )

            probability = (
                home_probability
                * away_probability
            )

            scorelines.append({
                "home_score": home_score,
                "away_score": away_score,
                "scoreline": (
                    f"{home_score}-{away_score}"
                ),
                "probability": probability,
                "result": get_result_name(
                    home_score=home_score,
                    away_score=away_score,
                ),
                "total_goals": (
                    home_score + away_score
                ),
                "both_teams_score": (
                    home_score > 0
                    and away_score > 0
                ),
            })

    probability_total = sum(
        item["probability"]
        for item in scorelines
    )

    if probability_total <= 0:
        raise ValueError(
            "比分概率总和无效"
        )

    # 截断到 max_goals 后重新归一化
    for item in scorelines:
        item["probability"] = (
            item["probability"]
            / probability_total
        )

    scorelines.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return scorelines


def predict_score_distribution(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 8,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    根据预期进球生成完整比分预测结果。
    """

    scorelines = calculate_scoreline_probabilities(
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        max_goals=max_goals,
    )

    home_win_probability = sum(
        item["probability"]
        for item in scorelines
        if item["result"] == "HOME_WIN"
    )

    draw_probability = sum(
        item["probability"]
        for item in scorelines
        if item["result"] == "DRAW"
    )

    away_win_probability = sum(
        item["probability"]
        for item in scorelines
        if item["result"] == "AWAY_WIN"
    )

    over_2_5_probability = sum(
        item["probability"]
        for item in scorelines
        if item["total_goals"] >= 3
    )

    under_2_5_probability = (
        1.0 - over_2_5_probability
    )

    both_teams_score_probability = sum(
        item["probability"]
        for item in scorelines
        if item["both_teams_score"]
    )

    most_likely = scorelines[0]

    result_probabilities = {
        "HOME_WIN": home_win_probability,
        "DRAW": draw_probability,
        "AWAY_WIN": away_win_probability,
    }

    predicted_result = max(
        result_probabilities,
        key=result_probabilities.get,
    )

    return {
        "predicted_home_score": (
            most_likely["home_score"]
        ),

        "predicted_away_score": (
            most_likely["away_score"]
        ),

        "predicted_scoreline": (
            most_likely["scoreline"]
        ),

        "scoreline_probability": round(
            most_likely["probability"],
            6,
        ),

        "predicted_result": predicted_result,

        "home_win_probability": round(
            home_win_probability,
            6,
        ),

        "draw_probability": round(
            draw_probability,
            6,
        ),

        "away_win_probability": round(
            away_win_probability,
            6,
        ),

        "over_2_5_probability": round(
            over_2_5_probability,
            6,
        ),

        "under_2_5_probability": round(
            under_2_5_probability,
            6,
        ),

        "both_teams_score_probability": round(
            both_teams_score_probability,
            6,
        ),

        "top_scorelines": [
            {
                "home_score": item["home_score"],
                "away_score": item["away_score"],
                "scoreline": item["scoreline"],
                "result": item["result"],
                "probability": round(
                    item["probability"],
                    6,
                ),
            }
            for item in scorelines[:top_n]
        ],
    }


def main() -> None:
    """
    单独测试比分计算器。
    """

    result = predict_score_distribution(
        expected_home_goals=1.45,
        expected_away_goals=0.92,
        max_goals=8,
        top_n=5,
    )

    print(result)


if __name__ == "__main__":
    main()

