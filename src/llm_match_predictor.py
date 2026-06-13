
"""
使用远程大模型和泊松分布预测足球比赛。

执行流程：

1. 读取并重建球队当前最新状态；
2. 根据近期攻防数据计算基础预期进球；
3. 大模型只输出有限范围内的进球修正值；
4. 将修正值加入基础预期进球；
5. 使用泊松分布生成完整比分概率；
6. 输出最可能比分、Top-5比分、胜平负概率。

大模型不直接决定最终比分。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from current_team_feature_builder import (
    CurrentTeamFeatureBuilder,
    create_current_feature_builder,
)
from scoreline_predictor import (
    predict_score_distribution,
)
from world_cup_teams import normalize_team_name


# ============================================================
# 一、路径与环境变量
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)


# ============================================================
# 二、大模型配置
# ============================================================

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
)

LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
)


# ============================================================
# 三、模型和比分参数
# ============================================================

# 大模型允许对基础预期进球做出的最大修正
MAX_GOAL_ADJUSTMENT = 0.35

# 最终预期进球允许的范围
MIN_EXPECTED_GOALS = 0.05
MAX_EXPECTED_GOALS = 4.50

# 泊松比分矩阵最大进球数
MAX_POISSON_GOALS = 8

# 输出前几个候选比分
TOP_SCORELINE_COUNT = 5


# ============================================================
# 四、工具函数
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    把数值限制在指定区间。
    """

    return max(
        minimum,
        min(value, maximum),
    )


def parse_json_content(
    content: str,
) -> dict[str, Any]:
    """
    将大模型返回内容解析为 Python 字典。

    兼容：
    1. 纯 JSON；
    2. ```json 代码块；
    3. 普通 Markdown 代码块。
    """

    if not content:
        raise ValueError(
            "大模型返回内容为空"
        )

    cleaned = content.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[
            len("```json"):
        ]

    elif cleaned.startswith("```"):
        cleaned = cleaned[
            len("```"):
        ]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError as error:
        raise ValueError(
            "大模型没有返回合法 JSON：\n"
            f"{content}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "大模型返回的 JSON 必须是对象"
        )

    return result


# ============================================================
# 五、构造当前比赛特征
# ============================================================

def build_prediction_features(
    feature_builder: CurrentTeamFeatureBuilder,
    home_team: str,
    away_team: str,
    prediction_date: str | None = None,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    weather: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    使用所有已结束比赛构造当前比赛特征。
    """

    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    home_features = (
        feature_builder.get_team_features(
            team_name=home_team,
            prediction_date=prediction_date,
        )
    )

    away_features = (
        feature_builder.get_team_features(
            team_name=away_team,
            prediction_date=prediction_date,
        )
    )

    return {
        "prediction_date": prediction_date,
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": neutral,

        "home_team_features": home_features,
        "away_team_features": away_features,

        "derived_features": {
            "elo_difference": round(
                home_features["elo"]
                - away_features["elo"],
                4,
            ),

            "win_rate_5_difference": round(
                home_features["win_rate_5"]
                - away_features["win_rate_5"],
                4,
            ),

            "win_rate_10_difference": round(
                home_features["win_rate_10"]
                - away_features["win_rate_10"],
                4,
            ),

            "avg_points_5_difference": round(
                home_features["avg_points_5"]
                - away_features["avg_points_5"],
                4,
            ),

            "avg_goals_for_5_difference": round(
                home_features["avg_goals_for_5"]
                - away_features["avg_goals_for_5"],
                4,
            ),

            "avg_goals_against_5_difference": round(
                home_features[
                    "avg_goals_against_5"
                ]
                - away_features[
                    "avg_goals_against_5"
                ],
                4,
            ),

            "rest_days_difference": (
                home_features["rest_days"]
                - away_features["rest_days"]
            ),
        },

        "weather": weather,
    }


# ============================================================
# 六、计算基础预期进球
# ============================================================

def get_tournament_goal_factor(
    tournament: str,
) -> float:
    """
    根据赛事类型设置进球系数。

    系数只是第一版经验值，后续应通过回测调整。
    """

    text = tournament.lower()

    if "world cup" in text:
        return 0.96

    if "friendly" in text:
        return 1.05

    if (
        "qualification" in text
        or "qualifier" in text
    ):
        return 1.00

    if "nations league" in text:
        return 0.98

    return 1.00


def calculate_base_expected_goals(
    features: dict[str, Any],
) -> dict[str, float]:
    """
    使用数值特征计算双方基础预期进球。

    主要使用：
    1. 本队最近5场、10场进攻；
    2. 对手最近5场、10场防守；
    3. Elo 差值；
    4. 主场或中立场；
    5. 赛事类型。
    """

    home = features["home_team_features"]
    away = features["away_team_features"]

    # --------------------------------------------------------
    # 1. 近期进攻能力
    # --------------------------------------------------------

    home_attack = (
        0.65 * float(
            home["avg_goals_for_10"]
        )
        + 0.35 * float(
            home["avg_goals_for_5"]
        )
    )

    away_attack = (
        0.65 * float(
            away["avg_goals_for_10"]
        )
        + 0.35 * float(
            away["avg_goals_for_5"]
        )
    )

    # --------------------------------------------------------
    # 2. 对手近期防守水平
    # --------------------------------------------------------

    home_defense_conceded = (
        0.65 * float(
            home["avg_goals_against_10"]
        )
        + 0.35 * float(
            home["avg_goals_against_5"]
        )
    )

    away_defense_conceded = (
        0.65 * float(
            away["avg_goals_against_10"]
        )
        + 0.35 * float(
            away["avg_goals_against_5"]
        )
    )

    # --------------------------------------------------------
    # 3. 进攻与对手防守融合
    # --------------------------------------------------------

    base_home_xg = (
        0.58 * home_attack
        + 0.42 * away_defense_conceded
    )

    base_away_xg = (
        0.58 * away_attack
        + 0.42 * home_defense_conceded
    )

    # --------------------------------------------------------
    # 4. Elo 修正
    # --------------------------------------------------------

    elo_difference = float(
        features[
            "derived_features"
        ]["elo_difference"]
    )

    elo_adjustment = clamp(
        elo_difference / 700.0,
        -0.30,
        0.30,
    )

    base_home_xg += elo_adjustment
    base_away_xg -= elo_adjustment

    # --------------------------------------------------------
    # 5. 场地修正
    # --------------------------------------------------------

    neutral = bool(
        features.get("neutral", True)
    )

    if not neutral:
        base_home_xg += 0.15
        base_away_xg -= 0.05

    # --------------------------------------------------------
    # 6. 赛事节奏修正
    # --------------------------------------------------------

    tournament_factor = (
        get_tournament_goal_factor(
            str(
                features.get(
                    "tournament",
                    "",
                )
            )
        )
    )

    base_home_xg *= tournament_factor
    base_away_xg *= tournament_factor

    # --------------------------------------------------------
    # 7. 限制合理范围
    # --------------------------------------------------------

    base_home_xg = clamp(
        base_home_xg,
        MIN_EXPECTED_GOALS,
        MAX_EXPECTED_GOALS,
    )

    base_away_xg = clamp(
        base_away_xg,
        MIN_EXPECTED_GOALS,
        MAX_EXPECTED_GOALS,
    )

    return {
        "home_attack": round(
            home_attack,
            4,
        ),

        "away_attack": round(
            away_attack,
            4,
        ),

        "home_defense_conceded": round(
            home_defense_conceded,
            4,
        ),

        "away_defense_conceded": round(
            away_defense_conceded,
            4,
        ),

        "elo_adjustment": round(
            elo_adjustment,
            4,
        ),

        "tournament_factor": round(
            tournament_factor,
            4,
        ),

        "base_home_expected_goals": round(
            base_home_xg,
            4,
        ),

        "base_away_expected_goals": round(
            base_away_xg,
            4,
        ),
    }


# ============================================================
# 七、大模型预测器
# ============================================================

class LLMMatchPredictor:
    """
    大模型辅助的足球比分预测器。

    大模型只负责对基础预期进球做有限修正，
    最终比分由泊松分布计算。
    """

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
    ):
        if not base_url:
            raise ValueError(
                "LLM_BASE_URL 不能为空"
            )

        if not api_key:
            raise ValueError(
                "LLM_API_KEY 或 ANTHROPIC_API_KEY 不能为空"
            )

        if not model:
            raise ValueError(
                "LLM_MODEL 不能为空"
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def build_prompt(
        self,
        features: dict[str, Any],
        base_xg: dict[str, float],
    ) -> str:
        """
        构造大模型修正提示词。

        大模型不输出最终比分，
        只输出有限范围内的预期进球修正。
        """

        feature_json = json.dumps(
            features,
            ensure_ascii=False,
            indent=2,
        )

        base_xg_json = json.dumps(
            base_xg,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
你是一名足球预期进球修正助手。

程序已经根据球队长期实力和近期攻防数据，
计算出了双方的基础预期进球。

你的职责不是直接猜测最终比分，
也不是重新计算完整预期进球。

你只能判断基础预期进球是否需要小幅修正。

【比赛特征】

{feature_json}

【程序计算出的基础预期进球】

{base_xg_json}

【修正规则】

1. home_goal_adjustment 和 away_goal_adjustment
   必须在 -0.35 到 +0.35 之间。

2. 当输入没有天气、伤停、首发或其他额外信息时，
   修正值应尽量接近 0。

3. 不得根据球队名气或外部知识修正。

4. neutral=true 时，不得给 home_team 额外主场优势。

5. Elo、最近5场和最近10场已经被基础公式使用，
   不应再次进行大幅重复调整。

6. 只有输入中存在明显冲突或基础公式可能高估、低估时，
   才做小幅修正。

7. 不要输出最终比分。

8. 不要输出胜平负概率。

9. confidence 表示你对修正必要性的信心。

只返回合法 JSON：

{{
  "home_goal_adjustment": 0.0,
  "away_goal_adjustment": 0.0,
  "confidence": 0.0,
  "adjustment_reasons": [
    "修正原因1",
    "修正原因2"
  ]
}}
"""

    def call_llm(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """
        调用远程 OpenAI 兼容接口。
        """

        url = (
            f"{self.base_url}"
            f"/chat/completions"
        )

        response = requests.post(
            url=url,

            headers={
                "Authorization": (
                    f"Bearer {self.api_key}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },

            json={
                "model": self.model,

                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你只负责对足球基础预期进球"
                            "做有限范围修正。"
                            "禁止直接输出最终比分。"
                            "必须只返回合法 JSON。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                "temperature": 0,
                "max_tokens": 500,
            },

            timeout=(20, 180),
        )

        if response.status_code != 200:
            raise RuntimeError(
                "大模型接口请求失败：\n"
                f"状态码：{response.status_code}\n"
                f"请求地址：{url}\n"
                f"请求模型：{self.model}\n"
                f"响应内容：{response.text}"
            )

        result = response.json()

        try:
            message = (
                result["choices"][0]["message"]
            )

            content = message["content"]

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise ValueError(
                "大模型返回格式不符合预期：\n"
                f"{result}"
            ) from error

        return parse_json_content(
            content
        )

    def validate_adjustments(
        self,
        adjustment_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        校验大模型返回的预期进球修正。
        """

        try:
            home_adjustment = float(
                adjustment_data.get(
                    "home_goal_adjustment",
                    0.0,
                )
            )

            away_adjustment = float(
                adjustment_data.get(
                    "away_goal_adjustment",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "大模型返回的进球修正值不是数字"
            ) from error

        home_adjustment = clamp(
            home_adjustment,
            -MAX_GOAL_ADJUSTMENT,
            MAX_GOAL_ADJUSTMENT,
        )

        away_adjustment = clamp(
            away_adjustment,
            -MAX_GOAL_ADJUSTMENT,
            MAX_GOAL_ADJUSTMENT,
        )

        confidence = adjustment_data.get(
            "confidence",
            0.0,
        )

        try:
            confidence = float(confidence)

        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = clamp(
            confidence,
            0.0,
            1.0,
        )

        reasons = adjustment_data.get(
            "adjustment_reasons",
            [],
        )

        if not isinstance(reasons, list):
            reasons = []

        return {
            "home_goal_adjustment": round(
                home_adjustment,
                4,
            ),

            "away_goal_adjustment": round(
                away_adjustment,
                4,
            ),

            "confidence": round(
                confidence,
                4,
            ),

            "adjustment_reasons": [
                str(reason)
                for reason in reasons[:5]
            ],
        }

    def predict(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        执行完整比分预测。
        """

        # 1. 程序计算基础预期进球
        base_xg = calculate_base_expected_goals(
            features
        )

        # 2. 大模型只提供有限修正
        prompt = self.build_prompt(
            features=features,
            base_xg=base_xg,
        )

        raw_adjustments = self.call_llm(
            prompt
        )

        adjustments = self.validate_adjustments(
            raw_adjustments
        )

        # 3. 合并基础预期进球与大模型修正
        final_home_xg = clamp(
            (
                base_xg[
                    "base_home_expected_goals"
                ]
                + adjustments[
                    "home_goal_adjustment"
                ]
            ),
            MIN_EXPECTED_GOALS,
            MAX_EXPECTED_GOALS,
        )

        final_away_xg = clamp(
            (
                base_xg[
                    "base_away_expected_goals"
                ]
                + adjustments[
                    "away_goal_adjustment"
                ]
            ),
            MIN_EXPECTED_GOALS,
            MAX_EXPECTED_GOALS,
        )

        # 4. 泊松分布生成比分概率
        score_distribution = (
            predict_score_distribution(
                expected_home_goals=(
                    final_home_xg
                ),
                expected_away_goals=(
                    final_away_xg
                ),
                max_goals=(
                    MAX_POISSON_GOALS
                ),
                top_n=(
                    TOP_SCORELINE_COUNT
                ),
            )
        )

        # 5. 整理最终输出
        return {
            "home_team": features[
                "home_team"
            ],

            "away_team": features[
                "away_team"
            ],

            "prediction_date": features.get(
                "prediction_date"
            ),

            "tournament": features.get(
                "tournament"
            ),

            "neutral": features.get(
                "neutral"
            ),

            "base_home_expected_goals": (
                base_xg[
                    "base_home_expected_goals"
                ]
            ),

            "base_away_expected_goals": (
                base_xg[
                    "base_away_expected_goals"
                ]
            ),

            "home_goal_adjustment": (
                adjustments[
                    "home_goal_adjustment"
                ]
            ),

            "away_goal_adjustment": (
                adjustments[
                    "away_goal_adjustment"
                ]
            ),

            "expected_home_goals": round(
                final_home_xg,
                4,
            ),

            "expected_away_goals": round(
                final_away_xg,
                4,
            ),

            "expected_total_goals": round(
                (
                    final_home_xg
                    + final_away_xg
                ),
                4,
            ),

            "predicted_home_score": (
                score_distribution[
                    "predicted_home_score"
                ]
            ),

            "predicted_away_score": (
                score_distribution[
                    "predicted_away_score"
                ]
            ),

            "predicted_scoreline": (
                score_distribution[
                    "predicted_scoreline"
                ]
            ),

            "scoreline_probability": (
                score_distribution[
                    "scoreline_probability"
                ]
            ),

            "predicted_result": (
                score_distribution[
                    "predicted_result"
                ]
            ),

            "home_win_probability": (
                score_distribution[
                    "home_win_probability"
                ]
            ),

            "draw_probability": (
                score_distribution[
                    "draw_probability"
                ]
            ),

            "away_win_probability": (
                score_distribution[
                    "away_win_probability"
                ]
            ),

            "over_2_5_probability": (
                score_distribution[
                    "over_2_5_probability"
                ]
            ),

            "under_2_5_probability": (
                score_distribution[
                    "under_2_5_probability"
                ]
            ),

            "both_teams_score_probability": (
                score_distribution[
                    "both_teams_score_probability"
                ]
            ),

            "alternate_scorelines": (
                score_distribution[
                    "top_scorelines"
                ]
            ),

            "confidence": adjustments[
                "confidence"
            ],

            "adjustment_reasons": adjustments[
                "adjustment_reasons"
            ],

            "base_xg_details": base_xg,

            "model": self.model,
        }


# ============================================================
# 八、打印结果
# ============================================================

def print_prediction(
    prediction: dict[str, Any],
) -> None:
    """
    格式化打印预测结果。
    """

    print("=" * 72)

    print(
        f"比赛："
        f"{prediction['home_team']} "
        f"vs "
        f"{prediction['away_team']}"
    )

    print(
        "预测日期：",
        prediction.get(
            "prediction_date"
        ),
    )

    print(
        "赛事：",
        prediction.get(
            "tournament"
        ),
    )

    print(
        "是否中立场：",
        prediction.get(
            "neutral"
        ),
    )

    print("-" * 72)

    print(
        "基础主队预期进球：",
        prediction[
            "base_home_expected_goals"
        ],
    )

    print(
        "大模型主队修正：",
        prediction[
            "home_goal_adjustment"
        ],
    )

    print(
        "最终主队预期进球：",
        prediction[
            "expected_home_goals"
        ],
    )

    print()

    print(
        "基础客队预期进球：",
        prediction[
            "base_away_expected_goals"
        ],
    )

    print(
        "大模型客队修正：",
        prediction[
            "away_goal_adjustment"
        ],
    )

    print(
        "最终客队预期进球：",
        prediction[
            "expected_away_goals"
        ],
    )

    print(
        "预期总进球：",
        prediction[
            "expected_total_goals"
        ],
    )

    print("-" * 72)

    print(
        "最可能比分：",
        prediction[
            "predicted_scoreline"
        ],
    )

    print(
        "该比分概率：",
        (
            f"{prediction['scoreline_probability'] * 100:.2f}%"
        ),
    )

    print(
        "预测结果：",
        prediction[
            "predicted_result"
        ],
    )

    print()

    print(
        "主胜概率：",
        (
            f"{prediction['home_win_probability'] * 100:.2f}%"
        ),
    )

    print(
        "平局概率：",
        (
            f"{prediction['draw_probability'] * 100:.2f}%"
        ),
    )

    print(
        "客胜概率：",
        (
            f"{prediction['away_win_probability'] * 100:.2f}%"
        ),
    )

    print()

    print(
        "大于2.5球概率：",
        (
            f"{prediction['over_2_5_probability'] * 100:.2f}%"
        ),
    )

    print(
        "小于等于2.5球概率：",
        (
            f"{prediction['under_2_5_probability'] * 100:.2f}%"
        ),
    )

    print(
        "双方都进球概率：",
        (
            f"{prediction['both_teams_score_probability'] * 100:.2f}%"
        ),
    )

    print("-" * 72)
    print("Top-5 比分：")

    for index, item in enumerate(
        prediction[
            "alternate_scorelines"
        ],
        start=1,
    ):
        print(
            f"{index}. "
            f"{item['scoreline']} "
            f"({item['result']}) "
            f"{item['probability'] * 100:.2f}%"
        )

    print("-" * 72)
    print("大模型修正原因：")

    reasons = prediction.get(
        "adjustment_reasons",
        []
    )

    if not reasons:
        print("- 无明显修正理由")

    else:
        for reason in reasons:
            print("-", reason)

    print(
        "修正置信度：",
        prediction.get(
            "confidence"
        ),
    )

    print(
        "调用模型：",
        prediction.get(
            "model"
        ),
    )

    print("=" * 72)


# ============================================================
# 九、主程序
# ============================================================

def main() -> None:
    """
    测试巴西对摩洛哥。
    """

    prediction_date = "2026-06-13"

    print("正在计算球队最新特征……")

    feature_builder = (
        create_current_feature_builder(
            cutoff_date=prediction_date
        )
    )

    print(
        f"读取到 "
        f"{len(feature_builder.matches)} 场"
        f"预测日期之前的已结束比赛"
    )

    match_features = build_prediction_features(
        feature_builder=feature_builder,
        home_team="Brazil",
        away_team="Morocco",
        prediction_date=prediction_date,
        neutral=True,
        tournament="FIFA World Cup",
        weather=None,
    )

    print("\n正在调用大模型进行有限修正……")

    predictor = LLMMatchPredictor()

    prediction = predictor.predict(
        match_features
    )

    print_prediction(
        prediction
    )


if __name__ == "__main__":
    main()
