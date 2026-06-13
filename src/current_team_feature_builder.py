"""
计算球队当前最新状态特征。

与历史训练特征的区别：

历史训练特征：
    某场比赛开赛前的球队状态。

当前球队特征：
    处理完所有已经结束的比赛后，
    球队在最新时间点的状态。

主要输出：
1. 最新 Elo；
2. 最近5场表现；
3. 最近10场表现；
4. 最近比赛日期；
5. 休息天数；
6. 总历史比赛数量。
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any, Deque

import pandas as pd

from match_feature_engineering import (
    DEFAULT_ELO,
    MAX_HISTORY_SIZE,
    calculate_recent_features,
    clean_matches,
    get_team_points,
    update_elo_ratings,
)
from world_cup_teams import normalize_team_name


# ============================================================
# 一、文件路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

HISTORICAL_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "international_results.csv"
)


# ============================================================
# 二、读取历史比赛
# ============================================================

def load_finished_matches(
    cutoff_date: str | datetime | date | None = None,
) -> pd.DataFrame:
    """
    读取已经结束的比赛。

    cutoff_date:
        可选的截止日期。

        例如传入 2026-06-13，则只使用：
        date < 2026-06-13

        这在历史回测时非常重要，可以防止使用未来比赛。

    当前实时预测时不传 cutoff_date，
    程序会使用 CSV 中所有已经有比分的比赛。
    """

    if not HISTORICAL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"没有找到历史比赛文件：{HISTORICAL_DATA_FILE}"
        )

    raw_matches = pd.read_csv(
        HISTORICAL_DATA_FILE
    )

    # 复用之前特征工程中的清洗逻辑：
    # 删除无比分的未来比赛、解析日期和比分。
    matches = clean_matches(raw_matches)

    if cutoff_date is not None:
        cutoff_timestamp = pd.Timestamp(
            cutoff_date
        )

        matches = matches[
            matches["date"] < cutoff_timestamp
        ].copy()

    return matches.sort_values(
        "date"
    ).reset_index(drop=True)


# ============================================================
# 四、当前球队状态构建器
# ============================================================

class CurrentTeamFeatureBuilder:
    """
    根据已结束比赛，计算球队当前最新特征。
    """

    def __init__(
        self,
        matches: pd.DataFrame,
    ):
        self.matches = matches

        # 保存球队最近最多20场比赛
        self.team_histories: dict[
            str,
            Deque[dict[str, Any]]
        ] = defaultdict(
            lambda: deque(
                maxlen=MAX_HISTORY_SIZE
            )
        )

        # 保存球队当前 Elo
        self.team_elos: dict[
            str,
            float
        ] = defaultdict(
            lambda: DEFAULT_ELO
        )

        # 保存球队最后一场比赛日期
        self.last_match_dates: dict[
            str,
            pd.Timestamp
        ] = {}

        # 保存球队总比赛数量
        self.total_match_counts: dict[
            str,
            int
        ] = defaultdict(int)

        # 避免重复计算
        self._built = False

    def build(self) -> None:
        """
        按时间顺序处理全部已结束比赛。

        这里与历史训练特征不同：

        历史训练特征是先生成赛前特征，再更新状态；
        当前状态只需要把全部比赛结果都更新进去。
        """

        if self._built:
            return

        for _, match in self.matches.iterrows():
            self._process_match(match)

        self._built = True

    def _process_match(
        self,
        match: pd.Series,
    ) -> None:
        """
        使用一场已结束比赛更新双方球队状态。
        """

        match_date = match["date"]

        home_team = normalize_team_name(
            str(match["home_team"])
        )

        away_team = normalize_team_name(
            str(match["away_team"])
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
        # 1. 获取比赛前 Elo
        # ----------------------------------------------------

        home_elo = self.team_elos[
            home_team
        ]

        away_elo = self.team_elos[
            away_team
        ]

        # ----------------------------------------------------
        # 2. 计算本场积分
        # ----------------------------------------------------

        home_points = get_team_points(
            goals_for=home_score,
            goals_against=away_score,
        )

        away_points = get_team_points(
            goals_for=away_score,
            goals_against=home_score,
        )

        # ----------------------------------------------------
        # 3. 把本场比赛加入近期历史
        # ----------------------------------------------------

        self.team_histories[
            home_team
        ].append({
            "date": match_date,
            "opponent": away_team,
            "goals_for": home_score,
            "goals_against": away_score,
            "points": home_points,
            "win": int(home_points == 3),
            "draw": int(home_points == 1),
            "loss": int(home_points == 0),
            "tournament": tournament,
            "neutral": neutral,
        })

        self.team_histories[
            away_team
        ].append({
            "date": match_date,
            "opponent": home_team,
            "goals_for": away_score,
            "goals_against": home_score,
            "points": away_points,
            "win": int(away_points == 3),
            "draw": int(away_points == 1),
            "loss": int(away_points == 0),
            "tournament": tournament,
            "neutral": neutral,
        })

        # ----------------------------------------------------
        # 4. 使用本场结果更新 Elo
        # ----------------------------------------------------

        new_home_elo, new_away_elo = (
            update_elo_ratings(
                home_elo=home_elo,
                away_elo=away_elo,
                home_score=home_score,
                away_score=away_score,
                tournament=tournament,
                neutral=neutral,
            )
        )

        self.team_elos[
            home_team
        ] = new_home_elo

        self.team_elos[
            away_team
        ] = new_away_elo

        # ----------------------------------------------------
        # 5. 更新最近比赛日期和比赛数量
        # ----------------------------------------------------

        self.last_match_dates[
            home_team
        ] = match_date

        self.last_match_dates[
            away_team
        ] = match_date

        self.total_match_counts[
            home_team
        ] += 1

        self.total_match_counts[
            away_team
        ] += 1

    def get_team_features(
        self,
        team_name: str,
        prediction_date: str | datetime | date | None = None,
    ) -> dict[str, Any]:
        """
        获取球队当前最新状态。

        prediction_date:
            待预测比赛日期，用于计算休息天数。

            不传时使用当前日期。
        """

        self.build()

        normalized_name = normalize_team_name(
            team_name
        )

        if normalized_name not in self.last_match_dates:
            raise ValueError(
                f"没有找到球队已结束的历史比赛：{team_name}"
            )

        history = self.team_histories[
            normalized_name
        ]

        recent_5 = calculate_recent_features(
            history=history,
            window=5,
        )

        recent_10 = calculate_recent_features(
            history=history,
            window=10,
        )

        last_match_date = self.last_match_dates[
            normalized_name
        ]

        if prediction_date is None:
            prediction_timestamp = pd.Timestamp.now().normalize()
        else:
            prediction_timestamp = pd.Timestamp(
                prediction_date
            )

        rest_days = (
            prediction_timestamp
            - last_match_date
        ).days

        # 避免日期异常产生负数
        rest_days = max(rest_days, 0)

        return {
            "team": team_name,
            "normalized_team": normalized_name,

            # 现在这个日期是真正的最近已结束比赛日期
            "latest_finished_match_date": (
                last_match_date.strftime(
                    "%Y-%m-%d"
                )
            ),

            # 这是处理完最近一场比赛后的 Elo
            "elo": round(
                float(
                    self.team_elos[
                        normalized_name
                    ]
                ),
                6,
            ),

            "total_historical_matches": int(
                self.total_match_counts[
                    normalized_name
                ]
            ),

            "rest_days": int(
                rest_days
            ),

            "matches_5": int(
                recent_5["matches"]
            ),
            "win_rate_5": recent_5[
                "win_rate"
            ],
            "draw_rate_5": recent_5[
                "draw_rate"
            ],
            "loss_rate_5": recent_5[
                "loss_rate"
            ],
            "avg_goals_for_5": recent_5[
                "avg_goals_for"
            ],
            "avg_goals_against_5": recent_5[
                "avg_goals_against"
            ],
            "avg_goal_difference_5": recent_5[
                "avg_goal_difference"
            ],
            "avg_points_5": recent_5[
                "avg_points"
            ],
            "clean_sheet_rate_5": recent_5[
                "clean_sheet_rate"
            ],
            "scoring_rate_5": recent_5[
                "scoring_rate"
            ],

            "matches_10": int(
                recent_10["matches"]
            ),
            "win_rate_10": recent_10[
                "win_rate"
            ],
            "draw_rate_10": recent_10[
                "draw_rate"
            ],
            "loss_rate_10": recent_10[
                "loss_rate"
            ],
            "avg_goals_for_10": recent_10[
                "avg_goals_for"
            ],
            "avg_goals_against_10": recent_10[
                "avg_goals_against"
            ],
            "avg_goal_difference_10": recent_10[
                "avg_goal_difference"
            ],
            "avg_points_10": recent_10[
                "avg_points"
            ],
            "clean_sheet_rate_10": recent_10[
                "clean_sheet_rate"
            ],
            "scoring_rate_10": recent_10[
                "scoring_rate"
            ],
        }


# ============================================================
# 五、快速创建构建器
# ============================================================

def create_current_feature_builder(
    cutoff_date: str | datetime | date | None = None,
) -> CurrentTeamFeatureBuilder:
    """
    创建当前球队特征构建器。
    """

    matches = load_finished_matches(
        cutoff_date=cutoff_date
    )

    builder = CurrentTeamFeatureBuilder(
        matches=matches
    )

    builder.build()

    return builder


# ============================================================
# 六、测试
# ============================================================

def main() -> None:
    """
    测试巴西和摩洛哥的最新状态。
    """

    print("正在读取所有已结束比赛……")

    builder = create_current_feature_builder()

    print(
        f"已结束比赛数量：{len(builder.matches)}"
    )

    print("\n巴西最新特征：")

    brazil = builder.get_team_features(
        team_name="Brazil",
        prediction_date="2026-06-13",
    )

    print(
        pd.Series(brazil).to_string()
    )

    print("\n摩洛哥最新特征：")

    morocco = builder.get_team_features(
        team_name="Morocco",
        prediction_date="2026-06-13",
    )

    print(
        pd.Series(morocco).to_string()
    )


if __name__ == "__main__":
    main()
