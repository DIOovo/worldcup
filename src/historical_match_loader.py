#要改一下

from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "international_results.csv"
)


def load_historical_matches() -> pd.DataFrame:
    """
    读取国家队历史比赛数据。
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"没有找到历史比赛文件：{DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    return data


def print_dataset_summary(data: pd.DataFrame) -> None:
    """
    输出数据集基本信息。
    """

    print("比赛总数：", len(data))
    print("最早比赛时间：", data["date"].min())
    print("最近比赛时间：", data["date"].max())
    print("国家队数量：", data["home_team"].nunique())

    print("\n字段：")
    print(data.columns.tolist())

    print("\n最近 10 场比赛：")

    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral"
    ]

    print(
        data[columns]
        .sort_values("date")
        .tail(10)
        .to_string(index=False)
    )
def clean_historical_matches(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    清洗国家队历史比赛数据。

    只保留：
    1. 日期有效的比赛；
    2. 主队和客队名称有效的比赛；
    3. 已经存在完整比分的比赛。
    """

    cleaned_data = data.copy()

    # 删除日期解析失败的数据
    cleaned_data = cleaned_data.dropna(
        subset=["date"]
    )

    # 删除主队、客队为空的数据
    cleaned_data = cleaned_data.dropna(
        subset=["home_team", "away_team"]
    )

    # 删除尚未结束、没有比分的未来比赛
    cleaned_data = cleaned_data.dropna(
        subset=["home_score", "away_score"]
    )

    # 将比分转换为整数
    cleaned_data["home_score"] = (
        cleaned_data["home_score"].astype(int)
    )

    cleaned_data["away_score"] = (
        cleaned_data["away_score"].astype(int)
    )

    # 按时间排序
    cleaned_data = cleaned_data.sort_values(
        "date"
    ).reset_index(drop=True)

    return cleaned_data

def add_match_result(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    根据全场比分生成比赛结果标签。

    H：主队获胜
    D：平局
    A：客队获胜
    """

    result_data = data.copy()

    def determine_result(row):
        if row["home_score"] > row["away_score"]:
            return "H"

        if row["home_score"] < row["away_score"]:
            return "A"

        return "D"

    result_data["result"] = result_data.apply(
        determine_result,
        axis=1
    )

    return result_data

def get_scheduled_matches(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    获取没有最终比分的未来比赛。
    """

    scheduled_matches = data[
        data["home_score"].isna()
        | data["away_score"].isna()
    ].copy()

    return scheduled_matches.sort_values(
        "date"
    ).reset_index(drop=True)
def save_cleaned_matches(
    data: pd.DataFrame
) -> None:
    """保存清洗后的比赛数据。"""

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "historical_matches_clean.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("清洗后数据保存到：", output_file)


if __name__ == "__main__":
    raw_matches = load_historical_matches()

    matches = clean_historical_matches(
        raw_matches
    )

    matches = add_match_result(matches)
    save_cleaned_matches(matches)
    print_dataset_summary(matches)

    print("\n最近 10 场已结束比赛：")

    print(
        matches[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "result"
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )