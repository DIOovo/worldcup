import os
from pathlib import Path

import requests
from dotenv import load_dotenv
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# 获取当前 Python 文件的绝对路径
CURRENT_FILE = Path(__file__).resolve()

# 当前文件在 src 目录中，因此 parent 是 src
SRC_DIR = CURRENT_FILE.parent

# src 的上一级是项目根目录 agent
PROJECT_ROOT = SRC_DIR.parent

# 拼接出 .env 文件地址
ENV_PATH = PROJECT_ROOT / ".env"

# 指定读取这个 .env 文件
load_dotenv(dotenv_path=ENV_PATH)

# 读取 Token
TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")

# API 基础地址
BASE_URL = "https://api.football-data.org/v4"

# 请求头
HEADERS = {
    "X-Auth-Token": '84f5a229f0184ba7a374d03362c6335d'
}


def get_world_cup_matches():
    """获取世界杯比赛数据。"""

    url = f"{BASE_URL}/competitions/WC/matches"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def print_matches(data):
    """打印世界杯比赛信息。"""

    competition = data.get("competition", {})
    filters = data.get("filters", {})
    result_set = data.get("resultSet", {})
    matches = data.get("matches", [])

    print("赛事名称：", competition.get("name"))
    print("筛选条件：", filters)
    print("数据范围：", result_set)
    print(f"共获取到 {len(matches)} 场世界杯比赛")
    print("=" * 80)

    for match in matches:
        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}
        score = match.get("score") or {}
        full_time_score = score.get("fullTime") or {}

        print(f"比赛 ID：{match.get('id')}")
        print(f"比赛时间：{match.get('utcDate')}")
        print(f"比赛状态：{match.get('status')}")
        print(f"比赛阶段：{match.get('stage')}")
        print(f"比赛轮次：{match.get('matchday')}")

        print(
            f"比赛双方："
            f"{home_team.get('name') or '待定'} "
            f"vs "
            f"{away_team.get('name') or '待定'}"
        )

        print(
            f"最终比分："
            f"{full_time_score.get('home', '-')}:"
            f"{full_time_score.get('away', '-')}"
        )

        print("-" * 80)

def save_matches_to_csv(data):
    """把世界杯比赛数据保存为 CSV 文件。"""

    matches = data.get("matches", [])

    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "world_cup_2026_matches.csv"

    fieldnames = [
        "match_id",
        "utc_date",
        "local_date",
        "status",
        "stage",
        "matchday",
        "group",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_score",
        "away_score",
        "winner"
    ]

    with output_file.open(
        mode="w",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for match in matches:
            home_team = match.get("homeTeam") or {}
            away_team = match.get("awayTeam") or {}

            score = match.get("score") or {}
            full_time = score.get("fullTime") or {}

            utc_date_text = match.get("utcDate")

            local_date_text = convert_utc_to_local(
                utc_date_text
            )

            writer.writerow({
                "match_id": match.get("id"),
                "utc_date": utc_date_text,
                "local_date": local_date_text,
                "status": match.get("status"),
                "stage": match.get("stage"),
                "matchday": match.get("matchday"),
                "group": match.get("group"),

                "home_team_id": home_team.get("id"),
                "home_team": home_team.get("name"),

                "away_team_id": away_team.get("id"),
                "away_team": away_team.get("name"),

                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),

                "winner": score.get("winner")
            })

    print(f"比赛数据已经保存到：{output_file}")

def convert_utc_to_local(utc_date_text):
    """将 UTC 比赛时间转换为新加坡/北京时间。"""

    if not utc_date_text:
        return None

    utc_datetime = datetime.fromisoformat(
        utc_date_text.replace("Z", "+00:00")
    )

    local_datetime = utc_datetime.astimezone(
        ZoneInfo("Asia/Singapore")
    )

    return local_datetime.strftime("%Y-%m-%d %H:%M:%S")
if __name__ == "__main__":
    # 临时打印路径，检查程序读取的是哪个 .env
    print("当前程序文件：", CURRENT_FILE)
    print("项目根目录：", PROJECT_ROOT)
    print(".env 文件地址：", ENV_PATH)
    print(".env 是否存在：", ENV_PATH.exists())

    if not TOKEN:
        raise ValueError(
            f"没有读取到 FOOTBALL_DATA_TOKEN，"
            f"请检查文件：{ENV_PATH}"
        )

    world_cup_data = get_world_cup_matches()

    print_matches(world_cup_data)

    save_matches_to_csv(world_cup_data)