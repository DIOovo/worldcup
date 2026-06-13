import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("LLM_MODEL")


def test_llm_api():
    if not BASE_URL:
        raise ValueError("缺少 LLM_BASE_URL")

    if not API_KEY:
        raise ValueError("缺少 ANTHROPIC_API_KEY")

    if not MODEL:
        raise ValueError("缺少 LLM_MODEL")

    # url = f"{BASE_URL.rstrip('/')}"
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    response = requests.post(
        url=url,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "你是什么模型"
                }
            ],
            "temperature": 0
        },
        timeout=120
    )

    print("请求地址：", url)
    print("状态码：", response.status_code)
    print("响应内容：", response.text)

    response.raise_for_status()


if __name__ == "__main__":
    test_llm_api()