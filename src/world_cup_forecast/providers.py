from __future__ import annotations

import json
import hashlib
import os
import random
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime

import httpx

from .models import (
    AgentForecast,
    Fixture,
    MatchPrediction,
    ProviderConfig,
)


class LLMProvider(ABC):
    def __init__(self, config: ProviderConfig, timeout_seconds: int):
        self.config = config
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def generate_json(self, system: str, prompt: str) -> dict:
        raise NotImplementedError

    def api_key(self) -> str:
        value = os.getenv(self.config.api_key_env)
        if not value:
            raise RuntimeError(
                f"Missing API key environment variable {self.config.api_key_env}"
            )
        return value

    @property
    def endpoint(self) -> str:
        return "local mock (no API request)"

    @property
    def is_mock(self) -> bool:
        return False


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class OpenAICompatibleProvider(LLMProvider):
    @property
    def endpoint(self) -> str:
        base_url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        return f"{base_url}/chat/completions"

    async def generate_json(self, system: str, prompt: str) -> dict:
        payload = {
            "model": self.config.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key()}"},
                json=payload,
            )
            response.raise_for_status()
        return _extract_json(response.json()["choices"][0]["message"]["content"])


class AnthropicProvider(LLMProvider):
    @property
    def endpoint(self) -> str:
        base_url = (
            self.config.base_url or "https://api.anthropic.com"
        ).rstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/messages"
        return f"{base_url}/v1/messages"

    async def generate_json(self, system: str, prompt: str) -> dict:
        payload = {
            "model": self.config.model,
            "max_tokens": 5000,
            "temperature": 0.2,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "x-api-key": self.api_key(),
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )
            response.raise_for_status()
        blocks = response.json()["content"]
        text = "".join(block["text"] for block in blocks if block["type"] == "text")
        return _extract_json(text)


class GeminiProvider(LLMProvider):
    @property
    def endpoint(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent"
        )

    async def generate_json(self, system: str, prompt: str) -> dict:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.endpoint, params={"key": self.api_key()}, json=payload
            )
            response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        return _extract_json("".join(part.get("text", "") for part in parts))


class MockProvider(LLMProvider):
    def __init__(
        self, config: ProviderConfig, timeout_seconds: int, fixtures: list[Fixture]
    ):
        super().__init__(config, timeout_seconds)
        self.fixtures = fixtures

    @property
    def is_mock(self) -> bool:
        return True

    async def generate_json(self, system: str, prompt: str) -> dict:
        seed_text = f"{self.config.id}\n{system}\n{prompt}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        matches = []
        for fixture in self.fixtures:
            home_goals = rng.randrange(0, 4)
            away_goals = rng.randrange(0, 4)
            probs = [rng.uniform(0.15, 0.65) for _ in range(3)]
            prob_total = sum(probs)
            matches.append(
                MatchPrediction(
                    match_id=fixture.match_id,
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_win_probability=probs[0] / prob_total,
                    draw_probability=probs[1] / prob_total,
                    away_win_probability=probs[2] / prob_total,
                    rationale="本地模拟结果，请配置真实大模型。",
                ).model_dump()
            )
        return AgentForecast(
            agent_id="placeholder",
            provider_id=self.config.id,
            generated_at=datetime.now(UTC),
            confidence=0.35,
            key_findings=["当前为演示模式，没有调用厂商大模型。"],
            risks=["比分与概率均为本地模拟数据。"],
            match_predictions=matches,
            champion_predictions=[],
            cited_urls=[],
        ).model_dump(mode="json")


def build_provider(
    config: ProviderConfig, timeout_seconds: int, fixtures: list[Fixture]
) -> LLMProvider:
    if config.type == "openai_compatible":
        return OpenAICompatibleProvider(config, timeout_seconds)
    if config.type == "anthropic":
        return AnthropicProvider(config, timeout_seconds)
    if config.type == "gemini":
        return GeminiProvider(config, timeout_seconds)
    return MockProvider(config, timeout_seconds, fixtures)
