from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import AppConfig


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    load_dotenv(config_path.parent / ".env")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(data)

    provider_ids = {provider.id for provider in config.providers}
    unknown = {agent.provider for agent in config.agents} - provider_ids
    if unknown:
        raise ValueError(f"Agents reference unknown providers: {sorted(unknown)}")
    return config
