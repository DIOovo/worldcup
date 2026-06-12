from __future__ import annotations

import argparse
import asyncio

from .pipeline import run_daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a daily multi-agent World Cup forecast"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML configuration"
    )
    args = parser.parse_args()
    markdown_path, json_path = asyncio.run(run_daily(args.config))
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
