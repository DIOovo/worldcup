from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from .models import Article, NewsSourceConfig


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _text(element: ElementTree.Element, *names: str) -> str:
    for name in names:
        child = element.find(name)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def _parse_feed(content: bytes, source: NewsSourceConfig) -> list[Article]:
    root = ElementTree.fromstring(content)
    items = root.findall(".//item")
    atom = "{http://www.w3.org/2005/Atom}"
    if not items:
        items = root.findall(f".//{atom}entry")

    articles: list[Article] = []
    for item in items:
        title = _text(item, "title", f"{atom}title")
        link = _text(item, "link")
        if not link:
            link_element = item.find(f"{atom}link")
            if link_element is not None:
                link = link_element.attrib.get("href", "")
        if not title or not link:
            continue
        published = _text(
            item,
            "pubDate",
            f"{atom}published",
            f"{atom}updated",
        )
        summary = _text(
            item,
            "description",
            f"{atom}summary",
            f"{atom}content",
        )
        articles.append(
            Article(
                source=source.name,
                title=title,
                url=link,
                published_at=_parse_datetime(published),
                summary=summary[:1200],
            )
        )
    return articles


async def _fetch_feed(
    client: httpx.AsyncClient, source: NewsSourceConfig
) -> list[Article]:
    response = await client.get(source.url, follow_redirects=True)
    response.raise_for_status()
    return _parse_feed(response.content, source)


async def collect_news(
    sources: list[NewsSourceConfig],
    lookback_hours: int,
    max_articles: int,
    timeout_seconds: int,
) -> tuple[list[Article], list[str]]:
    warnings: list[str] = []
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        headers={"User-Agent": "world-cup-forecast/0.1"},
    ) as client:
        results = await asyncio.gather(
            *[_fetch_feed(client, source) for source in sources],
            return_exceptions=True,
        )

    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
    deduped: dict[str, Article] = {}
    for source, result in zip(sources, results, strict=True):
        if isinstance(result, BaseException):
            warnings.append(f"新闻源 {source.name} 获取失败：{result}")
            continue
        for article in result:
            if article.published_at and article.published_at < cutoff:
                continue
            fingerprint = hashlib.sha256(
                article.title.casefold().encode("utf-8")
            ).hexdigest()
            deduped.setdefault(fingerprint, article)

    articles = sorted(
        deduped.values(),
        key=lambda item: item.published_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    return articles[:max_articles], warnings
