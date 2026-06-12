from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class Article(BaseModel):
    source: str
    title: str
    url: HttpUrl
    published_at: datetime | None = None
    summary: str = ""


class Fixture(BaseModel):
    match_id: str
    kickoff: datetime
    stage: str
    home_team: str
    away_team: str
    status: str = "scheduled"


class RecentMatch(BaseModel):
    date: datetime
    opponent: str
    venue: Literal["home", "away", "neutral"]
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    expected_goals_for: float | None = Field(default=None, ge=0)
    expected_goals_against: float | None = Field(default=None, ge=0)
    shots_for: int | None = Field(default=None, ge=0)
    shots_against: int | None = Field(default=None, ge=0)


class TeamMetrics(BaseModel):
    team: str
    fifa_rank: int | None = Field(default=None, ge=1)
    elo_rating: float | None = Field(default=None, ge=0)
    recent_matches: list[RecentMatch] = Field(default_factory=list)
    recent_goals_for_per_game: float | None = Field(default=None, ge=0)
    recent_goals_against_per_game: float | None = Field(default=None, ge=0)
    recent_xg_for_per_game: float | None = Field(default=None, ge=0)
    recent_xg_against_per_game: float | None = Field(default=None, ge=0)
    home_or_away_win_rate: float | None = Field(default=None, ge=0, le=1)
    clean_sheet_rate: float | None = Field(default=None, ge=0, le=1)
    rest_days: int | None = Field(default=None, ge=0)


class PlayerStatus(BaseModel):
    player: str
    status: Literal["available", "doubtful", "injured", "suspended", "unknown"]
    reason: str = ""
    importance: Literal["key", "starter", "rotation", "unknown"] = "unknown"


class TeamAvailability(BaseModel):
    team: str
    expected_lineup: list[str] = Field(default_factory=list)
    confirmed_lineup: list[str] = Field(default_factory=list)
    players: list[PlayerStatus] = Field(default_factory=list)


class HeadToHeadMatch(BaseModel):
    date: datetime
    home_team: str
    away_team: str
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    competition: str = ""


class OddsSnapshot(BaseModel):
    captured_at: datetime
    bookmaker: str
    home_decimal: float = Field(gt=1)
    draw_decimal: float = Field(gt=1)
    away_decimal: float = Field(gt=1)


class WeatherForecast(BaseModel):
    temperature_c: float | None = None
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    precipitation_mm: float | None = Field(default=None, ge=0)
    wind_kph: float | None = Field(default=None, ge=0)
    condition: str = ""


class VenueContext(BaseModel):
    stadium: str = ""
    city: str = ""
    country: str = ""
    surface: str = ""
    altitude_m: float | None = None
    capacity: int | None = Field(default=None, ge=0)
    neutral_venue: bool = True


class TravelContext(BaseModel):
    home_distance_km: float | None = Field(default=None, ge=0)
    away_distance_km: float | None = Field(default=None, ge=0)
    home_timezone_shift_hours: float | None = None
    away_timezone_shift_hours: float | None = None


class MatchIntelligence(BaseModel):
    match_id: str
    updated_at: datetime
    sources: list[str] = Field(default_factory=list)
    home_metrics: TeamMetrics
    away_metrics: TeamMetrics
    home_availability: TeamAvailability
    away_availability: TeamAvailability
    head_to_head: list[HeadToHeadMatch] = Field(default_factory=list)
    odds: list[OddsSnapshot] = Field(default_factory=list)
    weather: WeatherForecast | None = None
    venue: VenueContext | None = None
    travel: TravelContext | None = None
    notes: list[str] = Field(default_factory=list)


class MatchPrediction(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    home_goals: int = Field(ge=0, le=20)
    away_goals: int = Field(ge=0, le=20)
    home_win_probability: float = Field(ge=0, le=1)
    draw_probability: float = Field(ge=0, le=1)
    away_win_probability: float = Field(ge=0, le=1)
    rationale: str = ""


class ChampionPrediction(BaseModel):
    team: str
    probability: float = Field(ge=0, le=1)
    rationale: str = ""


class AgentForecast(BaseModel):
    agent_id: str
    provider_id: str
    generated_at: datetime
    confidence: float = Field(ge=0, le=1)
    key_findings: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    match_predictions: list[MatchPrediction] = Field(default_factory=list)
    champion_predictions: list[ChampionPrediction] = Field(default_factory=list)
    cited_urls: list[str] = Field(default_factory=list)


class ConsensusForecast(BaseModel):
    generated_at: datetime
    champion_predictions: list[ChampionPrediction]
    match_predictions: list[MatchPrediction]
    agreement_score: float = Field(ge=0, le=1)
    agent_count: int
    warnings: list[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    id: str
    type: Literal["openai_compatible", "anthropic", "gemini", "mock"]
    model: str
    api_key_env: str
    base_url: str | None = None
    weight: float = Field(default=1.0, gt=0)


class AgentConfig(BaseModel):
    id: str
    role: str
    provider: str
    weight: float = Field(default=1.0, gt=0)


class NewsSourceConfig(BaseModel):
    name: str
    url: str


class DataSourceConfig(BaseModel):
    id: str
    type: Literal["local_json", "http_json"]
    enabled: bool = True
    path: str | None = None
    url: str | None = None
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"


class CompetitionConfig(BaseModel):
    name: str
    timezone: str = "UTC"
    fixtures_file: str


class RunConfig(BaseModel):
    news_lookback_hours: int = Field(default=36, gt=0)
    max_articles: int = Field(default=80, gt=0)
    max_articles_per_agent: int = Field(default=35, gt=0)
    output_dir: str = "reports"
    request_timeout_seconds: int = Field(default=90, gt=0)


class AppConfig(BaseModel):
    competition: CompetitionConfig
    run: RunConfig = Field(default_factory=RunConfig)
    news_sources: list[NewsSourceConfig]
    data_sources: list[DataSourceConfig] = Field(default_factory=list)
    providers: list[ProviderConfig]
    agents: list[AgentConfig]
