from datetime import UTC, datetime

from world_cup_forecast.consensus import build_consensus
from world_cup_forecast.models import (
    AgentConfig,
    AgentForecast,
    ChampionPrediction,
    MatchPrediction,
    ProviderConfig,
)


def _forecast(agent_id: str, brazil: float, france: float) -> AgentForecast:
    return AgentForecast(
        agent_id=agent_id,
        provider_id="p",
        generated_at=datetime.now(UTC),
        confidence=0.8,
        champion_predictions=[
            ChampionPrediction(team="Brazil", probability=brazil),
            ChampionPrediction(team="France", probability=france),
        ],
        match_predictions=[
            MatchPrediction(
                match_id="m1",
                home_team="Brazil",
                away_team="France",
                home_goals=2 if agent_id == "a" else 1,
                away_goals=1,
                home_win_probability=0.5,
                draw_probability=0.3,
                away_win_probability=0.2,
            )
        ],
    )


def test_weighted_consensus_normalizes_probabilities():
    agents = [
        AgentConfig(id="a", role="a", provider="p", weight=2),
        AgentConfig(id="b", role="b", provider="p", weight=1),
    ]
    providers = [
        ProviderConfig(id="p", type="mock", model="m", api_key_env="X")
    ]
    result = build_consensus(
        [_forecast("a", 0.8, 0.2), _forecast("b", 0.2, 0.8)],
        agents,
        providers,
    )
    assert result.champion_predictions[0].team == "Brazil"
    assert sum(p.probability for p in result.champion_predictions) == 1
    assert result.match_predictions[0].home_goals == 2
    match = result.match_predictions[0]
    assert round(
        match.home_win_probability
        + match.draw_probability
        + match.away_win_probability,
        10,
    ) == 1


def test_agreement_uses_weighted_match_outcome_votes():
    agents = [
        AgentConfig(id="a", role="a", provider="p", weight=2),
        AgentConfig(id="b", role="b", provider="p", weight=1),
    ]
    providers = [
        ProviderConfig(id="p", type="mock", model="m", api_key_env="X")
    ]
    home = _forecast("a", 0.5, 0.5)
    away = _forecast("b", 0.5, 0.5)
    away.match_predictions[0].home_win_probability = 0.1
    away.match_predictions[0].draw_probability = 0.2
    away.match_predictions[0].away_win_probability = 0.7

    result = build_consensus([home, away], agents, providers)

    assert result.match_predictions[0].home_win_probability > 0.3
    assert result.agreement_score == 2 / 3
