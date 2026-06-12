from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from .models import (
    AgentConfig,
    AgentForecast,
    ChampionPrediction,
    ConsensusForecast,
    MatchPrediction,
    ProviderConfig,
)


def _weights(
    agents: list[AgentConfig], providers: list[ProviderConfig]
) -> dict[str, float]:
    provider_weight = {provider.id: provider.weight for provider in providers}
    return {
        agent.id: agent.weight * provider_weight[agent.provider] for agent in agents
    }


def _match_outcome(prediction: MatchPrediction) -> str:
    return max(
        ("home", prediction.home_win_probability),
        ("draw", prediction.draw_probability),
        ("away", prediction.away_win_probability),
        key=lambda item: item[1],
    )[0]


def build_consensus(
    forecasts: list[AgentForecast],
    agents: list[AgentConfig],
    providers: list[ProviderConfig],
    warnings: list[str] | None = None,
) -> ConsensusForecast:
    if not forecasts:
        raise RuntimeError("All forecasting agents failed")
    weights = _weights(agents, providers)

    champion_scores: dict[str, float] = defaultdict(float)
    champion_rationales: dict[str, list[str]] = defaultdict(list)
    total_weight = sum(weights.get(item.agent_id, 1.0) for item in forecasts)
    for forecast in forecasts:
        weight = weights.get(forecast.agent_id, 1.0)
        for prediction in forecast.champion_predictions:
            champion_scores[prediction.team] += prediction.probability * weight
            if prediction.rationale:
                champion_rationales[prediction.team].append(prediction.rationale)

    score_total = sum(champion_scores.values()) or 1.0
    champions = sorted(
        [
            ChampionPrediction(
                team=team,
                probability=score / score_total,
                rationale=" | ".join(champion_rationales[team][:2]),
            )
            for team, score in champion_scores.items()
        ],
        key=lambda item: item.probability,
        reverse=True,
    )[:10]

    match_values: dict[str, list[tuple[MatchPrediction, float]]] = defaultdict(list)
    for forecast in forecasts:
        weight = weights.get(forecast.agent_id, 1.0)
        for prediction in forecast.match_predictions:
            match_values[prediction.match_id].append((prediction, weight))

    matches: list[MatchPrediction] = []
    for predictions in match_values.values():
        weight_sum = sum(weight for _, weight in predictions)
        first = predictions[0][0]
        avg = lambda field: sum(
            getattr(item, field) * weight for item, weight in predictions
        ) / weight_sum
        home_probability = avg("home_win_probability")
        draw_probability = avg("draw_probability")
        away_probability = avg("away_win_probability")
        probability_total = (
            home_probability + draw_probability + away_probability
        ) or 1.0
        home_probability /= probability_total
        draw_probability /= probability_total
        away_probability /= probability_total

        home_goals = round(avg("home_goals"))
        away_goals = round(avg("away_goals"))
        leading_outcome = max(
            ("home", home_probability),
            ("draw", draw_probability),
            ("away", away_probability),
            key=lambda item: item[1],
        )[0]
        if leading_outcome == "draw":
            home_goals = away_goals = round((home_goals + away_goals) / 2)
        elif leading_outcome == "home" and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif leading_outcome == "away" and away_goals <= home_goals:
            away_goals = home_goals + 1

        matches.append(
            MatchPrediction(
                match_id=first.match_id,
                home_team=first.home_team,
                away_team=first.away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_win_probability=home_probability,
                draw_probability=draw_probability,
                away_win_probability=away_probability,
                rationale=f"Weighted consensus from {len(predictions)} agents.",
            )
        )

    agreement_score = 0.0
    if matches:
        consensus_match = matches[0]
        consensus_outcome = _match_outcome(consensus_match)
        agreeing_weight = 0.0
        voting_weight = 0.0
        for forecast in forecasts:
            agent_weight = weights.get(forecast.agent_id, 1.0)
            prediction = next(
                (
                    item
                    for item in forecast.match_predictions
                    if item.match_id == consensus_match.match_id
                ),
                None,
            )
            if prediction is None:
                continue
            voting_weight += agent_weight
            if _match_outcome(prediction) == consensus_outcome:
                agreeing_weight += agent_weight
        if voting_weight:
            agreement_score = agreeing_weight / voting_weight

    output_warnings = list(warnings or [])
    if total_weight <= 0:
        output_warnings.append("Invalid ensemble weight; equal weighting was used.")
    return ConsensusForecast(
        generated_at=datetime.now(UTC),
        champion_predictions=champions,
        match_predictions=sorted(matches, key=lambda item: item.match_id),
        agreement_score=agreement_score,
        agent_count=len(forecasts),
        warnings=output_warnings,
    )
