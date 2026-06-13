"""2026 FIFA World Cup team names and normalization helpers."""

from __future__ import annotations


WORLD_CUP_2026_TEAMS: set[str] = {
    "Mexico",
    "South Africa",
    "South Korea",
    "Czech Republic",
    "Canada",
    "Bosnia and Herzegovina",
    "Qatar",
    "Switzerland",
    "Brazil",
    "Morocco",
    "Haiti",
    "Scotland",
    "United States",
    "Paraguay",
    "Australia",
    "Turkey",
    "Germany",
    "Curaçao",
    "Ivory Coast",
    "Ecuador",
    "Netherlands",
    "Japan",
    "Sweden",
    "Tunisia",
    "Belgium",
    "Egypt",
    "Iran",
    "New Zealand",
    "Spain",
    "Cape Verde",
    "Saudi Arabia",
    "Uruguay",
    "France",
    "Senegal",
    "Iraq",
    "Norway",
    "Argentina",
    "Algeria",
    "Austria",
    "Jordan",
    "Portugal",
    "DR Congo",
    "Uzbekistan",
    "Colombia",
    "England",
    "Croatia",
    "Ghana",
    "Panama",
}


TEAM_NAME_ALIASES = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curaçao",
    "Türkiye": "Turkey",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
}


_CANONICAL_NAMES = {
    team.casefold(): team
    for team in WORLD_CUP_2026_TEAMS
}

_NORMALIZED_ALIASES = {
    alias.strip().casefold(): canonical
    for alias, canonical in TEAM_NAME_ALIASES.items()
}


def normalize_team_name(team_name: str) -> str:
    """Return the canonical historical-dataset name for a team."""

    cleaned_name = str(team_name).strip()
    normalized_key = cleaned_name.casefold()

    if normalized_key in _NORMALIZED_ALIASES:
        return _NORMALIZED_ALIASES[normalized_key]

    return _CANONICAL_NAMES.get(
        normalized_key,
        cleaned_name,
    )


def is_world_cup_2026_team(team_name: str) -> bool:
    """Return whether a team belongs to the configured 2026 World Cup field."""

    return normalize_team_name(team_name) in WORLD_CUP_2026_TEAMS


def validate_world_cup_team_count() -> None:
    """Ensure the configured World Cup field contains exactly 48 teams."""

    team_count = len(WORLD_CUP_2026_TEAMS)
    if team_count != 48:
        raise ValueError(
            "2026 世界杯球队数量必须为 48，"
            f"当前为 {team_count}"
        )


def main() -> None:
    """Validate and print the configured World Cup team list."""

    validate_world_cup_team_count()
    print(f"2026 世界杯球队数量：{len(WORLD_CUP_2026_TEAMS)}")
    print("\n".join(sorted(WORLD_CUP_2026_TEAMS)))


if __name__ == "__main__":
    main()
