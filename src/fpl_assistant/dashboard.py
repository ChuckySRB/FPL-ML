"""Pure data helpers for the Streamlit FPL assistant dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_FILES = {
    "players": "predictions_gw{gameweek}.csv",
    "fixtures": "fixture_predictions_gw{gameweek}.csv",
    "report": "report_gw{gameweek}.md",
    "json": "report_gw{gameweek}.json",
    "prompt": "chat_prompt_gw{gameweek}.md",
}


def package_paths(
    output_root: Path,
    season: str,
    gameweek: int,
) -> dict[str, Path]:
    """Return the expected files for one generated weekly package."""
    directory = output_root / season / f"gw{int(gameweek)}"
    paths = {
        name: directory / filename.format(gameweek=int(gameweek))
        for name, filename in PACKAGE_FILES.items()
    }
    return {"directory": directory, **paths}


def discover_seasons(raw_data_root: Path) -> list[str]:
    """Find seasons that have enough official inputs for prediction."""
    seasons = []
    if not raw_data_root.exists():
        return seasons
    for path in raw_data_root.glob("20??-??"):
        required = ("players_raw.csv", "teams.csv", "fixtures.csv")
        if path.is_dir() and all((path / name).exists() for name in required):
            seasons.append(path.name)
    return sorted(seasons)


def available_gameweeks(raw_data_root: Path, season: str) -> list[int]:
    """Read selectable gameweeks from the official fixture list."""
    path = raw_data_root / season / "fixtures.csv"
    if not path.exists():
        return []
    fixtures = pd.read_csv(path, usecols=lambda column: column == "event")
    if "event" not in fixtures:
        return []
    events = pd.to_numeric(fixtures["event"], errors="coerce").dropna()
    return sorted(events.astype(int).unique().tolist())


def _truthy(values: pd.Series) -> pd.Series:
    """Normalize booleans read from CSV."""
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def infer_default_gameweek(
    raw_data_root: Path,
    season: str,
    selectable_gameweeks: list[int],
) -> int | None:
    """Choose the official next/current GW, with deterministic fallbacks."""
    if not selectable_gameweeks:
        return None
    events_path = raw_data_root / season / "events.csv"
    if not events_path.exists():
        return selectable_gameweeks[0]
    events = pd.read_csv(events_path)
    if "id" not in events:
        return selectable_gameweeks[0]
    event_ids = pd.to_numeric(events["id"], errors="coerce")
    for column in ("is_next", "is_current"):
        if column not in events:
            continue
        matches = event_ids[_truthy(events[column])].dropna().astype(int)
        matches = [value for value in matches if value in selectable_gameweeks]
        if matches:
            return matches[0]
    if "finished" in events:
        unfinished = event_ids[~_truthy(events["finished"])].dropna().astype(int)
        unfinished = [
            value for value in unfinished if value in selectable_gameweeks
        ]
        if unfinished:
            return unfinished[0]
    return selectable_gameweeks[-1]


def load_weekly_package(
    output_root: Path,
    season: str,
    gameweek: int,
) -> dict[str, Any]:
    """Load a complete generated package for display and download."""
    paths = package_paths(output_root, season, gameweek)
    missing = [
        path.name
        for name, path in paths.items()
        if name != "directory" and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"GW{gameweek} package is incomplete: {', '.join(missing)}"
        )
    return {
        "paths": paths,
        "players": pd.read_csv(paths["players"], low_memory=False),
        "fixtures": pd.read_csv(paths["fixtures"], low_memory=False),
        "structured": json.loads(paths["json"].read_text(encoding="utf-8")),
        "report": paths["report"].read_text(encoding="utf-8"),
        "prompt": paths["prompt"].read_text(encoding="utf-8"),
        "generated_at": max(
            path.stat().st_mtime
            for name, path in paths.items()
            if name != "directory"
        ),
    }


def availability_mask(players: pd.DataFrame) -> pd.Series:
    """Apply the same conservative availability rule as weekly reports."""
    status = players.get(
        "status",
        pd.Series("a", index=players.index, dtype="string"),
    ).fillna("a")
    chance = pd.to_numeric(
        players.get(
            "chance_of_playing_next_round",
            pd.Series(np.nan, index=players.index),
        ),
        errors="coerce",
    )
    return ~status.isin(["i", "u", "s"]) & (chance.isna() | chance.ge(50))


def rank_players(
    players: pd.DataFrame,
    prediction_column: str,
    positions: list[str] | None = None,
    teams: list[str] | None = None,
    maximum_price: float | None = None,
    available_only: bool = True,
) -> pd.DataFrame:
    """Filter and rank player predictions for one dashboard horizon."""
    if prediction_column not in players:
        raise ValueError(f"Missing prediction column: {prediction_column}")
    ranked = players.copy()
    if available_only:
        ranked = ranked[availability_mask(ranked)]
    if positions:
        ranked = ranked[ranked["position_label"].isin(positions)]
    if teams:
        ranked = ranked[ranked["team_name"].isin(teams)]
    if maximum_price is not None and "current_price" in ranked:
        price = pd.to_numeric(ranked["current_price"], errors="coerce")
        ranked = ranked[price.le(maximum_price)]
    ranked[prediction_column] = pd.to_numeric(
        ranked[prediction_column],
        errors="coerce",
    )
    ranked["value_score"] = np.where(
        pd.to_numeric(ranked.get("current_price"), errors="coerce").gt(0),
        ranked[prediction_column]
        / pd.to_numeric(ranked["current_price"], errors="coerce"),
        np.nan,
    )
    return (
        ranked.sort_values(prediction_column, ascending=False)
        .reset_index(drop=True)
    )


def build_strategy_prompt(
    base_prompt: str,
    squad: pd.DataFrame,
    bank: float,
    free_transfers: int,
    chips: str,
    risk_profile: str,
    external_notes: str,
) -> str:
    """Append structured account context to the generated weekly prompt."""
    if squad.empty:
        squad_text = "Тим није унет."
    else:
        lines = []
        for position in ("GK", "DEF", "MID", "FWD"):
            names = squad.loc[
                squad["position_label"].eq(position),
                "name",
            ].astype(str)
            lines.append(f"{position}: {', '.join(names) or '—'}")
        squad_text = "\n".join(lines)
    notes = external_notes.strip() or "Нису приложене додатне белешке."
    return (
        f"{base_prompt.rstrip()}\n\n"
        "## Подаци унети у апликацији\n\n"
        f"```text\n{squad_text}\n"
        f"Новац у банци: £{bank:.1f}m\n"
        f"Број free transfer-а: {free_transfers}\n"
        f"Расположиви chip-ови: {chips.strip() or 'није наведено'}\n"
        f"Профил ризика: {risk_profile}\n```\n\n"
        "### Спољни извори и белешке\n\n"
        f"{notes}\n"
    )
