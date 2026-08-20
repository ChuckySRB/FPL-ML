"""Pure data helpers for the Streamlit FPL assistant dashboard."""

from __future__ import annotations

import json
from io import BytesIO
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
    watchlist: pd.DataFrame,
    bank: float,
    free_transfers: int,
    chips: str,
    risk_profile: str,
    external_notes: str,
    attachment_filename: str,
    data_timestamp: str,
) -> str:
    """Build a complete user prompt with draft, watchlist, and attachment."""
    if squad.empty:
        squad_text = "Тим није унет."
    else:
        lines = []
        for position in ("GK", "DEF", "MID", "FWD"):
            position_rows = squad[squad["position_label"].eq(position)]
            player_details = [
                _prompt_player_line(row)
                for _, row in position_rows.iterrows()
            ]
            lines.append(
                f"{position}: " + ("; ".join(player_details) or "—")
            )
        squad_text = "\n".join(lines)
    watchlist_text = (
        "\n".join(
            f"- {_prompt_player_line(row)}"
            for _, row in watchlist.iterrows()
        )
        if not watchlist.empty
        else "- Watchlist је празан."
    )
    notes = external_notes.strip() or "Нису приложене додатне белешке."
    questions_start = base_prompt.find("## Конкретна питања")
    questions = (
        base_prompt[questions_start:].rstrip()
        if questions_start >= 0
        else base_prompt.rstrip()
    )
    return (
        "# FPL GW анализа — комплетан кориснички захтев\n\n"
        f"Приложен је `{attachment_filename}`. Прво прочитај лист `UPUTSTVO`, "
        "затим `DRAFT_TIM`, `WATCHLIST`, обе TOP 50 листе, `RIZICI` и "
        "`SVE_PROGNOZE`. Excel је примарни машински прилог; немој "
        "претпостављати да је играч у мом тиму ако није у `DRAFT_TIM`. "
        f"Snapshot података и прогнозе је генерисан: {data_timestamp}.\n\n"
        "## Подаци унети у апликацији\n\n"
        f"```text\n{squad_text}\n"
        f"Новац у банци: £{bank:.1f}m\n"
        f"Број free transfer-а: {free_transfers}\n"
        f"Расположиви chip-ови: {chips.strip() or 'није наведено'}\n"
        f"Профил ризика: {risk_profile}\n```\n\n"
        "### Watchlist\n\n"
        f"{watchlist_text}\n\n"
        "### Спољни извори и белешке\n\n"
        f"{notes}\n\n"
        f"{questions}\n\n"
        "## Обавезна завршна провера\n\n"
        "Пре препоруке провери валидност draft-а (2 GK, 5 DEF, 5 MID, "
        "3 FWD, максимум три из клуба и буџет), минуте, повреде, "
        "суспензије и најновије вести. Посебно оцени сваког играча са "
        "watchlist-е као `довести`, `пратити` или `избацити са листе`, уз "
        "једну реченицу образложења. На крају дај конкретан draft XI, "
        "клупу по редоследу, капитена, заменика и план до петог GW-а.\n"
    )


def _prompt_player_line(row: pd.Series) -> str:
    """Format one compact player record for the copy/paste prompt."""
    values = {
        "price": pd.to_numeric(row.get("current_price"), errors="coerce"),
        "current": pd.to_numeric(
            row.get("predicted_points_current_gw"), errors="coerce"
        ),
        "five": pd.to_numeric(
            row.get("predicted_average_next_5_gws"), errors="coerce"
        ),
    }
    price = f"£{values['price']:.1f}m" if pd.notna(values["price"]) else "цена ?"
    current = f"GW {values['current']:.2f}" if pd.notna(values["current"]) else "GW ?"
    five = f"5GW {values['five']:.2f}/GW" if pd.notna(values["five"]) else "5GW ?"
    status = str(row.get("status", "?") or "?")
    return (
        f"{row.get('name', '?')} ({row.get('team_name', '?')}, "
        f"{row.get('position_label', '?')}, {price}, {current}, {five}, "
        f"status {status})"
    )


def build_ai_workbook(
    season: str,
    gameweek: int,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    structured: dict[str, Any],
    squad: pd.DataFrame,
    watchlist: pd.DataFrame,
    account: dict[str, Any],
) -> bytes:
    """Create the single Excel attachment consumed by the AI assistant."""
    current = rank_players(
        players,
        "predicted_points_current_gw",
        available_only=True,
    )
    if "current_gw_fixtures" in current:
        current = current[current["current_gw_fixtures"].gt(0)]
    long_term = rank_players(
        players,
        "predicted_average_next_5_gws",
        available_only=True,
    )
    unavailable = pd.DataFrame(structured.get("excluded_unavailable", []))
    disagreements = pd.DataFrame(
        structured.get("large_model_ep_next_disagreements", [])
    )
    risks = pd.concat(
        [
            unavailable.assign(risk_type="недоступан/суспендован"),
            disagreements.assign(risk_type="модел vs FPL xP > 2"),
        ],
        ignore_index=True,
    )
    guide = pd.DataFrame(
        {
            "поље": [
                "сезона",
                "gameweek",
                "snapshot генерисан",
                "важно",
                "краткорочна прогноза",
                "дугорочна прогноза",
                "draft",
                "watchlist",
                "bank",
                "free transfers",
                "chip-ови",
                "профил ризика",
                "спољне белешке",
            ],
            "вредност": [
                season,
                gameweek,
                account.get("generated_at", ""),
                "Прогнозе су очекиване вредности, не гаранције.",
                "predicted_points_current_gw: модел једне утакмице, сабран само за DGW",
                "predicted_average_next_5_gws: директан засебан 5GW просек",
                "Само играчи са листа DRAFT_TIM чине тренутни тим.",
                "Кандидати за посебну процену су на листу WATCHLIST.",
                account.get("bank", 0),
                account.get("free_transfers", 1),
                account.get("chips", ""),
                account.get("risk_profile", "Умерен"),
                account.get("external_notes", ""),
            ],
        }
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        guide.to_excel(writer, sheet_name="UPUTSTVO", index=False)
        squad.to_excel(writer, sheet_name="DRAFT_TIM", index=False)
        watchlist.to_excel(writer, sheet_name="WATCHLIST", index=False)
        current.head(50).to_excel(writer, sheet_name="TOP50_GW", index=False)
        long_term.head(50).to_excel(
            writer,
            sheet_name="TOP50_5GW",
            index=False,
        )
        risks.to_excel(writer, sheet_name="RIZICI", index=False)
        players.to_excel(writer, sheet_name="SVE_PROGNOZE", index=False)
        fixtures.to_excel(writer, sheet_name="UTAKMICE", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.column_dimensions["A"].width = 22
            for column in worksheet.iter_cols(
                min_col=2,
                max_col=min(worksheet.max_column, 12),
            ):
                worksheet.column_dimensions[column[0].column_letter].width = 18
    return buffer.getvalue()


def load_user_profile(path: Path) -> dict[str, Any]:
    """Load the locally persisted draft/watchlist selection."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def save_user_profile(path: Path, profile: dict[str, Any]) -> None:
    """Persist account selections so they survive application restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
