"""Weekly report and AI-prompt generation for the FPL assistant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional dependencies."""
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if "fixture" in column or column == "element":
            continue
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| "
        + " | ".join(
            "" if pd.isna(value) else str(value)
            for value in row
        )
        + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _likely_available(frame: pd.DataFrame) -> pd.Series:
    """Return a conservative availability mask for captain selection."""
    status = frame.get(
        "status",
        pd.Series("a", index=frame.index, dtype="string"),
    ).fillna("a")
    chance = pd.to_numeric(
        frame.get(
            "chance_of_playing_next_round",
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    )
    return ~status.isin(["i", "u", "s"]) & (chance.isna() | chance.ge(50))


def _captain_available(frame: pd.DataFrame) -> pd.Series:
    """Require a fully available status for captain recommendations."""
    status = frame.get(
        "status",
        pd.Series("a", index=frame.index, dtype="string"),
    ).fillna("a")
    return status.isin(["a", ""])


def _fixture_run(
    fixture_predictions: pd.DataFrame,
    element: int,
) -> str:
    """Format a player's five-GW fixture run."""
    player = fixture_predictions[
        fixture_predictions["element"].eq(element)
    ].sort_values(["gw", "fixture"])
    parts = []
    for gameweek, rows in player.groupby("gw"):
        fixtures = rows[rows["has_fixture"].astype(bool)]
        if fixtures.empty:
            parts.append(f"GW{gameweek}: BLANK")
            continue
        opponents = []
        for _, row in fixtures.iterrows():
            venue = "H" if row.get("was_home") == 1 else "A"
            opponents.append(
                f"{row.get('opponent_name', '?')} {venue} "
                f"(FDR {row.get('opponent_difficulty', np.nan):.0f})"
            )
        parts.append(f"GW{gameweek}: " + " + ".join(opponents))
    return "; ".join(parts)


def create_weekly_report(
    prediction_result: dict[str, Any],
    output_root: Path,
    prompt_template: Path | None = None,
) -> dict[str, Path]:
    """Save full predictions, ranked lists, report, JSON, and chat prompt."""
    season = prediction_result["season"]
    gameweek = int(prediction_result["gameweek"])
    player_predictions = prediction_result["player_predictions"].copy()
    fixture_predictions = prediction_result["fixture_predictions"].copy()
    output_directory = output_root / season / f"gw{gameweek}"
    output_directory.mkdir(parents=True, exist_ok=True)

    player_predictions["fixture_run"] = player_predictions["element"].map(
        lambda element: _fixture_run(fixture_predictions, int(element))
    )
    if "ep_next" in player_predictions:
        player_predictions["ep_next"] = pd.to_numeric(
            player_predictions["ep_next"], errors="coerce"
        )
        player_predictions["model_vs_ep_next_delta"] = (
            player_predictions["predicted_points_current_gw"]
            - player_predictions["ep_next"]
        )
    else:
        player_predictions["model_vs_ep_next_delta"] = np.nan
    availability = _likely_available(player_predictions)
    available_predictions = player_predictions[availability].copy()
    current_top = (
        available_predictions[
            available_predictions["current_gw_fixtures"].gt(0)
        ]
        .sort_values("predicted_points_current_gw", ascending=False)
        .head(25)
        .copy()
    )
    five_top = (
        available_predictions.sort_values(
            "predicted_average_next_5_gws",
            ascending=False,
        )
        .head(25)
        .copy()
    )
    captain_pool = player_predictions[_captain_available(player_predictions)]
    current_captain = captain_pool[
        captain_pool["current_gw_fixtures"].gt(0)
    ].nlargest(1, "predicted_points_current_gw")
    five_captain = captain_pool.nlargest(
        1,
        "predicted_average_next_5_gws",
    )

    player_path = output_directory / f"predictions_gw{gameweek}.csv"
    fixture_path = output_directory / f"fixture_predictions_gw{gameweek}.csv"
    report_path = output_directory / f"report_gw{gameweek}.md"
    json_path = output_directory / f"report_gw{gameweek}.json"
    prompt_path = output_directory / f"chat_prompt_gw{gameweek}.md"
    player_predictions.to_csv(player_path, index=False)
    fixture_predictions.to_csv(fixture_path, index=False)

    current_columns = [
        column
        for column in [
            "name",
            "position_label",
            "team_name",
            "current_price",
            "current_gw_fixtures",
            "predicted_points_current_gw",
            "ep_next",
            "model_vs_ep_next_delta",
            "status",
            "news",
        ]
        if column in current_top
    ]
    five_columns = [
        column
        for column in [
            "name",
            "position_label",
            "team_name",
            "current_price",
            "fixtures_next_5_gws",
            "blank_gws_next_5",
            "fdr_mean_next_5",
            "predicted_average_next_5_gws",
            "ep_next",
            "status",
        ]
        if column in five_top
    ]
    captain_current_name = (
        current_captain.iloc[0]["name"] if not current_captain.empty else "N/A"
    )
    captain_current_points = (
        current_captain.iloc[0]["predicted_points_current_gw"]
        if not current_captain.empty
        else np.nan
    )
    captain_five_name = (
        five_captain.iloc[0]["name"] if not five_captain.empty else "N/A"
    )
    captain_five_points = (
        five_captain.iloc[0]["predicted_average_next_5_gws"]
        if not five_captain.empty
        else np.nan
    )
    state_metadata = prediction_result.get("model_metadata", {}).get(
        "player_state",
        {},
    )
    gw1_history_note = ""
    if gameweek == 1 and state_metadata:
        gw1_history_note = (
            "- За GW1 је последња прошлосезонска форма пренета за "
            f"{state_metadata.get('previous_season_history_players', 0)} "
            "повратника повезана FPL `code` идентификатором; за "
            f"{state_metadata.get('imputed_history_players', 0)} нових или "
            "неповезаних играча недостајућу историју попуњава production "
            "imputer.\n"
        )
    excluded_unavailable = player_predictions[~availability].sort_values(
        "predicted_points_current_gw",
        ascending=False,
    )
    unavailable_note = ""
    if not excluded_unavailable.empty:
        names = ", ".join(excluded_unavailable.head(10)["name"].astype(str))
        unavailable_note = (
            f"- Из top листа су због статуса изостављени: {names}.\n"
        )
    large_disagreements = player_predictions[
        player_predictions["model_vs_ep_next_delta"].abs().gt(2.0)
    ].sort_values("model_vs_ep_next_delta", key=lambda values: values.abs(), ascending=False)
    disagreement_note = ""
    if not large_disagreements.empty:
        disagreement_note = (
            f"- За {len(large_disagreements)} играча GW модел и званични "
            "`ep_next` се разликују за више од 2 поена; третирај их као "
            "sanity-check заставице, не као сигурне изборе.\n"
        )
    model_metadata = prediction_result.get("model_metadata", {})
    forecast_mode = model_metadata.get("forecast_mode", "unspecified")
    history_weight = model_metadata.get("current_history_weight", 1.0)
    mode_note = {
        "preseason_gw1": (
            "Засебан cross-season preseason модел; актуелна сезона још нема "
            "историју наступа."
        ),
        "early_season_blend": (
            f"Blend preseason и актуелног модела; тежина актуелне форме је "
            f"{history_weight:.0%}."
        ),
        "production_gw6_plus": "Пуни модел актуелне сезоне, валидиран од GW6.",
    }.get(forecast_mode, forecast_mode)
    disagreement_columns = [
        column
        for column in [
            "name",
            "team_name",
            "status",
            "predicted_points_current_gw",
            "ep_next",
            "model_vs_ep_next_delta",
        ]
        if column in large_disagreements
    ]
    unavailable_columns = [
        column
        for column in ["name", "team_name", "status", "news"]
        if column in excluded_unavailable
    ]
    report = f"""# FPL Assistant Report — {season}, GW{gameweek}

## Режим прогнозе

- `{forecast_mode}` — {mode_note}

## Како читати две прогнозе

- `predicted_points_current_gw` долази из модела обученог за једну утакмицу. Код double gameweek-а fixture прогнозе се сабирају само у оквиру текућег GW-а.
- `predicted_average_next_5_gws` долази из засебног модела чији је директан target просек стварних GW повраћаја од GW{gameweek} до GW{gameweek + 4}.
- Друга прогноза **није** збир, нити просек пет једнокорачних прогноза. Позната тежина целог распореда, blank и double GW улазе као посебна обележја.

## Капитени

- Најбољи моделски капитен за GW{gameweek}: **{captain_current_name}** — {captain_current_points:.2f} очекиваних поена.
- Најбољи дугорочни капитен/ослонац за пет GW: **{captain_five_name}** — {captain_five_points:.2f} очекиваних поена по GW.

Оба избора су филтрирана на играче који су по локалним FPL подацима доступни или имају најмање 50% шансе за наступ. Вести након последњег освежавања података могу променити одлуку.

## Топ 25 за GW{gameweek}

{_markdown_table(current_top, current_columns)}

## Топ 25 по директном просеку за GW{gameweek}–GW{gameweek + 4}

{_markdown_table(five_top, five_columns)}

## Распоред водећих дугорочних избора

{chr(10).join(f"- **{row['name']}**: {row['fixture_run']}" for _, row in five_top.head(10).iterrows())}

## Sanity-check заставице

Највећа одступања од званичног FPL `ep_next` не значе аутоматски да је модел погрешан, али захтевају ручну проверу:

{_markdown_table(large_disagreements.head(10), disagreement_columns)}

Недоступни играчи изостављени из top листа:

{_markdown_table(excluded_unavailable.head(10), unavailable_columns)}

## Ограничења

{gw1_history_note}{unavailable_note}{disagreement_note}- Модел не зна вести објављене после последњег прикупљања података.
- Предвиђа очекивану вредност; не приказује цео распон неизвесности.
- Историјски defensive-contribution bonus за сезоне без CBIT/CBIRT компоненти је процењен вероватносним моделом и означен је као процена.
- Коначна одлука мора укључити ваш тим, буџет, free transfer-е, chip стратегију, повреде, конференције тренера и поуздане изворе.
"""
    report_path.write_text(report, encoding="utf-8")

    structured = {
        "season": season,
        "gameweek": gameweek,
        "forecast_gameweeks": prediction_result["gameweeks"],
        "definitions": {
            "current_gw": "one-fixture model aggregated within current GW",
            "next_5_average": (
                "separate direct five-GW average model; not summed "
                "one-fixture forecasts"
            ),
        },
        "captain_current_gw": current_captain.to_dict(orient="records"),
        "captain_next_5_gws": five_captain.to_dict(orient="records"),
        "top_25_current_gw": current_top.to_dict(orient="records"),
        "top_25_next_5_average": five_top.to_dict(orient="records"),
        "excluded_unavailable": excluded_unavailable.to_dict(orient="records"),
        "large_model_ep_next_disagreements": large_disagreements.to_dict(
            orient="records"
        ),
        "model_metadata": prediction_result["model_metadata"],
    }
    json_path.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    template_text = (
        prompt_template.read_text(encoding="utf-8")
        if prompt_template is not None and prompt_template.exists()
        else (
            "# Недељни FPL захтев\n\n"
            "Приложен је report за GW{gameweek}. Анализирај га уз мој тим "
            "и спољне изворе, па предложи краткорочну и петонедељну стратегију."
        )
    )
    prompt_text = template_text.format(
        season=season,
        gameweek=gameweek,
        end_gameweek=gameweek + 4,
        report_filename=report_path.name,
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")
    return {
        "directory": output_directory,
        "report": report_path,
        "json": json_path,
        "prompt": prompt_path,
        "players_csv": player_path,
        "fixtures_csv": fixture_path,
    }
