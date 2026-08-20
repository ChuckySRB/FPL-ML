"""Streamlit views for the production dual-horizon FPL assistant."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configs.config import MODELS_DIR, OUTPUTS_DIR, RAW_DATA_DIR
from src.fpl_assistant.dashboard import (
    availability_mask,
    available_gameweeks,
    build_ai_workbook,
    build_strategy_prompt,
    discover_seasons,
    infer_default_gameweek,
    load_user_profile,
    load_weekly_package,
    package_paths,
    rank_players,
    save_user_profile,
)
from src.fpl_assistant.service import generate_weekly_package
from src.optimization import optimize_fpl_squad


ROOT = Path(__file__).resolve().parents[2]
ASSISTANT_OUTPUTS = OUTPUTS_DIR / "assistant"
PROMPT_TEMPLATE = ROOT / "prompts" / "weekly_analysis_template.md"
SYSTEM_PROMPT = ROOT / "prompts" / "fpl_assistant_system_prompt.md"
USER_PROFILE = ASSISTANT_OUTPUTS / "user_profile.json"

POSITION_COLORS = {
    "GK": "#f7c948",
    "DEF": "#33d69f",
    "MID": "#42a5f5",
    "FWD": "#ff5c8a",
}

MODE_LABELS = {
    "preseason_gw1": "Preseason модел · GW1",
    "early_season_blend": "Preseason + актуелна форма",
    "production_gw6_plus": "Пуни сезонски модел",
}

DASHBOARD_RANKING_LIMIT = 50


def inject_theme() -> None:
    """Apply compact FPL-inspired styling without external assets."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 78% -10%, rgba(91,255,159,.12), transparent 28rem),
                radial-gradient(circle at 10% 18%, rgba(123,72,255,.13), transparent 26rem),
                #07140e;
        }
        [data-testid="stSidebar"] {
            background: rgba(10,29,20,.96);
            border-right: 1px solid rgba(167,255,194,.10);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1480px; padding-top: 2rem; }
        .hero {
            padding: 1.65rem 1.8rem;
            border: 1px solid rgba(159,255,190,.14);
            border-radius: 22px;
            background: linear-gradient(125deg, rgba(17,49,33,.95), rgba(23,23,54,.92));
            box-shadow: 0 22px 60px rgba(0,0,0,.24);
            margin-bottom: 1rem;
        }
        .eyebrow {
            color: #8dffb0; font-size: .74rem; font-weight: 800;
            letter-spacing: .14em; text-transform: uppercase;
        }
        .hero h1 {
            margin: .35rem 0 .3rem;
            font-size: clamp(2rem, 4vw, 3.4rem);
        }
        .hero p { color: #b7c9bf; max-width: 760px; margin: 0; }
        .mode-badge {
            display: inline-block; margin-top: .85rem; padding: .35rem .7rem;
            border-radius: 999px; background: rgba(137,255,172,.10);
            color: #a4ffbe; font-size: .78rem; font-weight: 700;
        }
        .captain-card {
            min-height: 152px; padding: 1.25rem 1.35rem;
            border-radius: 18px; border: 1px solid rgba(255,255,255,.09);
            background: linear-gradient(145deg, rgba(21,55,38,.96), rgba(16,31,26,.96));
        }
        .captain-card.long {
            background: linear-gradient(145deg, rgba(46,34,89,.96), rgba(22,25,45,.96));
        }
        .captain-card .label {
            color: #9db2a6; font-size: .78rem; font-weight: 700;
        }
        .captain-card h3 { font-size: 1.55rem; margin: .35rem 0 .1rem; }
        .captain-card .score {
            color: #93ffb3; font-size: 1.2rem; font-weight: 800;
        }
        .captain-card .meta {
            color: #91a49a; font-size: .82rem; margin-top: .35rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,.075);
            border-radius: 16px; padding: .85rem 1rem;
            background: rgba(13,34,23,.74);
        }
        .section-copy { color: #9eb2a6; margin-top: -.35rem; }
        .status-dot {
            display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; margin-right: .4rem; background: #66f29a;
            box-shadow: 0 0 10px #66f29a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_assistant_package(
    season: str,
    gameweek: int,
) -> dict[str, Any]:
    """Cache a generated package between Streamlit reruns."""
    return load_weekly_package(ASSISTANT_OUTPUTS, season, gameweek)


def _chart_layout(figure: go.Figure, height: int = 430) -> go.Figure:
    """Apply the shared transparent chart treatment."""
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="rgba(255,255,255,.07)")
    figure.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return figure


def _first_record(records: Any) -> dict[str, Any]:
    """Return the first structured-report record."""
    return records[0] if isinstance(records, list) and records else {}


def _render_hero(
    season: str,
    gameweek: int,
    metadata: dict[str, Any],
) -> None:
    """Render the active season and model route."""
    mode = metadata.get("forecast_mode", "unspecified")
    mode_label = MODE_LABELS.get(mode, mode)
    weight_copy = ""
    if mode == "early_season_blend":
        weight = float(metadata.get("current_history_weight", 0))
        weight_copy = f" · актуелна форма {weight:.0%}"
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">FPL decision workspace · {season}</div>
          <h1>Gameweek {gameweek} command center</h1>
          <p>Две независне прогнозе, распоред, ризици и контекст твог тима — на једном месту пре deadline-а.</p>
          <span class="mode-badge"><span class="status-dot"></span>{mode_label}{weight_copy}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_captains(structured: dict[str, Any]) -> None:
    """Show separate captain recommendations for both model horizons."""
    current = _first_record(structured.get("captain_current_gw"))
    long_term = _first_record(structured.get("captain_next_5_gws"))
    cards = [
        (
            "",
            "КАПИТЕН ЗА ТЕКУЋИ GW",
            current,
            "predicted_points_current_gw",
            "очекиваних поена",
        ),
        (
            " long",
            "ПЕТОНЕДЕЉНИ ОСЛОНАЦ",
            long_term,
            "predicted_average_next_5_gws",
            "поена / GW",
        ),
    ]
    for column, card in zip(st.columns(2), cards):
        css_class, label, player, score_key, suffix = card
        with column:
            score = float(player.get(score_key, 0))
            price = float(player.get("current_price", 0))
            st.markdown(
                f"""
                <div class="captain-card{css_class}">
                  <div class="label">{label}</div>
                  <h3>{player.get('name', 'N/A')}</h3>
                  <div class="score">{score:.2f} {suffix}</div>
                  <div class="meta">{player.get('team_name', '—')} · {player.get('position_label', '—')} · £{price:.1f}m</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _ranking_filters(
    players: pd.DataFrame,
    key: str,
) -> tuple[list[str], list[str], float, bool]:
    """Render reusable ranking filters."""
    with st.expander("Филтери"):
        columns = st.columns([1.2, 1.6, 1, 1])
        positions = columns[0].multiselect(
            "Позиције",
            ["GK", "DEF", "MID", "FWD"],
            default=["GK", "DEF", "MID", "FWD"],
            key=f"{key}_positions",
        )
        teams = columns[1].multiselect(
            "Клубови",
            sorted(players["team_name"].dropna().astype(str).unique()),
            key=f"{key}_teams",
        )
        prices = pd.to_numeric(players["current_price"], errors="coerce")
        maximum = float(prices.max()) if prices.notna().any() else 20.0
        maximum_price = columns[2].slider(
            "Макс. цена",
            4.0,
            max(4.0, maximum),
            max(4.0, maximum),
            0.5,
            key=f"{key}_price",
        )
        available_only = columns[3].toggle(
            "Само доступни",
            value=True,
            key=f"{key}_available",
        )
    return positions, teams, maximum_price, available_only


def _render_ranked_table(
    ranked: pd.DataFrame,
    prediction_column: str,
    current_horizon: bool,
) -> None:
    """Render the top dashboard rankings and price/projection chart."""
    if ranked.empty:
        st.warning("Нема играча за изабране филтере.")
        return
    top = ranked.head(DASHBOARD_RANKING_LIMIT).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    if current_horizon:
        columns = [
            "rank",
            "name",
            "position_label",
            "team_name",
            "current_price",
            prediction_column,
            "ep_next",
            "model_vs_ep_next_delta",
            "selected_by_percent",
            "fixture_run",
        ]
    else:
        columns = [
            "rank",
            "name",
            "position_label",
            "team_name",
            "current_price",
            prediction_column,
            "fixtures_next_5_gws",
            "blank_gws_next_5",
            "fdr_mean_next_5",
            "value_score",
            "fixture_run",
        ]
    columns = [column for column in columns if column in top]
    labels = {
        "rank": "#",
        "name": "Играч",
        "position_label": "Поз.",
        "team_name": "Клуб",
        "current_price": "Цена",
        prediction_column: "Модел" if current_horizon else "Просек / GW",
        "ep_next": "FPL xP",
        "model_vs_ep_next_delta": "Δ vs xP",
        "selected_by_percent": "Власн. %",
        "fixtures_next_5_gws": "Ут.",
        "blank_gws_next_5": "Blank",
        "fdr_mean_next_5": "FDR",
        "value_score": "Вредност",
        "fixture_run": "Распоред",
    }
    st.dataframe(
        top[columns].rename(columns=labels),
        width="stretch",
        hide_index=True,
        height=980,
        column_config={
            "Цена": st.column_config.NumberColumn(format="£%.1fm"),
            "Модел": st.column_config.NumberColumn(format="%.2f"),
            "FPL xP": st.column_config.NumberColumn(format="%.2f"),
            "Δ vs xP": st.column_config.NumberColumn(format="%+.2f"),
            "Просек / GW": st.column_config.NumberColumn(format="%.2f"),
            "FDR": st.column_config.NumberColumn(format="%.2f"),
            "Вредност": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    chart = px.scatter(
        top,
        x="current_price",
        y=prediction_column,
        size=(
            "selected_by_percent"
            if "selected_by_percent" in top
            else None
        ),
        color="position_label",
        color_discrete_map=POSITION_COLORS,
        hover_name="name",
        hover_data=["team_name", "fixture_run"],
        labels={
            "current_price": "Цена (£m)",
            prediction_column: (
                "Прогноза за GW" if current_horizon else "Просек / GW"
            ),
            "position_label": "Позиција",
        },
        title="Однос цене и моделске вредности",
    )
    st.plotly_chart(
        _chart_layout(chart),
        width="stretch",
        config={"displayModeBar": False},
    )


def _render_current(players: pd.DataFrame) -> None:
    """Render current-gameweek rankings."""
    st.subheader("Најбољи избори за текући GW")
    st.markdown(
        '<p class="section-copy">Прогноза једне утакмице; код DGW-а се сабирају само утакмице у овом колу.</p>',
        unsafe_allow_html=True,
    )
    positions, teams, price, available_only = _ranking_filters(
        players,
        "current",
    )
    ranked = rank_players(
        players,
        "predicted_points_current_gw",
        positions,
        teams,
        price,
        available_only,
    )
    ranked = ranked[ranked["current_gw_fixtures"].gt(0)]
    _render_ranked_table(ranked, "predicted_points_current_gw", True)


def _render_five(players: pd.DataFrame) -> None:
    """Render direct five-gameweek average rankings."""
    st.subheader("Петонедељни план")
    st.markdown(
        '<p class="section-copy">Засебан директни модел просека. Ово није збир пет једнокорачних прогноза.</p>',
        unsafe_allow_html=True,
    )
    positions, teams, price, available_only = _ranking_filters(
        players,
        "five",
    )
    ranked = rank_players(
        players,
        "predicted_average_next_5_gws",
        positions,
        teams,
        price,
        available_only,
    )
    _render_ranked_table(
        ranked,
        "predicted_average_next_5_gws",
        False,
    )


def _player_options(players: pd.DataFrame) -> dict[int, str]:
    """Build stable player choices keyed by FPL element ID."""
    options = {}
    for row in players.itertuples():
        price = float(getattr(row, "current_price", 0) or 0)
        options[int(row.element)] = (
            f"{row.name} · {row.team_name} · "
            f"{row.position_label} · £{price:.1f}m"
        )
    return options


def _render_comparison(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> None:
    """Compare up to four players across predictions and FDR."""
    st.subheader("Поређење играча")
    available = players[availability_mask(players)].copy()
    options = _player_options(available)
    defaults = (
        available.nlargest(3, "predicted_average_next_5_gws")["element"]
        .astype(int)
        .tolist()
    )
    selected = st.multiselect(
        "Изабери до четири играча",
        list(options),
        default=defaults,
        format_func=lambda element: options[element],
        max_selections=4,
        key="comparison_players",
    )
    if not selected:
        st.info("Изабери бар једног играча.")
        return
    comparison = available[available["element"].isin(selected)].copy()
    scores = comparison.melt(
        id_vars=["name"],
        value_vars=[
            "predicted_points_current_gw",
            "predicted_average_next_5_gws",
            "ep_next",
        ],
        var_name="forecast",
        value_name="points",
    )
    scores["forecast"] = scores["forecast"].map(
        {
            "predicted_points_current_gw": "Модел · овај GW",
            "predicted_average_next_5_gws": "Модел · 5GW просек",
            "ep_next": "Званични FPL xP",
        }
    )
    chart = px.bar(
        scores,
        x="name",
        y="points",
        color="forecast",
        barmode="group",
        color_discrete_sequence=["#65f09a", "#8c6dfd", "#f7c948"],
        labels={
            "name": "Играч",
            "points": "Поени",
            "forecast": "Прогноза",
        },
        title="Краткорочни сигнал, дугорочни сигнал и FPL референца",
    )
    st.plotly_chart(
        _chart_layout(chart),
        width="stretch",
        config={"displayModeBar": False},
    )
    has_fixture = (
        fixtures["has_fixture"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    selected_fixtures = fixtures[
        fixtures["element"].isin(selected) & has_fixture
    ]
    if not selected_fixtures.empty:
        heatmap = selected_fixtures.pivot_table(
            index="name",
            columns="gw",
            values="opponent_difficulty",
            aggfunc="mean",
        )
        figure = px.imshow(
            heatmap,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale=["#1fb978", "#e2c044", "#d95068"],
            labels={
                "x": "Gameweek",
                "y": "Играч",
                "color": "FDR",
            },
            title="Тежина распореда — ниже је лакше",
        )
        st.plotly_chart(
            _chart_layout(figure, 320),
            width="stretch",
            config={"displayModeBar": False},
        )
    st.dataframe(
        comparison[
            [
                "name",
                "team_name",
                "position_label",
                "current_price",
                "predicted_points_current_gw",
                "predicted_average_next_5_gws",
                "fdr_mean_next_5",
                "selected_by_percent",
                "fixture_run",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def _render_risks(
    players: pd.DataFrame,
    structured: dict[str, Any],
) -> None:
    """Expose availability and model/reference disagreements."""
    st.subheader("Ризици које треба ручно проверити")
    unavailable = pd.DataFrame(structured.get("excluded_unavailable", []))
    disagreements = pd.DataFrame(
        structured.get("large_model_ep_next_disagreements", [])
    )
    status = players.get(
        "status",
        pd.Series("a", index=players.index),
    )
    doubtful = players[status.eq("d")]
    metrics = st.columns(3)
    metrics[0].metric("Недоступни / суспендовани", len(unavailable))
    metrics[1].metric("Одступање од xP > 2", len(disagreements))
    metrics[2].metric("Означени као doubtful", len(doubtful))
    left, right = st.columns(2)
    with left:
        st.markdown("#### Статус и вести")
        status_rows = pd.concat(
            [unavailable, doubtful],
            ignore_index=True,
        )
        if status_rows.empty:
            st.success("Локални snapshot нема availability ризика.")
        else:
            status_rows = status_rows.drop_duplicates(subset=["element"])
            columns = [
                column
                for column in [
                    "name",
                    "team_name",
                    "status",
                    "chance_of_playing_next_round",
                    "news",
                ]
                if column in status_rows
            ]
            st.dataframe(
                status_rows[columns],
                width="stretch",
                hide_index=True,
                height=420,
            )
    with right:
        st.markdown("#### Модел насупрот FPL xP")
        valid = players.dropna(
            subset=["ep_next", "predicted_points_current_gw"]
        )
        chart = px.scatter(
            valid,
            x="ep_next",
            y="predicted_points_current_gw",
            color="position_label",
            color_discrete_map=POSITION_COLORS,
            hover_name="name",
            hover_data=["team_name", "news"],
            labels={
                "ep_next": "Званични FPL xP",
                "predicted_points_current_gw": "Наш модел",
                "position_label": "Позиција",
            },
        )
        maximum = max(
            valid["ep_next"].max(),
            valid["predicted_points_current_gw"].max(),
        )
        chart.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=maximum,
            y1=maximum,
            line={"dash": "dash", "color": "#8da397"},
        )
        st.plotly_chart(
            _chart_layout(chart, 420),
            width="stretch",
            config={"displayModeBar": False},
        )
    if not disagreements.empty:
        st.markdown("#### Највећа неслагања — sanity check")
        columns = [
            column
            for column in [
                "name",
                "team_name",
                "status",
                "predicted_points_current_gw",
                "ep_next",
                "model_vs_ep_next_delta",
                "news",
            ]
            if column in disagreements
        ]
        st.dataframe(
            disagreements[columns].head(25),
            width="stretch",
            hide_index=True,
        )


def _optimizer_candidates(
    players: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Adapt assistant rows to the legal-squad optimizer."""
    available = players[availability_mask(players)].copy()
    return available.rename(
        columns={
            "current_price": "price",
            prediction_column: "projected_points",
        }
    )[
        [
            "element",
            "name",
            "position_label",
            "team_name",
            "price",
            "projected_points",
        ]
    ]


def _render_squad_ai(
    players: pd.DataFrame,
    package: dict[str, Any],
    season: str,
    gameweek: int,
) -> None:
    """Collect team context and prepare the complete AI handoff."""
    st.subheader("Мој тим и AI handoff")
    st.markdown(
        '<p class="section-copy">Унеси стварно стање тима; апликација га додаје уз report и prompt.</p>',
        unsafe_allow_html=True,
    )
    profile = load_user_profile(USER_PROFILE)
    options = _player_options(players)
    valid_elements = set(options)
    squad_key = f"squad_{season}"
    watchlist_key = f"watchlist_{season}"
    if squad_key not in st.session_state:
        st.session_state[squad_key] = [
            int(element)
            for element in profile.get("squad_elements", [])
            if int(element) in valid_elements
        ]
    if watchlist_key not in st.session_state:
        st.session_state[watchlist_key] = [
            int(element)
            for element in profile.get("watchlist_elements", [])
            if int(element) in valid_elements
        ]
    selected = st.multiselect(
        "Тренутни састав (до 15 играча)",
        list(options),
        format_func=lambda element: options[element],
        max_selections=15,
        key=squad_key,
    )
    squad = players[players["element"].isin(selected)].copy()
    watchlist_selected = st.multiselect(
        "Watchlist — додај играче за посебну AI процену",
        list(options),
        format_func=lambda element: options[element],
        key=watchlist_key,
        help=(
            "Кликни × поред имена да га уклониш. Листа се чува локално "
            "када генеришеш AI пакет."
        ),
    )
    watchlist = players[
        players["element"].isin(watchlist_selected)
    ].copy()
    price_series = pd.to_numeric(
        squad.get("current_price"),
        errors="coerce",
    )
    metrics = st.columns(6)
    metrics[0].metric("Играча", f"{len(squad)}/15")
    for index, position in enumerate(
        ("GK", "DEF", "MID", "FWD"),
        start=1,
    ):
        metrics[index].metric(
            position,
            int(squad["position_label"].eq(position).sum()),
        )
    metrics[5].metric("Тренутна цена", f"£{price_series.sum():.1f}m")
    left, middle, right = st.columns(3)
    bank = left.number_input(
        "Новац у банци (£m)",
        0.0,
        20.0,
        float(profile.get("bank", 0.0)),
        0.1,
        key=f"bank_{season}",
    )
    free_transfers = middle.number_input(
        "Free transfer-и",
        0,
        5,
        int(profile.get("free_transfers", 1)),
        1,
        key=f"fts_{season}",
    )
    risk_profile = right.selectbox(
        "Профил ризика",
        ["Умерен", "Конзервативан", "Агресиван"],
        index=["Умерен", "Конзервативан", "Агресиван"].index(
            profile.get("risk_profile", "Умерен")
            if profile.get("risk_profile", "Умерен")
            in ["Умерен", "Конзервативан", "Агресиван"]
            else "Умерен"
        ),
        key=f"risk_{season}",
    )
    chips = st.text_input(
        "Расположиви chip-ови",
        value=str(profile.get("chips", "")),
        placeholder="Wildcard, Free Hit, Bench Boost, Triple Captain",
        key=f"chips_{season}",
    )
    notes = st.text_area(
        "Белешке из конференција, објава и видео-анализа",
        value=str(profile.get("external_notes", "")),
        placeholder=(
            "За сваки извор унеси датум, аутора/линк, "
            "кључну информацију и процену поузданости."
        ),
        height=160,
        key=f"notes_{season}",
    )
    with st.expander("Моделски wildcard предлог"):
        st.caption(
            "Поштује £100m, позиције и највише три играча по клубу. "
            "Не зна продајне цене, казне трансфера ни најновије вести."
        )
        horizon = st.radio(
            "Циљ",
            ["Текући GW", "5GW просек"],
            horizontal=True,
            key=f"optimizer_horizon_{season}_{gameweek}",
        )
        budget = st.slider(
            "Буџет",
            80.0,
            110.0,
            100.0,
            0.5,
            key=f"optimizer_budget_{season}_{gameweek}",
        )
        if st.button(
            "Оптимизуј 15 играча",
            key=f"optimize_{season}_{gameweek}",
        ):
            column = (
                "predicted_points_current_gw"
                if horizon == "Текући GW"
                else "predicted_average_next_5_gws"
            )
            try:
                optimized = optimize_fpl_squad(
                    _optimizer_candidates(players, column),
                    budget=budget,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.dataframe(
                    optimized,
                    width="stretch",
                    hide_index=True,
                )
    st.markdown("#### Генериши пакет за GPT")
    expected_positions = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    actual_positions = {
        position: int(squad["position_label"].eq(position).sum())
        for position in expected_positions
    }
    invalid_positions = [
        f"{position} {actual_positions[position]}/{expected}"
        for position, expected in expected_positions.items()
        if actual_positions[position] != expected
    ]
    club_counts = squad["team_name"].value_counts()
    over_limit = club_counts[club_counts.gt(3)]
    validation_messages = []
    if len(squad) != 15:
        validation_messages.append(f"изабрано је {len(squad)}/15 играча")
    if invalid_positions:
        validation_messages.append("позиције: " + ", ".join(invalid_positions))
    if not over_limit.empty:
        validation_messages.append(
            "више од три из клуба: "
            + ", ".join(
                f"{club} ({count})" for club, count in over_limit.items()
            )
        )
    if price_series.sum() + float(bank) > 100.05:
        validation_messages.append(
            f"тим + банка = £{price_series.sum() + float(bank):.1f}m"
        )
    if validation_messages:
        st.warning(
            "Draft још није валидан: " + "; ".join(validation_messages) + "."
        )
    else:
        st.success(
            f"Draft је структурно валидан. Watchlist: {len(watchlist)} играча."
        )
    system_prompt = (
        SYSTEM_PROMPT.read_text(encoding="utf-8")
        if SYSTEM_PROMPT.exists()
        else ""
    )
    attachment_filename = f"fpl_assistant_{season}_gw{gameweek}.xlsx"
    generated_at = datetime.fromtimestamp(package["generated_at"]).isoformat(
        timespec="minutes"
    )
    artifact_key = f"ai_artifacts_{season}_{gameweek}"
    if st.button(
        "Генериши AI Excel и комплетан prompt",
        type="primary",
        width="stretch",
        key=f"generate_ai_{season}_{gameweek}",
    ):
        account = {
            "bank": float(bank),
            "free_transfers": int(free_transfers),
            "chips": chips,
            "risk_profile": risk_profile,
            "external_notes": notes,
            "generated_at": generated_at,
        }
        user_prompt = build_strategy_prompt(
            package["prompt"],
            squad,
            watchlist,
            float(bank),
            int(free_transfers),
            chips,
            risk_profile,
            notes,
            attachment_filename,
            generated_at,
        )
        combined_prompt = (
            "# SYSTEM INSTRUCTIONS\n\n"
            f"{system_prompt.rstrip()}\n\n"
            "---\n\n# USER REQUEST\n\n"
            f"{user_prompt.rstrip()}\n"
        )
        workbook = build_ai_workbook(
            season,
            gameweek,
            players,
            package["fixtures"],
            package["structured"],
            squad,
            watchlist,
            account,
        )
        save_user_profile(
            USER_PROFILE,
            {
                "season": season,
                "squad_elements": [int(value) for value in selected],
                "watchlist_elements": [
                    int(value) for value in watchlist_selected
                ],
                **account,
            },
        )
        st.session_state[artifact_key] = {
            "workbook": workbook,
            "user_prompt": user_prompt,
            "combined_prompt": combined_prompt,
        }
        st.success(
            "Пакет је генерисан и draft/watchlist су сачувани локално."
        )
    artifacts = st.session_state.get(artifact_key)
    if artifacts:
        st.info(
            "Најједноставније: окачи Excel и копирај `Комплетан prompt`. "
            "Он већ садржи и system улогу и твој недељни захтев."
        )
        columns = st.columns(4)
        downloads = [
            (
                "1 · Excel за upload",
                artifacts["workbook"],
                attachment_filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            (
                "2 · Комплетан prompt",
                artifacts["combined_prompt"],
                f"complete_prompt_gw{gameweek}.md",
                "text/markdown",
            ),
            (
                "Само user prompt",
                artifacts["user_prompt"],
                f"user_prompt_gw{gameweek}.md",
                "text/markdown",
            ),
            (
                "Само system role",
                system_prompt,
                "fpl_assistant_system_prompt.md",
                "text/markdown",
            ),
        ]
        for column, download in zip(columns, downloads):
            label, data, filename, mime = download
            column.download_button(
                label,
                data,
                file_name=filename,
                mime=mime,
                width="stretch",
            )
        with st.expander("Копирај комплетан prompt", expanded=True):
            st.code(artifacts["combined_prompt"], language="markdown")
    with st.expander("Преглед моделског report-а"):
        st.markdown(package["report"])


def render_weekly_assistant() -> None:
    """Render the production assistant and its generation controls."""
    seasons = discover_seasons(RAW_DATA_DIR)
    if not seasons:
        st.error(
            "Нема сезоне са players, teams и fixtures подацима у data/raw."
        )
        return
    with st.sidebar:
        st.markdown("### Контрола прогнозе")
        season = st.selectbox(
            "Сезона",
            seasons,
            index=len(seasons) - 1,
        )
        gameweeks = available_gameweeks(RAW_DATA_DIR, season)
        if not gameweeks:
            st.error("У fixtures.csv нема gameweek ознака.")
            return
        inferred = infer_default_gameweek(
            RAW_DATA_DIR,
            season,
            gameweeks,
        )
        default_index = (
            gameweeks.index(inferred)
            if inferred in gameweeks
            else 0
        )
        gameweek = st.selectbox(
            "Gameweek",
            gameweeks,
            index=default_index,
        )
        refresh = st.toggle(
            "Освежи званичне FPL податке",
            value=True,
            help=(
                "Укључено: пре прогнозе преузима players, teams и "
                "fixtures. Искључено: користи локални snapshot."
            ),
        )
        run = st.button(
            "Покрени прогнозу",
            type="primary",
            width="stretch",
        )
        paths = package_paths(ASSISTANT_OUTPUTS, season, gameweek)
        package_ready = all(
            path.exists()
            for name, path in paths.items()
            if name != "directory"
        )
        st.caption(
            "Пакет је већ генерисан."
            if package_ready
            else "За овај GW још нема локалног пакета."
        )
    if run:
        message = (
            "Освежавам податке и рачунам обе прогнозе..."
            if refresh
            else "Рачунам обе прогнозе из локалног snapshot-а..."
        )
        with st.spinner(message):
            try:
                generate_weekly_package(
                    season=season,
                    gameweek=gameweek,
                    refresh_official_data=refresh,
                    model_directory=MODELS_DIR / "fpl_assistant",
                    output_directory=ASSISTANT_OUTPUTS,
                    prompt_template=PROMPT_TEMPLATE,
                )
            except Exception as error:
                st.error(f"Генерисање није успело: {error}")
            else:
                load_assistant_package.clear()
                st.session_state["generation_success"] = (
                    f"GW{gameweek} пакет је успешно генерисан."
                )
                st.rerun()
    success = st.session_state.pop("generation_success", None)
    if success:
        st.success(success)
    try:
        package = load_assistant_package(season, gameweek)
    except FileNotFoundError:
        st.markdown(
            f"## {season} · GW{gameweek}\n"
            "Изабрани пакет не постоји. Кликни "
            "**Покрени прогнозу** у sidebar-у."
        )
        return
    structured = package["structured"]
    metadata = structured.get("model_metadata", {})
    players = package["players"]
    fixtures = package["fixtures"]
    _render_hero(season, gameweek, metadata)
    _render_captains(structured)
    single_validation = (
        metadata.get("one_fixture", {}).get("validation", {})
    )
    five_validation = metadata.get("five_gw", {}).get("validation", {})
    metrics = st.columns(4)
    metrics[0].metric("Анализирано играча", f"{len(players):,}")
    metrics[1].metric(
        "1GW validation MAE",
        f"{float(single_validation.get('mae', 0)):.3f}",
    )
    metrics[2].metric(
        "5GW validation MAE",
        f"{float(five_validation.get('mae', 0)):.3f}",
    )
    generated_at = datetime.fromtimestamp(package["generated_at"])
    metrics[3].metric(
        "Пакет освежен",
        generated_at.strftime("%d.%m.%Y. %H:%M"),
    )
    tabs = st.tabs(
        [
            "Овај GW",
            "Наредних 5",
            "Поређење",
            "Ризици",
            "Мој тим + AI",
        ]
    )
    with tabs[0]:
        _render_current(players)
    with tabs[1]:
        _render_five(players)
    with tabs[2]:
        _render_comparison(players, fixtures)
    with tabs[3]:
        _render_risks(players, structured)
    with tabs[4]:
        _render_squad_ai(
            players,
            package,
            season,
            gameweek,
        )
