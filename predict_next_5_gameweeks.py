#!/usr/bin/env python3
"""
FPL Prediction Script — Next 5 Gameweeks
=========================================
Single-run script that:
  1. Downloads latest FPL data (historical seasons + current 2025-26 from the API)
  2. Engineers Tier-3 features per-season (no data leakage)
  3. Trains an XGBoost model on ALL available data and saves it
  4. Predicts total_points for every player for each of the next 5 gameweeks
  5. Saves predictions to CSV in outputs/predictions/
  6. Displays a structured terminal report and saves a PNG summary chart

Usage:
    python predict_next_5_gameweeks.py

Outputs (in outputs/predictions/):
    predictions_gw<N>_to_gw<M>.csv   — full predictions table
    predictions_gw<N>_to_gw<M>.png   — visual summary chart
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# ── Project root & imports ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from configs.config import RAW_DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_collection.current_season_collector import CurrentSeasonCollector
from src.data_collection.historical_data_downloader import download_season_data
from src.preprocessing.data_loader import FPLDataLoader
from src.preprocessing.feature_engineering import (
    FPLFeatureEngineer,
    TIER3_FEATURES,
    TIER2_FEATURES,
    prepare_training_data,
)

# ── Configuration ─────────────────────────────────────────────────────────────
CURRENT_SEASON = "2025-26"
TRAIN_SEASONS  = ["2021-22", "2022-23", "2023-24", "2024-25"]

PREDICTION_DIR = OUTPUTS_DIR / "predictions"
MODEL_PATH     = MODELS_DIR / "xgboost_next5gw.json"
IMPUTER_PATH   = MODELS_DIR / "imputer_next5gw.pkl"
N_FUTURE_GW    = 5

POSITION_MAP   = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_COLORS     = {"GK": "#f9c74f", "DEF": "#4cc9f0", "MID": "#7bed9f", "FWD": "#ff6b6b"}

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Download
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_data() -> None:
    """Download historical seasons (skip if cached) and refresh current season from the FPL API."""
    _header("STEP 1: Downloading Data")

    # Historical seasons — skip if merged_gw already exists
    for season in TRAIN_SEASONS:
        merged = RAW_DATA_DIR / season / "gws" / "merged_gw.csv"
        if merged.exists():
            print(f"  ⊙ {season}: already downloaded, skipping")
        else:
            print(f"  → Downloading {season}…")
            download_season_data(season)

    # Current season — always refresh so we have the latest GW
    print(f"\n  → Refreshing current season ({CURRENT_SEASON}) from FPL API…")
    collector = CurrentSeasonCollector(season=CURRENT_SEASON)
    collector.collect_all(include_player_details=True)
    print("  ✓ Current season data refreshed")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load Data & Engineer Features
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_player_info(gw_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Add player name, position_label, element_type and team to a GW DataFrame.

    Always refreshes name/position/element_type from players_raw.csv so that
    real player names (not NaN or element IDs) appear in every output.
    team is only added when absent (it may already be correctly set by the
    data loader).
    """
    players_path = RAW_DATA_DIR / season / "players_raw.csv"

    # Try players_raw.csv first, then fall back to bootstrap_static.json
    meta: Optional[pd.DataFrame] = None
    if players_path.exists():
        try:
            players_df = pd.read_csv(players_path)
            cols_needed = {"id", "web_name", "element_type", "team"}
            if cols_needed.issubset(players_df.columns):
                meta = players_df[["id", "web_name", "element_type", "team",
                                    "now_cost"]].copy() \
                    if "now_cost" in players_df.columns \
                    else players_df[["id", "web_name", "element_type", "team"]].copy()
        except Exception:
            pass

    if meta is None:
        bootstrap_path = RAW_DATA_DIR / season / "bootstrap_static.json"
        if bootstrap_path.exists():
            try:
                with open(bootstrap_path, encoding="utf-8") as fh:
                    bdata = json.load(fh)
                elements = bdata.get("elements", [])
                meta = pd.DataFrame([
                    {"id": e["id"], "web_name": e["web_name"],
                     "element_type": e.get("element_type"),
                     "team": e.get("team"),
                     "now_cost": e.get("now_cost")}
                    for e in elements
                ])
            except Exception:
                pass

    if meta is None or meta.empty:
        return gw_df

    meta = meta.rename(columns={"id": "element", "web_name": "name"})
    meta["position_label"] = meta["element_type"].map(POSITION_MAP)
    meta["team"] = pd.to_numeric(meta["team"], errors="coerce")

    # Always drop existing name/position/element_type so real values replace any
    # stale or missing data; only add team when genuinely absent.
    gw_df = gw_df.copy()
    for col in ["name", "position_label", "element_type"]:
        if col in gw_df.columns:
            gw_df = gw_df.drop(columns=[col])

    add_cols = ["name", "position_label", "element_type"]
    if "team" not in gw_df.columns:
        add_cols.append("team")
    if "now_cost" not in gw_df.columns and "now_cost" in meta.columns:
        add_cols.append("now_cost")

    return gw_df.merge(
        meta[["element"] + add_cols],
        on="element",
        how="left",
    )


def _load_and_engineer_season(
    loader: FPLDataLoader,
    engineer: FPLFeatureEngineer,
    season: str,
    tier: int = 3,
) -> Optional[pd.DataFrame]:
    """Load one season's raw data, enrich it and engineer features."""
    try:
        gw_df = loader.load_gameweeks(season)
    except FileNotFoundError:
        print(f"  ! {season}: no gameweek file found, skipping")
        return None

    gw_df["season"] = season

    # Enrich current-season data (no name/position/team in its merged_gw.csv)
    if season == CURRENT_SEASON:
        gw_df = _enrich_with_player_info(gw_df, season)

    if gw_df.empty or "element" not in gw_df.columns:
        print(f"  ! {season}: empty or missing element column, skipping")
        return None

    # Load teams and fixtures for opponent-difficulty features
    teams_df: Optional[pd.DataFrame] = None
    fixtures_df: Optional[pd.DataFrame] = None
    try:
        teams_df = loader.load_teams(season)
    except FileNotFoundError:
        pass
    try:
        fixtures_df = loader.load_fixtures(season)
    except FileNotFoundError:
        pass

    try:
        df = engineer.create_all_features(
            gw_df, teams_df=teams_df, fixtures_df=fixtures_df, tier=tier
        )
    except Exception as exc:
        print(f"  ! {season}: feature engineering failed ({exc}), skipping")
        return None

    return df


def build_training_data(
    tier: int = 3,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build train X / y from all available seasons.

    Features are engineered per-season (to avoid cross-season rolling leakage)
    then concatenated.

    Returns:
        X_train   — feature DataFrame
        y_train   — target Series
        feat_list — ordered list of feature column names used
    """
    _header("STEP 2: Building Training Dataset")

    loader   = FPLDataLoader()
    engineer = FPLFeatureEngineer()
    all_dfs: List[pd.DataFrame] = []

    for season in TRAIN_SEASONS + [CURRENT_SEASON]:
        print(f"\n  Loading {season}…")
        df = _load_and_engineer_season(loader, engineer, season, tier=tier)
        if df is None:
            continue
        df = prepare_training_data(df, min_gw=4)
        if df.empty:
            print(f"  ! {season}: no rows after min_gw filter, skipping")
            continue
        all_dfs.append(df)
        print(f"  ✓ {season}: {len(df):,} training rows")

    if not all_dfs:
        raise RuntimeError("No training data available — run data download first.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Choose feature list based on tier; fall back gracefully for missing columns
    candidate = TIER3_FEATURES if tier == 3 else TIER2_FEATURES
    feat_list = [f for f in candidate if f in combined.columns]
    missing   = [f for f in candidate if f not in combined.columns]
    if missing:
        print(f"\n  ⚠ Features not found (will be skipped): {missing}")

    print(f"\n  Total training records : {len(combined):,}")
    print(f"  Features used          : {len(feat_list)}")
    print(f"  Target mean / std      : {combined['total_points'].mean():.2f} / "
          f"{combined['total_points'].std():.2f}")

    X = combined[feat_list].copy()
    y = combined["total_points"].copy()
    return X, y, feat_list


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train XGBoost Model
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feat_list: List[str],
) -> Tuple[xgb.XGBRegressor, SimpleImputer]:
    """Train XGBoost on the full training set, save model and imputer."""
    _header("STEP 3: Training XGBoost Model")

    # Impute missing values with per-column medians
    print("  Imputing missing values (median strategy)…")
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=feat_list,
    )

    print(f"  Training on {len(X_imp):,} samples × {len(feat_list)} features…")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_imp, y_train)

    # In-sample metrics (informational only)
    y_pred_is = model.predict(X_imp)
    mae  = mean_absolute_error(y_train, y_pred_is)
    rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_is)))
    print(f"  In-sample  MAE={mae:.3f}  RMSE={rmse:.3f}  (train set, informational)")

    # Persist
    model.save_model(str(MODEL_PATH))
    joblib.dump(imputer, IMPUTER_PATH)
    print(f"  ✓ Model  saved → {MODEL_PATH.relative_to(ROOT)}")
    print(f"  ✓ Imputer saved → {IMPUTER_PATH.relative_to(ROOT)}")

    return model, imputer


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Current Player State (rolling features snapshot)
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_player_state(feat_list: List[str]) -> pd.DataFrame:
    """Return the latest engineered feature row for each player in 2025-26.

    The rolling / form features in this row represent the player's state
    *going into* the next gameweek and will be held constant across all
    5 future GW predictions (only was_home / opponent features change).
    """
    print("\n  Extracting current player state from 2025-26 data…")

    loader   = FPLDataLoader()
    engineer = FPLFeatureEngineer()

    df = _load_and_engineer_season(loader, engineer, CURRENT_SEASON, tier=3)
    if df is None or df.empty:
        return pd.DataFrame()

    # Keep only rows where the player actually played (minutes > 0) so that
    # the last row reflects real form, not a DNP bench appearance
    if "minutes" in df.columns:
        played = df[df["minutes"] > 0].copy()
        # If some players only ever played 0 mins, keep their latest row anyway
        never_played_ids = set(df["element"].unique()) - set(played["element"].unique())
        if never_played_ids:
            fallback = df[df["element"].isin(never_played_ids)].copy()
            played = pd.concat([played, fallback], ignore_index=True)
        df = played

    df = df.sort_values(["element", "round"])
    latest = df.groupby("element").last().reset_index()

    print(f"  ✓ {len(latest)} players with current state")
    return latest


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build Future Prediction Rows
# ═══════════════════════════════════════════════════════════════════════════════

def _get_upcoming_gws(n: int = N_FUTURE_GW) -> List[int]:
    """Return the next N unfinished gameweek numbers from events.csv.

    Caps at GW38 (standard Premier League season length).
    Returns fewer than N if the season is almost over.
    """
    MAX_GW = 38  # Premier League seasons always have exactly 38 rounds
    events_path = RAW_DATA_DIR / CURRENT_SEASON / "events.csv"
    if events_path.exists():
        try:
            ev = pd.read_csv(events_path)
            if "finished" in ev.columns and "id" in ev.columns:
                upcoming = (ev[~ev["finished"].astype(bool)]
                            .sort_values("id")["id"]
                            .tolist())
                upcoming = [int(g) for g in upcoming if int(g) <= MAX_GW]
                if upcoming:
                    return upcoming[:n]
        except Exception:
            pass

    # Fallback: use the GW after the latest GW in the current season data
    try:
        loader = FPLDataLoader()
        gw_df  = loader.load_gameweeks(CURRENT_SEASON)
        max_gw = int(gw_df["round"].max())
        return [gw for gw in range(max_gw + 1, max_gw + n + 1) if gw <= MAX_GW]
    except Exception:
        return list(range(1, n + 1))


def _build_fixture_lookup(
    fixtures_df: pd.DataFrame,
    upcoming_gws: List[int],
) -> Dict[int, Dict[int, List[Tuple[int, float, int]]]]:
    """Build a nested lookup:  team_id -> gw -> [(opponent_id, fdr, was_home)].

    For double-gameweek teams, the inner list has two entries.
    """
    lookup: Dict[int, Dict[int, List[Tuple[int, float, int]]]] = {}
    if fixtures_df.empty or "event" not in fixtures_df.columns:
        return lookup

    for _, row in fixtures_df.iterrows():
        gw = row.get("event")
        if pd.isna(gw):
            continue
        gw = int(gw)
        if gw not in upcoming_gws:
            continue

        team_h = row.get("team_h")
        team_a = row.get("team_a")
        fdr_h  = float(row.get("team_h_difficulty", 3) or 3)
        fdr_a  = float(row.get("team_a_difficulty", 3) or 3)

        if pd.notna(team_h):
            tid = int(team_h)
            opp = int(team_a) if pd.notna(team_a) else 0
            lookup.setdefault(tid, {}).setdefault(gw, []).append((opp, fdr_h, 1))

        if pd.notna(team_a):
            tid = int(team_a)
            opp = int(team_h) if pd.notna(team_h) else 0
            lookup.setdefault(tid, {}).setdefault(gw, []).append((opp, fdr_a, 0))

    return lookup


def build_prediction_rows(
    player_state: pd.DataFrame,
    upcoming_gws: List[int],
    feat_list: List[str],
) -> pd.DataFrame:
    """Create one feature row per player per upcoming GW.

    Strategy:
    - All rolling/form features are held at their current (latest) values.
    - ``was_home``, ``opponent_difficulty``, and ``opponent_strength`` are
      updated from the upcoming fixture for that specific GW.
    - Players with no fixture in a given GW (blank GW) get a row with
      ``has_fixture=False`` and ``predicted_points`` will be zeroed later.
    - Double-GW players get two rows for that GW (one per fixture).
    """
    # Load fixtures for the current season
    fixtures_df = pd.DataFrame()
    try:
        loader = FPLDataLoader()
        fixtures_df = loader.load_fixtures(CURRENT_SEASON)
    except FileNotFoundError:
        pass

    fixture_lookup = _build_fixture_lookup(fixtures_df, upcoming_gws)

    # Build team-strength map
    team_strength_map: Dict[int, float] = {}
    try:
        loader      = FPLDataLoader()
        teams_df    = loader.load_teams(CURRENT_SEASON)
        if "id" in teams_df.columns and "strength" in teams_df.columns:
            team_strength_map = teams_df.set_index("id")["strength"].to_dict()
    except FileNotFoundError:
        pass

    rows: List[dict] = []

    for _, player in player_state.iterrows():
        pid  = int(player["element"])
        try:
            team_id = int(player["team"]) if pd.notna(player.get("team")) else 0
        except (ValueError, TypeError):
            team_id = 0

        # Base feature dict from the player's latest state
        base: dict = {}
        for feat in feat_list:
            base[feat] = player[feat] if feat in player.index else np.nan

        # Metadata (not features, used for display)
        for col in ("name", "position_label", "team"):
            if col in player.index:
                base[col] = player[col]

        for gw in upcoming_gws:
            fixtures_in_gw = fixture_lookup.get(team_id, {}).get(gw, [])

            if not fixtures_in_gw:
                # Blank GW — no match for this team
                row = {**base, "element": pid, "gw": gw, "has_fixture": False}
                rows.append(row)
            else:
                for opp_id, fdr, is_home in fixtures_in_gw:
                    row = {**base, "element": pid, "gw": gw, "has_fixture": True}
                    row["was_home"]            = float(is_home)
                    row["opponent_difficulty"] = fdr
                    row["opponent_team"]       = opp_id
                    row["opponent_strength"]   = float(
                        team_strength_map.get(opp_id, 3)
                    )
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Generate Predictions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_predictions(
    model: xgb.XGBRegressor,
    imputer: SimpleImputer,
    pred_rows: pd.DataFrame,
    feat_list: List[str],
) -> pd.DataFrame:
    """Run the trained model on prediction rows; return enriched DataFrame."""
    _header("STEP 6: Generating Predictions")

    if pred_rows.empty:
        print("  ! No prediction rows — cannot predict.")
        return pd.DataFrame()

    # Build feature matrix — ensure all model features are present
    X = pd.DataFrame(index=pred_rows.index)
    for feat in feat_list:
        X[feat] = pred_rows[feat] if feat in pred_rows.columns else np.nan

    X_imp = pd.DataFrame(
        imputer.transform(X[feat_list]),
        columns=feat_list,
        index=pred_rows.index,
    )

    raw_preds = model.predict(X_imp).astype(float)
    raw_preds = np.clip(raw_preds, 0.0, None)   # no negative points

    # Zero out blank-GW rows
    no_fixture = ~pred_rows.get("has_fixture", pd.Series(True, index=pred_rows.index)).astype(bool)
    raw_preds[no_fixture.values] = 0.0

    out = pred_rows.copy()
    out["predicted_points"] = raw_preds

    n_players = out["element"].nunique()
    n_rows    = len(out[out["has_fixture"] != False])  # noqa: E712
    print(f"  Players predicted : {n_players}")
    print(f"  Fixture rows      : {n_rows:,}")
    print(f"  Predicted pts range: {raw_preds.min():.2f} – {raw_preds.max():.2f}")
    print(f"  Mean predicted pts : {raw_preds[raw_preds > 0].mean():.2f}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save Predictions
# ═══════════════════════════════════════════════════════════════════════════════

def save_predictions(pred_df: pd.DataFrame, upcoming_gws: List[int]) -> Path:
    """Save the full predictions table to CSV."""
    gw0, gwN = upcoming_gws[0], upcoming_gws[-1]
    fname    = f"predictions_gw{gw0}_to_gw{gwN}.csv"
    out_path = PREDICTION_DIR / fname

    keep = ["element", "name", "position_label", "price", "team", "gw",
            "predicted_points", "was_home", "opponent_difficulty", "has_fixture"]
    keep = [c for c in keep if c in pred_df.columns]

    pred_df[keep].sort_values(
        ["gw", "predicted_points"], ascending=[True, False]
    ).to_csv(out_path, index=False)

    print(f"\n  ✓ Predictions CSV → {out_path.relative_to(ROOT)}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Display Results (terminal + chart)
# ═══════════════════════════════════════════════════════════════════════════════

def _totals_df(pred_df: pd.DataFrame, upcoming_gws: List[int]) -> pd.DataFrame:
    """Aggregate predicted points per player across all upcoming GWs."""
    group_cols = ["element", "name", "position_label"]
    agg_cols: dict = {"predicted_points": "sum"}
    # price is aggregated (not grouped) so each player row keeps its price
    if "price" in pred_df.columns:
        agg_cols["price"] = "first"

    totals = (
        pred_df.groupby(group_cols).agg(agg_cols).reset_index()
        .rename(columns={"predicted_points": "total_predicted"})
    )
    totals["avg_per_gw"] = totals["total_predicted"] / len(upcoming_gws)
    return totals.sort_values("total_predicted", ascending=False).reset_index(drop=True)


def display_results(pred_df: pd.DataFrame, upcoming_gws: List[int]) -> None:
    """Print ranked tables to terminal and save a PNG summary chart."""
    if pred_df.empty:
        print("No predictions to display.")
        return

    gw0, gwN = upcoming_gws[0], upcoming_gws[-1]
    totals   = _totals_df(pred_df, upcoming_gws)

    has_price = "price" in totals.columns

    # ── 1. Top-20 by total predicted points ───────────────────────────────────
    print("\n" + "═" * 72)
    print(f"  FPL PREDICTIONS  ·  GW{gw0} → GW{gwN}  ·  ({len(upcoming_gws)} gameweeks)")
    print("═" * 72)

    price_hdr = "  Price" if has_price else ""
    print(f"\n📊 TOP 20 PLAYERS — TOTAL PREDICTED POINTS (Next {len(upcoming_gws)} GWs)")
    _hline(62)
    print(f"  {'#':>3}  {'Player':<26} {'Pos':>4}{price_hdr}  {'Total':>6}  {'Avg/GW':>7}")
    _hline(62)
    for i, r in totals.head(20).iterrows():
        pos   = str(r.get("position_label") or "?")
        price = f"  £{r['price']:>4.1f}" if has_price and pd.notna(r.get("price")) else ""
        print(f"  {i+1:>3}. {str(r['name']):<26} {pos:>4}{price}  "
              f"{r['total_predicted']:>6.1f}  {r['avg_per_gw']:>7.2f}")
    _hline(62)

    # ── 2. Best-5 per GW ──────────────────────────────────────────────────────
    print(f"\n📅 BEST 5 PLAYERS PER GAMEWEEK")
    for gw in upcoming_gws:
        gw_rows = pred_df[pred_df["gw"] == gw]
        gw_agg  = (gw_rows.groupby(["element", "name", "position_label"])
                   ["predicted_points"].sum()
                   .reset_index()
                   .sort_values("predicted_points", ascending=False)
                   .reset_index(drop=True))
        print(f"\n  ── GW{gw} ──")
        print(f"  {'#':>2}  {'Player':<26} {'Pos':>4}  {'Pts':>5}")
        print(f"  {'─'*42}")
        for j, r in gw_agg.head(5).iterrows():
            pos = str(r.get("position_label") or "?")
            print(f"  {j+1:>2}. {str(r['name']):<26} {pos:>4}  "
                  f"{r['predicted_points']:>5.1f}")

    # ── 3. Top-15 by average pts/GW ───────────────────────────────────────────
    print(f"\n⭐  TOP 15 BY AVERAGE PREDICTED POINTS / GW")
    _hline(60)
    print(f"  {'#':>3}  {'Player':<26} {'Pos':>4}{price_hdr}  {'Avg/GW':>7}  {'Total':>6}")
    _hline(60)
    for i, (_, r) in enumerate(totals.nlargest(15, "avg_per_gw").iterrows(), 1):
        pos   = str(r.get("position_label") or "?")
        price = f"  £{r['price']:>4.1f}" if has_price and pd.notna(r.get("price")) else ""
        print(f"  {i:>3}. {str(r['name']):<26} {pos:>4}{price}  "
              f"{r['avg_per_gw']:>7.2f}  {r['total_predicted']:>6.1f}")
    _hline(60)

    # ── 4. Top-5 per position ─────────────────────────────────────────────────
    print(f"\n🏟  TOP 5 BY POSITION — Total Predicted Points")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_rows = totals[totals["position_label"] == pos].head(5)
        if pos_rows.empty:
            continue
        print(f"\n  {pos}:")
        print(f"  {'#':>2}  {'Player':<26}  {'Total':>6}  {'Avg/GW':>7}")
        print(f"  {'─'*45}")
        for j, (_, r) in enumerate(pos_rows.iterrows(), 1):
            print(f"  {j:>2}. {str(r['name']):<26}  "
                  f"{r['total_predicted']:>6.1f}  {r['avg_per_gw']:>7.2f}")

    # ── 5. Save visual chart ──────────────────────────────────────────────────
    _save_chart(pred_df, totals, upcoming_gws)

    print("\n" + "═" * 72)
    print("  ✅  Done!  Check outputs/predictions/ for CSV and PNG.")
    print("═" * 72 + "\n")


def _safe_name(name) -> str:
    """Return a clean string for a player name, truncated if too long."""
    s = str(name) if pd.notna(name) else "Unknown"
    return s[:22] + "…" if len(s) > 22 else s


def _save_chart(
    pred_df: pd.DataFrame,
    totals: pd.DataFrame,
    upcoming_gws: List[int],
) -> None:
    """Create and save a 4-panel summary chart with real player names.

    Panels:
      A (top, full-width): horizontal bar — top-20 total predicted points
      B (bottom-left):     heatmap — top-15 players × each upcoming GW
      C (bottom-middle):   best-5 players per GW as a visual table
      D (bottom-right):    position distribution pie (top-30 players)
    """
    gw0, gwN = upcoming_gws[0], upcoming_gws[-1]
    n_gws     = len(upcoming_gws)
    has_price = "price" in totals.columns

    try:
        # Wider figure and more height for the bar chart so names are legible
        fig = plt.figure(figsize=(26, 20))
        fig.patch.set_facecolor("#1e1e2e")
        title_color = "#cdd6f4"

        fig.suptitle(
            f"FPL Predictions  ·  GW{gw0}–GW{gwN}  "
            f"(generated {datetime.now().strftime('%Y-%m-%d %H:%M')})",
            fontsize=16, fontweight="bold", color=title_color, y=0.995,
        )

        gs = gridspec.GridSpec(
            2, 3, figure=fig,
            height_ratios=[1.0, 0.9],
            hspace=0.42, wspace=0.38,
        )

        panel_bg  = "#313244"
        text_col  = "#cdd6f4"
        grid_col  = "#45475a"

        def _style_ax(ax):
            ax.set_facecolor(panel_bg)
            ax.tick_params(colors=text_col, labelsize=9)
            ax.xaxis.label.set_color(text_col)
            ax.yaxis.label.set_color(text_col)
            ax.title.set_color(title_color)
            for spine in ax.spines.values():
                spine.set_color(grid_col)

        # ── Panel A: horizontal bar — top-20 total predicted points ───────────
        ax_bar = fig.add_subplot(gs[0, :])  # spans all 3 columns
        _style_ax(ax_bar)

        top20      = totals.head(20).copy()
        top20_rev  = top20.iloc[::-1].reset_index(drop=True)  # bottom→top order

        bar_colors = [POS_COLORS.get(str(p), "#adb5bd") for p in top20_rev["position_label"]]
        y_pos      = list(range(len(top20_rev)))

        bars = ax_bar.barh(
            y_pos,
            top20_rev["total_predicted"].values,
            color=bar_colors,
            alpha=0.92,
            edgecolor=panel_bg,
            linewidth=0.8,
            height=0.72,
        )
        # Y-axis labels: rank + name (+ price if available)
        y_labels = []
        for _, r in top20_rev.iterrows():
            nm = _safe_name(r["name"])
            if has_price and pd.notna(r.get("price")):
                y_labels.append(f"{nm}  £{r['price']:.1f}m")
            else:
                y_labels.append(nm)

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(y_labels, fontsize=10, color=text_col)
        ax_bar.set_xlabel("Total Predicted Points", fontsize=11, color=text_col)
        ax_bar.set_title(
            f"Top 20 Players — Total Predicted Points  (GW{gw0}–GW{gwN})",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax_bar.grid(axis="x", alpha=0.2, linestyle="--", color=grid_col)
        ax_bar.spines[["top", "right"]].set_visible(False)

        # Value labels on each bar
        max_val = top20_rev["total_predicted"].max()
        for bar in bars:
            w = bar.get_width()
            ax_bar.text(
                w + max_val * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}",
                va="center", ha="left", fontsize=9, color=text_col,
                fontweight="bold",
            )

        legend_patches = [Patch(facecolor=v, label=k, alpha=0.9)
                          for k, v in POS_COLORS.items()]
        ax_bar.legend(
            handles=legend_patches, loc="lower right",
            fontsize=9, framealpha=0.5,
            facecolor=panel_bg, labelcolor=text_col,
        )

        # ── Panel B: heatmap — top-15 players × GW ────────────────────────────
        ax_heat = fig.add_subplot(gs[1, 0])
        _style_ax(ax_heat)

        n_heat    = min(15, len(totals))
        top_slice = totals.head(n_heat)
        top_ids   = top_slice["element"].tolist()
        top_names = [_safe_name(n) for n in top_slice["name"].tolist()]

        heat_data = np.zeros((n_heat, n_gws))
        for i, eid in enumerate(top_ids):
            for j, gw in enumerate(upcoming_gws):
                pts = pred_df[(pred_df["element"] == eid) &
                              (pred_df["gw"] == gw)]["predicted_points"].sum()
                heat_data[i, j] = pts

        im = ax_heat.imshow(
            heat_data, aspect="auto", cmap="YlOrRd",
            vmin=0, vmax=max(float(heat_data.max()), 1),
        )
        ax_heat.set_xticks(range(n_gws))
        ax_heat.set_xticklabels([f"GW{g}" for g in upcoming_gws],
                                 fontsize=9, color=text_col)
        ax_heat.set_yticks(range(n_heat))
        ax_heat.set_yticklabels(top_names, fontsize=9, color=text_col)
        ax_heat.set_title("GW-by-GW Heatmap (Top 15)",
                           fontsize=11, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax_heat, shrink=0.85)
        cbar.set_label("Pred. Points", color=text_col, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=text_col)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_col)

        vmax = float(heat_data.max()) if heat_data.max() > 0 else 1
        for i in range(n_heat):
            for j in range(n_gws):
                val = heat_data[i, j]
                fc  = "white" if val > vmax * 0.62 else "#1e1e2e"
                ax_heat.text(j, i, f"{val:.1f}", ha="center", va="center",
                             fontsize=8, color=fc, fontweight="bold")

        # ── Panel C: best players per GW (visual table) ────────────────────────
        ax_tbl = fig.add_subplot(gs[1, 1])
        ax_tbl.set_facecolor(panel_bg)
        ax_tbl.axis("off")
        ax_tbl.set_title(f"Best Players per Gameweek (Top 3)",
                          fontsize=11, fontweight="bold", color=title_color)

        top_n_per_gw = 3
        row_h  = 1.0 / (n_gws * (top_n_per_gw + 1.5) + 0.5)
        y_cur  = 0.97

        for gw in upcoming_gws:
            gw_rows = pred_df[pred_df["gw"] == gw]
            gw_agg  = (gw_rows.groupby(["element", "name", "position_label"])
                       ["predicted_points"].sum()
                       .reset_index()
                       .sort_values("predicted_points", ascending=False)
                       .head(top_n_per_gw))

            # GW header
            ax_tbl.text(0.03, y_cur, f"GW{gw}",
                        transform=ax_tbl.transAxes,
                        fontsize=10, fontweight="bold", color="#89b4fa",
                        va="top")
            y_cur -= row_h * 0.9

            for _, r in gw_agg.iterrows():
                nm  = _safe_name(r["name"])
                pos = str(r.get("position_label") or "?")
                pts = r["predicted_points"]
                pc  = POS_COLORS.get(pos, "#adb5bd")

                ax_tbl.text(0.03, y_cur, f"  {nm}",
                            transform=ax_tbl.transAxes,
                            fontsize=9, color=text_col, va="top")
                ax_tbl.text(0.72, y_cur, pos,
                            transform=ax_tbl.transAxes,
                            fontsize=8, color=pc, va="top",
                            fontweight="bold")
                ax_tbl.text(0.87, y_cur, f"{pts:.1f}",
                            transform=ax_tbl.transAxes,
                            fontsize=9, color="#a6e3a1", va="top",
                            fontweight="bold")
                y_cur -= row_h * 0.85

            y_cur -= row_h * 0.4  # spacer between GWs

        # ── Panel D: position pie — top-30 ────────────────────────────────────
        ax_pie = fig.add_subplot(gs[1, 2])
        ax_pie.set_facecolor(panel_bg)
        ax_pie.set_title("Position Mix — Top 30 Players",
                          fontsize=11, fontweight="bold", color=title_color)

        pos_counts = totals.head(30)["position_label"].value_counts()
        pie_colors = [POS_COLORS.get(str(p), "#adb5bd") for p in pos_counts.index]

        _, texts, autotexts = ax_pie.pie(
            pos_counts.values,
            labels=pos_counts.index,
            colors=pie_colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 11, "color": text_col},
            pctdistance=0.76,
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
            at.set_color("#1e1e2e")

        plt.tight_layout(rect=[0, 0, 1, 0.995])

        chart_path = PREDICTION_DIR / f"predictions_gw{gw0}_to_gw{gwN}.png"
        fig.savefig(
            chart_path, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        print(f"\n  ✓ Chart saved → {chart_path.relative_to(ROOT)}")

    except Exception as exc:
        print(f"\n  ⚠ Could not create chart: {exc}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _header(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def _hline(width: int = 60) -> None:
    print("  " + "─" * width)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    warnings.filterwarnings("ignore")

    print("\n" + "#" * 68)
    print("#  FPL NEXT-5-GAMEWEEK PREDICTOR")
    print(f"#  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("#" * 68)

    # ── 1. Download ────────────────────────────────────────────────────────────
    download_all_data()

    # ── 2. Build training data ─────────────────────────────────────────────────
    X_train, y_train, feat_list = build_training_data(tier=3)

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    model, imputer = train_model(X_train, y_train, feat_list)

    # ── 4. Extract current player state ───────────────────────────────────────
    _header("STEP 4: Extracting Current Player State")
    player_state = get_current_player_state(feat_list)

    if player_state.empty:
        print("  ✗ No current-season player state found.  "
              "Ensure 2025-26 data was downloaded successfully.")
        sys.exit(1)

    # ── 5. Find upcoming GWs & build prediction rows ──────────────────────────
    _header("STEP 5: Building Future Prediction Rows")
    upcoming_gws = _get_upcoming_gws(N_FUTURE_GW)
    print(f"  Upcoming GWs: {upcoming_gws}")

    pred_rows = build_prediction_rows(player_state, upcoming_gws, feat_list)
    if pred_rows.empty:
        print("  ✗ No prediction rows could be built.  Check fixture data.")
        sys.exit(1)
    print(f"  Prediction rows: {len(pred_rows):,}  "
          f"(includes DGW rows where applicable)")

    # ── 6. Generate predictions ───────────────────────────────────────────────
    pred_df = generate_predictions(model, imputer, pred_rows, feat_list)

    # ── Always resolve real player names & positions from bootstrap ──────────
    # Runs unconditionally so names are never blank or numeric element IDs even
    # if the enrichment inside _load_and_engineer_season() partially failed.
    _header("STEP 6b: Resolving Player Names from Bootstrap Data")
    bootstrap_path = RAW_DATA_DIR / CURRENT_SEASON / "bootstrap_static.json"
    if bootstrap_path.exists():
        try:
            with open(bootstrap_path, encoding="utf-8") as fh:
                bdata = json.load(fh)
            elements = bdata.get("elements", [])
            name_map  = {e["id"]: e["web_name"] for e in elements}
            pos_map   = {e["id"]: POSITION_MAP.get(e.get("element_type"), "?")
                         for e in elements}
            price_map = {e["id"]: round(e.get("now_cost", 0) / 10, 1)
                         for e in elements}

            # Always apply name map; overwrite NaN / numeric strings
            def _resolve_name(row):
                existing = str(row.get("name", "")).strip()
                if not existing or existing.lower() == "nan" or existing.isdigit():
                    return name_map.get(int(row["element"]), str(int(row["element"])))
                return existing

            pred_df["name"] = pred_df.apply(_resolve_name, axis=1)

            if "position_label" not in pred_df.columns or pred_df["position_label"].isna().any():
                pred_df["position_label"] = pred_df["element"].map(pos_map).fillna(
                    pred_df.get("position_label", pd.Series("?", index=pred_df.index))
                )

            # Add price column for practical FPL use
            if "price" not in pred_df.columns:
                pred_df["price"] = pred_df["element"].map(price_map)

            n_named = pred_df["name"].notna().sum()
            print(f"  ✓ Resolved names for {n_named}/{len(pred_df)} prediction rows")
        except Exception as exc:
            print(f"  ⚠ Name resolution from bootstrap failed: {exc}")
    else:
        print("  ⚠ bootstrap_static.json not found — names may show as element IDs")
        # Fallback: try players_raw.csv
        players_path = RAW_DATA_DIR / CURRENT_SEASON / "players_raw.csv"
        if players_path.exists():
            try:
                praw = pd.read_csv(players_path)
                if {"id", "web_name"}.issubset(praw.columns):
                    nm = praw.set_index("id")["web_name"].to_dict()
                    pred_df["name"] = pred_df["element"].map(nm).fillna(
                        pred_df["element"].astype(str)
                    )
            except Exception:
                pass
        if "name" not in pred_df.columns:
            pred_df["name"] = pred_df["element"].astype(str)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    _header("STEP 7: Saving Results")
    save_predictions(pred_df, upcoming_gws)

    # ── 8. Display ────────────────────────────────────────────────────────────
    display_results(pred_df, upcoming_gws)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n\n✗  Fatal error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
