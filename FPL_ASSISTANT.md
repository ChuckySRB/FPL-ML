# FPL Assistant — Architecture and Workflow

## Scope of This Phase

This phase implements the prediction core and the Streamlit decision workspace.
The production service supports a selected season/GW, two independently trained
forecasts, ranked recommendations, captain candidates, a weekly report, and
reusable chat prompts.

## Two Separate Forecasts

### Current GW

`one_fixture_model.joblib` predicts points for one scheduled player-fixture.
Predictions are summed only when a player has two fixtures in the selected GW.
Blank players receive zero.

### Direct Five-GW Average

`five_gw_average_model.joblib` predicts the directly observed average return
from GW `n` through GW `n+4`. Historical DGW fixture points are first summed
inside their GW; blank GWs contribute zero. The target is not constructed from
five predictions of the one-fixture model.

The long-horizon model sees known schedule descriptors:

- fixture, blank-GW, and double-GW counts;
- mean/min/max FDR;
- home-fixture rate;
- mean opponent strength.

This prevents a strong current form value from being copied blindly across five
fixtures and lets the model learn regression toward a five-week average.

## Early-Season Routing

GW1 does not use the mature in-season models. Two dedicated preseason models
are trained on five historical season transitions using only the preceding
season's final rolling form, observed GW1 price/ownership/club, and the known
new-season schedule. The newest transition, 2025/26, is held out during
validation. GW2–GW5 blend preseason estimates with current-season estimates at
20%, 40%, 60%, and 80% current-history weight. From GW6 onward, the fully
validated in-season models are used alone.

## Defensive-Contribution Scoring

The official current rule is:

- DEF: two points for at least 10 CBIT actions;
- MID/FWD: two points for at least 12 CBIRT actions;
- maximum two points per match.

The 2025/26 `total_points` already includes this bonus and is never incremented
again. Older local seasons do not contain the required match components. Their
training target receives an expected bonus, `2 × P(qualifies)`, from a separate
reconstruction model trained on exact 2025/26 components. Provenance is stored
as `dc_adjustment_source`; the estimate must not be described as observed data.

The DC estimator's chronological GW31–38 check currently gives ROC AUC 0.981,
average precision 0.757, and Brier score 0.0217.

## Train the Production Models

```powershell
python scripts/train_fpl_assistant.py
```

The final artifacts use all locally available rows from 2020/21–2025/26:

- one-fixture model: 144,399 rows;
- direct five-GW model: 118,802 rows.
- preseason one-fixture model: 3,093 rows across five transitions;
- preseason direct five-GW model: 3,091 rows across five transitions.

The 2025/26 diagnostic holdout produced:

| Target | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| One fixture | 0.969 | 1.918 | 0.328 |
| Direct five-GW average | 0.712 | 1.142 | 0.538 |

The dedicated preseason models were evaluated only on the unseen 2025/26
transition:

| Target | MAE | RMSE | R² | Spearman |
| --- | ---: | ---: | ---: | ---: |
| GW1 fixture | 1.355 | 2.275 | 0.194 | 0.560 |
| Direct GW1–GW5 average | 0.960 | 1.409 | 0.366 | 0.707 |

For the same GW1 fixture rows, official FPL `xP` gave MAE 1.501, RMSE 2.322,
R² 0.160, and Spearman 0.468. It is retained as a sanity reference rather than
silently blended into the model target.

The two metric rows are not directly comparable because the averaged target is
statistically smoother.

## Generate a Weekly Package

```powershell
python scripts/generate_gw_report.py --season 2026-27 --gw 1
```

The command refreshes `players_raw.csv`, `teams.csv`, and `fixtures.csv` from
the official FPL API before predicting. Add `--offline` to reuse the local
cache when network access is unavailable.

The command creates `outputs/assistant/<season>/gw<n>/` containing:

- `report_gw<n>.md`;
- full player and fixture prediction CSVs;
- structured report JSON;
- `chat_prompt_gw<n>.md`.

Top lists exclude injured, unavailable, and suspended players. The report and
JSON preserve excluded players separately and flag every current-GW prediction
that differs from official `ep_next` by more than two points.

The committed prompt sources are:

- `prompts/fpl_assistant_system_prompt.md`;
- `prompts/weekly_analysis_template.md`.

Attach the generated report to a chat, paste the system role once, then fill in
the weekly template with the current squad, bank, free transfers, chips, and
dated external-source summaries.

## Interface

Run `streamlit run app.py`. The default Weekly Assistant workspace provides:

1. season/GW selection, package freshness, model route, and validation MAE;
2. an explicit **Run predictions** action with online refresh or offline cache;
3. separate current-GW and direct next-five-GW top-25 rankings;
4. captain cards, price/value plots, player comparison, and FDR heatmaps;
5. availability, news, and large model-versus-`ep_next` risk flags;
6. current-squad, bank, free-transfer, chip, and source-note input;
7. report, completed strategy prompt, system role, CSV, and JSON downloads;
8. an optional legal 15-player wildcard optimizer for either horizon.

The application never calls the official API on an ordinary rerun. Network
refresh and prediction occur only after the user presses **Run predictions**.
The older academic evaluation dashboard remains under Model Research.

GW1 is supported before a `gws/` history exists. Returning players are matched
to the preceding season by stable FPL `code`; unseen players follow the
preseason imputation pattern learned from historical new arrivals. Current
ownership is reconstructed on the same log-selected scale used in training.
When preseason API team strengths are empty, fixture difficulty provides the
fallback club rating. Refresh completed player histories for GW2 and later so
the early-season blend can progressively use current evidence.
