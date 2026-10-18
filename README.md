# EcoTwin AI + PI-AGMT

Streamlit digital-twin dashboard for campus carbon emissions. The prediction engine is
**PI-AGMT: Physics-Informed Adaptive Graph-Mamba Transformer** (the old Random Forest / Gradient
Boosting / Ridge comparison has been replaced).

```
final prediction = physics baseline (activity x standard emission factor)
                 + residual learned by [feature embedding -> Mamba-style state-space block over the last 7 days
                                        -> attention across the 14 buildings (adaptive graph)]
```

## Quick start

| Goal | Windows | macOS / Linux |
|---|---|---|
| Open the dashboard | `run.bat` | `./run.sh` |
| Train + validate on the Sona files, write results | `train.bat` | `./train.sh` |
| External test on Dataset B | `run_external.bat` | `./run_external.sh` |
| Robustness test (18 damaged copies of Dataset B) | `run_robustness.bat` | `./run_robustness.sh` |
| Run the tests | `test.bat` | `./test.sh` |

All scripts create `.venv`, install `requirements.txt` and run. First install downloads PyTorch (large).
Python 3.10+ is required. On a Linux machine without a GPU you can install a smaller CPU build first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Using the dashboard

1. **Dataset page** > upload `data/Sona_Campus_Combined_for_App.csv` (train + validation joined, ready to use).
2. Check the guessed settings:
   * Date column: `Date`
   * Target: `Next_Day_CO2_kg`
   * Building / entity column: `Building_ID`
   * Validation starts on: `2026-03-13` (leave empty for an automatic 80/20 split by date)
3. **Build workspace.** Training takes roughly 1 to 3 minutes on a laptop CPU.
4. Every page (Overview, Forecast, Anomalies, Risk, Carbon Graph, What-if, Roadmap, Reports) now runs on PI-AGMT.
   The Carbon Graph page also shows the building-to-building attention learned by the graph layer.

The bundled demo (`sample_data/`, one computer lab, no building column) still works: the graph layer then has a single
node and the model is physics + temporal learning.

## Train / validate from the command line

`./train.sh` (or `train.bat`) reads `data/Sona_Campus_Training_Dataset.csv` and `data/Sona_Campus_Validation_Dataset.csv`
and writes to `outputs/`:

| File | Content |
|---|---|
| `validation_results.csv` | Persistence and physics-only references, PI-AGMT, and the ablation runs |
| `validation_predictions.csv` | Actual vs predicted for every validation record |
| `feature_importance.csv` | Top drivers (permutation importance on held-out data) |
| `building_attention.csv` | 14 x 14 attention matrix from the graph layer |
| `error_by_building.csv` | Mean absolute error per building |
| `external_results.csv` / `external_predictions.csv` | Only with `--external` (see below) |
| `external_robustness.csv` | From `python scripts/robustness_test.py` |

Options: `--epochs 30`, `--no-ablation` (4x faster), `--no-tomorrow`, `--external data/DatasetB.csv`.

**Dataset B (final external test):** `data/DatasetB.csv` is a *synthetic* stress-test set made by
`scripts/make_dataset_b.py` (181 days from 2027-01-01, same 14 buildings, plus 3 columns the training file does not have: `EV_Charging_kWh`, `Campus_Zone`, `Data_Source`, hotter weather, hybrid attendance,
car-heavy commuting, 40% bigger solar plants, a generator-outage wave, per-building consumption shifts). It is derived
from the Sona data, so it tests robustness to change, not independent real-world accuracy. Use real data from another
period or campus if you have it. Run: `python scripts/train_and_evaluate.py --no-ablation --external data/DatasetB.csv`.
Pass your own file with `--external`. It is only scored, never used for fitting or
for filling gaps. It needs the same columns and the same 14 `Building_ID` values.

**The external test does not crash on a damaged Dataset B.** `python scripts/robustness_test.py` trains once and scores 18
damaged copies of Dataset B (columns removed, buildings missing, text in numeric columns, no target, shuffled rows...).
Result: 0 crashes, written to `outputs/external_robustness.csv`. Repairs always use *training* statistics and are printed
as `note:` lines. Missing columns are filled with training medians, so the score drops when important columns are gone
(re-run `scripts/robustness_test.py` for the current figures). The only things it cannot work without are the `Date` and
`Building_ID` columns; it then stops with a clear message instead of a traceback.

## What goes into the model

* **Target:** `Next_Day_CO2_kg`. Row = one building on one date.
* **Inputs:** the columns listed in your feature plan, plus *tomorrow's calendar flags* (`Tomorrow_Holiday`,
  `Tomorrow_Event_Day`), which ship in the new data files. The calendar is known in advance, so this is not leakage.
* **Never inputs:** every CO2 column, per-person ratios, `Recycling_Rate`, `Sustainability_Score`, all `Next_Day_*`
  columns, and metadata (`Building_Name`, `Department`, `Month_Name`).
* **Missing values:** the command-line script fills gaps with *training-set* medians. The dashboard's cleaning step
  fills with column medians. The new data files have no missing values.
* **Physics baseline:** `0.71 x electricity kWh + 2.68 x generator litres + 2.68 x bus diesel litres +
  0.00035 x water litres + 0.5 x waste kg + 1.2 x vehicles` (the standard factors of the Daily Emission Calculator).
  On the Sona data this reproduces today's CO2 exactly (max error 0.02 kg; the new files have no `Total_CO2_kg` column, today's value is the previous row's `Next_Day_CO2_kg`). For tomorrow's target it is today's emission,
  and the network learns the change.

## Honest notes for your report

* The ablation table is the evidence for each component. If a part does not improve validation scores,
  say so; do not claim it does.
* The Forecast page projects the *total* series with a separate lag-feature model, because multi-day forecasting
  needs future activity values that PI-AGMT does not predict.
* What-if / Roadmap score a single row: the row replaces one building on the latest day, earlier days come
  from the stored history.
* Claim "a hybrid physics-informed graph-temporal architecture for campus carbon forecasting", not "a new algorithm".
* The "Mamba" block is a compact selective state-space layer in plain PyTorch (CPU friendly), not the official
  CUDA `mamba-ssm` package.

## Project layout

```
app.py                 entry point
core/pi_agmt.py        PI-AGMT model, training, importance, external scoring
core/modeling.py       train_models / predict_rows (what-if) + the Forecast-page model
core/data.py           upload, schema + building-column detection, cleaning, quality report
core/analytics.py      KPIs, anomalies, risk, budget, what-if, roadmap
core/emission_calc.py  standard emission factors
views/                 one module per page
scripts/train_and_evaluate.py
data/                  Sona train / validation / combined-for-app CSVs
tests/                 test_pipeline.py, test_piagmt.py
```

## Dashboard label

The Command Center's first metric card is labelled `Total_CO2_kg` (display text only). The target column, model and
other pages still use `Next_Day_CO2_kg`.
