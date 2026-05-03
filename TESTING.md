# Application testing guide

This document is the **single place** for how to verify the **AI for Predicting and Preventing Road Accidents** MVP: environment, data pipeline, training, prediction (CLI), Streamlit UI, automated tests, error cases, and troubleshooting.

**Important:** Run all commands from the **project root** (the folder that contains `main.py`, `train.py`, `requirements.txt`). On Windows, use **PowerShell** unless noted.

---

## 1. What you are testing

| Layer | What “good” means |
|-------|-------------------|
| **Environment** | Python venv works; `pandas` and `scikit-learn` import. |
| **Data** | Raw CSV loads; cleaned CSV exists; features preprocess without errors. |
| **Model** | Training completes; `models/accident_risk.joblib` exists; reload works. |
| **Prediction** | JSON has `risk_level` (Low/Medium/High), `severity_class` (0–2), non-empty `recommendation`. |
| **UI** | Streamlit loads; Predict shows a result or a clear “train first” error. |
| **Regression** | Pytest and smoke script pass after changes. |

This app is **local-only** and **not** safety-certified. Testing checks that the **software behaves as designed**, not that predictions are legally or medically valid.

---

## 2. Prerequisites

1. **Python 3.10+** installed (`python --version`).
2. **Project folder** available (path may contain spaces — keep quotes in `cd`).
3. **Network** only needed for `pip install` (not for running predictions after install).

---

## 3. First-time setup (do once per machine / clone)

Open PowerShell and run (adjust the path if your folder is elsewhere):

```powershell
cd "C:\Users\BHUMI\OneDrive\Desktop\Ai for prediction and prevention of road accidents"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Verify install:** you should see no errors at the end of `pip install`.

---

## 4. Activate the venv (every new terminal)

```powershell
cd "C:\Users\BHUMI\OneDrive\Desktop\Ai for prediction and prevention of road accidents"
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation, run once in an elevated PowerShell (or for current user):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 5. Recommended test order (“full pass”)

Run these **in order** the first time, or after you change code/data/features.

| Step | Command | What to check |
|------|---------|----------------|
| 1 | `python scripts/check_env.py` | Prints success for imports. |
| 2 | `python scripts/run_phase2_eda.py` | Prints EDA; creates `data/processed/clean_accidents.csv`. |
| 3 | `python scripts/run_phase3_features.py` | Train/test shapes and feature names print without error. |
| 4 | `python main.py train` | Accuracy + report print; **Saved pipeline** message; file `models/accident_risk.joblib` exists. |
| 5 | `python scripts/verify_model.py` | **Loaded** + steps `preprocess`, `classifier`. |
| 6 | `python main.py predict --hour 14 --day-of-week 2 --weather Clear --area-type Urban --traffic-level Medium` | JSON with three keys (see §8). |
| 7 | `python scripts/run_phase7_smoke.py` | All lines **PASS**; checklist prints at end. |
| 8 | `pytest tests/test_mvp_phase7.py -v` | **7 passed** (or skips if model missing — then fix step 4). |
| 9 | `python main.py ui` | Browser opens; see §9. |

---

## 6. Environment & dependency tests

### 6.1 Import check

```powershell
python scripts/check_env.py
```

**Expected:** A line indicating `pandas` and `scikit-learn` import successfully.

**Failure:** `ModuleNotFoundError` → run `pip install -r requirements.txt` inside the activated venv.

### 6.2 One-liner (optional)

```powershell
python -c "import pandas, sklearn; print('ok')"
```

**Expected:** `ok`

---

## 7. Data pipeline tests

### 7.1 Phase 2 — load, EDA, clean CSV

```powershell
python scripts/run_phase2_eda.py
```

**Expected:**

- Sections **Raw** and **After cleaning** with `shape`, `head`, `dtypes`, missing counts.
- Message **Saved:** pointing to `data/processed/clean_accidents.csv`.

**Optional flags:**

```powershell
python scripts/run_phase2_eda.py --input data/raw/sample_accidents.csv --output data/processed/clean_accidents.csv
python scripts/run_phase2_eda.py --no-save
```

### 7.2 Phase 3 — features and train/test split

```powershell
python scripts/run_phase3_features.py
```

**Expected:**

- `X_train` / `X_test` shapes consistent with `test_size=0.25`.
- Transformed shapes match on train and test (same number of engineered columns).
- **First 15 feature names** listed (e.g. `num__hour`, `cat__weather_Rain`, …).

---

## 8. Model training tests

### 8.1 Train via unified CLI (preferred)

```powershell
python main.py train
```

**Expected:**

- `Classifier: lr`
- `Test accuracy:` between 0 and 1 (value depends on data split).
- **Classification report** table for Low / Medium / High.
- `Saved pipeline:` path to `models/accident_risk.joblib`.
- `Smoke test:` one line with a predicted class.

### 8.2 Train standalone script (equivalent)

```powershell
python train.py
```

### 8.3 Optional: decision tree artifact

```powershell
python main.py train --model dt --output models/accident_tree.joblib
```

**Expected:** File created at `models/accident_tree.joblib`. Default predict path remains `models/accident_risk.joblib` unless you pass `--model-path` when predicting.

### 8.4 Verify saved artifact

```powershell
python scripts/verify_model.py
python scripts/verify_model.py --model-path models/accident_risk.joblib
```

**Expected:** `Loaded:` … and `Steps: ['preprocess', 'classifier']`.

---

## 9. Prediction (CLI) tests

Predict requires **five** inputs aligned with training:

| Argument | Meaning | Sample values |
|----------|---------|----------------|
| `--hour` | 0–23 | `10`, `22` |
| `--day-of-week` | 0 = Sunday … 6 = Saturday | `2` |
| `--weather` | Category string | `Clear`, `Rain`, `Fog`, `Snow` |
| `--area-type` | `Urban`, `Highway`, `Rural` | `Urban` |
| `--traffic-level` | `Low`, `Medium`, `High` | `Medium` |

### 9.1 Via `main.py`

```powershell
python main.py predict --hour 22 --day-of-week 5 --weather Rain --area-type Highway --traffic-level High
```

### 9.2 Via `predict.py` (same flags)

```powershell
python predict.py --hour 22 --day-of-week 5 --weather Rain --area-type Highway --traffic-level High
```

**Expected JSON (structure):**

```json
{
  "risk_level": "Low | Medium | High",
  "severity_class": 0,
  "recommendation": "non-empty string"
}
```

**Manual checks:**

- `risk_level` is exactly one of: `Low`, `Medium`, `High`.
- `severity_class` is `0`, `1`, or `2`.
- `recommendation` is not empty and reads as safety advice.

### 9.3 Custom model path

If you trained to another file:

```powershell
python predict.py --model-path models/accident_tree.joblib --hour 10 --day-of-week 1 --weather Clear --area-type Urban --traffic-level Low
```

**Expected:** Same JSON shape; values may differ from the default model.

### 9.4 Help / argument errors

```powershell
python main.py predict --help
```

Omit a required flag (e.g. omit `--hour`):

**Expected:** `argparse` usage error — confirms CLI wiring works.

---

## 10. Streamlit UI tests

### 10.1 Launch

**Option A**

```powershell
python main.py ui
```

**Option B**

```powershell
streamlit run app_streamlit.py
```

**Expected:** Terminal shows a **Local URL** (usually `http://localhost:8501`). Browser opens (or open the URL manually).

### 10.2 Manual UI checklist

Do these in the browser:

1. **No model file:** Temporarily rename `models/accident_risk.joblib` → refresh app → you should see an error telling you to run `python main.py train`. Rename back and refresh.
2. **Default state:** Click **Predict risk** — a **risk level** appears (green / amber / red style) plus **Suggestions** text.
3. **Change inputs:** Set hour to **22**, weather **Rain**, area **Highway**, traffic **High** → Predict again → you should still get a valid `risk_level` and non-empty text.
4. **Stop server:** In the terminal where Streamlit runs, press **Ctrl+C**.

---

## 11. Automated tests

### 11.1 Install pytest (if not already)

```powershell
pip install -r requirements.txt
```

### 11.2 MVP / Phase 7 tests

```powershell
pytest tests/test_mvp_phase7.py -v
```

**Expected:** `7 passed` when `models/accident_risk.joblib` exists.

**If tests are skipped:** Message will mention missing model — run `python main.py train` first.

### 11.3 Smoke script (no pytest)

```powershell
python scripts/run_phase7_smoke.py
```

**Expected:**

- `Column order OK`
- Five lines starting with `PASS [...]`
- `Done: all scenarios passed.`
- Pre-demo checklist printed at the end.

---

## 12. Testing after you change something

| You changed… | Re-run at minimum |
|--------------|-------------------|
| `data/raw/*.csv` or cleaning in `utils.py` | `run_phase2_eda.py` → `main.py train` → `pytest …` |
| Features / `preprocess.py` | `run_phase3_features.py` → `main.py train` → predict + pytest |
| `train.py` model type / hyperparameters | `main.py train` → predict + pytest |
| `predict.py` recommendations only | `run_phase7_smoke.py` (no retrain unless you want) |
| Column names in `FEATURE_COLUMNS` | **Retrain** + run **full pass** (§5) |

**Rule of thumb:** If **training columns** or **preprocessing** change, always **retrain** and run **pytest** + **smoke script**.

---

## 13. Negative & error tests

| Test | Command / action | Expected behavior |
|------|------------------|-------------------|
| Missing model | Delete or rename `models/accident_risk.joblib`, then `python main.py predict ...` | JSON `error` on stderr or non-zero exit; message says to train first. |
| Missing cleaned data | Delete `data/processed/clean_accidents.csv`, run `main.py train` | Clear `FileNotFoundError` telling you to run Phase 2 EDA. |
| Wrong working directory | Run `python main.py train` from **outside** project root | May fail imports or paths — always `cd` to project root first. |

---

## 14. Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `python` not found | Python not on PATH | Install Python; use **py** launcher: `py -3.12 -m venv .venv` |
| `No module named 'sklearn'` | Wrong interpreter / venv off | Activate `.venv`; reinstall requirements. |
| `Missing model` | Never trained or wrong path | `python main.py train` |
| `Missing … clean_accidents.csv` | Phase 2 not run | `python scripts/run_phase2_eda.py` |
| Streamlit won’t start | Port in use | Close other Streamlit tabs; or `streamlit run app_streamlit.py --server.port 8502` |
| Pytest **skipped** | No `.joblib` | Train first (§8). |
| Predict crashes with shape / feature error | Column mismatch | Run `pytest tests/test_mvp_phase7.py::test_inputs_dataframe_matches_training_columns -v`; align `FEATURE_COLUMNS` in `utils.py` with training CSV columns. |

---

## 15. Quick command reference

```powershell
# Setup
cd "<PROJECT_ROOT>"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Health
python scripts/check_env.py

# Data
python scripts/run_phase2_eda.py
python scripts/run_phase3_features.py

# Train + verify
python main.py train
python scripts/verify_model.py

# Predict
python main.py predict --hour 14 --day-of-week 2 --weather Clear --area-type Urban --traffic-level Medium

# Tests
python scripts/run_phase7_smoke.py
pytest tests/test_mvp_phase7.py -v

# UI
python main.py ui
```

---

## 16. Sign-off before a demo or submission

Use this checklist literally (tick when done):

- [ ] Fresh terminal; venv **activated**; `cd` to project root  
- [ ] `python scripts/check_env.py` OK  
- [ ] `data/processed/clean_accidents.csv` exists (or re-run Phase 2)  
- [ ] `models/accident_risk.joblib` exists (or `python main.py train`)  
- [ ] `python scripts/run_phase7_smoke.py` → all **PASS**  
- [ ] `pytest tests/test_mvp_phase7.py -v` → **7 passed**  
- [ ] One manual `main.py predict` command returns valid JSON  
- [ ] (If demoing UI) `python main.py ui` opens and **Predict risk** works once  

When all boxes are checked, the application is in a **tested, demo-ready** state for this MVP repository.
