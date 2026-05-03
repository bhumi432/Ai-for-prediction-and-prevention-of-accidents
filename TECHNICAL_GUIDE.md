# Technical Components & Execution Guide

**Project:** AI for Predicting and Preventing Road Accidents (beginner MVP)

This guide is implementation-focused: set up the repo, implement modules, train, predict, and debug.

---

## 1. Tech Stack

| Area | Choice |
|------|--------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn (Logistic Regression, Decision Tree, `train_test_split`, metrics) |
| **Persistence** | joblib (preferred) or pickle for model + encoders |
| **Viz (optional)** | Matplotlib / Seaborn for EDA only |
| **Interface** | **CLI** via `argparse` or **Streamlit** for a simple UI—pick one for MVP |
| **Editor / explore** | VS Code + optional Jupyter for scratch EDA (keep production path in `.py` scripts) |
| **Terminal** | PowerShell (Windows) or bash |

---

## 2. Project Setup (Step-by-Step)

### 2.1 Create virtual environment

```powershell
cd "path\to\your\project"
python -m venv .venv
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2 Install dependencies

Create `requirements.txt` (see [Section 11](#11-dependencies-file)) then:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Folder setup

Create this layout (adjust names if you already use `src/`):

```text
project/
├── data/
│   ├── raw/           # Original CSV (add to .gitignore if huge)
│   └── processed/     # Cleaned train-ready CSV
├── models/            # Saved .joblib files (often gitignored)
├── TECHNICAL_GUIDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── utils.py
├── train.py
├── predict.py
├── main.py            # CLI entry: train or predict
└── app_streamlit.py   # Optional; only if using Streamlit
```

**`.gitignore` hints:** `.venv/`, `data/raw/*.csv` (if large), `models/*.joblib`, `__pycache__/`, `.ipynb_checkpoints/`.

---

## 3. Core Components Breakdown

### a. Data Ingestion Module

**Responsibility:** Load CSV into a DataFrame; validate required columns exist.

```python
import pandas as pd
from pathlib import Path

def load_accidents_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    return df
```

**Tip:** After loading, print `df.shape`, `df.dtypes`, and `df.isnull().sum()` once to sanity-check.

---

### b. Data Preprocessing Module

**Responsibility:** Missing values, categorical encoding, optional feature selection, **consistent** transforms for train and predict.

**Typical steps:**

1. Drop or fill missing values (e.g., `SimpleImputer` with most_frequent / median).
2. Encode categoricals: `OneHotEncoder` (sparse_output=False) or `OrdinalEncoder` for ordered bins.
3. Keep a **list of feature column names** used at training time—prediction must use the same.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def build_preprocess_pipeline(categorical_cols: list[str], numeric_cols: list[str]):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor
```

**Feature selection (simple):** Drop IDs and free-text columns; keep weather, hour, day_of_week, road/area type, etc. If you have too many columns, start with 5–10 strong candidates.

---

### c. Model Training Module

**Responsibility:** Train/test split, fit preprocessor + model, save **both** preprocessor and estimator (or a single `Pipeline`).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

def train_and_save(X, y, model_path: str, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Example: full pipeline with preprocessor already attached as first step
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    # clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=random_state)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return X_test, y_test
```

**Saving:** Prefer **one `Pipeline`** object: `preprocessor` → `classifier`. Then one `joblib.dump(pipeline, "models/accident_risk.joblib")`.

---

### d. Prediction Module

**Responsibility:** Load artifact(s), accept **same schema** as training features, output discrete risk.

**Risk labels:** Map model output to strings, e.g. `0 → Low`, `1 → Medium`, `2 → High` (or binary then bucket probabilities).

```python
import joblib
import numpy as np

RISK_NAMES = {0: "Low", 1: "Medium", 2: "High"}

def predict_risk(pipeline, features_df):
    proba = pipeline.predict_proba(features_df)
    class_idx = int(np.argmax(proba, axis=1)[0])
    return RISK_NAMES.get(class_idx, "Medium")
```

If the model is binary (`high_risk` vs not), use `predict_proba` threshold or `predict()` and map accordingly.

---

### e. Recommendation Module

**Responsibility:** Deterministic rules from **risk level + raw inputs** (weather, time, etc.).

```python
def recommend(risk_level: str, weather: str, hour: int) -> str:
    tips = []
    if risk_level == "High":
        tips.append("Delay trip if possible; risk is elevated.")
    if weather.lower() in ("rain", "heavy rain", "snow", "fog"):
        tips.append("Reduce speed and increase following distance.")
    if hour >= 20 or hour <= 5:
        tips.append("Night driving: use headlights and stay alert.")
    if not tips:
        tips.append("Maintain safe speed and avoid distractions.")
    return " ".join(tips)
```

---

## 4. File-by-File Implementation Guide

| File | Role |
|------|------|
| **`main.py`** | Entry point: subcommands `train` and `predict` (or flags). Parses CLI args, calls `train.py` / `predict.py` logic. |
| **`train.py`** | Load raw CSV → preprocess → split → fit pipeline → `joblib.dump` to `models/accident_risk.joblib` → print accuracy/F1 on holdout. |
| **`predict.py`** | Load joblib, build one-row DataFrame from CLI args or JSON, run `pipeline.predict` / `predict_proba`, call recommendation helper, print JSON. |
| **`utils.py`** | `load_data`, `build_preprocess_pipeline`, column lists, `risk_label_from_prediction`, `recommend`, paths like `MODEL_PATH`. |
| **`data/raw/`** | Place `accidents.csv` (your chosen public dataset). |
| **`data/processed/`** | Optional: save `train_clean.csv` after cleaning for faster re-runs. |
| **`models/`** | Store `accident_risk.joblib` only (gitignore if repo size matters). |

**Optional:** `app_streamlit.py` imports `predict_risk` logic from `utils` / `predict` and uses `st.selectbox`, `st.slider`, `st.json`.

---

## 5. Execution Flow

1. **Load data** — Read CSV from `data/raw/`.
2. **Preprocess** — Select columns, handle missing, encode; define `X` and `y` (target: severity bucket or `high_risk` flag).
3. **Train model** — `train_test_split` → fit `Pipeline` → evaluate on test set.
4. **Save model** — `joblib.dump(pipeline, "models/accident_risk.joblib")`.
5. **Run prediction** — Load pipeline, construct input row matching training columns, predict + recommend.

```text
data/raw/*.csv  →  preprocess  →  train  →  models/*.joblib
                                                      ↓
 user input  →  same preprocess shape  →  predict  →  JSON output
```

---

## 6. Sample Code Snippets

### Training (minimal)

```python
# train.py (conceptual minimal flow)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/raw/accidents.csv")
# Example: target column must exist in YOUR dataset
y = df["severity_class"]  # e.g. 0=Low, 1=Med, 2=High — align with your labeling
X = df.drop(columns=["severity_class", "id"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Replace with ColumnTransformer if you have mixed types
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
joblib.dump(pipe, "models/accident_risk.joblib")
```

### Prediction (minimal)

```python
# predict.py (conceptual)
import joblib
import pandas as pd

pipe = joblib.load("models/accident_risk.joblib")
# Columns must match training X exactly (order and names)
row = pd.DataFrame([{
    "hour": 21,
    "day_of_week": 5,
    "weather": "Rain",
    "area_type": "Urban",
}])
risk_class = int(pipe.predict(row)[0])
print({"risk_level": ["Low", "Medium", "High"][risk_class]})
```

Adjust column names and encodings to match **your** pipeline (one-hot will expect the same categories learned at fit time).

---

## 7. Input/Output Format

### Example input (dict / JSON)

```json
{
  "hour": 22,
  "day_of_week": 6,
  "weather": "Heavy Rain",
  "area_type": "Highway",
  "traffic_level": "High"
}
```

CLI equivalent:

```bash
python main.py predict --hour 22 --day-of-week 6 --weather "Heavy Rain" --area-type Highway --traffic-level High
```

### Example output

```json
{
  "risk_level": "High",
  "recommendation": "Avoid driving in heavy rain at night; reduce speed and increase following distance."
}
```

Implement `print(json.dumps(result))` or return dict from a function for Streamlit.

---

## 8. Running the Project

```bash
# Activate venv first
python train.py
```

```bash
python main.py predict --hour 20 --weather Rain --area-type Urban
# or
python predict.py
```

**Optional Streamlit:**

```bash
streamlit run app_streamlit.py
```

Use **one** trained artifact path everywhere (`models/accident_risk.joblib`).

---

## 9. Debugging Tips

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `ValueError: Found unknown categories` | Predict-time category not seen in training | Use `OneHotEncoder(handle_unknown='ignore')`; widen training categories |
| `X has N features; model expects M` | Wrong columns or order | Save feature list in training; build DataFrame with same columns |
| Very high accuracy (~99%) | Target leakage | Remove columns that “explain” the label after the fact (e.g., police report fields) |
| `sklearn` version mismatch | Model pickled on different version | Pin versions in `requirements.txt`; retrain after upgrade |
| All predictions one class | Imbalanced data | `class_weight='balanced'`; check `value_counts()` on `y` |
| `FileNotFoundError` for model | Not trained yet | Run `train.py` first; check `models/` path |

**Quick checks:** `pipeline.named_steps` (if using Pipeline), `print(X_train.columns)`, compare to prediction row.

---

## 10. Testing (Basic)

Use `pytest` or plain `assert` in `tests/test_predict.py`.

```python
import joblib
import pandas as pd

def test_model_loads():
    pipe = joblib.load("models/accident_risk.joblib")
    assert pipe is not None

def test_predict_shape():
    pipe = joblib.load("models/accident_risk.joblib")
    # Use one real row schema from your training data
    X = pd.read_csv("data/processed/sample_row.csv")
    pred = pipe.predict(X.head(1))
    assert pred.shape == (1,)
```

Smoke-test: after training, run predict with **one row copied from training** (held-out) and verify output is finite and label in allowed set.

---

## 11. Dependencies File

Example `requirements.txt`:

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
streamlit>=1.28.0
pytest>=7.4.0
```

Pin exact versions in team settings: `pip freeze > requirements-lock.txt`.

---

## 12. Future Enhancements (Optional)

- **APIs:** Open-Meteo or national weather API + schedule refresh (rate limits, API keys in `.env`).
- **Model:** Try `RandomForestClassifier` or **HistGradientBoostingClassifier** (still sklearn, tabular).
- **Frontend:** Static React page calling a small **FastAPI** `/predict` endpoint.
- **Monitoring:** Log inputs/outputs locally for drift awareness (privacy-sensitive—avoid PII).

---

## Quick checklist before “done”

- [ ] `train.py` runs end-to-end on your CSV  
- [ ] `models/accident_risk.joblib` loads on another machine with same `requirements.txt`  
- [ ] Predict script uses **identical** feature construction as training  
- [ ] README / TECH guide document dataset source and license  

This MVP stays simple: **one pipeline object, one dataset, one interface path**—expand only after that works reliably.
