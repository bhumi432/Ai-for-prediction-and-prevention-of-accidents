# AI for Predicting and Preventing Road Accidents

---

## 1. Project Overview

### Problem statement

Road accidents cause injuries, deaths, and economic loss worldwide. Many crashes are influenced by **predictable factors**: poor visibility (weather, time of day), congestion, road type, and historical hotspots. Drivers and planners often lack a **simple, localized risk signal** before they choose a route or time to travel.

Common contributing factors include:

- **Environmental**: rain, fog, low light, slippery surfaces  
- **Temporal**: rush hour, night driving, weekends vs. weekdays  
- **Infrastructure / location**: intersections, highway merges, known black spots  
- **Behavior** (hard to measure in MVP): speeding, distraction — *acknowledged but not fully modeled in v1*

### Proposed solution

Build a **lightweight AI assistant** that:

1. Takes **simple inputs** (weather category, time bucket, location type, optional traffic level).  
2. Outputs a **risk level**: Low / Medium / High.  
3. Suggests **short prevention tips** (e.g., slow down, delay trip, avoid a route type).

This is **not** a replacement for safe driving or official traffic management — it is a **decision-support tool** for awareness and planning.

### Target users

| User | How they use it |
|------|------------------|
| **Everyday drivers** | Check risk before commuting or a long trip; get plain-language tips. |
| **City / transport students & analysts** | Explore risk patterns on a simple dashboard (optional). |
| **Small municipalities / civic tech demos** | Prototype “risk heat” by area using public data (non-production). |

---

## 2. Goals and Non-Goals

### Goals (MVP)

- Train a **simple, interpretable** model from tabular data (no deep learning).  
- Expose **risk level + suggestions** via **CLI** or a **minimal web/dashboard** (pick one for MVP).  
- Document data sources, features, and how to reproduce training locally.  
- Keep the repo runnable on a **normal laptop** with clear instructions.

### Non-Goals (explicitly out of scope for MVP)

- Real-time GPS routing, live traffic fusion, or production-grade map services  
- Deep learning (CNNs, Transformers, etc.)  
- Mobile apps, IoT sensors, or in-vehicle hardware  
- Legal liability, insurance integration, or claims automation  
- Personalized driver profiling or surveillance-grade tracking  
- Guaranteed accident prevention — the product **informs**, it does not **control** vehicles  

---

## 3. Key Features (MVP)

| Feature | Description |
|---------|-------------|
| **Accident risk prediction** | Given inputs (e.g., weather, time of day, day of week, road / area type, optional traffic), predict relative risk. |
| **Risk level output** | Discrete label: **Low**, **Medium**, **High** (or 3-class classification). |
| **Prevention suggestions** | Rule-based messages mapped from risk + factors, e.g., “Reduce speed in wet conditions,” “Consider delaying until daylight,” “Avoid known high-risk corridor if alternatives exist.” |
| **Simple interface** | **CLI** (recommended for beginners) *or* a **small dashboard** (e.g., Streamlit) — one primary interface for MVP. |

---

## 4. User Flow

**Path A — CLI user**

1. User installs the project and runs the CLI.  
2. User enters or selects: approximate **location type** (urban / highway / rural), **weather** (clear / rain / fog), **time** (hour + weekday/weekend), optional **traffic** level.  
3. App loads the trained model and outputs **Low / Medium / High** plus **2–4 short tips**.  
4. User adjusts inputs to compare scenarios (e.g., same route, different weather).

**Path B — Dashboard user (optional)**

1. User opens the local web app.  
2. User adjusts sliders / dropdowns for the same factors.  
3. Dashboard shows **risk gauge**, **tips**, and optionally a **static chart** from sample or aggregated historical data.  

---

## 5. Data Requirements

### Types of data

| Category | Examples | MVP usage |
|----------|----------|-----------|
| **Historical accidents** | Counts or binary “accident occurred” per segment/time | Labels or proxy labels for risk |
| **Weather** | Rain, fog, temperature bins | Features |
| **Time** | Hour, day of week, month | Features |
| **Location / road** | Urban vs rural, road category, intersection flag | Features |
| **Traffic (optional)** | Volume bucket, congestion level | Features if available |

### Suggested public datasets (starting points)

> Availability and licenses change — verify each source before use.

- **Government open data**: national or regional road safety / crash statistics (often CSV).  
- **Weather archives**: open meteorological datasets with date + region (merge by time/area).  
- **Kaggle**: search for “road accidents,” “US accidents,” “traffic accidents” — useful for **learning** and prototyping; cite the dataset in your README.  

Use **one primary dataset** for MVP to avoid merge hell.

### Input features for the model (example set)

Categorical or binned numeric features work well for logistic regression and trees:

- `hour` (0–23 or binned: morning / midday / evening / night)  
- `day_of_week` (or `is_weekend`)  
- `weather` (clear / rain / fog / snow — one-hot encoded)  
- `road_type` or `area_type` (highway / urban / rural)  
- `light_conditions` (daylight / dark-lit / dark-unlit) if available  
- `traffic_level` (low / medium / high) if available  

**Label example**: `high_risk` = 1 if accident severity or count exceeds a threshold in that bin; or use **severity** as multi-class if the dataset supports it (keep MVP binary or 3-class).

---

## 6. Technical Approach (Beginner Friendly)

### Suggested tech stack

| Layer | Tool |
|-------|------|
| Language | **Python 3.10+** |
| Data | **Pandas**, **NumPy** |
| ML | **Scikit-learn** |
| Config / CLI | `pyyaml` or env vars; **Typer** or `argparse` for CLI |
| Optional UI | **Streamlit** or **Flask + HTML** (minimal) |
| Notebooks (optional) | **Jupyter** for exploration only — keep final pipeline in scripts |

### Model choices

| Model | Why use it |
|-------|------------|
| **Logistic Regression** | Fast, interpretable coefficients; strong baseline for tabular data. |
| **Decision Tree** | Easy to visualize; handles non-linear splits; watch for overfitting. |

**Recommendation**: Train both; compare with the metrics in Section 8; ship the simpler one that meets your accuracy goal.

### High-level pipeline

1. **Data collection** — Download one curated dataset; store raw files under `data/raw/`.  
2. **Data preprocessing** — Clean missing values, encode categories, define the label, split train/validation/test **by time** if dates exist (avoids “future leakage”).  
3. **Model training** — Fit logistic regression and/or decision tree; save with **joblib** or **pickle** under `models/`.  
4. **Prediction** — Load model + preprocessor in CLI/app; map probabilities to Low/Medium/High with chosen thresholds; attach suggestion rules.  

---

## 7. System Architecture (Simple)

### Components in plain English

- **Data folder**: Raw and processed CSVs; nothing secret here — only public data.  
- **Training script**: Reads processed data, trains the model, saves artifacts.  
- **Inference module**: Loads artifacts, applies same preprocessing, returns risk + text tips.  
- **Interface**: CLI or small web app — calls inference only.  

### ASCII diagram

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Raw data   │────▶│  Preprocess +    │────▶│  Trained    │
│  (CSV)      │     │  train script    │     │  model +    │
└─────────────┘     └──────────────────┘     │  encoders   │
                                              └──────┬──────┘
                                                     │
┌─────────────┐     ┌──────────────────┐             │
│  User input │────▶│  Inference       │◀────────────┘
│  (CLI/UI)   │     │  + suggestions   │
└─────────────┘     └────────┬─────────┘
                             │
                             ▼
                     Low / Med / High + tips
```

---

## 8. Evaluation Metrics

| Metric | What it measures | Brief interpretation |
|--------|------------------|----------------------|
| **Accuracy** | Fraction of correct class predictions | Easy to read; can mislead if classes are imbalanced. |
| **Precision** | Of all predicted “high risk,” how many were truly high | Important if false alarms are costly. |
| **Recall** | Of all actual high-risk cases, how many you caught | Important if missing danger is worse than extra warnings. |

For accident risk, stakeholders often care about **recall for high-risk** (catch real danger) while keeping **false positives** tolerable. Report **confusion matrix** on the validation set and tune class thresholds for Low/Medium/High.

---

## 9. Risks and Limitations

| Risk | Impact | Mitigation (MVP) |
|------|--------|------------------|
| **Data quality** | Missing fields, inconsistent reporting | Single dataset; document cleaning; simple imputation |
| **Geographic bias** | Model only reflects training region | State clearly in README; don’t claim universal accuracy |
| **Label noise** | “Accident” definitions vary | Use consistent severity field; binary/3-class only |
| **Simplicity of model** | Cannot capture all real-world nuance | Position as educational / prototype; show metrics |
| **Deployment** | No live infra in MVP | Local-only; no SLA |
| **Over-trust** | Users might rely on output too much | Disclaimer: **assistive only**, not legal or safety certification |

---

## 10. Future Improvements

- **Real-time data**: Weather APIs + traffic APIs (rate limits, keys, cost).  
- **Mobile app**: Same backend logic; adds UX and distribution complexity.  
- **Richer ML**: Gradient boosting (**XGBoost** / **LightGBM**) — still tabular, not deep learning.  
- **Maps**: Visualize historical hotspots (static tiles or embed).  
- **Fairness & ethics review**: If deployed beyond demo, review bias and messaging.  

---

## 11. Project Milestones (Week-by-Week Roadmap)

| Week | Focus | Deliverable |
|------|--------|-------------|
| **1** | Problem + data | One dataset downloaded; EDA notebook or script; feature list frozen |
| **2** | Preprocessing | Clean pipeline script; train/val/test split; saved processed CSV |
| **3** | Modeling | Trained sklearn model(s); metrics + confusion matrix logged |
| **4** | Product | CLI or Streamlit; suggestion rules; README “How to Run” verified |
| **5** | Polish | Folder structure cleanup; sample inputs; limitations documented |

*Adjust pacing if part-time — milestones matter more than calendar weeks.*

---

## 12. Folder Structure (Starter)

```text
road-accident-ai/
├── README.md                 # This PRD + setup (you are here)
├── requirements.txt          # Pinned dependencies
├── .gitignore
├── data/
│   ├── raw/                  # Original downloads (gitignored if large)
│   └── processed/            # Cleaned datasets for training
├── notebooks/                # Optional EDA only
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Cleaning & feature encoding
│   ├── train.py              # Train + evaluate + save model
│   ├── predict.py            # Load model, score inputs
│   └── suggestions.py        # Rule-based tips from risk + features
├── models/                   # Saved .pkl / joblib (gitignored or store small demo)
├── tests/                    # Optional smoke tests
│   └── test_predict.py
└── scripts/
    └── run_cli.py            # Entry point for CLI
```

---

## 13. How to Run (Basic Setup)

### Prerequisites

- **Python 3.10+** installed  
- **pip** for packages  

### Installation

```bash
# Clone or open the project folder
cd road-accident-ai

# Create a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running locally (after you implement scripts)

```bash
# Example — adjust to your actual entry point after implementation
python scripts/run_cli.py --help

# Train model (once data is in place)
python src/train.py

# Predict (example)
python src/predict.py --weather rain --hour 20 --area urban
```

> **Note**: Training requires placing dataset files under `data/raw/` and implementing `preprocess.py` / `train.py` as described in this PRD. The commands above are the **intended** interface once those files exist.

---

## Testing this repository

For **end-to-end test steps** (environment, data scripts, train, predict, Streamlit, pytest, troubleshooting), see **[TESTING.md](TESTING.md)**.

---

## Disclaimer

This project is for **education and prototyping**. Predictions are **not** certified for safety-critical or regulatory decisions. Always follow traffic laws and official advisories.

---

## License

Specify your license in this repository (e.g., MIT) when you publish.
