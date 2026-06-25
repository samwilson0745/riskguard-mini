# RiskGuard-mini

A self-contained fintech fraud detection demo. It generates synthetic transaction data, trains an XGBoost model to score each transaction by fraud risk, and surfaces the results in a Streamlit dashboard — including an AI-generated compliance memo powered by Gemini.

---

## What it does

### 1. Data generation (`scripts/generate_data.py`)

Creates 3,000 synthetic payment transactions across ~200 cards and 10 countries. About 8% (240 transactions) are labelled fraud. The fraud records have realistic signals baked in:

- **High velocity** — fraud cards fire 4–12 transactions in the same hour vs. 1–3 for legit cards
- **Large or round amounts** — fraud amounts cluster at $500, $1k, $2k, $5k, $9,999; legit amounts follow a log-normal distribution capped at $1,999
- **Odd hours** — fraud transactions skew toward 1–4 AM; legit ones are daytime
- **Country mismatch** — fraud merchants are in a random country, not the card's home country
- **High-risk merchant categories** — fraud skews toward gambling, crypto, and electronics
- **New merchant flag** — fraud cards more often transact with merchants the card has never used

15% noise is added to both classes so the signals aren't perfectly clean.

Output: `data/transactions.csv` with 14 columns including the ground-truth `is_fraud` label.

---

### 2. Model training (`scripts/train.py`)

Trains an XGBoost binary classifier on 9 features derived from the transaction data:

| Feature | Description |
|---|---|
| `amount` | Transaction dollar amount |
| `hour` | Hour of day (0–23) |
| `velocity_1h` | # of transactions from the same card in the past hour |
| `new_merchant` | 1 if the card has never transacted with this merchant |
| `round_amount` | 1 if amount is a round multiple of $100 |
| `country_mismatch` | 1 if card country ≠ merchant country |
| `odd_hour` | 1 if transaction occurred between 1–4 AM |
| `large_amount` | 1 if amount > $2,000 |
| `merchant_category_enc` | Label-encoded merchant category |

The model uses `scale_pos_weight` to handle the class imbalance (92% legit vs 8% fraud). It trains on 80% of the data and evaluates on the remaining 20%.

Outputs:
- `data/scored_transactions.csv` — the full 3,000-row dataset with a `risk_score` column (0–1 probability of fraud)
- `models/model.joblib` — the saved model + label encoder
- `models/feature_importance.json` — feature importance scores ranked by gain

---

### 3. Streamlit dashboard (`app.py`)

A single-page app with four sections:

**Sidebar filters**
- Risk score threshold slider (default 0.5) — only transactions at or above this score are flagged
- Top-N slider — how many of the riskiest transactions to show (10–100)
- Merchant category multi-select — filter by category

**KPI row**
Four live metrics: total transactions in view, number flagged, flag rate %, and average risk score of flagged transactions.

**Riskiest transactions table**
Sorted by risk score descending. Each row shows: TXN ID, card, amount, merchant category, card/merchant countries, hour, velocity, all binary fraud signals, risk score, and ground-truth label. A colour-coded risk level badge (🔴 Critical ≥0.8 / 🟠 High ≥0.6 / 🟡 Medium ≥0.4 / 🟢 Low) appears on every row.

**Feature importance chart**
Horizontal bar chart (Plotly) showing each feature's importance score from XGBoost, ordered highest to lowest. Colour intensity scales with importance.

**Compliance flag memo**
Shows the raw fields of the single highest-risk transaction. A "Generate compliance memo" button calls `gemini-3.1-flash-lite` and renders a formal 3-sentence compliance memo covering: the primary risk indicators, the relevant regulatory concern (AML / PSD2 SCA / BSA), and the recommended immediate action. Requires `GEMINI_API_KEY` — the section degrades gracefully with a warning if the key is absent.

---

## Quick start

```bash
pip install -r requirements.txt

# Step 1 — generate data
python scripts/generate_data.py

# Step 2 — train model + score transactions
python scripts/train.py

# Step 3 — launch dashboard
streamlit run app.py
```

For the AI compliance memo, set your Gemini API key first:

```bash
export GEMINI_API_KEY=AIza...
```

---

## Project structure

```
riskguard-mini/
├── app.py                        # Streamlit dashboard
├── scripts/
│   ├── generate_data.py          # Synthetic dataset generator
│   └── train.py                  # XGBoost training + scoring
├── data/
│   ├── transactions.csv          # Raw generated data (git-ignored)
│   └── scored_transactions.csv   # With risk_score column (git-ignored)
├── models/
│   ├── model.joblib              # Saved model artifact (git-ignored)
│   └── feature_importance.json   # Feature gain scores (git-ignored)
├── requirements.txt
└── README.md
```

## Stack

- **Python** — data generation, model training, app logic
- **scikit-learn** — train/test split, label encoding, evaluation metrics
- **XGBoost** — gradient boosted tree classifier for fraud scoring
- **Streamlit** — dashboard UI (no frontend code)
- **Plotly** — feature importance chart
- **Google Gemini** (`gemini-3.1-flash-lite`) — compliance memo generation
- **joblib** — model serialisation
