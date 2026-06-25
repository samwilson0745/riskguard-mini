"""
RiskGuard-mini — Streamlit compliance dashboard.

Run:
    streamlit run app.py

Requires:
    data/scored_transactions.csv  (from scripts/train.py)
    models/feature_importance.json
    ANTHROPIC_API_KEY env var (optional — memo section degrades gracefully)
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskGuard-mini",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ RiskGuard-mini")
st.caption("Synthetic fraud detection · XGBoost risk scores · LLM compliance memos")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = BASE / "data" / "scored_transactions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_importance():
    path = BASE / "models" / "feature_importance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())

df = load_data()
importance = load_importance()

if df is None:
    st.error(
        "No scored data found. Run the pipeline first:\n\n"
        "```bash\npython scripts/generate_data.py\npython scripts/train.py\n```"
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    threshold = st.slider("Risk score threshold", 0.0, 1.0, 0.5, 0.01)
    top_n = st.slider("Show top N transactions", 10, 100, 25, 5)
    cats = st.multiselect(
        "Merchant categories",
        options=sorted(df["merchant_category"].unique()),
        default=sorted(df["merchant_category"].unique()),
    )

# ── Filter & sort ─────────────────────────────────────────────────────────────
filtered = df[df["merchant_category"].isin(cats)].copy()
flagged = filtered[filtered["risk_score"] >= threshold].sort_values("risk_score", ascending=False)
top = flagged.head(top_n).reset_index(drop=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total transactions", f"{len(filtered):,}")
col2.metric("Flagged (≥ threshold)", f"{len(flagged):,}")
col3.metric("Flag rate", f"{len(flagged)/len(filtered)*100:.1f}%")
col4.metric("Avg risk score (flagged)", f"{flagged['risk_score'].mean():.3f}" if len(flagged) else "–")

st.divider()

# ── Top risky transactions table ──────────────────────────────────────────────
st.subheader(f"Top {min(top_n, len(flagged))} Riskiest Transactions")

def risk_badge(score):
    if score >= 0.8:
        return "🔴 Critical"
    elif score >= 0.6:
        return "🟠 High"
    elif score >= 0.4:
        return "🟡 Medium"
    return "🟢 Low"

display = top[[
    "txn_id", "card_id", "amount", "merchant_category",
    "card_country", "merchant_country", "hour",
    "velocity_1h", "country_mismatch", "odd_hour",
    "large_amount", "new_merchant", "risk_score", "is_fraud",
]].copy()

display.insert(0, "risk_level", display["risk_score"].apply(risk_badge))
display["amount"] = display["amount"].map("${:,.2f}".format)
display["risk_score"] = display["risk_score"].map("{:.4f}".format)
display["is_fraud"] = display["is_fraud"].map({1: "✅ Fraud", 0: "–"})
display = display.rename(columns={
    "txn_id": "TXN ID", "card_id": "Card", "amount": "Amount",
    "merchant_category": "Category", "card_country": "Card Country",
    "merchant_country": "Merch Country", "hour": "Hour",
    "velocity_1h": "Velocity (1h)", "country_mismatch": "Country ≠",
    "odd_hour": "Odd Hour", "large_amount": "Large Amt",
    "new_merchant": "New Merch", "risk_score": "Risk Score",
    "is_fraud": "Label", "risk_level": "Level",
})

st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# ── Feature importance chart ──────────────────────────────────────────────────
st.subheader("Feature Importance")

if importance:
    features = list(importance.keys())
    scores = list(importance.values())

    FEATURE_LABELS = {
        "amount": "Transaction Amount",
        "hour": "Hour of Day",
        "velocity_1h": "Velocity (1-hour window)",
        "new_merchant": "New Merchant",
        "round_amount": "Round Amount",
        "country_mismatch": "Country Mismatch",
        "odd_hour": "Odd Hour (1–4 AM)",
        "large_amount": "Large Amount (>$2k)",
        "merchant_category_enc": "Merchant Category",
    }
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale="Reds",
            showscale=False,
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Importance (gain)",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=60, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No feature importance data found — run scripts/train.py first.")

st.divider()

# ── Compliance memo (LLM) ──────────────────────────────────────────────────────
st.subheader("🤖 Compliance Flag Memo")
st.caption("AI-generated memo for the top-flagged transaction")

if len(top) == 0:
    st.info("No transactions meet the current threshold.")
else:
    top_raw = flagged.head(1).iloc[0]

    memo_col, detail_col = st.columns([3, 2])

    with detail_col:
        st.markdown("**Transaction under review**")
        st.json({
            "txn_id": top_raw["txn_id"],
            "amount": f"${top_raw['amount']:,.2f}",
            "card_country": top_raw["card_country"],
            "merchant_country": top_raw["merchant_country"],
            "merchant_category": top_raw["merchant_category"],
            "hour": int(top_raw["hour"]),
            "velocity_1h": int(top_raw["velocity_1h"]),
            "risk_score": float(top_raw["risk_score"]),
        })

    with memo_col:
        api_key = os.environ.get("GEMINI_API_KEY", "")

        if not api_key:
            st.warning(
                "Set `GEMINI_API_KEY` to generate LLM memos.\n\n"
                "```bash\nexport GEMINI_API_KEY=AIza...\n```"
            )
        else:
            if st.button("Generate compliance memo", type="primary"):
                prompt = f"""You are a compliance officer at a fintech firm. Write a concise 3-sentence compliance flag memo for the following suspicious transaction. Use formal language appropriate for a regulatory audit trail.

Transaction details:
- ID: {top_raw['txn_id']}
- Amount: ${top_raw['amount']:,.2f}
- Card issued in: {top_raw['card_country']} | Merchant in: {top_raw['merchant_country']}
- Merchant category: {top_raw['merchant_category']}
- Hour of transaction: {int(top_raw['hour'])}:00
- Velocity (transactions in last hour from same card): {int(top_raw['velocity_1h'])}
- ML risk score: {float(top_raw['risk_score']):.4f} (scale 0–1)
- Country mismatch: {'Yes' if top_raw['country_mismatch'] else 'No'}
- Odd-hour transaction: {'Yes' if top_raw['odd_hour'] else 'No'}
- Large amount flag: {'Yes' if top_raw['large_amount'] else 'No'}

Write exactly 3 sentences. First sentence: state the transaction and the primary risk indicators. Second sentence: cite the specific compliance concern (e.g. AML, PSD2 SCA, BSA). Third sentence: recommend immediate action."""

                with st.spinner("Drafting memo…"):
                    try:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-3.1-flash-lite",
                            contents=prompt,
                        )
                        memo_text = response.text
                        st.markdown(
                            f"""<div style="background:#1e1e2e;border-left:4px solid #e05252;
                            padding:1rem 1.2rem;border-radius:6px;font-size:0.95rem;
                            line-height:1.6;color:#f0f0f0;">
                            {memo_text}
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
