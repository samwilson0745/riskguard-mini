"""
Generate synthetic transaction dataset with baked-in fraud signals.
Outputs: data/transactions.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 3000
FRAUD_RATE = 0.08  # ~8% fraud

COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "SG", "NG", "RU", "CN"]
MERCHANT_CATS = ["retail", "food", "travel", "electronics", "gambling", "crypto", "fuel", "healthcare"]


def generate():
    n_fraud = int(N * FRAUD_RATE)
    n_legit = N - n_fraud
    labels = [0] * n_legit + [1] * n_fraud
    RNG.shuffle(labels)
    labels = np.array(labels)

    # Base fields
    txn_ids = [f"TXN{str(i).zfill(6)}" for i in range(N)]
    card_ids = RNG.integers(1000, 1200, size=N)  # ~200 unique cards
    amounts = np.where(
        labels == 1,
        RNG.choice([500, 1000, 2000, 5000, 9999], size=N) + RNG.uniform(0, 1, N),
        RNG.lognormal(mean=4.5, sigma=1.2, size=N).clip(1, 1999),
    )
    amounts = np.round(amounts, 2)

    hours = np.where(
        labels == 1,
        RNG.choice([1, 2, 3, 4], size=N),          # fraud: odd hours
        RNG.integers(6, 23, size=N),                 # legit: daytime
    )
    # Mix in some legit odd-hour txns and some daytime fraud
    noise_mask = RNG.random(N) < 0.15
    hours[noise_mask] = RNG.integers(0, 24, size=noise_mask.sum())

    card_country = RNG.choice(COUNTRIES, size=N, p=[0.6, 0.08, 0.06, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02])
    merchant_country = np.where(
        labels == 1,
        RNG.choice(COUNTRIES, size=N),              # fraud: random country
        card_country,                                # legit: same country
    )
    # add some legit cross-border
    cross_mask = RNG.random(N) < 0.05
    merchant_country[cross_mask] = RNG.choice(COUNTRIES, size=cross_mask.sum())

    merchant_cat = np.where(
        labels == 1,
        RNG.choice(["gambling", "crypto", "electronics"], size=N),
        RNG.choice(MERCHANT_CATS, size=N, p=[0.35, 0.25, 0.1, 0.1, 0.05, 0.03, 0.07, 0.05]),
    )

    # Velocity: transactions per card in a rolling 1-hour window (simulated)
    velocity = np.where(labels == 1, RNG.integers(4, 12, size=N), RNG.integers(1, 4, size=N))

    # New merchant flag
    new_merchant = np.where(labels == 1, RNG.integers(0, 2, size=N), RNG.binomial(1, 0.1, size=N))

    # Round amount flag
    round_amount = (amounts % 100 == 0).astype(int)

    df = pd.DataFrame({
        "txn_id": txn_ids,
        "card_id": card_ids,
        "amount": amounts,
        "hour": hours,
        "card_country": card_country,
        "merchant_country": merchant_country,
        "merchant_category": merchant_cat,
        "velocity_1h": velocity,
        "new_merchant": new_merchant,
        "round_amount": round_amount,
        "is_fraud": labels,
    })

    # Derived feature
    df["country_mismatch"] = (df["card_country"] != df["merchant_country"]).astype(int)
    df["odd_hour"] = df["hour"].isin([1, 2, 3, 4]).astype(int)
    df["large_amount"] = (df["amount"] > 2000).astype(int)

    out = Path(__file__).parent.parent / "data" / "transactions.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")
    print(df["is_fraud"].value_counts().rename({0: "legit", 1: "fraud"}))


if __name__ == "__main__":
    generate()
