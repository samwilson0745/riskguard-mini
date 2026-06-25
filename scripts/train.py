"""
Train XGBoost fraud classifier → attach risk scores to every transaction.
Outputs: models/model.joblib, data/scored_transactions.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE = Path(__file__).parent.parent
DATA_IN = BASE / "data" / "transactions.csv"
DATA_OUT = BASE / "data" / "scored_transactions.csv"
MODEL_OUT = BASE / "models" / "model.joblib"
META_OUT = BASE / "models" / "feature_importance.json"

FEATURES = [
    "amount",
    "hour",
    "velocity_1h",
    "new_merchant",
    "round_amount",
    "country_mismatch",
    "odd_hour",
    "large_amount",
    "merchant_category_enc",
]


def train():
    df = pd.read_csv(DATA_IN)

    le = LabelEncoder()
    df["merchant_category_enc"] = le.fit_transform(df["merchant_category"])

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle imbalance
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nAUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    # Score full dataset
    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["risk_score"] = df["risk_score"].round(4)
    df.to_csv(DATA_OUT, index=False)
    print(f"Scored data → {DATA_OUT}")

    # Save model
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "features": FEATURES}, MODEL_OUT)
    print(f"Model → {MODEL_OUT}")

    # Feature importance
    importance = dict(zip(FEATURES, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    META_OUT.write_text(json.dumps(importance, indent=2))
    print(f"Feature importance → {META_OUT}")


if __name__ == "__main__":
    train()
