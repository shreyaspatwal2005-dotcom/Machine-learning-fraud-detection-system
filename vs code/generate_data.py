"""Generate a synthetic, imbalanced transaction dataset (~1.5% fraud).

Use this only if you don't have a real CSV. Any CSV with numeric features
and a binary `is_fraud` column works with fraud_detection.py.
"""
import argparse
import numpy as np
import pandas as pd


def generate(n=20000, fraud_rate=0.015, seed=42):
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    f = is_fraud == 1

    amount = np.where(f, rng.lognormal(4.6, 1.1, n), rng.lognormal(3.8, 1.0, n))
    night = rng.random(n) < 0.5  # only half of frauds happen at night; some legit txns do too
    hour = np.where(f & night, rng.choice([0, 1, 2, 3, 4, 23], n),
                    np.where(rng.random(n) < 0.05, rng.choice([0, 1, 2, 3, 4, 23], n),
                             rng.integers(6, 23, n)))
    distance_from_home = np.where(f, rng.exponential(45, n), rng.exponential(18, n))
    txn_last_24h = np.where(f, rng.poisson(3.5, n), rng.poisson(2.2, n))
    account_age_days = np.where(f, rng.integers(1, 900, n), rng.integers(10, 3650, n))
    avg_amount_30d = rng.lognormal(3.8, 0.6, n)
    amount_ratio = amount / avg_amount_30d

    df = pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "distance_from_home": distance_from_home.round(2),
        "txn_last_24h": txn_last_24h,
        "account_age_days": account_age_days,
        "avg_amount_30d": avg_amount_30d.round(2),
        "amount_ratio": amount_ratio.round(3),
        "is_fraud": is_fraud,
    })
    # Add a few missing values to exercise validation/imputation
    for col in ["amount", "distance_from_home"]:
        df.loc[rng.choice(n, 20, replace=False), col] = np.nan
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="transactions.csv")
    p.add_argument("--rows", type=int, default=20000)
    a = p.parse_args()
    d = generate(a.rows)
    d.to_csv(a.out, index=False)
    print(f"Saved {len(d)} rows to {a.out}; fraud rate = {d.is_fraud.mean():.2%}")
