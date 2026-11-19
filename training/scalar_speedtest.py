"""Inference speed test for the three scalar models.

Loads the saved linear regression, random forest and XGBoost models, times
prediction on the 100-run test set (10 warm-up calls, then 10,000 timed
calls) and reports time per sample and speed-up over a ~120 s Radiance run.

Run linear_regression.py, random_forest.py and xgboost_model.py first.

Reads: data/DNN_SerraGlaze_Final_TestingData.txt
       training/outputs/linear_model.pkl, rf_model.pkl, xgb_model.json
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"

# --- load test data (same as training scripts) ---
Test = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TestingData.txt', header=None)
X_test = Test.iloc[:, :4].values

n_samples = X_test.shape[0]
print(f"Test set size: {n_samples} samples\n")

# --- load all three models ---
lr_model = joblib.load(OUT_DIR / "linear_model.pkl")
rf_model = joblib.load(OUT_DIR / "rf_model.pkl")

xgb_booster = xgb.Booster()
xgb_booster.load_model(str(OUT_DIR / "xgb_model.json"))


# Wrap XGBoost so all three share a predict(X) interface.
# DMatrix construction is part of the realistic inference path,
# so it stays inside the timed call.
def xgb_predict(X):
    return xgb_booster.predict(xgb.DMatrix(X))


predictors = {
    "Linear Regression": lr_model.predict,
    "Random Forest":     rf_model.predict,
    "XGBoost":           xgb_predict,
}


def time_inference(predict_fn, X, n_repeats=10000, n_warmup=10):
    """Time a predict function. Returns mean and std per sample, in microseconds."""
    # Warm-up: the first call pays for cache / lazy init
    for _ in range(n_warmup):
        _ = predict_fn(X)

    times = np.empty(n_repeats)
    for i in range(n_repeats):
        t0 = time.perf_counter()
        _ = predict_fn(X)
        times[i] = time.perf_counter() - t0

    n = X.shape[0]
    return {
        "per_sample_us_mean": (times.mean() / n) * 1e6,
        "per_sample_us_std":  (times.std() / n) * 1e6,
        "batch_ms_mean":       times.mean() * 1e3,
    }


# --- run benchmark ---
print(f"{'Model':<20} {'per-sample (µs)':>22} {'batch of 100 (ms)':>22}")
print("-" * 66)

results = {}
for name, fn in predictors.items():
    r = time_inference(fn, X_test)
    results[name] = r
    print(f"{name:<20} "
          f"{r['per_sample_us_mean']:>10.2f} ± {r['per_sample_us_std']:>5.2f}      "
          f"{r['batch_ms_mean']:>10.3f}")

# --- Radiance comparison (the headline ratio) ---
radiance_seconds_per_sim = 120  # 2 minutes per sim, conservative
print(f"\nRadiance reference: ~{radiance_seconds_per_sim} s/sim "
      f"= {radiance_seconds_per_sim*1e6:.0f} µs/sim")
print("\nSpeedup factors vs Radiance:")
for name, r in results.items():
    speedup = (radiance_seconds_per_sim * 1e6) / r["per_sample_us_mean"]
    print(f"  {name:<20} {speedup:>12,.0f}×")
