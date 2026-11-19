"""Radiance noise-floor analysis (preliminary; not part of the report).

Radiance's Monte Carlo ray tracing gives slightly different results each run.
This script uses repeated simulations of nine fixed configurations (six LHS
points, two edge cases and one central case) to measure that run-to-run
noise:

1. Per configuration: mean lux, sigma and coefficient of variation (CoV) of
   the room-average illuminance, plus a pooled per-sensor sigma. The worst-case
   CoV is compared with XGBoost's error to show how much of the model error
   could be simulation noise.
2. For the central configuration: per-sensor mean, sigma and CoV maps on the
   13 x 13 grid.

Reads:  data/Training_Outputs_matrix_Noise.csv
Writes: training/outputs/noise/ (scalar table CSV, .npy grids, 3-panel figure)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "Training_Outputs_matrix_Noise.csv"
OUT_DIR = HERE / "outputs" / "noise"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_PARAMS = 4               # window_height, angle, room_depth, window_size
GRID = 13                  # 13x13 sensor grid
XGB_MAPE = 0.054           # XGBoost reference for the comparison

# Known configs (for human-readable labels)
LABELS = {
    (0.226022, 1.634803, 1.204374, 1.690790): "LHS_1",
    (-0.188047, 0.675902, 0.163712, 1.022045): "LHS_2",
    (0.037674, 5.811452, 0.534466, 1.364213): "LHS_3",
    (0.407636, 3.327212, 0.781775, 0.432127): "LHS_4",
    (-0.459703, 4.214268, 1.479292, 0.931789): "LHS_5",
    (-0.015734, 3.074762, 0.907777, 0.210504): "LHS_6",
    (0.0, np.pi, 1.7, 0.1):                    "EDGE_high_noise",
    (0.0, np.pi, 0.1, 1.9):                    "EDGE_low_noise",
    (0.0, np.pi, 0.9, 1.4):                    "VAR_central",
}


def label_for(params, tol=1e-3):
    for k, v in LABELS.items():
        if all(abs(a - b) < tol for a, b in zip(params, k)):
            return v
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, header=None)
params = df.iloc[:, :N_PARAMS].round(6)
sensors = df.iloc[:, N_PARAMS + 1:].values        # skip the rep-index column
assert sensors.shape[1] == GRID * GRID, f"Expected {GRID*GRID} sensors, got {sensors.shape[1]}"
print(f"Loaded {len(df)} rows, {sensors.shape[1]} sensors\n")

# ---------------------------------------------------------------------------
# Group by parameters
# ---------------------------------------------------------------------------
groups = {}
for (key, idx) in params.groupby(list(params.columns)).groups.items():
    groups[key] = sensors[idx]                  # shape (n_reps, 169)

# ---------------------------------------------------------------------------
# Tier 1a: scalar sigma per config
# ---------------------------------------------------------------------------
rows = []
for key, arr in groups.items():
    n_reps = arr.shape[0]
    rep_means = arr.mean(axis=1)                # mean lux per rep
    mean_lux = rep_means.mean()
    sigma_mean = rep_means.std(ddof=1)
    cov_mean = sigma_mean / mean_lux

    # Pooled per-sensor sigma (single representative noise level for the config)
    sigma_per_sensor = arr.std(axis=0, ddof=1)
    pooled_sigma = np.sqrt(np.mean(sigma_per_sensor**2))   # RMS across 169 sensors
    pooled_cov = pooled_sigma / arr.mean()

    rows.append({
        "config":          label_for(key),
        "n_reps":          n_reps,
        "mean_lux":        mean_lux,
        "sigma_mean_lux":  sigma_mean,
        "CoV_mean_lux":    cov_mean,
        "pooled_sigma":    pooled_sigma,
        "pooled_CoV":      pooled_cov,
    })

table = pd.DataFrame(rows).sort_values("mean_lux").reset_index(drop=True)
table.to_csv(OUT_DIR / "tier1_scalar_table.csv", index=False)

print("=" * 95)
print("Tier 1a — scalar noise per config")
print("=" * 95)
fmt = "{:<18} {:>5} {:>10} {:>12} {:>10} {:>12} {:>10}"
print(fmt.format("config", "reps", "mean_lux", "sigma_avg", "CoV_avg", "pooled_sig", "pool_CoV"))
print("-" * 95)
for _, r in table.iterrows():
    print(fmt.format(
        r["config"], int(r["n_reps"]),
        f"{r['mean_lux']:.2f}", f"{r['sigma_mean_lux']:.3f}",
        f"{100*r['CoV_mean_lux']:.3f}%",
        f"{r['pooled_sigma']:.3f}", f"{100*r['pooled_CoV']:.2f}%"))
print("-" * 95)
print(f"\nXGBoost reference MAPE: {100*XGB_MAPE:.1f}%")
print(f"Worst-case scalar CoV:  {100*table['CoV_mean_lux'].max():.3f}%")
print(f"=> Headroom factor:     {XGB_MAPE / table['CoV_mean_lux'].max():.1f}x\n")

# ---------------------------------------------------------------------------
# Tier 1b: VAR_central per-sensor maps
# ---------------------------------------------------------------------------
var_key = next(k for k in groups if label_for(k) == "VAR_central")
arr = groups[var_key]                          # (n_reps, 169)
print(f"VAR_central: {arr.shape[0]} reps")

mean_grid = arr.mean(axis=0).reshape(GRID, GRID)
sigma_grid = arr.std(axis=0, ddof=1).reshape(GRID, GRID)
cov_grid = sigma_grid / mean_grid

np.save(OUT_DIR / "var_central_mean.npy",  mean_grid)
np.save(OUT_DIR / "var_central_sigma.npy", sigma_grid)
np.save(OUT_DIR / "var_central_cov.npy",   cov_grid)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im0 = axes[0].imshow(mean_grid, cmap="hot", origin="upper")
axes[0].set_title(f"Mean lux  (n={arr.shape[0]})")
fig.colorbar(im0, ax=axes[0], label="lux")

im1 = axes[1].imshow(sigma_grid, cmap="viridis", origin="upper")
axes[1].set_title("σ per sensor (lux)")
fig.colorbar(im1, ax=axes[1], label="σ [lux]")

im2 = axes[2].imshow(100 * cov_grid, cmap="plasma", origin="upper")
axes[2].set_title("CoV = σ/mean  (%)")
fig.colorbar(im2, ax=axes[2], label="CoV [%]")

for ax in axes:
    ax.set_xlabel("window-wall axis")
    ax.set_ylabel("depth axis")

plt.suptitle("VAR_central noise floor (per-sensor)")
plt.tight_layout()
plt.savefig(OUT_DIR / "var_central_maps.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  mean lux : range [{mean_grid.min():.1f}, {mean_grid.max():.1f}], avg {mean_grid.mean():.1f}")
print(f"  sigma    : range [{sigma_grid.min():.2f}, {sigma_grid.max():.2f}], avg {sigma_grid.mean():.2f} lux")
print(f"  CoV      : range [{100*cov_grid.min():.2f}%, {100*cov_grid.max():.2f}%], avg {100*cov_grid.mean():.2f}%")
print(f"\nSaved to {OUT_DIR}")
