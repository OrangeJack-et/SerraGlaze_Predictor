"""Inference speed test for the deployed spatial DNN.

Loads the deployed weights and pipeline from the repo root and times, over
1,000 trials on the 100-run test inputs: the network forward pass alone, the
full pipeline (network + inverse PCA + inverse standardisation) for a single
input, and batched prediction. Reports speed-up over a ~120 s Radiance run.

Reads: serraglaze_dnn_weights.pth, pipeline.joblib (repo root)
       data/DNN_SerraGlaze_Final_TestingData.txt
"""
import platform
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"


class SerraluxDNN(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),   nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_components)
        )

    def forward(self, x):
        return self.net(x)


# ─── Load model + pipeline ───────────────────────────────────────────
pipeline = joblib.load(REPO_ROOT / 'pipeline.joblib')
pca = pipeline['pca']
scaler_Inputs = pipeline['scaler_Inputs']
scaler_Outputs = pipeline['scaler_Outputs']

model = SerraluxDNN(n_components=pca.n_components_)
model.load_state_dict(torch.load(REPO_ROOT / 'serraglaze_dnn_weights.pth'))
model.eval()

# ─── Load test set (so we time on representative inputs) ─────────────
test_file = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TestingData.txt', header=None)
test_inputs = test_file.iloc[:, :4].values

# Pre-scale once (scaling cost is negligible vs network; do not include in loop)
Inputs_scaled = scaler_Inputs.transform(test_inputs)
Inputs_tensor = torch.tensor(Inputs_scaled, dtype=torch.float32)

# ─── Warmup ──────────────────────────────────────────────────────────
# The first inference call carries one-off setup overhead.
with torch.no_grad():
    _ = model(Inputs_tensor[:1])

N_TRIALS = 1000

# ─── 1. Network forward pass only (single input) ─────────────────────
start = time.perf_counter()
with torch.no_grad():
    for _ in range(N_TRIALS):
        _ = model(Inputs_tensor[:1])
network_ms = (time.perf_counter() - start) / N_TRIALS * 1000

# ─── 2. Full pipeline (network + inverse PCA + inverse standardise) ──
start = time.perf_counter()
for _ in range(N_TRIALS):
    with torch.no_grad():
        y_pca = model(Inputs_tensor[:1]).numpy()
    _ = scaler_Outputs.inverse_transform(pca.inverse_transform(y_pca))
pipeline_ms = (time.perf_counter() - start) / N_TRIALS * 1000

# ─── 3. Batched inference (full 100-sim test set in one call) ────────
start = time.perf_counter()
with torch.no_grad():
    y_pca_batch = model(Inputs_tensor).numpy()
_ = scaler_Outputs.inverse_transform(pca.inverse_transform(y_pca_batch))
batch_total_ms = (time.perf_counter() - start) * 1000
batched_per_pred_ms = batch_total_ms / len(Inputs_tensor)

# ─── 4. Comparison vs Radiance ───────────────────────────────────────
radiance_per_sim_s = 120  # ~2 minutes per simulation (as in the report)
speedup_single = (radiance_per_sim_s * 1000) / pipeline_ms

# ─── Report ──────────────────────────────────────────────────────────
print(f"Hardware: {platform.processor() or platform.machine()}")
print(f"Trials per timing: {N_TRIALS}\n")
print(f"Network forward pass (single input): {network_ms:.4f} ms")
print(f"Full pipeline       (single input):  {pipeline_ms:.4f} ms")
print(f"Batched (per prediction over {len(Inputs_tensor)}): {batched_per_pred_ms:.4f} ms")
print(f"\nRadiance equivalent: ~{radiance_per_sim_s:.0f} s per simulation")
print(f"Speedup (full pipeline vs Radiance): ~{speedup_single:,.0f}x")
