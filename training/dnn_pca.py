"""Spatial DNN: predicts the full 13 x 13 illuminance grid via PCA.

1. PCA sweep: on an internal 800/200 split of the 1,000 training runs, tries
   several variance thresholds and records the compression ceiling R² and the
   DNN's validation R² (written to training/pca_sweep_results.csv).
2. Final model: standardises inputs and outputs, compresses the 169 outputs
   with PCA at 99.3 % variance, and trains a 4-128-256-128-n DNN (Adam,
   lr 1e-3, MSE in PCA space, 4,000 epochs) on all 1,000 runs.
3. Evaluates R² and RMSE on the 100 held-out runs, after inverse PCA and
   inverse standardisation, so compression loss counts against the score.
4. Saves the weights and preprocessing pipeline.

Reads:  data/DNN_SerraGlaze_Final_TrainingData.txt, ..._TestingData.txt
Writes: training/pca_sweep_results.csv
        training/outputs/serraglaze_dnn_weights.pth, pipeline.joblib
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Config ──────────────────────────────────────────────────────────
SEED_TORCH = 2
SEED_NP = 2
SEED_SPLIT = 42                          # 800/200 internal split
VARIANCE_THRESHOLDS = [0.99, 0.993, 0.995, 0.999]
N_EPOCHS = 4000
RUN_SWEEP = True
CHOSEN_VARIANCE = 0.993                  # set after inspecting sweep table

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────
# Columns: 4 inputs, 1 run index (skipped), 169 sensor values (13 x 13 grid).
train_file = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TrainingData.txt', header=None)
Inputs = train_file.iloc[:, :4].values
Outputs = train_file.iloc[:, 5:].values

test_file = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TestingData.txt', header=None)
Testing_Inputs = test_file.iloc[:, :4].values
Testing_Outputs = test_file.iloc[:, 5:].values


# ─── DNN definition ──────────────────────────────────────────────────
class SerraluxDNN(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_components)
        )

    def forward(self, x):
        return self.net(x)


# ─── Helper: train DNN (resets seed for fair comparison) ─────────────
def train_dnn(X_tr_t, y_tr_t, n_components, n_epochs=N_EPOCHS):
    torch.manual_seed(SEED_TORCH)
    np.random.seed(SEED_NP)
    model = SerraluxDNN(n_components=n_components)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(n_epochs):
        model.train()
        pred = model(X_tr_t)
        loss = loss_fn(pred, y_tr_t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    model.eval()
    return model, losses


# ─── 1. PCA sweep on internal 800/200 split ──────────────────────────
if RUN_SWEEP:
    print("\n" + "=" * 64)
    print(" PCA component sweep — internal 800/200 split (no test-set leakage)")
    print("=" * 64)

    X_tr, X_val, y_tr, y_val = train_test_split(
        Inputs, Outputs, test_size=200, random_state=SEED_SPLIT)

    sweep_results = []
    for var in VARIANCE_THRESHOLDS:
        # scaler + PCA fit on the 800 only
        sX = StandardScaler().fit(X_tr)
        sY = StandardScaler().fit(y_tr)
        X_tr_s, X_val_s = sX.transform(X_tr), sX.transform(X_val)
        y_tr_s, y_val_s = sY.transform(y_tr), sY.transform(y_val)

        pca_local = PCA(n_components=var)
        y_tr_pca = pca_local.fit_transform(y_tr_s)
        n_comp = pca_local.n_components_

        # Ceiling R²: PCA round-trip on the 200 — upper bound from compression alone
        y_val_recon = sY.inverse_transform(
            pca_local.inverse_transform(pca_local.transform(y_val_s)))
        ceiling_r2 = r2_score(y_val, y_val_recon)

        # Train DNN on the 800
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_pca, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        model_local, _ = train_dnn(X_tr_t, y_tr_t, n_comp)

        # Model R² on the 200
        with torch.no_grad():
            y_val_pred_pca = model_local(X_val_t).numpy()
        y_val_pred = sY.inverse_transform(pca_local.inverse_transform(y_val_pred_pca))
        model_r2 = r2_score(y_val, y_val_pred)

        sweep_results.append({
            'variance': var,
            'n_components': n_comp,
            'ceiling_R2': round(ceiling_r2, 4),
            'model_R2':   round(model_r2,   4),
            'gap':        round(ceiling_r2 - model_r2, 4),
        })
        print(f"  var={var:.3f}  n_comp={n_comp:3d}  "
              f"ceiling R²={ceiling_r2:.4f}  model R²={model_r2:.4f}  "
              f"gap={ceiling_r2 - model_r2:.4f}")

    sweep_df = pd.DataFrame(sweep_results)
    sweep_path = HERE / 'pca_sweep_results.csv'
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSweep table written to {sweep_path}")
    print("\n" + sweep_df.to_string(index=False))

# ─── 2. Final training on FULL 1000-sim set, chosen threshold ────────
print("\n" + "=" * 64)
print(f" Final model — full 1000-sim set, variance threshold = {CHOSEN_VARIANCE}")
print("=" * 64)

scaler_Inputs = StandardScaler().fit(Inputs)
scaler_Outputs = StandardScaler().fit(Outputs)
Inputs_scaled = scaler_Inputs.transform(Inputs)
Outputs_scaled = scaler_Outputs.transform(Outputs)
Testing_Inputs_scaled = scaler_Inputs.transform(Testing_Inputs)

pca = PCA(n_components=CHOSEN_VARIANCE)
Outputs_pca = pca.fit_transform(Outputs_scaled)
print(f"PCA components selected: {pca.n_components_}")

Input_tr = torch.tensor(Inputs_scaled,         dtype=torch.float32)
Input_te = torch.tensor(Testing_Inputs_scaled, dtype=torch.float32)
Output_tr = torch.tensor(Outputs_pca,          dtype=torch.float32)

model, train_losses = train_dnn(Input_tr, Output_tr, pca.n_components_)

plt.plot(train_losses, label='Train')
plt.xlabel('Epoch')
plt.ylabel('MSE loss (PCA space)')
plt.title('Final DNN training — 1000 simulations')
plt.legend()
plt.show()

# ─── 3. Test set evaluation (genuinely held out) ─────────────────────
with torch.no_grad():
    Output_pred_pca = model(Input_te).numpy()

Output_pred = scaler_Outputs.inverse_transform(pca.inverse_transform(Output_pred_pca))
Output_true = Testing_Outputs                       # original — never through PCA

r2_test = r2_score(Output_true, Output_pred)
rmse_test = np.sqrt(np.mean((Output_true - Output_pred)**2))
print(f"\nTest set (100 sims, untouched):  R² = {r2_test:.4f}   RMSE = {rmse_test:.1f} lux")

# Quick heatmap sanity check
pred_grid = Output_pred[0].reshape(13, 13)
true_grid = Output_true[0].reshape(13, 13)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(true_grid, cmap='hot')
axes[0].set_title('True')
axes[1].imshow(pred_grid, cmap='hot')
axes[1].set_title('Predicted')
plt.show()

# ─── 4. Save model + pipeline ────────────────────────────────────────
torch.save(model.state_dict(), OUT_DIR / 'serraglaze_dnn_weights.pth')
joblib.dump(
    {'pca': pca, 'scaler_Inputs': scaler_Inputs, 'scaler_Outputs': scaler_Outputs},
    OUT_DIR / 'pipeline.joblib')
print(f"Model + pipeline saved to {OUT_DIR}")
