"""XGBoost model for average room illuminance (the primary scalar model).

Trains gradient-boosted trees on the four room parameters to predict the
mean of the 169 sensor values. The number of trees is chosen by early
stopping on an 80/20 split of the training set; the 100-run test set is used
only for the final R², MAE and RMSE. Also runs 5-fold cross-validation and
plots feature importance (gain) and actual vs predicted.

Reads:  data/DNN_SerraGlaze_Final_TrainingData.txt, ..._TestingData.txt
Writes: training/outputs/xgb_model.json, xgb_predictions.npz
        training/figures/xgb_feature_importances.png, xgb_actual_vs_predicted.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# --- load data ---
# Columns: 4 inputs, 1 run index (skipped), 169 sensor values (13 x 13 grid).
feature_names = ["vertical_window_placement", "angle", "room_depth", "window_size"]

Train = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TrainingData.txt', header=None)
X_train = Train.iloc[:, :4].values
Y_train = Train.iloc[:, 5:].values

Test = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TestingData.txt', header=None)
X_test = Test.iloc[:, :4].values
Y_test = Test.iloc[:, 5:].values

y_train = Y_train.mean(axis=1)
y_test = Y_test.mean(axis=1)

# --- split training set for early stopping (test set stays untouched) ---
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# --- params ---
params = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.05,
    "max_depth":        4,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "rmse",
    "verbosity":        0,
}

# --- train with early stopping on validation slice ---
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

best_ntree = bst.best_iteration + 1
print(f"Best n_estimators: {best_ntree}")
print(f"Best validation RMSE: {bst.best_score:.3f}")

# --- evaluate on held-out test set (touched only here) ---
y_pred = bst.predict(dtest)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest R²   : {r2:.3f}")
print(f"Test MAE  : {mae:.3f}")
print(f"Test RMSE : {rmse:.3f}")

# --- cross-validation on training set ---
sk_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=best_ntree,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(sk_model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
print(f"\nCV R² scores: {np.round(cv_scores, 3)}")
print(f"Mean CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# --- feature importances (gain) ---
importance_gain = bst.get_score(importance_type="gain")
mapped = {feature_names[int(k[1:])]: v for k, v in importance_gain.items()}

print("\nFeature importances (gain):")
for name in feature_names:
    print(f"  {name}: {mapped.get(name, 0):.3f}")

# --- feature importance plot (percent of total gain, sorted) ---
display_names = {
    "vertical_window_placement": "Vertical window\nplacement",
    "angle":                     "Azimuth angle",
    "room_depth":                "Room depth",
    "window_size":               "Window size",
}

gains = np.array([mapped.get(n, 0) for n in feature_names])
gains_pct = 100 * gains / gains.sum()

order = np.argsort(gains_pct)[::-1]
sorted_names = [display_names[feature_names[i]] for i in order]
sorted_values = gains_pct[order]

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(sorted_names, sorted_values, color="steelblue", edgecolor="black", linewidth=0.5)
ax.set_ylabel("Feature importance (% of total gain)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", alpha=0.5)

for i, v in enumerate(sorted_values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "xgb_feature_importances.png", dpi=300, bbox_inches="tight")
plt.show()

# --- actual vs predicted plot ---
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.6, s=20)
lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
plt.xlabel("Actual lux")
plt.ylabel("Predicted lux")
plt.gca().set_aspect("equal")
plt.title(f"XGBoost (R² = {r2:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "xgb_actual_vs_predicted.png", dpi=300)
plt.show()

# --- save ---
bst.save_model(str(OUT_DIR / "xgb_model.json"))
np.savez(OUT_DIR / "xgb_predictions.npz", y_test=y_test, y_pred=y_pred)
print("\nSaved model and predictions.")
