"""Linear regression baseline for average room illuminance.

Fits a linear model from the four room parameters to the mean of the 169
sensor values, then reports R², MAE and RMSE on the 100 held-out runs, the
coefficients and 5-fold cross-validation on the training set.

Reads:  data/DNN_SerraGlaze_Final_TrainingData.txt, ..._TestingData.txt
Writes: training/outputs/linear_model.pkl, linear_predictions.npz
        training/figures/linear_actual_vs_predicted.png
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# --- load data ---
# Columns: 4 inputs, 1 run index (skipped), 169 sensor values (13 x 13 grid).
feature_names = ["window_height", "angle", "room_depth", "window_size"]
Train = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TrainingData.txt', header=None)
X_train = Train.iloc[:, :4].values
Y_train = Train.iloc[:, 5:].values
Test = pd.read_csv(DATA_DIR / 'DNN_SerraGlaze_Final_TestingData.txt', header=None)
X_test = Test.iloc[:, :4].values
Y_test = Test.iloc[:, 5:].values

y_train = Y_train.mean(axis=1)
y_test = Y_test.mean(axis=1)

# --- train and predict ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- evaluation ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# --- coefficients (the linear-model equivalent of feature importances) ---
print("\nCoefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# --- cross-validation ---
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print(f"\nCV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.3f}")

# --- save model and predictions ---
joblib.dump(model, OUT_DIR / "linear_model.pkl")
np.savez(OUT_DIR / "linear_predictions.npz", y_test=y_test, y_pred=y_pred)

# --- plot ---
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.6, s=20)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', lw=1)
plt.xlabel("Actual lux")
plt.ylabel("Predicted lux")
plt.gca().set_aspect("equal")
plt.title(f"Linear Regression (R² = {r2:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "linear_actual_vs_predicted.png", dpi=300)
plt.show()
