"""Random forest model for average room illuminance.

Trains a 300-tree random forest on the four (unscaled) room parameters to
predict the mean of the 169 sensor values. Reports R², MAE and RMSE on the
100 held-out runs, feature importances and 5-fold cross-validation.

Reads:  data/DNN_SerraGlaze_Final_TrainingData.txt, ..._TestingData.txt
Writes: training/outputs/rf_model.pkl, rf_predictions.npz
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

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

# --- model ---
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

# --- train and predict ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- evaluation ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

# --- cross-validation and feature importances ---
scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring="r2"
)

for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")

print("CV R² scores:", scores)
print("Mean CV R²:", scores.mean())

# --- save model and predictions ---
joblib.dump(model, OUT_DIR / "rf_model.pkl")
np.savez(OUT_DIR / "rf_predictions.npz", y_test=y_test, y_pred=y_pred)
