# SerraGlaze Predictor

A machine-learned surrogate for daylighting simulation. SerraGlaze is
SerraLux's window film, which redirects daylight deeper into a room.
Predicting its effect normally takes Radiance about two minutes per room;
this model does it in under a millisecond.

**Live app:** https://orange-jacket-serraglaze-predictor.hf.space/

## Results
Trained on 1,000 Radiance simulations and tested on 100 held-out ones.

| Model | Predicts | R² | Speed-up over Radiance |
|---|---|---|---|
| Neural network (PyTorch) | full 13 × 13 illuminance grid | 0.988 | ~4 × 10⁵ |
| XGBoost | average room illuminance | 0.986 | ~7 × 10⁶ |

Inputs: window size, vertical window placement, room depth and azimuth.

## Limitations
- Validated against Radiance simulations, not physical measurements.
- Azimuth is not encoded as periodic, so 0 and 2π differ by about 7%.

## Files
- `app.py`: the Streamlit app
- `serraglaze_dnn_weights.pth`: trained network weights
- `pipeline.joblib`: fitted preprocessing pipeline

This repository holds the deployed app and trained model.

## Running it
    pip install -r requirements.txt
    streamlit run app.py

## Context
Final-year industrial project with SerraLux, University of Oxford Department
of Physics, 2025–26. Awarded the Department of Physics prize for best group
project presentation.
