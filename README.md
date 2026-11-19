# SerraGlaze Predictor

A machine-learned surrogate for daylighting simulation. SerraGlaze is
SerraLux's window film, which redirects daylight deeper into a room.
Predicting its effect normally takes Radiance about two minutes per room;
this model does it in under a millisecond.

**Live app:** https://serraglaze.jackmarsters.com/ (runs entirely in your browser)

The original Streamlit version is still on Hugging Face:
https://orange-jacket-serraglaze-predictor.hf.space/

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
- At some extreme settings the network predicts slightly negative illuminance
  for a few sensors. The web app shows these as 0 lux.

## Files
- `docs/`: the static web app, served by GitHub Pages
  - `index.html`: the page
  - `serraglaze-model.js`: the network and preprocessing in plain JavaScript
  - `model.json`: weights and fitted preprocessing parameters
- `tools/`: scripts to regenerate `model.json` and check the JavaScript against PyTorch
- `app.py`: the Streamlit app
- `serraglaze_dnn_weights.pth`: trained network weights
- `pipeline.joblib`: fitted preprocessing pipeline (input scaler, PCA, output scaler)

This repository holds the deployed apps and trained model.

## Running it
Web app: serve the repo root and open `/docs/`, for example

    python -m http.server
    # then visit http://localhost:8000/docs/

Streamlit app:

    pip install -r requirements.txt
    streamlit run app.py

## Updating the web model
After retraining, regenerate the browser model and check it still matches:

    python tools/export_model.py
    python tools/make_test_cases.py
    # serve the repo root and open /tools/verify.html: it should say PASS

The JavaScript matches a float64 evaluation of the network to about 1e-12 lux,
and PyTorch to within PyTorch's own float32 rounding (about 3e-4 lux).

## Context
Final-year industrial project with SerraLux, University of Oxford Department
of Physics, 2025–26. Awarded the Department of Physics prize for best group
project presentation.
