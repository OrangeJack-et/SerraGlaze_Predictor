# SerraGlaze Predictor

A machine-learned surrogate for daylighting simulation. It predicts how light
is distributed across a room fitted with SerraGlaze, SerraLux's glazing film,
in a fraction of the time a full Radiance simulation takes.

**Live app:** https://orange-jacket-serraglaze-predictor.hf.space/

**Headline result:** R² = 0.99 against held-out Radiance simulations, at
roughly 4 × 10⁵ times the speed, which makes real-time design exploration
possible in the browser. [CHECK: name the model this R² belongs to]

Final-year industrial project, University of Oxford Department of Physics,
with SerraLux (Oct 2025 to May 2026). Awarded the Department of Physics prize
for best group project presentation.

## The problem
Radiance gives accurate daylighting results but is too slow to explore many
room configurations interactively. This project trains models on Radiance
output so a new configuration can be evaluated almost instantly.

## Data
- 1,000 room configurations chosen by Latin hypercube sampling
- Each simulated parametrically in Grasshopper and Radiance
- Inputs: [FILL: the parameters varied, e.g. room dimensions, orientation, sun position]
- Outputs: a 13 × 13 grid of lux values across the room, plus [FILL: the scalar target]

## Models
| Model | Predicts | Held-out R² |
|---|---|---|
| Linear regression | scalar output | [FILL] |
| Random forest | scalar output | [FILL] |
| XGBoost | scalar output | [FILL] |
| Deep neural network (PyTorch) | full 13 × 13 lux grid | [FILL] |

## Results
- R² = 0.99 on simulations held out from training [CHECK: which model]
- Evaluation about 4 × 10⁵ times faster per room configuration than Radiance
- Deployed as a Streamlit app as a working proof of concept

## Repository
- `[FILL: file]`: data generation and sampling
- `[FILL: file]`: model training and evaluation
- `[FILL: file]`: the Streamlit app

## Running it
    pip install -r requirements.txt
    streamlit run [FILL: app file]

## Team and my contribution
Group project of [FILL: number] students. My part: [FILL: what you did
yourself, e.g. dataset generation, the DNN, the app].

## Limitations
[FILL: one or two honest limits, e.g. the range of room geometries covered,
or that it was trained on simulation and not measured data]
