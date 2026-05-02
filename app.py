import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# Model definition — must match the architecture used in training
# ============================================================
class SerraluxDNN(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_components)
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# Load model + preprocessing pipeline (cached — runs once)
# ============================================================
@st.cache_resource
def load_model():
    pipeline = joblib.load('pipeline.joblib')
    model = SerraluxDNN(n_components=pipeline['pca'].n_components_)
    model.load_state_dict(torch.load('serraglaze_dnn_weights.pth', map_location='cpu'))
    model.eval()
    return model, pipeline

model, pipeline = load_model()
scaler_Inputs = pipeline['scaler_Inputs']
scaler_Outputs = pipeline['scaler_Outputs']
pca = pipeline['pca']


# ============================================================
# Page layout
# ============================================================
st.set_page_config(page_title="SerraGlaze Daylight Predictor", layout="wide")
st.title("SerraGlaze Daylight Predictor")
st.write(
    "Predicts the spatial distribution of daylight across a 13×13 sensor grid "
    "for a room fitted with SerraGlaze micro-structured glazing film. "
    "Trained on 1000 Radiance simulations using the SerraGlaze BSDF."
)

# Two-column layout: inputs on the left, heatmap on the right
col_inputs, col_output = st.columns([1, 2])

l_bounds = [-0.5, 0.0, 0.1, 0.1]        # lower bounds
u_bounds = [ 0.5, 2*np.pi, 1.7, 1.9]    # upper bounds

with col_inputs:
    st.subheader("Room parameters")
    window_height = st.slider(
        "Vertical window placement",
        min_value=-0.5, max_value=0.5, value=0, step=0.01,
        help="Fractional height of the window centre on the wall (0 = floor, 1 = ceiling)."
    )
    angle = st.slider(
        "Solar angle (rad)",
        min_value=0.0, max_value=2*np.pi, value=np.pi, step=0.01,
        help="Solar azimuth angle in radians."
    )
    room_depth = st.slider(
        "Room depth (m)",
        min_value=0.1, max_value=1.7, value=0.9, step=0.01
    )
    window_size = st.slider(
        "Window size",
        min_value=0.1, max_value=1.9, value=1, step=0.01
    )

    predict_button = st.button("Predict", type="primary")

with col_output:
    if predict_button:
        # ============================================================
        # Inference
        # ============================================================
        new_input = np.array([[window_height, angle, room_depth, window_size]])
        scaled = scaler_Inputs.transform(new_input)
        tensor = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_pca = model(tensor).numpy()

        lux_matrix = scaler_Outputs.inverse_transform(pca.inverse_transform(pred_pca))
        grid = lux_matrix[0].reshape(13, 13)

        # ============================================================
        # Display
        # ============================================================
        st.subheader("Predicted lux distribution")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap='hot')
        ax.set_xlabel("Sensor column")
        ax.set_ylabel("Sensor row")
        plt.colorbar(im, ax=ax, label='Lux')
        st.pyplot(fig)

        # Summary stats
        st.metric("Average lux", f"{grid.mean():.1f}")
        st.metric("Max lux", f"{grid.max():.1f}")
        st.metric("Min lux", f"{grid.min():.1f}")
    else:
        st.info("Set room parameters and click Predict to see the lux distribution.")