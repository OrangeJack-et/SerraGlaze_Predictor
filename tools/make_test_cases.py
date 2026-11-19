"""Run random inputs through the original PyTorch + joblib pipeline (exactly as
app.py does) and save them as reference outputs for checking the JS port.

Run from the repo root:  python tools/make_test_cases.py
Then open tools/verify.html via a local server at the repo root.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

N = 20
L = np.array([-0.5, 0.0, 0.1, 0.1])          # bounds from app.py
U = np.array([0.5, 2 * np.pi, 1.7, 1.9])


class SerraluxDNN(torch.nn.Module):  # copied from app.py (importing it would start Streamlit)
    def __init__(self, n_components):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, n_components),
        )

    def forward(self, x):
        return self.net(x)


def main():
    pipeline = joblib.load(ROOT / "pipeline.joblib")
    model = SerraluxDNN(pipeline["pca"].n_components_)
    model.load_state_dict(torch.load(ROOT / "serraglaze_dnn_weights.pth", map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(0)
    X = L + rng.random((N, 4)) * (U - L)
    X[0] = [0, np.pi, 0.9, 1.0]                 # app.py defaults
    X[1] = L                                     # corners
    X[2] = U

    # Same trained float32 weights, evaluated in float64 (exact reference)
    Ws = [(model.net[i].weight.detach().double().numpy(), model.net[i].bias.detach().double().numpy())
          for i in (0, 2, 4, 6)]

    def to_lux(z):
        return pipeline["scaler_Outputs"].inverse_transform(pipeline["pca"].inverse_transform(z[None, :]))[0]

    cases = []
    for x in X:
        scaled = pipeline["scaler_Inputs"].transform(x[None, :])
        # 1) exactly what app.py does (float32 PyTorch)
        with torch.no_grad():
            z32 = model(torch.tensor(scaled, dtype=torch.float32)).numpy()[0]
        # 2) float64 evaluation of the same model
        h = scaled[0].astype(np.float32).astype(np.float64)
        for k, (W, b) in enumerate(Ws):
            h = W @ h + b
            if k < len(Ws) - 1:
                h = np.maximum(h, 0)
        cases.append({"input": x.tolist(),
                      "output": to_lux(z32).tolist(),
                      "output_f64": to_lux(h).tolist()})

    out = ROOT / "tools" / "test_cases.json"
    out.write_text(json.dumps(cases))
    print(f"wrote {N} cases to {out}")


if __name__ == "__main__":
    main()
