"""Export the trained DNN and preprocessing pipeline to docs/model.json
so the static site can run inference in the browser.

Run from the repo root:  python tools/export_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def f32(a):
    # Shortest decimal strings that round-trip to the same float32 value.
    # The JS side applies Math.fround() on load to recover the exact weights.
    a = np.asarray(a, dtype=np.float32)
    return json.loads(json.dumps(
        np.vectorize(lambda x: float(np.format_float_positional(x, unique=True)))(a).tolist()
    ))


def f64(a):
    return np.asarray(a, dtype=np.float64).tolist()


def main():
    pipeline = joblib.load(ROOT / "pipeline.joblib")
    sx, sy, pca = pipeline["scaler_Inputs"], pipeline["scaler_Outputs"], pipeline["pca"]
    state = torch.load(ROOT / "serraglaze_dnn_weights.pth", map_location="cpu")

    # nn.Sequential indices of the Linear layers (ReLU between each, none after the last)
    layers = []
    for i in (0, 2, 4, 6):
        layers.append({
            "W": f32(state[f"net.{i}.weight"].numpy()),  # shape (out, in)
            "b": f32(state[f"net.{i}.bias"].numpy()),
        })

    assert not pca.whiten, "PCA whitening not handled"
    model = {
        "inputs": ["window_height", "angle", "room_depth", "window_size"],
        "input_scaler": {"mean": f64(sx.mean_), "scale": f64(sx.scale_)},
        "layers": layers,
        "pca": {"components": f64(pca.components_), "mean": f64(pca.mean_)},
        "output_scaler": {"mean": f64(sy.mean_), "scale": f64(sy.scale_)},
        "grid": [13, 13],
    }
    out = ROOT / "docs" / "model.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(model, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
