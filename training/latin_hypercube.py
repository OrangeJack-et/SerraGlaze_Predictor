"""Latin hypercube sampling of the four room parameters.

Draws 1,000 space-filling samples over window height (vertical window
placement), azimuth angle, room depth and window size. Each row is one room
configuration, simulated in Radiance via Grasshopper/Honeybee to build the
training set. The 100-run test set came from a separate Latin hypercube
sample.

Output: training/outputs/Matrix_Input_LHS_1000.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

param_names = ['window_height', 'angle', 'room_depth', 'window_size']
# room_depth is a multiplier on the 6 m base room depth (0.1-1.7 -> 0.6-10.2 m).
l_bounds = [-0.5, 0.0, 0.1, 0.1]
u_bounds = [0.5, 2 * np.pi, 1.7, 1.9]

n_samples = 1000

sampler = qmc.LatinHypercube(d=len(param_names), scramble=True, seed=42)
sample_unit = sampler.random(n=n_samples)          # samples in [0, 1)
sample_scaled = qmc.scale(sample_unit, l_bounds, u_bounds)

df = pd.DataFrame(sample_scaled, columns=param_names)
out_path = OUT_DIR / 'Matrix_Input_LHS_1000.txt'
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with {len(df)} rows")
