// Browser re-implementation of the SerraGlaze predictor in app.py:
//   StandardScaler(inputs) -> MLP 4-128-256-128-20 (ReLU) -> PCA inverse -> StandardScaler inverse
// Weights are the exact trained float32 values; arithmetic is float64, so this is
// an exact evaluation of the trained model. PyTorch's float32 arithmetic differs
// from it by ~1e-4 lux (rounding noise), which is far below anything visible.

function loadSerraGlazeModel(json) {
  const layers = json.layers.map(l => ({
    W: l.W.map(row => Float64Array.from(row, Math.fround)),
    b: Float64Array.from(l.b, Math.fround),
  }));
  const { input_scaler: sx, output_scaler: sy, pca } = json;

  function linear(x, { W, b }, relu) {
    const out = new Float64Array(W.length);
    for (let i = 0; i < W.length; i++) {
      const row = W[i];
      let s = b[i];
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      out[i] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  // inputs: [window_height, angle, room_depth, window_size]
  // returns Float64Array(169), row-major 13x13 lux grid
  function predict(inputs) {
    // PyTorch receives the scaled inputs as float32, so round them the same way
    let x = Float64Array.from(inputs, (v, i) => Math.fround((v - sx.mean[i]) / sx.scale[i]));
    layers.forEach((layer, k) => { x = linear(x, layer, k < layers.length - 1); });

    const n = pca.mean.length;
    const y = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let s = pca.mean[j];
      for (let c = 0; c < x.length; c++) s += x[c] * pca.components[c][j];
      y[j] = s * sy.scale[j] + sy.mean[j];
    }
    return y;
  }

  return { predict, grid: json.grid };
}

if (typeof module !== "undefined") module.exports = { loadSerraGlazeModel };
