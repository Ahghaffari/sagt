# SAGT (Spectral Adaptive Graph Transformer)

Reference implementation of *Spectral Adaptive Graph Transformer for Efficient and Scalable Traffic Forecasting* together with the baselines used in the paper's comparison.


## How to use

```python
from models.sagt.transformer import SAGTransformer

model = SAGTransformer(
    num_nodes=N, in_steps=12, out_steps=12, input_dim=3,
    steps_per_day=288,          # 144 for 10-minute data
    k=32,                       # spectral rank
    num_layers=3, num_heads=4, feed_forward_dim=256, dropout=0.1,
    hybrid=True,                # hybrid spectral spatial layer
    freq_bin_attn=True,         # frequency-bin attention
    adj=adj,                    # (N, N) float array, A[i, j] > 0 means i -> j
    device="cuda:0",
)
y = model(x)
```

**Input** `x` of shape `(B, T, N, 3)`:

| channel | contents |
| --- | --- |
| 0 | traffic measurement, **z-scored using training-split statistics only** |
| 1 | time of day, a fraction in `[0, 1)` |
| 2 | day of week, an integer in `{0, ..., 6}` |

**Output** of shape `(B, out_steps, N, 1)`, **in the normalised space** of channel 0.
Apply `sigma * y + mu` before computing any loss or metric.

Only channel 0 is standardized, and only channel 0 is filtered by frequency-bin
attention; the calendar channels pass through untouched.

## Layout

```
models/sagt/        SAGT (transformer.py holds variants D-F; sagt.py the earlier A-C)
models/staeformer/  SelfAttentionLayer, the temporal backbone SAGT adopts
models/            baselines used in the comparison
common/            shared data pipeline, metrics and training loop
```

