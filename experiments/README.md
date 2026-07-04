# Gate v2: Fixed LN→Gate Order (bias=2.0)

## Status: ❌ Failed (ori_ap collapsed, gate stuck at init)

## Architecture (fix applied)
```
R = W_out(W_m(M) * W_t(T_pool))
R = LN(R)                       # LN FIRST
R = R * sigmoid(gate_net(M))    # gate AFTER LN
M' = M + gamma * R
```

## Results (3 epoch)
| | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| Gate v2 | 0.380 | 0.250 | 0.363 |
| Bilinear+LN (no gate) | 0.466 | 0.339 | 0.450 |

## Diagnostics
- gate_net.2.bias = 2.0004 after 3 epochs (init = 2.0)
- Gate didn't learn AT ALL — gradient too weak
- Measured gradient: gate_bias_grad = 2.68e-3 at loss~10
  → per-step move = 1.3e-7 → 7500 steps = 0.0010 total (essentially zero)
- Root cause: sigmoid saturation (0.88^2=0.106) × small perturbation (RMS t=1.0)
  → gradient 20x too small for meaningful learning
- 100x LR would give only 0.0375 bias movement in 3 epochs — still useless
