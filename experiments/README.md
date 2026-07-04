# TC-MLP: Text-Conditioned MLP

## Status: ❌ Failed (old-class collapse)

## Architecture
```
R = LN(MLP(concat(M, T_pool)))
M' = M + gamma * R
```
Concat memory with text → MLP → residual. Text conditions the injection.

## Results (3 epoch)
| | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| TC-MLP | 0.353 | 0.329 | 0.350 |
| Bilinear (no gate) | 0.466 | 0.343 | 0.450 |

## Root Cause
- Concat doesn't prevent old-class perturbation. MLP output is position-agnostic.
- Unlike Bilinear's multiplicative interaction (M×T → natural selectivity),
  concat+MLP applies the same transformation to all tokens.
- LayerNorm forces all tokens to norm=16, destroying any residual selectivity.
