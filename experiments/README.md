# Cross-Attention Residual Injection

## Status: ❌ Failed (old-class collapse)

## Architecture
```
Q = LN(W_q(T_new))
K = W_k(memory)
A = softmax(Q @ K.T / sqrt(d))
R = LN(A.T @ Q)
M' = M + gamma * R
```
Attention from text to memory — residual = text-weighted memory combination.

## Results (3 epoch)
| | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| CrossAttn | 0.353 | 0.329 | 0.350 |
| Bilinear | 0.466 | 0.343 | 0.450 |

## Root Cause
- Attention weights from T_new to all M tokens are diffuse — T_new is broad, not selective.
- Softmax distributes attention across all positions → all tokens receive residual.
- LN post-attention equalizes magnitudes.
- Same collapse pattern as TC-MLP.
