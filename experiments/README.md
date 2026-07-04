# MLP + LayerNorm Residual Injection

## Status: ❌ Failed (progressive collapse)

## Architecture
```
R = LN(MLP(M))
M' = M + gamma * R
```
Simplest form: MLP on memory only, no text conditioning.

## Results (3 epoch)
| Ep | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| 1 | 0.373 | 0.280 | 0.361 |
| 2 | 0.354 | 0.309 | 0.349 |
| 3 | 0.329 | 0.331 | 0.329 |

## Root Cause
- Without LN: MLP output grows unbounded (rm_norm 3.6 → 36, epoch 6 crash)
- With LN: norm locked at 16 → large perturbation from epoch 1
- LN prevents explosion but creates immediate high-SNR perturbation
- No text conditioning → residual is random noise for all positions
- Worse than TC-MLP: both old and new collapse simultaneously

## Comparison
| 3e | ori_ap | new_ap |
|:--|:--|:--|
| MLP+LN | 0.329 | 0.331 |
| TC-MLP | 0.353 | 0.329 |
| Bilinear | 0.466 | 0.343 |

Bilinear is the ONLY architecture that preserves old-class features.
