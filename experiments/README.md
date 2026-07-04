# Bilinear + per-token LayerNorm

## Code Changes
- ResidualInject: Bilinear mode with per-token LN
- R = LN(W_out(W_m(M) * W_t(T_pool)))
- Per-token LN forces all tokens to norm=16, destroying amplitude selectivity

## Configs
- `configs/gdino_inc/70+10/grmi_bilinear_3e_clean.py`: 3e reproduction
- `configs/gdino_inc/70+10/grmi_bilinear_12e_clean.py`: 12e full training

## Key Results
| 12e | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| Bilinear+LN | 0.460 | 0.403 | 0.453 |
| GRMI | 0.465 | 0.398 | 0.459 |
| GCD | 0.474 | 0.391 | 0.464 |

## Diagnostics
- new_ap +1.2pt over GCD but ori_ap -1.4pt
- LN eliminates M×T amplitude ratio: pre-LN CV=0.159, post-LN CV=0.0
- Motivated RMS (global norm) replacement
