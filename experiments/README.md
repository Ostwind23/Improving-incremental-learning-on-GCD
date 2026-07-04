# S2: Bilinear with no normalization (init_std=0.01 only)

## Code Changes
- norm_mode='none'
- init_std=0.01 (critical: prevents R norm from exploding to 448)
- No per-token LN, no global RMS

## Configs
- `configs/gdino_inc/70+10/grmi_nonorm_t1e.py`: 1e test

## Key Results (1 epoch)
| 1e | ori_ap | new_ap |
|:--|:--|:--|
| S2 (no norm) | 0.475 | 0.275 |
| B2 (RMS t=1.0) | 0.474 | 0.291 |
| Bilinear+LN | 0.470 | 0.306 |

## Notes
- Highest ori_ap of all variants (0.475, beating GCD baseline 0.474)
- Lowest new_ap (0.275) due to larger perturbation (2.25 vs 0.5 for RMS)
- Viable but RMS variants give better new/old balance
