# GRMI: Gated Residual Memory Injection (MLP)

## Code
- `mmdet/models/utils/residual_inject.py`: MLP residual injection
- `mmdet/models/detectors/gdino_inc_gcd.py`: detector integration with pre_decoder hook

## Configs
- `configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_grmi_12e_coco.py`: GRMI 12e (gamma=0.01, MLP mode)

## Key Results
| 3e | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| GRMI | 0.466 | 0.350 | 0.451 |
| GCD baseline | 0.474 | 0.328 | — |

| 12e | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| GRMI | 0.465 | 0.398 | 0.459 |
| GCD baseline | 0.474 | 0.391 | 0.464 |

## Notes
- First architecture adding residual injection to GroundingDINO encoder
- gamma=0.01 provides minimal perturbation (~0.16), preserving old classes
- Limitations: new_ap ceiling ~0.398, lower than Bilinear variants
