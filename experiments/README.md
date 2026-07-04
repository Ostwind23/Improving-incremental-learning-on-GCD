# B2: Bilinear + Global RMS (target=1.0)

## Code Changes
- norm_mode='rms', rms_target=1.0
- init_std=0.01 (solves weight explosion from std=0.1 init)
- Global RMS preserves token amplitude ratios

## Configs
- `configs/gdino_inc/70+10/grmi_rms_t1_3e.py`: 3e test
- `configs/gdino_inc/70+10/grmi_rms_t1_12e.py`: 12e full (running)

## Key Results
| 3e | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| B2 (RMS t=1.0) | 0.466 | 0.356 | 0.452 |
| Bilinear+LN | 0.466 | 0.339 | 0.450 |
| GRMI | 0.466 | 0.350 | 0.451 |

| 12e | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| B2 (pending) | ? | ? | ? |

## Notes
- Best 3e result of all architectures
- ori_ap matches LN (0.466), new_ap exceeds LN (+1.7pt at 3e)
- 12e pending (est. completion ~15:00 Jul 4)
