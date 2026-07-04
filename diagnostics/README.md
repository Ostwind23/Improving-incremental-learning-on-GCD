# Diagnostic Scripts & Analysis Archive

## Script Index

### Gradient & Architecture Analysis
| Script | Purpose | Key Finding |
|:--|:--|:--|
| `diag_gate_grad_v2.py` | Gate gradient measurement | gate_bias_grad=2.68e-3, 1000x too weak |
| `diag_grad_scaled.py` | Gate gradient at loss~10 | Decoder attenuation factor = 0.4x |
| `diag_fwd_all.py` | Test A/B/D/E normalization | Only RMS viable, init_std=0.01 critical |
| `diag_fwd_fresh.py` | Fresh init RMS tests | std=0.1→R=448, std=0.01→R=4.5 |

### LN Selectivity Analysis
| Script | Purpose | Key Finding |
|:--|:--|:--|
| `diag_ln_bilinear_12e.py` | Pre/post LN norm | pre-LN CV=0.159, post-LN CV=0.0 |
| `diag_ln_quick.py` | Quick LN diagnosis | Random M+T: ||R|| goes from 2.37→16.00 |
| `diag_direction.py` | cos(R,M) old vs new | R direction ~0 for all positions |

### R-norm Mask Feasibility
| Script | Purpose | Key Finding |
|:--|:--|:--|
| `diag_r_mask.py` | R-norm threshold test | R norm doesn't separate old/new |
| `diag_r_mask_cpu.py` | CPU version | init_std=0.01→|R|=0.19, no selectivity |

### Architecture Comparison
| Script | Purpose | Key Finding |
|:--|:--|:--|
| `diag_rm_architecture.py` | Compare R(M) modes | Only Bilinear preserves old-class |
| `diag_rm_synth.py` | Synthetic architecture test | Multiplicative interaction is the key |
| `diag_rm_v2.py` | Architecture validation v2 | LN elementwise=False critical |

### Environment & Config
| Script | Purpose |
|:--|:--|
| `check_cfg.py` | Config chain resolution |
| `check_enc_grid.py` | Encoder feature map sizes |
| `check_lm.py` | Language model inspection |
| `check_text.py` | Text feat attributes |
| `check_token_boundary.py` | Token position mapping |
| `check_gate_ckpt.py` | Gate weight inspection from checkpoint |
