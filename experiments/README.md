# Diagnostic Scripts & Analysis

## Key Diagnostics

### Gradient flow analysis
- `diag_gate_grad_v2.py`: Measures gate_net gradient magnitude vs W_out/W_m
- `diag_grad_scaled.py`: Gate gradient at realistic loss scales (~10)
- Found: gate_bias_grad ~ 2.68e-3, sigmoid saturation (0.106) kills learning

### LN selectivity analysis  
- `diag_ln_bilinear_12e.py`: Pre-LN vs post-LN norm distributions
- Found: pre-LN CV=0.159, post-LN CV=0.000 — LN destroys amplitude selectivity

### R-norm mask feasibility
- `diag_r_mask.py`: Tests if |R| can separate old/new tokens
- `diag_r_mask_cpu.py`: CPU version with extracted weights
- Found: R norm does NOT separate old/new (Bilinear doesn't encode amplitude selectivity)

### Forward diagnostics
- `diag_fwd_all.py`: Tests A/B/D/E normalization variants
- Found: init_std=0.01 + RMS is the only viable path

### Encoder grid analysis
- `check_enc_grid.py`: Multi-scale feature map resolution
- Found: 4 levels (54×80, 27×40, 14×20, 7×10)

## Config comparison tools
- `check_cfg.py`, `check_text.py`, `check_lm.py`: Environment inspection
