# Gate v3: Maximized Gradient (bias=0, RMS target=4.0)

## Status: ❌ Failed (gate bias moved but too slowly)

## Architecture
```
R = W_out(W_m(M) * W_t(T_pool))
R = R * (4.0 / mean(|R|))      # RMS target=4.0 (4x perturbation)
R = R * sigmoid(gate_net(M.detach()))  # detach to prevent gradient loop
M' = M + gamma * R
```
- bias=0 → sigmoid=0.5 → gate*(1-gate)=0.25 (2.4x better than 0.106)
- RMS=4.0 → 4x more perturbation → 4x stronger gradient
- memory.detach() → prevents self-referential gradient loop

## Results (1 epoch)
| | ori_ap | new_ap | gate bias |
|:--|:--|:--|:--|
| Gate v3 ep1 | 0.468 | 0.286 | -0.0022 |

- Bias moved from 0.0 to -0.0022 (visible gradient!)
- But 5.5x improvement still too slow: projected 12e bias move = 0.026
  → sigmoid(0.0→-0.026) ≈ 0.5→0.493 (essentially no change)
- Gate pushed toward CLOSING (bias negative) — old-class gradient dominates

## Final Verdict
Learned spatial gate is structurally infeasible in this architecture:
1. Detection loss gradient through 6-layer decoder → gate is ~1000x too weak
2. Sigmoid saturation kills what little gradient remains
3. Non-learned R-norm threshold also fails (R norm doesn't separate old/new)
4. Pivot to orthogonal projection (ortho-l1) — no learnable parameters needed
