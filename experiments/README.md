# Gate v1: Original Old-Class Gate (bias=2.0, LN after gate)

## Status: ❌ Failed (ori_ap collapsed from 0.465 → 0.298)

## Architecture
```
R = W_out(W_m(M) * W_t(T_pool))
R = R * sigmoid(gate_net(M))    # gate BEFORE LN
R = LN(R)                         # LN AFTER gate
M' = M + gamma * R
```

## Bugs Identified
1. **LN after gate**: LN normalizes per-token to norm=16, ERASING any gate effect.
   Gate=0.88 → LN → norm=16. Gate=0.01 → LN → norm=16. Gate is useless.
2. **init_std=0.1**: Without LN, R norm explodes to 448.
3. **bias=2.0**: Sigmoid(2.0)=0.88, saturated. Gradient factor = 0.88×0.12 = 0.106.

## Results (3 epoch)
| | ori_ap | new_ap | mAP |
|:--|:--|:--|:--|
| Gate v1 | 0.298 | 0.317 | 0.300 |
| Bilinear+LN (no gate) | 0.466 | 0.339 | 0.450 |

## Key Insight
- The gate had ZERO effect due to LN ordering.
- This led to fixing LN→gate order in v2.
- Also exposed the init_std problem that led to RMS.
