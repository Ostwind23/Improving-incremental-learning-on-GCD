# Filter Old Pseudo-Gradient

## Status: ❌ Mixed (ep5 peak, ep6 crash)

## Architecture
Standard MLP residual + gradient filtering: zero out gradients from old-class pseudo-labels

## Results (6 epoch)
| Ep | ori_ap | new_ap | rm_norm |
|:--|:--|:--|:--|
| 1 | ~0.45 | — | 3.6 |
| 5 | 0.391 | 0.311 | 18 |
| 6 | CRASH | — | 36 |

## Key Finding
- rm_norm grows monotonically: 3.6 → 18 → 36 → CRASH
- Without LN, MLP's Linear→ReLU→Linear chain amplifies output each epoch
- filter_old_pseudo_grad slows but doesn't prevent the divergence
- This observation DIRECTLY motivated adding LayerNorm → led to LN experiments
- Also motivated replacing MLP with Bilinear (multiplicative interaction can't diverge)
