# Gate Experiment gate-v2 (ABANDONED)

## Status: ❌ Failed

## What was attempted
Learned spatial gate: gate = sigmoid(gate_net(memory))
Used to selectively suppress Bilinear residual at old-class positions.

## Root cause of failure
- sigmoid saturation at 0.88 → gradient factor 0.106
- gate gradient ~1000x weaker than main model gradients  
- gate_net 16K params cannot learn in 12 epochs

## Key findings
- gate_bias_grad = 2.68e-3 (at loss~10) → per-step move = 1.3e-7
- Even 100x LR would only move bias 0.16 units in 12 epochs
- Non-learned threshold mask also infeasible (R norm doesn't encode selectivity)

## Decision
Gate direction abandoned. Pivot to non-learned orthogonal projection (ortho-l1).
