# mamba-metal Architecture

## Selective Scan – Associative Formulation

The classic recurrence

```
h_t = Ā_t h_{t-1} + B̄_t x_t
```

is rewritten as the affine map

```
h' = a·h + b
```

with the monoid operation

```
(a₂,b₂) ⊗ (a₁,b₁) = (a₂ a₁, a₂ b₁ + b₂)
```

Because the operator is associative, a parallel prefix-scan can be used:

1. **Thread-local** – each thread computes its own (a,b)
2. **SIMD-group scan** – Hillis-Steele or Kogge-Stone inside the SIMD group (width obtained from `threadExecutionWidth`)
3. **Threadgroup scan** – scan the SIMD-group carries in threadgroup memory
4. **Inter-tile carry** – for sequences longer than one threadgroup, a second kernel or host-side reduction stitches tiles

This yields O(log L) depth instead of O(L) while remaining numerically equivalent to the serial recurrence (within floating-point associativity differences).

## Memory Hierarchy

| Address space   | Usage                                      |
|-----------------|--------------------------------------------|
| `device`        | input/output tensors, weights, final state |
| `constant`      | small parameter structs                    |
| `threadgroup`   | SIMD-group carries, shared tiles           |
| `thread`        | per-timestep affines, A coefficients       |

## Precision Policy

- Weights may be FP16 or FP32
- Δ, B, C projections may be FP16
- **State accumulation stays in FP32** (configurable) to protect the long-running product of Ā factors
- Final output can be cast back to FP16

## Incremental Path

`ssm_step` and `causal_conv1d_step` keep persistent

```
conv_state  [B, d_inner, d_conv-1]
ssm_state   [B, d_inner, d_state]
```

so that each new token costs O(d_inner · d_state) work independent of sequence length.

## Fusion Levels

- **L0** – individual operators (silu, discretize, rmsnorm, …)
- **L1** – selective_scan, causal_conv
- **L2** – full Mamba block (projected + conv + scan + gate + out-proj)
- **L3** – single-token `mamba_step`
- **L4** – multi-layer model pipeline

## Hardware Adaptation

At pipeline creation time the runtime queries:

```
pso->threadExecutionWidth()
pso->maxTotalThreadsPerThreadgroup()
```

and chooses threadgroup sizes that are multiples of the SIMD width. No CUDA warp-size assumptions are hard-coded.
