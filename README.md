```text
+--------------------------------------+
|             MAMBA METAL              |
|   Native Mamba SSM on Apple Metal    |
+--------------------------------------+
```

![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Objective--C%20%7C%20MSL-blue)
![Kernels](https://img.shields.io/badge/kernels-10%20hand--written%20MSL-critical)
![LOC](https://img.shields.io/badge/lines%20of%20code-1%2C706-brightgreen)
![Precision](https://img.shields.io/badge/precision-FP32%20%7C%20FP16-blueviolet)
![Status](https://img.shields.io/badge/status-complete%20implementation-orange)

![Snapkitty Apple emblem with silver cat ears — Mamba Metal](assets/snapkitty-apple-metal.png)

---

## What is this?

Native Metal implementation of the Mamba / Selective State Space Model computational substrate. Not a wrapper around CUDA, PyTorch, or any existing Mamba extension. From-scratch implementation of the selective SSM recurrence, causal convolution, and full Mamba block written directly in Metal Shading Language with a minimal native host runtime.

```
MSL kernels → Metal compiler → Apple GPU (SIMD groups + threadgroups)
                                → associative selective scan + fused operators
```

No framework dependency. Pure Metal + C++/Objective-C runtime.

---

## Core Mathematical Primitive

Each timestep is an affine transformation of the hidden state:

```
h' = ā · h + b̄
```

Represented as the pair `T = (ā, b̄)`. Composition is associative:

```
(a₂, b₂) ⊗ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)
```

This allows the classic recurrent form `h_t = Ā_t h_{t-1} + B̄_t x_t` to be evaluated with a parallel prefix-scan (local → SIMD → threadgroup → global) instead of a purely serial loop. O(log L) depth instead of O(L).

---

## Architecture

```mermaid
flowchart TD
    subgraph "Host Runtime (C++)"
        CFG[MambaConfig<br/>d_model, d_state, d_conv, expand]
        RT[MambaRuntime<br/>pipeline creation + dispatch]
        STATE[MambaState<br/>conv_state + ssm_state]
        CAPI[C API<br/>mamba_create / forward / step]
    end

    subgraph "Metal Runtime"
        DEV[MetalDevice<br/>device + queue]
        BUF[MetalBuffer<br/>shared storage]
        PIPE[MetalPipeline<br/>PSO + threadgroup sizing]
    end

    subgraph "MSL Kernels (10)"
        K1[selective_scan<br/>FP32 + FP16]
        K2[causal_conv1d<br/>full + incremental]
        K3[discretize<br/>ZOH: Ā = exp Δ·A]
        K4[silu<br/>x·σ x]
        K5[rmsnorm]
        K6[gated_mlp<br/>SwiGLU-style]
        K7[projection<br/>linear]
        K8[prefix_scan<br/>parallel associative]
        K9[mamba_block<br/>fused scan+gate]
        K10[state_update<br/>incremental SSM step]
    end

    subgraph "CPU Reference"
        REF[selective_scan_cpu<br/>scalar verification oracle]
    end

    CFG --> RT
    RT --> DEV & PIPE
    PIPE --> K1 & K2 & K3 & K4 & K5 & K6 & K7 & K8 & K9 & K10
    DEV --> BUF
    STATE --> BUF
    CAPI --> RT
    K1 -.->|verify| REF
```

## Selective Scan Pipeline

```mermaid
flowchart LR
    IN([Input x, Δ, B, C]) --> DISC[Discretize<br/>Ā = exp Δ·A<br/>B̄ = ZOH B·x]
    DISC --> AFFINE[Build Affine Pairs<br/>T_t = Ā_t, B̄_t·x_t]
    AFFINE --> SIMD[SIMD-Group Scan<br/>Hillis-Steele]
    SIMD --> TG[Threadgroup Scan<br/>carry propagation]
    TG --> TILE[Inter-Tile Carry<br/>global stitching]
    TILE --> OUTPUT[Output<br/>y_t = C·h_t + D·x_t]
    OUTPUT --> STATE_WB[State Writeback<br/>final h for next step]
```

## Kernel Levels

```mermaid
graph TD
    subgraph "L0 — Primitives"
        L0A[discretize]
        L0B[silu]
        L0C[rmsnorm]
        L0D[prefix_scan]
        L0E[projection]
    end

    subgraph "L1 — Operator-Fused"
        L1A[selective_scan<br/>FP32 + FP16]
        L1B[causal_conv1d<br/>full + step]
    end

    subgraph "L2 — Block-Fused"
        L2A[mamba_block<br/>scan + gate]
        L2B[gated_mlp<br/>SwiGLU]
    end

    subgraph "L3 — Incremental"
        L3A[state_update<br/>single-token SSM step]
    end

    subgraph "L4 — Pipeline"
        L4A[MambaRuntime<br/>host orchestration]
    end

    L0A & L0B & L0C & L0D & L0E --> L1A & L1B
    L1A & L1B --> L2A & L2B
    L2A & L2B --> L3A
    L3A --> L4A

    style L0A fill:#1a1a2e,stroke:#60a5fa,color:#e2e8f0
    style L0B fill:#1a1a2e,stroke:#60a5fa,color:#e2e8f0
    style L0C fill:#1a1a2e,stroke:#60a5fa,color:#e2e8f0
    style L0D fill:#1a1a2e,stroke:#60a5fa,color:#e2e8f0
    style L0E fill:#1a1a2e,stroke:#60a5fa,color:#e2e8f0
    style L1A fill:#0d3320,stroke:#22c55e,color:#e2e8f0
    style L1B fill:#0d3320,stroke:#22c55e,color:#e2e8f0
    style L2A fill:#0d3320,stroke:#22c55e,color:#e2e8f0
    style L2B fill:#0d3320,stroke:#22c55e,color:#e2e8f0
    style L3A fill:#0d3320,stroke:#22c55e,color:#e2e8f0
    style L4A fill:#0d3320,stroke:#22c55e,color:#e2e8f0
```

---

## Memory Hierarchy

| Address space | Usage |
|--------------|-------|
| `device` | Input/output tensors, weights, final state |
| `constant` | Small parameter structs (ScanParams) |
| `threadgroup` | SIMD-group carries, shared tiles |
| `thread` | Per-timestep affines, A coefficients |

## Precision Policy

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Weights | FP16 or FP32 | Configurable |
| Δ, B, C projections | FP16 | Bandwidth reduction |
| **State accumulation** | **FP32** | Protects long-running product of Ā factors |
| Output | FP16 castable | Optional downcast |

---

## C API

```c
MambaConfig cfg = {
    .d_model = 512,
    .d_state = 16,
    .d_conv  = 4,
    .expand  = 2,
    .dtype   = MAMBA_F16
};

MambaModel *model = mamba_create(&cfg);
mamba_load_weights(model, weight_blob, weight_size);

// Full sequence
mamba_forward(model, input, output, batch, seq_len);

// Incremental (persistent state)
MambaStateHandle *state = mamba_create_state(model, batch);
for (int t = 0; t < seq_len; ++t)
    mamba_step(model, state, &input[t], &output[t]);
mamba_destroy_state(state);
mamba_destroy(model);
```

---

## Project Structure

```
mamba-metal/
├── Sources/
│   ├── Kernels/                  10 hand-written MSL shaders
│   │   ├── selective_scan.metal  Associative scan (FP32 + FP16) — 360 LOC
│   │   ├── causal_conv1d.metal   Full-sequence + incremental — 100 LOC
│   │   ├── mamba_block.metal     Fused scan + gate — 128 LOC
│   │   ├── rmsnorm.metal         RMS normalization — 81 LOC
│   │   ├── state_update.metal    Single-token SSM step — 57 LOC
│   │   ├── projection.metal      Linear projection — 49 LOC
│   │   ├── gated_mlp.metal       SwiGLU-style gate — 41 LOC
│   │   ├── prefix_scan.metal     Parallel associative scan — 40 LOC
│   │   ├── silu.metal            SiLU activation — 25 LOC
│   │   └── discretize.metal      ZOH discretization — 19 LOC
│   │
│   ├── Mamba/                    High-level runtime
│   │   ├── MambaConfig.hpp       Model configuration
│   │   ├── MambaState.hpp        Persistent inference state
│   │   ├── MambaRuntime.hpp      Runtime + C API — 87 LOC
│   │   └── MambaRuntime.cpp      Implementation — 181 LOC
│   │
│   ├── Runtime/                  Metal host runtime
│   │   ├── MetalDevice.hpp       Device + command queue
│   │   ├── MetalDevice.cpp       Device management — 58 LOC
│   │   ├── MetalBuffer.hpp       Shared storage buffers
│   │   ├── MetalBuffer.cpp       Buffer management — 27 LOC
│   │   ├── MetalPipeline.hpp     PSO + threadgroup sizing
│   │   └── MetalPipeline.cpp     Pipeline management — 39 LOC
│   │
│   └── CPURef/                   Verification oracle
│       ├── selective_scan_cpu.hpp CPU reference header — 39 LOC
│       └── selective_scan_cpu.cpp Scalar reference — 48 LOC
│
├── Tests/
│   ├── test_selective_scan.cpp   Scan correctness tests — 36 LOC
│   └── test_numerical.cpp        Associativity / zero / impulse — 70 LOC
│
├── Benchmarks/
│   └── bench_scan.cpp            Scan throughput microbenchmark — 47 LOC
│
├── Examples/
│   └── simple_forward.c          Minimal C API usage — 42 LOC
│
└── docs/
    └── architecture.md           Scan formulation + memory hierarchy
```

---

## Design Principles

| Principle | Enforcement |
|-----------|-------------|
| No framework dependency | Pure Metal + C++/ObjC runtime |
| Associative scan formulation | Parallel prefix over affine monoid |
| GPU-aware parallel scan | SIMD groups + threadgroup memory + hierarchical composition |
| Fused kernels | Minimize global memory traffic |
| Incremental inference | Persistent conv_state + ssm_state per batch |
| Configurable precision | FP32/FP16 weights; state always FP32 |
| Hardware adaptation | Runtime query of threadExecutionWidth + maxTotalThreadsPerThreadgroup |

---

## Build

Requires macOS + Xcode with Metal support.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## License

See [LICENSE](LICENSE) for details.
