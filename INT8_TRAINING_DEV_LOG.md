# Pure INT8 Training Development Log

## Project Objective
**Goal:** Implement **Pure INT8 Training** (both Forward and Backward passes) for the `super-res_torchlight` project on the Frontier supercomputer (AMD MI250x).
**Primary Driver:** Maximize training throughput and minimize memory bandwidth usage by enforcing INT8 arithmetic for gradient computations ($dX$, $dW$), explicitly rejecting FP16/BF16 compromises in the backward pass.

## Strategic Direction
We are adopting an aggressive **"Pure INT8 Backward"** strategy. Unlike conventional hybrid approaches (e.g., SwitchBack) that revert to FP16 for gradients, we will strictly enforce INT8 GEMM for backpropagation.

### Core Technical Pillars
1.  **Pure INT8 GEMM for Gradients:**
    -   Compute $dX = dY \times W^T$ and $dW = X^T \times dY$ using INT8 operations.
    -   Leverage MI250x Matrix Cores for INT8 acceleration.
2.  **Integer-only Scaling (Bit-Shift):**
    -   Replace floating-point scaling factors with bit-wise shift operations (`<<`, `>>`) to maintain integer domain processing and avoid FP overhead.
    -   Dynamic range management to prevent gradient underflow/overflow.
3.  **Stochastic Rounding:**
    -   Implement stochastic rounding during quantization to preserve gradient information that would otherwise be lost due to low precision.

## Development Roadmap

### Phase 1: Prototyping `PureInt8Linear`
- [x] Analyze `src/climate_learn/models/hub/components/vit_blocks.py`.
- [x] Analyze `src/climate_learn/models/hub/components/mlp.py` and `src/climate_learn/models/hub/components/attention.py`.
- [x] Implement a custom `torch.autograd.Function` for the Linear layer (`PureInt8Matmul`).
    -   **Forward:** INT8 Input $\times$ INT8 Weight $\rightarrow$ INT32 Output.
    -   **Backward:** INT8 GradOutput $\times$ INT8 Weight/Input $\rightarrow$ INT32 Gradients.
- [x] Implement `PureInt8Linear` module that uses `PureInt8Matmul`.
- [x] Integrate Bit-Shift scaling logic within `PureInt8Matmul` (using `get_scale_shift`, `quantize_to_int8_shifted`, `dequantize_from_int8_shifted`).
- [ ] Implement (or placeholder for) Stochastic Rounding within `PureInt8Matmul`. (Currently a basic version in `quantize_to_int8_shifted`)

### Phase 2: Integration & Stability
- [x] Replace standard `nn.Linear` in `Mlp` and `Attention` blocks with `PureInt8Linear`.
- [x] Initial stability test run (Failed silently, suspecting segmentation fault or NaN explosion).
- [x] **Debugging:** Instrument `PureInt8Linear` with logs and NaN checks to pinpoint failure location.
- [ ] Verify numerical stability (check for loss crashes or stagnation).

### Phase 3: Optimization & Benchmarking
- [ ] Profile memory usage and throughput on target hardware.
- [ ] Iterate on scaling strategies if convergence issues arise.

## Log Entries

### 2025-12-09: Debugging & Critical Fixes
-   **Critical Fix:** Resolved `NameError` in `PureInt8Matmul.backward` where `bias_fp32` was not accessible. Added `ctx.has_bias` to track bias existence.
-   **Critical Fix:** Fixed precision loss bug where INT32 accumulators were cast to INT8 *before* dequantization. Changed to cast to FP32 first to preserve accumulator values during scaling.
-   **Instrumentation:** Uncommented `debug_print` logs and NaN checks to capture detailed state during the next Frontier run.

### 2025-12-09: Initiative Kick-off
-   Rejected standard QAT and Hybrid (SwitchBack) approaches.
-   Defined the "Pure INT8 Backward" requirement.
-   Identified `src/climate_learn/models/hub/components/vit_blocks.py`, `mlp.py`, and `attention.py` as primary modification targets.
-   Started drafting the `PureInt8Linear` module structure.

### 2025-12-09: `PureInt8Linear` Integration
-   Created `src/climate_learn/models/hub/components/pure_int8_linear.py` with initial `PureInt8Linear` and `PureInt8Matmul` structure.
-   Modified `src/climate_learn/models/hub/components/mlp.py` to use `PureInt8Linear`.
-   Modified `src/climate_learn/models/hub/components/attention.py` to use `PureInt8Linear`.

### 2025-12-09: Bit-Shift Scaling Implementation
-   Refined `quantize_to_int8_shifted` and `dequantize_from_int8_shifted` to use bit-shift like scaling.
-   Updated `PureInt8Matmul` to incorporate the new scaling functions.
-   Completed initial implementation for Phase 1.

### 2025-12-09: First Stability Test & Debugging
-   Submitted initial test job to Frontier.
-   Result: Job failed silently without python traceback. Ominstat report showed low GPU utilization.
-   Action: Instrumented `pure_int8_linear.py` with `debug_print` and critical NaN checks to catch silent failures in C++ land or distributed training hang.
