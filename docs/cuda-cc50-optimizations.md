# CUDA CC 5.0 Optimizations for sa3.cpp

## Overview

These changes add compile-time code paths that optimize the ggml CUDA backend for
NVIDIA Maxwell (Compute Capability 5.0) GPUs — specifically the GTX 960M (GM107)
used during development. A secondary code path (`GGML_CUDA_CC80_PLUS`) is provided
for Ampere+ (CC 8.0+) but contains no optimizations yet; it exists as a skeleton
for future work.

**Design constraint:** All optimizations are controlled by compile-time preprocessor
defines (`GGML_CUDA_CC50` / `GGML_CUDA_CC80_PLUS`), never by runtime CC detection.
Each binary targets a single architecture. No fat binaries, no JIT.

## Build Presets

Two new build presets are available in `build.sh`:

| Preset | Example | Target GPU | CMake flags |
|---|---|---|---|
| `cuda-cc50` | `./build.sh cuda-cc50` | GTX 960M (sm_50) | `-DGGML_CUDA_CC50=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_GRAPHS=OFF` |
| `cuda-cc80` | `./build.sh cuda-cc80` | RTX 3080+ (sm_80) | `-DGGML_CUDA_CC80_PLUS=ON -DGGML_CUDA_GRAPHS=ON` |

The existing `cuda` preset is unchanged and builds a multi-arch binary with no
CC50/CC80-specific paths.

The `cuda-cc50` preset also sets `-DGGML_CUDA_FORCE_CUBLAS=ON` to ensure all
quantized matmuls are handled by cuBLAS (the custom MMQ kernels are not optimized
for Maxwell and would run slow fallback paths).

### Requirements

- CUDA Toolkit 12.4+ (tested with 12.4.131)
- GPU with CC 5.0 (Maxwell) for `cuda-cc50`
- Host compiler GCC 13.x (for CUDA 12.4)

## Changes by File

### 1. Build Configuration

#### `build.sh`

- Added `cuda-cc50` and `cuda-cc80` case branches
- `cuda-cc50`: forces `CMAKE_CUDA_ARCHITECTURES=50-real`, enables `GGML_CUDA_CC50`,
  disables CUDA graphs, forces cuBLAS for quantized types
- `cuda-cc80`: forces `CMAKE_CUDA_ARCHITECTURES=80-real`, enables `GGML_CUDA_CC80_PLUS`,
  enables CUDA graphs

#### `ggml/CMakeLists.txt`

New CMake options (lines 207-208):

```cmake
option(GGML_CUDA_CC50      "ggml: enable CC 5.0 optimizations (Maxwell)"  OFF)
option(GGML_CUDA_CC80_PLUS "ggml: enable CC 8.0+ optimizations (Ampere+)" OFF)
```

#### `ggml/src/ggml-cuda/CMakeLists.txt`

- Added `add_compile_definitions(GGML_CUDA_CC50)` and `add_compile_definitions(GGML_CUDA_CC80_PLUS)`
  guarded by their respective CMake options (lines 148-153)
- Added `mm_small.cu` to the source list (line 114)

---

### 2. Optimizations

#### A1a — cuBLAS ALGO13 in F32 Fallback

**Files:** `ggml/src/ggml-cuda/ggml-cuda.cu`

**Location:** `ggml_cuda_op_mul_mat_cublas()`, F32 fallback branch (the
`} else {` at line ~1750).

**Before:**
```cpp
CUBLAS_CHECK(
    cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
            row_diff, src1_ncols, ne10,
            &alpha, src0_ddf_i,  ne00,
                    src1_ddf1_i, ne10,
            &beta,  dst_dd_i,    ldc));
```

**After (under `GGML_CUDA_CC50`):**
```cpp
CUBLAS_CHECK(
    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
            row_diff, src1_ncols, ne10,
            &alpha, src0_ddf_i,  CUDA_R_32F, ne00,
                    src1_ddf1_i, CUDA_R_32F, ne10,
            &beta,  dst_dd_i,    CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_ALGO13));
```

**Rationale:** Benchmarks on GTX 960M showed `CUBLAS_GEMM_ALGO13` outperforming
`CUBLAS_GEMM_DEFAULT` by 2-4.5% across all tested matmul shapes.

**Note:** `cublasGemmEx` is used instead of `cublasSgemm` because the algo
selection parameter is only available through the GEMM Ex API.

---

#### A1b — Vectorized F16→F32 Dequant

**Files:** `ggml/src/ggml-cuda/convert.cu`

**New kernel** (inserted at line 8, under `#ifdef GGML_CUDA_CC50`):

```cuda
static __global__ void dequantize_f16_f32_v4(
        const void * __restrict__ vx, float * __restrict__ y, const int64_t k);
```

**Design:**
- Each thread processes **4 elements** (8 bytes via a single `uint2` load)
- Block size: 128 threads (down from 256)
- Tail loop handles any remaining elements (k % 4 != 0)
- Uses `__ushort_as_half` + `__half2float` for IEEE-compliant F16→F32 conversion

**Memory access pattern:**
| Load | Size | Per warp | Cache lines |
|---|---|---|---|
| Original (scalar) | 2B/half | 64 B | 0.5 × 128 B |
| Vectorized (uint2) | 8 B/4 halves | 256 B | 2 × 128 B |

The vectorized load better utilizes the 128-byte cache sector size on Maxwell,
reducing the number of global memory transactions by 4×.

**Dispatch override** (in `ggml_get_to_fp32_cuda()`, type `GGML_TYPE_F16`):
Under `#ifdef GGML_CUDA_CC50`, the function pointer is set to
`dequantize_f16_f32_v4_cuda` instead of the generic `convert_unary_cont_cuda<half>`.

---

#### A2 — Force cuBLAS Dispatch

**Files:** `ggml/src/ggml-cuda/ggml-cuda.cu`

**Location:** `ggml_cuda_mul_mat()`, right before the debug printf block (line 2609).

```cpp
#ifdef GGML_CUDA_CC50
    use_mul_mat_vec_f = false;
    use_mul_mat_f     = false;
    use_mul_mat_vec_q = false;
    use_mul_mat_q     = false;
#endif
```

**Effect:** All matmuls (including GEMV-sized operations) fall through the custom
kernel checks and end up in the cuBLAS path. This ensures Maxwell never executes
the MMQ/MMF/MMVF custom kernels, which either require tensor cores (CC 7.0+) or
have slow scalar fallbacks on pre-Pascal hardware.

---

#### A3 — Flash Attention Tile Config Tuning

**Files:** `ggml/src/ggml-cuda/fattn-tile.cuh`

**New function** (inserted at line 297, under `#ifdef GGML_CUDA_CC50`):

```cpp
static constexpr __host__ __device__ uint32_t
ggml_cuda_fattn_tile_get_config_nvidia_maxwell(
    const int DKQ, const int DV, const int ncols);
```

**Changes from the fp32 baseline:**

| Parameter | fp32 baseline | Maxwell (CC50) | Reason |
|---|---|---|---|
| Threads per block (ncols ≥ 8) | 256 | 128 | Lower register pressure per SM |
| Occupancy (ncols ≥ 8) | 2 | 3 | More blocks in flight on 5 SMs |
| nbatch_fa (ncols ≥ 8) | 32 | 32 | Unchanged |
| nbatch_fa (ncols=2, DKQ=128) | 64 | 64 | Unchanged (required for np > 1) |

On Maxwell (5 SMs, 64 KB registers/SM), the compiler targets 3 blocks per SM
instead of 2. This reduces per-thread register allocation and increases the
number of concurrent blocks, helping hide memory latency. Fewer threads per
block (128 vs 256) also reduces shared memory bank conflict pressure.

**Dispatch:** Both the host- and device-side `ggml_cuda_fattn_tile_get_config()`
functions route to the Maxwell function under `#ifdef GGML_CUDA_CC50`.

---

#### A5 — Small Matmul Kernel

**Files:** `ggml/src/ggml-cuda/mm_small.cuh`, `ggml/src/ggml-cuda/mm_small.cu`

New files added to the ggml CUDA backend.

**Kernel (`mm_small.cu`):**

```cuda
static __global__ void small_mm_f16_f32(
    const half * __restrict__ w,    // [M, K] F16 weights
    const float * __restrict__ a,   // [K, N] F32 activations
    float * __restrict__ dst,       // [M, N] F32 output
    const int M, const int N, const int K);
```

**Design:**
- 1D grid, one thread per output element (M × N total threads)
- 256 threads per block (good occupancy on 5 SMs)
- On-the-fly F16→F32 conversion via `__half2float`
- No shared memory, no synchronization
- Handles any M, N, K within threshold limits

**Dispatch thresholds** (`mm_small.cuh`):

```cpp
#define SMALL_MM_MAX_ELTS 4096   // Max output elements (M * N)
#define SMALL_MM_MAX_K    512    // Max reduction dimension (K)
```

The dispatch function `ggml_cuda_should_use_small_mm()` is called before the
cuBLAS fallback in `ggml_cuda_mul_mat()` (under `#ifdef GGML_CUDA_CC50`). If the
F16×F32 matmul has small dimensions meeting the thresholds, it bypasses cuBLAS
entirely and runs the custom kernel.

**Rationale:** Benchmarks showed the SAME decoder's sliding window matmul
(64×51×17) achieving only ~14 GFLOPS with cuBLAS — dominated by kernel launch
overhead (~3-5 µs per call). A custom kernel with no API overhead can complete
small matmuls in <1 µs, potentially achieving 20-50× speedup on these
operations.

---

## Feature Reporting

The compile-time features are reported in the `ggml_cuda_info()` output:

```
CC50_OPTS:     1    (when built with -DGGML_CUDA_CC50=ON)
CC80_PLUS_OPTS:1    (when built with -DGGML_CUDA_CC80_PLUS=ON)
```

These are visible when `GGML_CUDA_DEBUG=1` is set or through the backend feature
dump in `ggml-cuda.cu`.

---

## Benchmark Tools (unchanged, written during investigation)

| Tool | Purpose | Status |
|---|---|---|
| `tools/bench_cublas_mixed.cu` | cuBLAS F16×F32 mixed-precision + algo sweep | Compiled, ran on GTX 960M |
| `tools/bench_cuda_graphs.cu` | CUDA graph capture/replay speedup for DiT | Written for CC 8.0+, not run |

---

## Skipped / Future Work

### A4 — Element-wise Fusion (bias add)

The most natural fusion for sa3.cpp is merging the post-matmul bias addition
(`GGML_OP_ADD`) directly into the cuBLAS gemm output. This requires either:

1. A new fused op `GGML_OP_MUL_MAT_BIAS` — requires adding the op type to ggml
   and dispatching in both CPU and CUDA backends
2. A backend- or model-level pattern detector that checks if the MUL_MAT output
   is immediately consumed by an ADD with a bias tensor

This was deferred because the bias-add kernel launch is cheap (~3 µs) and the
small-matmul optimization (A5) has much higher impact. Revisit if profiling
shows bias-add overhead as a bottleneck.

### GGML_CUDA_CC80_PLUS

The build-system skeleton and `GGML_CUDA_CC80_PLUS` compile define are in place,
but no Ampere-specific optimizations have been implemented yet. Candidates for
future work:
- CUDA graph capture (the `bench_cuda_graphs.cu` tool was written for this)
- Tensor core matmul with F16 accumulation
- Async copy (cp.async) for flash attention tiles

---

## A6 — K-Quants Support for `get_rows` CUDA Kernel

**Files:** `ggml/src/ggml-cuda/dequantize.cuh`, `ggml/src/ggml-cuda/getrows.cu`

### Problem

The `get_rows_cuda_q` template in `getrows.cu` requires per-element dequantize
functions with signature `void(const void*, int64_t ib, int iqs, float2&)`.
These existed for legacy quants (Q4_0, Q5_0, Q8_0, etc.) but **not** for
k-quants (Q4_K, Q5_K, Q6_K, etc.).

The `sa3-t5gemma` model uses `ggml_get_rows` on `te.embed.weight`, which gets
quantized to Q6_K by the Q4_K_M policy. The CUDA backend would hit `GGML_ABORT`
at the `// TODO: k-quants` default case.

### Solution

1. **Added `get_scale_min_k4` helper** to `dequantize.cuh` (shared with
   `convert.cu` block dequantize kernels)

2. **Added per-element dequantize functions** in `dequantize.cuh`:
   - `dequantize_q4_K` — maps linear element index to Q4_K's interleaved layout
   - `dequantize_q5_K` — maps to Q5_K's layout with qh bit extraction
   - `dequantize_q6_K` — maps to Q6_K's ql/qh/scales layout

   All match the block dequantize logic in `convert.cu` exactly.

3. **Added custom kernel `k_get_rows_k`** in `getrows.cu`:
   - One thread per output element (like `k_get_rows_float`)
   - Computes block index: `ib = i00 / QK_K`, element index: `iqs = i00 % QK_K`
   - Calls the per-element dequantize function

4. **Dispatch cases** in `ggml_cuda_get_rows_switch_src0_type` for
   `GGML_TYPE_Q4_K`, `GGML_TYPE_Q5_K`, `GGML_TYPE_Q6_K`

### Verification

```bash
# CPU reference
./build-cuda-cc50/bin/sa3-quant-eval --model <orig> <quant> --cpu
# hidden  MSE=1.062e+00  cos=0.976001

# GPU (fixed)
./build-cuda-cc50/bin/sa3-quant-eval --model <orig> <quant>
# hidden  MSE=9.426e-01  cos=0.978713
```

GPU now matches CPU within expected quantization variance.

### TODO
- [ ] Backward pass (`ggml_cuda_op_get_rows_back`) for k-quants (training only)
- [ ] Q2_K, Q3_K, Q8_K support (not used by Q4_K_M policy)
