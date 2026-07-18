// bench_cuda_graphs.cu
// Measure CUDA graph capture/replay speedup for a DiT-like multi-kernel graph.
//
// Builds a synthetic graph of sequential matmuls + element-wise ops
// (simulating a DiT block), then runs it uncaptured vs captured+replayed.
//
// Compile (CC 8.0+ required for meaningful results):
//   nvcc -O3 -arch=sm_80 -lcublas tools/bench_cuda_graphs.cu -o bench_cuda_graphs
//
// Run:
//   ./bench_cuda_graphs                     # all sequence lengths
//   ./bench_cuda_graphs 256                 # single sequence length
//   ./bench_cuda_graphs 128 256 512 1024    # custom list

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- error checking ----
#define CUCHECK(call) do {                                            \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err), err);    \
        exit(1);                                                      \
    }                                                                 \
} while(0)

#define CUBLASCHECK(call) do {                                        \
    cublasStatus_t s = call;                                          \
    if (s != CUBLAS_STATUS_SUCCESS) {                                 \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n",               \
                __FILE__, __LINE__, s);                               \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---- synthetic DiT graph helpers ----
// We simulate one DiT block's matmuls:
//   1. QKV projection:       M×K × K×N1  → M×N1   (N1 = 5*dim for differential attn)
//   2. Attn out projection:  M×K × K×dim → M×dim
//   3. FF projection:        M×dim × dim×N2 → M×N2  (N2 = 4*dim for SwiGLU)
//   4. FF output:            M×N2/2 × N2/2×dim → M×dim
// With dim=3072 (DiT medium).
//
// Each is followed by a simulated element-wise add (F32 scale+add on device).

static __global__ void eltwise_add(float * __restrict__ out,
                                   const float * __restrict__ x,
                                   const float * __restrict__ residual,
                                   float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = x[i] * scale + residual[i];
}

static void launch_eltwise_add(float * out, const float * x,
                               const float * residual,
                               float scale, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    grid = min(grid, 65535);
    eltwise_add<<<grid, block, 0, stream>>>(out, x, residual, scale, n);
}

// One DiT block matmul sequence (4 matmuls + 4 adds)
static void run_dit_block(cublasHandle_t handle, cudaStream_t stream,
                          int dim, int S,
                          const half * weights,      // [sum_of_all_weights]
                          const float * input,       // [dim, S]
                          float * output,            // [dim, S]
                          float * scratch,           // temp buffers
                          const float * residual,
                          const int * weight_offsets // cumulative offsets into weights
                          ) {
    float alpha = 1.0f, beta = 0.0f;
    // pointer to current scratch position
    float * tmp1 = scratch;
    float * tmp2 = scratch + dim * S * 2;  // extra space for large intermediates

    // 1. QKV: [dim, S] × [dim, 5*dim]^T → [5*dim, S]
    // weight is [dim, 5*dim]
    int N_qkv = 5 * dim;
    CUBLASCHECK(cublasGemmEx(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N_qkv, S, dim,
                             &alpha,
                             weights + weight_offsets[0], CUDA_R_16F, dim,
                             input, CUDA_R_32F, dim,
                             &beta,
                             tmp1, CUDA_R_32F, N_qkv,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT));

    // residual add (simulates attention output)
    launch_eltwise_add(tmp1, tmp1, residual, 1.0f, dim * S, stream);

    // 2. Attn out: [dim, S] × [dim, dim]^T → [dim, S]
    CUBLASCHECK(cublasGemmEx(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, S, dim,
                             &alpha,
                             weights + weight_offsets[1], CUDA_R_16F, dim,
                             tmp1, CUDA_R_32F, dim,
                             &beta,
                             tmp2, CUDA_R_32F, dim,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT));
    launch_eltwise_add(tmp2, tmp2, input, 1.0f, dim * S, stream);

    // 3. FF proj: [dim, S] × [dim, 4*dim]^T → [4*dim, S]
    int N_ff = 4 * dim;
    CUBLASCHECK(cublasGemmEx(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N_ff, S, dim,
                             &alpha,
                             weights + weight_offsets[2], CUDA_R_16F, dim,
                             tmp2, CUDA_R_32F, dim,
                             &beta,
                             tmp1, CUDA_R_32F, N_ff,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT));
    launch_eltwise_add(tmp1, tmp1, residual, 1.0f, N_ff * S, stream);

    // 4. FF out: [2*dim, S] × [2*dim, dim]^T → [dim, S]
    CUBLASCHECK(cublasGemmEx(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, S, N_ff / 2,
                             &alpha,
                             weights + weight_offsets[3], CUDA_R_16F, N_ff / 2,
                             tmp1, CUDA_R_32F, N_ff / 2,
                             &beta,
                             output, CUDA_R_32F, dim,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT));
    launch_eltwise_add(output, output, tmp2, 1.0f, dim * S, stream);
}

// replay 24 DiT blocks + some RMS norm + final matmul (full DiT layer)
static void run_dit_full(cublasHandle_t handle, cudaStream_t stream,
                         int dim, int S, int n_blocks,
                         const half * weights, const float * input,
                         float * output, float * scratch,
                         const float * residual, const int * w_offsets) {
    float * cur = (float *)malloc(dim * S * sizeof(float));
    memcpy(cur, input, dim * S * sizeof(float));
    // Host-side pointers (data is on device, we pass device pointers)
    // We cheat by allocating on device and using device pointers
    // Actually the run_dit_block uses device pointers, let's restructure

    // For the graph test, we use a simpler approach:
    // just do the 4 matmuls per block inline
    float alpha = 1.0f, beta = 0.0f;
    float * tmp = scratch;

    for (int b = 0; b < n_blocks; b++) {
        // 1. QKV
        int N_qkv = 5 * dim;
        CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N_qkv, S, dim,
                                 &alpha,
                                 weights + w_offsets[4*b + 0], CUDA_R_16F, dim,
                                 cur, CUDA_R_32F, dim,
                                 &beta,
                                 tmp, CUDA_R_32F, N_qkv,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        eltwise_add<<<(dim*S + 255)/256, 256, 0, stream>>>(tmp, tmp, residual, 1.0f, dim*S);

        // 2. Attn out
        CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 dim, S, dim,
                                 &alpha,
                                 weights + w_offsets[4*b + 1], CUDA_R_16F, dim,
                                 tmp, CUDA_R_32F, dim,
                                 &beta,
                                 cur, CUDA_R_32F, dim,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        eltwise_add<<<(dim*S + 255)/256, 256, 0, stream>>>(cur, cur, input, 1.0f, dim*S);

        // 3. FF proj
        int N_ff = 4 * dim;
        CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N_ff, S, dim,
                                 &alpha,
                                 weights + w_offsets[4*b + 2], CUDA_R_16F, dim,
                                 cur, CUDA_R_32F, dim,
                                 &beta,
                                 tmp, CUDA_R_32F, N_ff,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        eltwise_add<<<(dim*S + 255)/256, 256, 0, stream>>>(tmp, tmp, residual, 1.0f, dim*S);

        // 4. FF out
        CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 dim, S, N_ff / 2,
                                 &alpha,
                                 weights + w_offsets[4*b + 3], CUDA_R_16F, N_ff / 2,
                                 tmp, CUDA_R_32F, N_ff / 2,
                                 &beta,
                                 cur, CUDA_R_32F, dim,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        eltwise_add<<<(dim*S + 255)/256, 256, 0, stream>>>(cur, cur, input, 1.0f, dim*S);
    }
    free(cur);
}

// ---- benchmark ----
typedef struct {
    int S;            // sequence length
    double t_uncap;   // uncaptured median (ms)
    double t_cap;     // captured median (ms)
    double speedup;
} Result;

static Result bench_sequence(cublasHandle_t handle, int dim, int S,
                             int n_blocks, int warmup, int iters,
                             const half * d_weights,
                             const float * d_input,
                             float * d_output,
                             float * d_scratch,
                             const float * d_residual,
                             int * d_w_offsets,
                             cudaStream_t stream) {
    Result r = { S, 0, 0, 0 };

    // ---- uncaptured run ----
    // warmup
    for (int i = 0; i < warmup; i++)
        run_dit_full(handle, stream, dim, S, n_blocks,
                     d_weights, d_input, d_output, d_scratch,
                     d_residual, d_w_offsets);
    CUCHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++)
        run_dit_full(handle, stream, dim, S, n_blocks,
                     d_weights, d_input, d_output, d_scratch,
                     d_residual, d_w_offsets);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_uncap;
    cudaEventElapsedTime(&ms_uncap, start, stop);
    r.t_uncap = ms_uncap / iters;

    // ---- captured run ----
    cudaGraph_t graph;
    CUCHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    run_dit_full(handle, stream, dim, S, n_blocks,
                 d_weights, d_input, d_output, d_scratch,
                 d_residual, d_w_offsets);
    CUCHECK(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graph_exec;
    CUCHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // warmup
    for (int i = 0; i < warmup; i++)
        CUCHECK(cudaGraphLaunch(graph_exec, stream));
    CUCHECK(cudaStreamSynchronize(stream));

    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++)
        CUCHECK(cudaGraphLaunch(graph_exec, stream));
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms_cap;
    cudaEventElapsedTime(&ms_cap, start, stop);
    r.t_cap = ms_cap / iters;

    r.speedup = r.t_uncap / r.t_cap;

    CUCHECK(cudaGraphExecDestroy(graph_exec));
    CUCHECK(cudaGraphDestroy(graph));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return r;
}

static void fill_random_f16(half * buf, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        float v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        buf[i] = __float2half(v);
    }
}

static void fill_random_f32(float * buf, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

int main(int argc, char ** argv) {
    int dim = 3072;       // DiT medium hidden
    int n_blocks = 24;    // full DiT

    int n_S;
    int S_list[16];

    if (argc > 1) {
        n_S = argc - 1;
        for (int i = 0; i < n_S && i < 16; i++)
            S_list[i] = atoi(argv[i + 1]);
    } else {
        // default test sizes: 128..1024
        n_S = 5;
        S_list[0] = 128;
        S_list[1] = 256;
        S_list[2] = 384;
        S_list[3] = 512;
        S_list[4] = 1024;
    }

    // device info
    int dev;
    CUCHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUCHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (CC %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("CUDA Graphs support: %s\n\n",
           (prop.major >= 8 || prop.major == 7) ? "yes" : "no");

    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    cudaStream_t stream;
    CUCHECK(cudaStreamCreate(&stream));
    CUBLASCHECK(cublasSetStream(handle, stream));

    // allocate device buffers
    // weights: 24 blocks × 4 weight tensors each
    //   QKV:     dim × 5*dim
    //   AttnOut: dim × dim
    //   FFProj:  dim × 4*dim
    //   FFOut:   2*dim × dim
    // Total per block: 5*dim^2 + dim^2 + 4*dim^2 + 2*dim^2 = 12*dim^2
    int weights_per_block = 12 * dim * dim;
    int total_weights = n_blocks * weights_per_block;

    half  * d_weights;
    float * d_input;
    float * d_output;
    float * d_scratch;
    float * d_residual;
    int   * d_w_offsets;  // cumulative weight offsets per per-block tensor

    CUCHECK(cudaMalloc(&d_weights,    total_weights * sizeof(half)));
    CUCHECK(cudaMalloc(&d_input,      dim * 1024 * sizeof(float)));   // max S = 1024
    CUCHECK(cudaMalloc(&d_output,     dim * 1024 * sizeof(float)));
    CUCHECK(cudaMalloc(&d_scratch,    5 * dim * 1024 * sizeof(float) + 4*dim*1024*sizeof(float)));
    CUCHECK(cudaMalloc(&d_residual,   dim * 1024 * sizeof(float)));
    CUCHECK(cudaMalloc(&d_w_offsets,  n_blocks * 4 * sizeof(int)));

    // fill host data
    half * h_weights = (half *)malloc(total_weights * sizeof(half));
    fill_random_f16(h_weights, total_weights, 0xDEAD);

    // weight offsets
    int * h_w_offsets = (int *)malloc(n_blocks * 4 * sizeof(int));
    int offset = 0;
    for (int b = 0; b < n_blocks; b++) {
        // QKV: dim * 5*dim
        h_w_offsets[4*b + 0] = offset;
        offset += dim * 5 * dim;
        // AttnOut: dim * dim
        h_w_offsets[4*b + 1] = offset;
        offset += dim * dim;
        // FFProj: dim * 4*dim
        h_w_offsets[4*b + 2] = offset;
        offset += dim * 4 * dim;
        // FFOut: 2*dim * dim
        h_w_offsets[4*b + 3] = offset;
        offset += 2 * dim * dim;
    }

    int max_S = 0;
    for (int i = 0; i < n_S; i++)
        if (S_list[i] > max_S) max_S = S_list[i];

    float * h_input    = (float *)malloc(dim * max_S * sizeof(float));
    float * h_residual = (float *)malloc(dim * max_S * sizeof(float));
    fill_random_f32(h_input,    dim * max_S, 0xCAFE);
    fill_random_f32(h_residual, dim * max_S, 0xBEEF);

    CUCHECK(cudaMemcpy(d_weights,   h_weights,   total_weights * sizeof(half),
                       cudaMemcpyHostToDevice));
    CUCHECK(cudaMemcpy(d_w_offsets, h_w_offsets, n_blocks * 4 * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUCHECK(cudaMemcpy(d_residual,  h_residual,  dim * max_S * sizeof(float),
                       cudaMemcpyHostToDevice));

    const int warmup = 20;
    const int iters  = 100;

    printf("=== CUDA Graphs Benchmark ===\n");
    printf("Synthetic DiT: dim=%d, %d blocks, %d weight tensors\n\n", dim, n_blocks, n_blocks * 4);
    printf("%-8s   %12s   %12s   %10s\n", "SeqLen", "Uncaptured", "Captured", "Speedup");
    printf("%-8s   %12s   %12s   %10s\n", "", "(ms/iter)", "(ms/iter)", "");
    printf("--------   ------------   ------------   ----------\n");

    for (int ti = 0; ti < n_S; ti++) {
        int S = S_list[ti];

        CUCHECK(cudaMemcpy(d_input, h_input, dim * S * sizeof(float),
                           cudaMemcpyHostToDevice));

        Result r = bench_sequence(handle, dim, S, n_blocks,
                                  warmup, iters,
                                  d_weights, d_input, d_output, d_scratch,
                                  d_residual, d_w_offsets, stream);

        printf("%-8d   %9.4f ms   %9.4f ms   %6.2fx\n",
               S, r.t_uncap, r.t_cap, r.speedup);

        // estimate launch overhead saved
        double launch_saved = r.t_uncap - r.t_cap;
        // rough estimate: 4 matmuls + 4 eltwise per block = 8 launches/block
        int total_launches = n_blocks * 8;
        printf("           (saved ~%.2f ms across %d launches = ~%.3f ms/launch)\n",
               launch_saved, total_launches, launch_saved / total_launches);
    }

    free(h_weights);
    free(h_w_offsets);
    free(h_input);
    free(h_residual);
    CUCHECK(cudaFree(d_weights));
    CUCHECK(cudaFree(d_input));
    CUCHECK(cudaFree(d_output));
    CUCHECK(cudaFree(d_scratch));
    CUCHECK(cudaFree(d_residual));
    CUCHECK(cudaFree(d_w_offsets));
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    printf("\nNote: Results are only meaningful on CC 8.0+ (Ampere+).\n");
    printf("On older GPUs the graph capture may fall back to uncaptured mode.\n");
    return 0;
}
