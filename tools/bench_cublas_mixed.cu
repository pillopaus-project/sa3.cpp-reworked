// bench_cublas_mixed.cu
// Compare cuBLAS mixed-precision matmul (F16 weights x F32 activations, F32 compute)
// vs explicit F16->F32 dequant + cublasSgemm (F32 x F32, F32 compute).
//
// Compile:
//   nvcc -O3 -arch=sm_50 -lcublas tools/bench_cublas_mixed.cu -o bench_cublas_mixed
//
// Run:
//   ./bench_cublas_mixed          # full suite, shapes from sa3.cpp pipeline
//   ./bench_cublas_mixed M N K    # single custom shape
//
// Each test case runs 100 warmup + 200 timed iterations.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ---- test case shapes (K = input features, N = output features, M = batch) ----
// In ggml: weight is [K, N], input is [K, M], output is [N, M].
// In cuBLAS: C = A^T * B   where A is K×N, B is K×M, C is N×M.
// cublasGemmEx(OP_T, OP_N, N, M, K, ...)
typedef struct {
    const char * name;
    int M, N, K;
} Shape;

static const Shape shapes[] = {
    { "dit_qkv",       192, 15360, 3072  },  // DiT QKV projection
    { "dit_ff_proj",   192, 12288, 3072  },  // DiT FF projection
    { "dit_ff_out",    192, 3072,  6144  },  // DiT FF output
    { "dit_out",       192, 3072,  3072  },  // DiT attention output
    { "same_qkv",      2176, 5120, 1024  },  // SAME QKV
    { "same_ff_proj",  2176, 8192, 1024  },  // SAME FF projection
    { "same_ff_out",   2176, 1024,  4096  },  // SAME FF output
    { "small_attn",    17,   51,    64    },  // SAME sliding window (per head)
    { NULL, 0, 0, 0 }
};

// ---- CUDA error checking ----
#define CUCHECK(call) do {                                         \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CUBLASCHECK(call) do {                                     \
    cublasStatus_t s = call;                                       \
    if (s != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n",            \
                __FILE__, __LINE__, s);                            \
        exit(1);                                                   \
    }                                                              \
} while(0)

// ---- simple F16 → F32 dequant kernel ----
__global__ static void dequant_f16_f32(const half * __restrict__ src,
                                        float * __restrict__ dst,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __half2float(src[i]);
}

// ---- helpers ----
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

// ---- function pointer type for timed runs ----
typedef int (*gemm_func_t)(cublasHandle_t, int M, int N, int K,
                           const void * A, const void * B, void * C,
                           cudaStream_t stream, void * scratch,
                           int scratch_size);

// ---- dequant + F32 gemm path ----
// runtime algo selection: 0-15 for specific algo, -1 for DEFAULT
static int g_gemm_algo = -1;

int run_dequant_gemm(cublasHandle_t handle, int M, int N, int K,
                     const void * A, const void * B, void * C,
                     cudaStream_t stream, void * scratch,
                     int scratch_size) {
    float * A_f32 = (float *)scratch;
    int n_weights = K * N;

    // dequant F16 → F32
    int block = 256;
    int grid  = (n_weights + block - 1) / block;
    grid = min(grid, 65535);
    dequant_f16_f32<<<grid, block, 0, stream>>>((const half *)A, A_f32, n_weights);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmAlgo_t algo = (g_gemm_algo < 0)
        ? CUBLAS_GEMM_DEFAULT
        : (cublasGemmAlgo_t)g_gemm_algo;
    CUBLASCHECK(cublasGemmEx(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             A_f32, CUDA_R_32F, K,
                             B, CUDA_R_32F, K,
                             &beta,
                             C, CUDA_R_32F, N,
                             CUBLAS_COMPUTE_32F,
                             algo));
    return 0;
}

// ---- mixed-precision cublasGemmEx path ----
int run_mixed_gemm(cublasHandle_t handle, int M, int N, int K,
                   const void * A, const void * B, void * C,
                   cudaStream_t stream, void * scratch,
                   int scratch_size) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLASCHECK(cublasGemmEx(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             A, CUDA_R_16F, K,
                             B, CUDA_R_32F, K,
                             &beta,
                             C, CUDA_R_32F, N,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT));
    return 0;
}

// ---- benchmark one shape + one path ----
static double bench_one(cublasHandle_t handle, cudaStream_t stream,
                        int M, int N, int K,
                        const half * d_A, const float * d_B, float * d_C,
                        gemm_func_t func, void * scratch, int scratch_size,
                        int warmup, int iters) {
    // warmup
    for (int i = 0; i < warmup; i++)
        func(handle, M, N, K, d_A, d_B, d_C, stream, scratch, scratch_size);
    CUCHECK(cudaStreamSynchronize(stream));

    // timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++)
        func(handle, M, N, K, d_A, d_B, d_C, stream, scratch, scratch_size);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

// ---- compute GFLOPS for a gemm: 2 * K * N * M ----
static double gflops(int M, int N, int K) {
    return 2.0 * K * N * M * 1e-9;
}

// ---- verify numerical difference ----
static double max_diff(const float * cpu_a, const float * cpu_b, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)cpu_a[i] - (double)cpu_b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ---- print device info ----
static void print_device_info() {
    int dev;
    CUCHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUCHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (CC %d.%d, %d SMs, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("CUDA %d.%d, Driver %d\n\n",
           prop.major, prop.minor, prop.major * 10 + prop.minor);
}

// ---- detect if mixed-precision F16×F32 gemm is supported ----
static int probe_mixed_gemm(cublasHandle_t handle, int M, int N, int K,
                            const half * d_A, const float * d_B) {
    float * d_tmp;
    CUCHECK(cudaMalloc(&d_tmp, (size_t)N * M * sizeof(float)));
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    d_A, CUDA_R_16F, K,
                                    d_B, CUDA_R_32F, K,
                                    &beta,
                                    d_tmp, CUDA_R_32F, N,
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT);
    CUCHECK(cudaFree(d_tmp));
    return (s == CUBLAS_STATUS_SUCCESS);
}

static void report_mixed_not_supported(void) {
    printf("  F16 x F32 mixed:     NOT SUPPORTED on this GPU\n");
    printf("                        (CC 5.0 exact does not support CUDA_R_16F in cublasGemmEx)\n");
}

int main(int argc, char ** argv) {
    // set a small L2 cache prefer split for more deterministic results
    CUCHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int n_test_shapes;
    const Shape * test_shapes;
    Shape single_shape;

    if (argc == 4) {
        single_shape.name = "custom";
        single_shape.M    = atoi(argv[1]);
        single_shape.N    = atoi(argv[2]);
        single_shape.K    = atoi(argv[3]);
        test_shapes  = &single_shape;
        n_test_shapes = 1;
    } else {
        test_shapes  = shapes;
        n_test_shapes = sizeof(shapes) / sizeof(shapes[0]) - 1; // skip sentinel
    }

    print_device_info();

    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));
    cudaStream_t stream;
    CUCHECK(cudaStreamCreate(&stream));
    CUBLASCHECK(cublasSetStream(handle, stream));

    const int warmup = 100;
    const int iters  = 200;

    printf("=== cuBLAS Mixed-Path Benchmark ===\n");
    printf("Shapes: M = batch seq, N = output features, K = input features\n\n");

    // probe mixed support once
    int mixed_supported = 0;

    for (int ti = 0; ti < n_test_shapes; ti++) {
        const Shape * s = &test_shapes[ti];
        int M = s->M, N = s->N, K = s->K;

        int n_weights = K * N;
        int n_inputs  = K * M;
        int n_outputs = N * M;

        // scratch for dequant path: F32 copy of weights
        int scratch_size = n_weights * sizeof(float);

        // allocate device memory
        half  * d_A;        // F16 weights
        float * d_B;        // F32 inputs
        float * d_C;        // F32 output (mixed path)
        float * d_C_ref;    // F32 output (dequant path)
        void  * d_scratch;

        CUCHECK(cudaMalloc(&d_A,       n_weights * sizeof(half)));
        CUCHECK(cudaMalloc(&d_B,       n_inputs  * sizeof(float)));
        CUCHECK(cudaMalloc(&d_C,       n_outputs * sizeof(float)));
        CUCHECK(cudaMalloc(&d_C_ref,   n_outputs * sizeof(float)));
        CUCHECK(cudaMalloc(&d_scratch, scratch_size));

        // host data
        half  * h_A = (half  *)malloc(n_weights * sizeof(half));
        float * h_B = (float *)malloc(n_inputs  * sizeof(float));
        fill_random_f16(h_A, n_weights, 42 + ti);
        fill_random_f32(h_B, n_inputs,  99 + ti);

        // upload
        CUCHECK(cudaMemcpyAsync(d_A, h_A, n_weights * sizeof(half),
                                cudaMemcpyHostToDevice, stream));
        CUCHECK(cudaMemcpyAsync(d_B, h_B, n_inputs * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
        CUCHECK(cudaStreamSynchronize(stream));
        free(h_A);
        free(h_B);

        // probe mixed support on first shape
        if (ti == 0) {
            mixed_supported = probe_mixed_gemm(handle, M, N, K, d_A, d_B);
            if (!mixed_supported)
                printf("[probe] cublasGemmEx(CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F) "
                       "NOT SUPPORTED on this GPU\n\n");
            else
                printf("[probe] mixed F16×F32 gemm IS SUPPORTED\n\n");
        }

        // warm device (only dequant path if mixed unsupported)
        run_dequant_gemm(handle, M, N, K, d_A, d_B, d_C_ref, stream,
                         d_scratch, scratch_size);
        CUCHECK(cudaStreamSynchronize(stream));

        // benchmark dequant path
        double t_dequant = bench_one(handle, stream, M, N, K,
                                     d_A, d_B, d_C_ref,
                                     run_dequant_gemm,
                                     d_scratch, scratch_size,
                                     warmup, iters);

        // break down dequant cost vs gemm cost for the dequant path
        // (single-shot measure of just the dequant kernel)
        float t_dequant_only = 0;
        {
            cudaEvent_t ds, de;
            cudaEventCreate(&ds);
            cudaEventCreate(&de);
            int block = 256;
            int grid  = (n_weights + block - 1) / block;
            grid = min(grid, 65535);
            cudaEventRecord(ds, stream);
            for (int i = 0; i < iters; i++)
                dequant_f16_f32<<<grid, block, 0, stream>>>((const half *)d_A,
                    (float *)d_scratch, n_weights);
            cudaEventRecord(de, stream);
            cudaEventSynchronize(de);
            float ms_dq;
            cudaEventElapsedTime(&ms_dq, ds, de);
            t_dequant_only = ms_dq / iters;
            cudaEventDestroy(ds);
            cudaEventDestroy(de);
        }

        printf("%-14s M=%5d N=%6d K=%5d  (%.1f GFLOP, W=%d MB)\n",
               s->name, M, N, K, gflops(M, N, K),
               (int)((size_t)n_weights * sizeof(half) / (1024*1024)));

        printf("  dequant + F32 gemm:  %8.4f ms  (%7.1f GFLOPS)\n",
               t_dequant, gflops(M, N, K) / (t_dequant * 1e-3));
        printf("    dequant kernel:    %8.4f ms  (%7.1f GB/s)\n",
               t_dequant_only,
               (double)n_weights * sizeof(half) / (t_dequant_only * 1e-3) / 1e9);
        printf("    F32 gemm only:     %8.4f ms  (%7.1f GFLOPS)\n",
               t_dequant - t_dequant_only,
               gflops(M, N, K) / ((t_dequant - t_dequant_only) * 1e-3));

        // benchmark mixed path (if supported)
        if (mixed_supported) {
            CUCHECK(cudaMemsetAsync(d_C, 0, n_outputs * sizeof(float), stream));
            CUCHECK(cudaStreamSynchronize(stream));

            double t_mixed = bench_one(handle, stream, M, N, K,
                                       d_A, d_B, d_C,
                                       run_mixed_gemm,
                                       d_scratch, scratch_size,
                                       warmup, iters);

            double gflops_mixed = gflops(M, N, K) / (t_mixed * 1e-3);

            // numerical comparison
            float * h_dequant = (float *)malloc(n_outputs * sizeof(float));
            float * h_mixed   = (float *)malloc(n_outputs * sizeof(float));
            CUCHECK(cudaMemcpy(h_dequant, d_C_ref, n_outputs * sizeof(float),
                               cudaMemcpyDeviceToHost));
            CUCHECK(cudaMemcpy(h_mixed,   d_C,     n_outputs * sizeof(float),
                               cudaMemcpyDeviceToHost));
            double err = max_diff(h_dequant, h_mixed, n_outputs);
            free(h_dequant);
            free(h_mixed);

            printf("  F16 x F32 mixed:     %8.4f ms  (%7.1f GFLOPS)"
                   "  %s%.2fx%s\n",
                   t_mixed, gflops_mixed,
                   t_dequant > t_mixed ? "\033[32m" : "\033[31m",
                   t_dequant / t_mixed,
                   "\033[0m");
            printf("  Max|diff|: %.2e  %s\n", err,
                   err < 1e-4 ? "(ok)" : "(WARNING)");
        } else {
            report_mixed_not_supported();
            printf("  Max|diff|: N/A\n");
        }
        printf("\n");

        CUCHECK(cudaFree(d_A));
        CUCHECK(cudaFree(d_B));
        CUCHECK(cudaFree(d_C));
        CUCHECK(cudaFree(d_C_ref));
        CUCHECK(cudaFree(d_scratch));
    }

    // ---- cuBLAS algorithm sweep ----
    printf("\n=== cuBLAS Algorithm Sweep (F32 gemm only, 50 iter each) ===\n\n");

    int algo_list[] = {
        CUBLAS_GEMM_DEFAULT,
        CUBLAS_GEMM_ALGO0,  CUBLAS_GEMM_ALGO1,  CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3,  CUBLAS_GEMM_ALGO4,  CUBLAS_GEMM_ALGO5,
        CUBLAS_GEMM_ALGO6,  CUBLAS_GEMM_ALGO7,  CUBLAS_GEMM_ALGO8,
        CUBLAS_GEMM_ALGO9,  CUBLAS_GEMM_ALGO10, CUBLAS_GEMM_ALGO11,
        CUBLAS_GEMM_ALGO12, CUBLAS_GEMM_ALGO13, CUBLAS_GEMM_ALGO14,
        CUBLAS_GEMM_ALGO15
    };
    int n_algos = sizeof(algo_list) / sizeof(algo_list[0]);
    const char * algo_names[] = {
        "DEFAULT","ALGO0","ALGO1","ALGO2","ALGO3","ALGO4","ALGO5",
        "ALGO6","ALGO7","ALGO8","ALGO9","ALGO10","ALGO11",
        "ALGO12","ALGO13","ALGO14","ALGO15"
    };
    // tensor-core algos not valid on CC 5.0, skip them

    const int sweep_iters = 50;
    // only sweep the first 4 representative shapes to keep runtime sane
    int sweep_ids[] = {0, 1, 3, 7};
    int n_sweep = sizeof(sweep_ids) / sizeof(sweep_ids[0]);

    for (int si = 0; si < n_sweep; si++) {
        const Shape * s = &test_shapes[sweep_ids[si]];
        int M = s->M, N = s->N, K = s->K;

        int n_weights = K * N;
        int n_inputs  = K * M;
        int n_outputs = N * M;
        int scratch_size = n_weights * sizeof(float);

        half  * d_A;
        float * d_B;
        float * d_C;
        float * d_C_ref_tmp;
        void  * d_scratch;

        CUCHECK(cudaMalloc(&d_A,          n_weights * sizeof(half)));
        CUCHECK(cudaMalloc(&d_B,          n_inputs  * sizeof(float)));
        CUCHECK(cudaMalloc(&d_C,          n_outputs * sizeof(float)));
        CUCHECK(cudaMalloc(&d_C_ref_tmp,  n_outputs * sizeof(float)));
        CUCHECK(cudaMalloc(&d_scratch,    scratch_size));

        half  * h_A = (half  *)malloc(n_weights * sizeof(half));
        float * h_B = (float *)malloc(n_inputs  * sizeof(float));
        fill_random_f16(h_A, n_weights, 42 + sweep_ids[si]);
        fill_random_f32(h_B, n_inputs,  99 + sweep_ids[si]);
        CUCHECK(cudaMemcpyAsync(d_A, h_A, n_weights * sizeof(half),
                                cudaMemcpyHostToDevice, stream));
        CUCHECK(cudaMemcpyAsync(d_B, h_B, n_inputs * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
        CUCHECK(cudaStreamSynchronize(stream));
        free(h_A);
        free(h_B);

        printf("%-14s M=%5d N=%6d K=%5d  (%.1f GFLOP)\n",
               s->name, M, N, K, gflops(M, N, K));

        double best_time = 1e99;
        int best_algo_idx = -1;

        for (int ai = 0; ai < n_algos; ai++) {
            g_gemm_algo = algo_list[ai];

            // test if algo is supported (one-shot try)
            {
                float ione = 1.0f, izero = 0.0f;
                cublasStatus_t st = cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K,
                    &ione, (float*)d_scratch, CUDA_R_32F, K,
                    d_B, CUDA_R_32F, K,
                    &izero, d_C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t)g_gemm_algo);
                CUCHECK(cudaStreamSynchronize(stream));
                if (st != CUBLAS_STATUS_SUCCESS) {
                    if (st != CUBLAS_STATUS_NOT_SUPPORTED)
                        fprintf(stderr, "  algo %s failed with error %d\n",
                                algo_names[ai], st);
                    printf("  %-10s: NOT SUPPORTED\n", algo_names[ai]);
                    continue;
                }
            }

            // warmup
            for (int w = 0; w < 10; w++)
                run_dequant_gemm(handle, M, N, K, d_A, d_B, d_C,
                                 stream, d_scratch, scratch_size);
            CUCHECK(cudaStreamSynchronize(stream));

            cudaEvent_t ds, de;
            cudaEventCreate(&ds);
            cudaEventCreate(&de);
            cudaEventRecord(ds, stream);
            for (int i = 0; i < sweep_iters; i++)
                run_dequant_gemm(handle, M, N, K, d_A, d_B, d_C,
                                 stream, d_scratch, scratch_size);
            cudaEventRecord(de, stream);
            cudaEventSynchronize(de);
            float ms;
            cudaEventElapsedTime(&ms, ds, de);
            cudaEventDestroy(ds);
            cudaEventDestroy(de);

            double t_avg = ms / sweep_iters;
            double gf    = gflops(M, N, K) / (t_avg * 1e-3);

            printf("  %-10s: %8.4f ms  (%7.1f GFLOPS)%s\n",
                   algo_names[ai], t_avg, gf,
                   t_avg < best_time ? "  <--" : "");
            if (t_avg < best_time) {
                best_time = t_avg;
                best_algo_idx = ai;
            }
        }

        printf("  >> Best: %s  (%.4f ms, %.1f GFLOPS)\n\n",
               algo_names[best_algo_idx], best_time,
               gflops(M, N, K) / (best_time * 1e-3));

        CUCHECK(cudaFree(d_A));
        CUCHECK(cudaFree(d_B));
        CUCHECK(cudaFree(d_C));
        CUCHECK(cudaFree(d_C_ref_tmp));
        CUCHECK(cudaFree(d_scratch));
    }

    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    return 0;
}
