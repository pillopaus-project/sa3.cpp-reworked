#!/usr/bin/env bash
# Build sa3.cpp for one backend into its own build dir (backends coexist for A/B testing).
#
# Usage: ./build.sh [cpu|cpu-variants|cuda|cuda-cc50|cuda-cc80|vulkan|hip|metal|all]   (default: cpu)
#   cpu           -> build/              native CPU, no GPU
#   cpu-variants  -> build-cpu-variants/ CPU-only runtime CPU variant selection
#   cuda          -> build-cuda/         NVIDIA (needs CUDA Toolkit; arch auto-detected)
#   cuda-cc50     -> build-cuda-cc50/    NVIDIA, single-arch sm_50 (GTX 960M / Maxwell)
#   cuda-cc80     -> build-cuda-cc80/    NVIDIA, single-arch sm_80 (Ampere+)
#   vulkan        -> build-vulkan/       any GPU (needs the Vulkan SDK to compile shaders)
#   hip           -> build-hip/          AMD/ROCm (needs ROCm/HIP)
#   metal         -> build-metal/        Apple GPU (macOS only)
#   all           -> build-all/          one binary, all GPU backends loaded at runtime (GGML_BACKEND_DL)
#
# On macOS, cpu/all also pick up Metal + Accelerate automatically via ggml.
set -eu

BACKEND="${1:-cpu}"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
UNAME="$(uname -s)"

case "$BACKEND" in
    cpu)    DIR=build         ; FLAGS="-DGGML_CUDA=OFF" ;;
    cpu-variants)
            DIR=build-cpu-variants
            FLAGS="-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=OFF -DSA3_CUDA=OFF -DSA3_VULKAN=OFF" ;;
    cuda)   DIR=build-cuda    ; FLAGS="-DSA3_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
            # CUDA graphs require Ampere (CC >= 8.0); disable for older GPUs to avoid
            # the runtime graph infrastructure overhead on every compute call.
            if command -v nvidia-smi &>/dev/null; then
                CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
                if [ -n "$CC_MAJOR" ] && [ "$CC_MAJOR" -lt 8 ] 2>/dev/null; then
                    FLAGS="$FLAGS -DGGML_CUDA_GRAPHS=OFF"
                    echo "[sa3] CUDA architecture $CC_MAJOR.x < 8.0 -> disabling CUDA graphs"
                fi
            fi ;;
    cuda-cc50)
            # Single-arch build for Maxwell (CC 5.0) with CC50-specific optimizations.
            DIR=build-cuda-cc50
            FLAGS="-DSA3_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50-real -DGGML_CUDA_CC50=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FORCE_CUBLAS=ON" ;;
    cuda-cc80)
            # Single-arch build for Ampere+ (CC 8.0+) with CC80-specific optimizations.
            DIR=build-cuda-cc80
            FLAGS="-DSA3_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80-real -DGGML_CUDA_CC80_PLUS=ON -DGGML_CUDA_GRAPHS=ON" ;;
    vulkan) DIR=build-vulkan  ; FLAGS="-DSA3_VULKAN=ON" ;;
    hip)    DIR=build-hip     ; FLAGS="-DSA3_HIP=ON" ;;
    metal)  DIR=build-metal   ; FLAGS="-DSA3_METAL=ON" ;;
    all)
        DIR=build-all
        # One binary that loads the GPU backends at runtime. On macOS that's Metal;
        # elsewhere CUDA + Vulkan cover NVIDIA/AMD/Intel.
        if [ "$UNAME" = "Darwin" ]; then
            FLAGS="-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DSA3_METAL=ON"
        else
            FLAGS="-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DSA3_CUDA=ON -DSA3_VULKAN=ON"
        fi ;;
    *) echo "unknown backend: '$BACKEND' (cpu|cpu-variants|cuda|cuda-cc50|cuda-cc80|vulkan|hip|metal|all)" >&2; exit 1 ;;
esac

echo "[sa3] configuring $BACKEND -> $DIR/"
cmake -S . -B "$DIR" -DCMAKE_BUILD_TYPE=Release $FLAGS
echo "[sa3] building (-j $JOBS) ..."
cmake --build "$DIR" --config Release -j "$JOBS"
echo "[sa3] done -> $DIR/bin/"
echo "[sa3]   e.g.:  $DIR/bin/sa3-generate --models-dir models --model medium --prompt \"...\" --out song.wav"
