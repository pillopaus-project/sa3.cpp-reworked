#!/usr/bin/env bash
# Benchmark flash-attention modes for DiT (--flash-attn) and SAME (--same-flash-attn).
# Runs sa3-generate with each combination and reports wall-clock time.
set -eu

BIN="./build-cuda-cc80/bin/sa3-generate"
MDIR="models"
PROMPT="upbeat funk groove with slap bass, bright horns, tight drums"
STEPS=8
DURATION=150
OUTDIR="bench_results"

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found — run ./build.sh cuda first" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

declare -a FLASH_OPTS=(0 1)
declare -a SAME_OPTS=(0 1 2)   # 0=off 1=full 2=local

# Prefer GNU time for reliable 'real' output, fall back to bash builtin
TIME_CMD=""
if command -v gtime &>/dev/null; then
  TIME_CMD="gtime"
elif command -v /usr/bin/time &>/dev/null; then
  TIME_CMD="/usr/bin/time"
fi

echo "Benchmarking flash-attention modes..."
echo "  Binary: $BIN"
echo "  Steps:  $STEPS   Duration: ${DURATION}s"
echo "  Prompt: \"$PROMPT\""
echo "  Results dir: $OUTDIR/"
echo ""

for FA in "${FLASH_OPTS[@]}"; do
  for SA in "${SAME_OPTS[@]}"; do
    OUTFILE="${OUTDIR}/fa${FA}_sa${SA}.wav"
    LOGFILE="${OUTDIR}/fa${FA}_sa${SA}.log"

    echo -n "fa=$FA sa=$SA ... "

    if [ -n "$TIME_CMD" ]; then
      "$TIME_CMD" -f "real %e" -o "$LOGFILE" \
        "$BIN" \
          --models-dir "$MDIR" \
          --model medium \
          --prompt "$PROMPT" \
          --steps "$STEPS" \
          --duration "$DURATION" \
          --flash-attn "$FA" \
          --same-flash-attn "$SA" \
          --out "$OUTFILE" \
          2>&1 | tee -a "$LOGFILE"
    else
      # bash builtin time — capture stderr trick
      { time "$BIN" \
          --models-dir "$MDIR" \
          --model medium \
          --prompt "$PROMPT" \
          --steps "$STEPS" \
          --duration "$DURATION" \
          --flash-attn "$FA" \
          --same-flash-attn "$SA" \
          --out "$OUTFILE" \
          2>&1 ; } 2>&1 | tee "$LOGFILE"
    fi

    REAL=$(grep -oP 'real \K[0-9.]+' "$LOGFILE" | tail -1 || echo "?")
    echo "  -> ${REAL}s"
  done
done

echo ""
echo "=== Summary ==="
echo "FlashAttn  SameAttn  Time(s)"
echo "--------  --------  -------"
for FA in "${FLASH_OPTS[@]}"; do
  for SA in "${SAME_OPTS[@]}"; do
    REAL=$(grep -oP 'real \K[0-9.]+' "${OUTDIR}/fa${FA}_sa${SA}.log" 2>/dev/null | tail -1 || echo "?")
    printf "    %d        %d      %s\n" "$FA" "$SA" "$REAL"
  done
done
