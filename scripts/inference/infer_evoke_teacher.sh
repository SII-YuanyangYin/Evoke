#!/bin/bash
# ============================================================================
# EvokeTeacher dual-expert sampling -- i2v example
#   Weights : models/evoke/evoke_teacher/{high,low}_noise  (+ models/evoke-base for vae/text encoder)
#   Engine  : scripts/inference/infer_evoke_teacher.py
#
#   This is an EXAMPLE, not a supported launcher. The teacher is normally only a frozen scorer
#   inside stage-3 DMD; the engine wraps that scoring forward in a flow-match Euler loop. It is
#   nocam / non-SP / single-process, and it is NOT numerically validated against the teacher's own
#   sampler. A/B one prompt+seed against the upstream sampler before reading anything into a result.
#
#   Memory: 2 x 14B is ~56 GB of bf16 weights. OFFLOAD=1 (the default) keeps only the routed
#   expert resident. SINGLE_EXPERT=high|low loads one expert -- a plumbing check, not a valid sample.
#
#   Sparse attention: the currently shipped weights are cs9/select1. loader.py's cs8/select4 is a
#   legacy fallback. These values do not alter parameter shapes, so checkpoint loading cannot catch
#   a mismatch; always check the engine's "sparse EFFECTIVE" line when changing them.
#
#   Usage: [CLIP_SECONDS=5] [STEPS=50] [OUT=...] bash scripts/inference/infer_evoke_teacher.sh
# ============================================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"      # repo root
[ -n "${EVOKE_PYTHON_BIN:-}" ] && export PATH="$EVOKE_PYTHON_BIN:$PATH"

if ! python -c 'import torch, cv2' 2>/dev/null; then
  echo "[preflight] FATAL: the python on PATH ($(command -v python || echo none)) cannot import torch + cv2." >&2
  echo "[preflight]        Set EVOKE_PYTHON_BIN=<env>/bin to prepend the right interpreter." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
infer_evoke_teacher.sh -- sample the dual-expert teacher (models/evoke/evoke_teacher), i2v

  Defaults to the bundled i2v example: examples/i2v/image.jpg + examples/i2v/prompt.txt, ~5s @24fps.
  Drop IMAGE (IMAGE= ) to sample t2v instead -- the engine conditions on the prompt alone.

Length
  CLIP_SECONDS (default 5) is rounded up to the nearest valid frame count: the VAE is temporal
  stride 4, so pixel frames must be 4k+1 and latent T = (frames-1)/4 + 1. 5s @24fps -> 121 frames.
  NUM_FRAMES overrides it directly; a value that is not 4k+1 wastes the remainder.

Knobs   PROMPT  PROMPT_FILE  IMAGE        prompt text / file / i2v reference frame
        CLIP_SECONDS  NUM_FRAMES  FPS      output length
        STEPS  GUIDANCE_SCALE  SHIFT  SEED  sampler (GUIDANCE_SCALE=1.0 disables CFG)
        HEIGHT  WIDTH  BOUNDARY            resolution / expert switch (t >= BOUNDARY*1000 -> high)
        CHUNK_SIZE (default 9)  NUM_SELECT_FRAMES (default 1)  sparse attention (shipped weights)
        OFFLOAD (default 1)  SINGLE_EXPERT  memory
        TEACHER_DIR  BASE  OUT  EVOKE_PYTHON_BIN
Output  $OUT                                 generated mp4
        logs/infer_evoke_teacher/run.log     engine log
EOF
}
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }

TEACHER_DIR=${TEACHER_DIR:-"models/evoke/evoke_teacher"}
BASE=${BASE:-"models/evoke-base"}
IMAGE=${IMAGE-"examples/i2v/image.jpg"}
PROMPT_FILE=${PROMPT_FILE-"examples/i2v/prompt.txt"}
HEIGHT=${HEIGHT:-384} WIDTH=${WIDTH:-640} FPS=${FPS:-24}
CLIP_SECONDS=${CLIP_SECONDS:-5}
STEPS=${STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
SHIFT=${SHIFT:-5.0}
BOUNDARY=${BOUNDARY:-0.9}
CHUNK_SIZE=${CHUNK_SIZE:-9}
NUM_SELECT_FRAMES=${NUM_SELECT_FRAMES:-1}
SEED=${SEED:-42}
OFFLOAD=${OFFLOAD:-1}
OUT=${OUT:-"output/evoke_teacher/i2v.mp4"}

if [ -z "${PROMPT:-}" ]; then
  if [ -n "$PROMPT_FILE" ] && [ -f "$PROMPT_FILE" ]; then
    PROMPT=$(tr '\n' ' ' < "$PROMPT_FILE" | sed 's/  *$//')
  else
    echo "[teacher] FATAL: no PROMPT and PROMPT_FILE='$PROMPT_FILE' is not a file." >&2
    exit 1
  fi
fi

if [ -z "${NUM_FRAMES:-}" ]; then
  _target=$((CLIP_SECONDS * FPS))
  NUM_FRAMES=$(( 4 * (( _target - 1 + 3 ) / 4) + 1 ))
fi
if [ $(( (NUM_FRAMES - 1) % 4 )) -ne 0 ]; then
  echo "[teacher] WARNING: NUM_FRAMES=$NUM_FRAMES is not 4k+1; the VAE is temporal stride 4 and the"
  echo "[teacher]          remainder is wasted. Nearest valid: $(( 4 * (( NUM_FRAMES - 1 + 3 ) / 4) + 1 ))."
fi

MODE_NOTE="i2v (reference frame $IMAGE)"
[ -z "$IMAGE" ] && MODE_NOTE="t2v (no reference frame)"
if [ -n "$IMAGE" ] && [ ! -f "$IMAGE" ]; then
  echo "[teacher] FATAL: IMAGE='$IMAGE' not found." >&2
  exit 1
fi

CFG_STATE=on
{ [ "$GUIDANCE_SCALE" = "1" ] || [ "$GUIDANCE_SCALE" = "1.0" ]; } && CFG_STATE=off

SLOG="logs/infer_evoke_teacher"; mkdir -p "$SLOG"
echo "=============================================================================="
echo "[teacher] WEIGHTS   : $TEACHER_DIR  (base: $BASE)"
echo "[teacher] EXAMPLE ONLY -- not validated against the teacher's own sampler; see -h"
echo "[teacher] mode=$MODE_NOTE"
echo "[teacher] frames=$NUM_FRAMES (~$(awk "BEGIN{printf \"%.2f\", $NUM_FRAMES/$FPS}")s @${FPS}fps, latent T=$(( (NUM_FRAMES - 1) / 4 + 1 )))  ${HEIGHT}x${WIDTH}"
echo "[teacher] steps=$STEPS shift=$SHIFT guidance_scale=$GUIDANCE_SCALE (CFG $CFG_STATE) boundary=$BOUNDARY seed=$SEED"
echo "[teacher] sparse requested: chunk_size=$CHUNK_SIZE num_select_frames=$NUM_SELECT_FRAMES (shipped: cs9/select1)"
echo "[teacher] offload=$OFFLOAD single_expert=${SINGLE_EXPERT:-none}"
echo "[teacher] out=$OUT"
echo "=============================================================================="

ARGS=(--teacher_dir "$TEACHER_DIR" --base "$BASE" --prompt "$PROMPT"
      --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" --fps "$FPS"
      --num_inference_steps "$STEPS" --shift "$SHIFT" --guidance_scale "$GUIDANCE_SCALE"
      --boundary "$BOUNDARY" --chunk_size "$CHUNK_SIZE" --num_select_frames "$NUM_SELECT_FRAMES"
      --seed "$SEED" --output "$OUT")
[ -n "$IMAGE" ] && ARGS+=(--image_path "$IMAGE")
[ "$OFFLOAD" = "1" ] && ARGS+=(--offload)
[ -n "${SINGLE_EXPERT:-}" ] && ARGS+=(--single_expert "$SINGLE_EXPERT")

set +e
python scripts/inference/infer_evoke_teacher.py "${ARGS[@]}" 2>&1 | tee "$SLOG/run.log"
fail=${PIPESTATUS[0]}
set -e
echo "[teacher] DONE rc=$fail -> $OUT"; exit $fail
