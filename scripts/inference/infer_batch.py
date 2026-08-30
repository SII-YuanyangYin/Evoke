"""jsonl-driven batched inference worker (one shard).

The three launchers share this driver. All knobs come in through environment variables;
for each case it shells out to
`scripts/inference/infer_single.py`:

  - scripts/inference/infer_stage1.sh       -> models/evoke/stage1_camera_control (multi-step, IS_STAGE2=0)
  - scripts/inference/infer_post_distill.sh -> models/evoke/stage3_post_distillation (3-step, long rollout)

Each launcher mirrors the knobs of its training config. A warp/attention mismatch between
training and inference degrades quality, so a new model means updating its launcher.
Sharding: SHARD / NSHARD env vars, case i goes to shard i % NSHARD; MAX_CASES>0 keeps the first N."""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # scripts/inference/ → repo root
os.chdir(REPO)

def _env(k, d): return os.environ.get(k, d)


def _drain_bg_postprocess(out_root: Path, timeout_s: float = 1800.0, poll: float = 3.0) -> int:
    """Block until every detached postprocess worker under out_root has finished.

    Required whenever BG_POSTPROC=1: the worker is what writes DONE_MARK (geo_pred.mp4), so checking it
    while workers are still in flight would report perfectly good cases as FAIL -- and, worse, delete
    their output directory as an empty crash leftover. Each live worker owns a <case>.lock holding its
    pid (infer_single._spawn_bg_postprocess); a lock whose pid is gone is a crashed worker, so it is
    pruned rather than waited on forever. Returns the number of locks still present at the timeout.
    """
    import time as _t
    wdir = out_root / ".pp_workers"
    if not wdir.is_dir():
        return 0
    deadline = _t.time() + timeout_s
    while True:
        live = 0
        for m in list(wdir.glob("*.lock")):
            try:
                pid = int((m.read_text().strip() or "0"))
            except (ValueError, OSError):
                pid = 0
            if pid <= 0:
                live += 1            # marker created but the pid is not written yet
                continue
            try:
                os.kill(pid, 0)
                live += 1
            except ProcessLookupError:
                try:
                    m.unlink()       # dead worker; reclaim so we do not wait on it
                except OSError:
                    pass
            except PermissionError:
                live += 1
        if live == 0:
            return 0
        if _t.time() >= deadline:
            print(f"[warn] {live} postprocess worker(s) still in flight after {timeout_s:.0f}s; "
                  f"their cases will be reported FAIL and can be resumed", flush=True)
            return live
        print(f"[bg-drain] waiting on {live} postprocess worker(s) ...", flush=True)
        _t.sleep(poll)

# ── shard ──
SHARD = int(_env("SHARD", "0")); NSHARD = int(_env("NSHARD", "1"))
# Per-case output goes to _logs/<case>.log. With one shard that is also the interactive case, so mirror
#   it to stdout as well: hiding it makes a live rollout look exactly like a hang. Several shards keep
#   the log-only behaviour, since interleaving their progress bars on one terminal is unreadable.
VERBOSE = _env("INFER_VERBOSE", "1" if NSHARD == 1 else "0") == "1"

# -- data --
JSONL = _env("JSONL", "examples/v2v/cases.jsonl")
MAX_CASES = int(_env("MAX_CASES", "0"))   # >0 = first N cases only (smoke); 0 = all
# Roots for jsonl paths. Empty = paths are repo-relative (how the bundled examples/* sets are
#   written). The full sekai sweep stores paths relative to a mounted dataset, so it needs both set.
VROOT = _env("VROOT", ""); AROOT = _env("AROOT", "")
OUT_ROOT = Path(_env("OUT_ROOT", "output_evoke/infer/batch"))

# -- checkpoint / backbone --
BASE_CKPT = _env("BASE_CKPT", "models/evoke-base")
# Parent of the transformer/ dir: loaded as from_pretrained(TRANSFORMER_PATH, subfolder="transformer").
TRANSFORMER_PATH = _env("TRANSFORMER_PATH", "")

# ── generation ──
# NUM_FRAMES only picks the chunk count, it does NOT bound the output length:
#   chunks = ceil(NUM_FRAMES/33)   (33 = (latent_window_size-1)*vae_stride+1, pipeline_evoke.py)
#   frames = 36*chunks - 3         (chunk0 decodes 33 px frames, the rest 36; no trim to NUM_FRAMES)
# So 721 -> 22 chunks -> 789 frames (32.9s @24fps), 1437 -> 44 chunks -> 1581 (65.9s).
# Set NUM_CHUNKS instead to get an exact chunk count (NUM_FRAMES = 33*NUM_CHUNKS).
HEIGHT = _env("HEIGHT", "384"); WIDTH = _env("WIDTH", "640"); FPS = _env("FPS", "24")
NUM_CHUNKS = int(_env("NUM_CHUNKS", "0"))
NUM_FRAMES = str(33 * NUM_CHUNKS) if NUM_CHUNKS > 0 else _env("NUM_FRAMES", "1437")
NUM_INFERENCE_STEPS = _env("NUM_INFERENCE_STEPS", "3")   # = validation num_inference_steps; overridden by sum(stage2_steps) when the pyramid is on
GUIDANCE_SCALE = _env("GUIDANCE_SCALE", "1.0"); SEED = _env("SEED", "44")
START_SECONDS = _env("START_SECONDS", "0.0"); REF_VIDEO_SEC = _env("REF_VIDEO_SEC", "5.0")

# -- 3-step pyramid (few-step distilled recipe, = training stage2) --
MODE = _env("MODE", "v2v")            # v2v | i2v | t2v (see the per-mode input table in the docstring)
if MODE not in ("v2v", "i2v", "t2v"):
    sys.exit(f"[ERROR] MODE must be one of v2v/i2v/t2v, got {MODE!r}")
# WARP=off runs i2v (or v2v) with geometric state switched OFF entirely: no --use_geometric_state, no
#   pose, no --visibility_aware_noise, no --geo_*. The rows then need no "pose_path".
#
#   This is NOT the same as feeding an identity pose track, and that difference is the whole reason the
#   switch exists. With a still camera ViGeo rejects every window as "zero baseline", no point cloud is
#   ever built, and the model is handed an all-black warp conditioning channel for the whole rollout
#   (`pool=0 cov=0.000`) -- a state that never occurs in training. Measured consequence on WorldScore's
#   dynamic split (all 1000 cases are camera_path "fixed"): motion_magnitude flow 0.327 against a
##   dataset mean of 3.24.
#
#   t2v is always warp-free (the CLI rejects warp+t2v), so WARP is a no-op there.
WARP = _env("WARP", "on")
if WARP not in ("on", "off"):
    sys.exit(f"[ERROR] WARP must be on|off, got {WARP!r}")
GEO_ON = MODE != "t2v" and WARP == "on"
IS_STAGE2 = _env("IS_STAGE2", "1")   # 1 = pyramid (distilled models); 0 = no pyramid (multi-step stage1 model)
STAGE2_NUM_STAGES = _env("STAGE2_NUM_STAGES", "3")
STAGE2_STEPS = _env("STAGE2_STEPS", "1 1 1").split()
STAGE2_STAGE_RANGE = _env("STAGE2_STAGE_RANGE", "0 0.3333333333333333 0.6666666666666666 1").split()
# DMD timestep schedule. AMPLIFY_FIRST_CHUNK=1 gives the FIRST chunk 2 sampling steps per pyramid stage
#   instead of 1 (pipeline_evoke.py; read only inside the use_dmd branch, hence the pairing).
#   These exist because the post_distill ckpts were trained that way -- configs/training/
#   stage3_post_distillation.yaml sets is_train_dmd/is_amplify_first_chunk true and dmd_num_latent_sections 1, so the
#   student's own rollout ran its single chunk at 2 steps/stage (utils_evoke_post.py).
# Inference stays at **1 step per stage for every chunk** regardless, so
#   both default to 0 and the scored configs do not set them; they are here for ablation only. Keeping
#   them off also keeps results comparable with every run made before they existed.
USE_DMD = _env("USE_DMD", "0")
AMPLIFY_FIRST_CHUNK = _env("AMPLIFY_FIRST_CHUNK", "0")
# 1 (default) = one interpreter per SHARD: the shard's cases are handed to
#   infer_single.run_argv_batch, which builds the pipeline once and loops. 0 = the historical
#   one-interpreter-per-case layout, kept as an escape hatch (and for bisecting a suspected
#   cross-case state leak). Measured on this ckpt: per-case wall time 251 s with per-case processes,
#   of which ~131 s was rebuilding the pipeline -- identical work for every case in the shard.
IN_PROCESS_BATCH = _env("IN_PROCESS_BATCH", "1")
# Optional file-queue transport used by the Director UI. A preload launcher starts one
# infer_single --argv_server process and blocks there; later launchers only translate their JSONL
# rows into argv and submit them to that already-loaded process. Empty keeps the regular CLI/batch
# behaviour unchanged.
ARGV_SERVER_DIR = _env("EVOKE_ARGV_SERVER_DIR", "")
SERVER_PRELOAD = _env("EVOKE_SERVER_PRELOAD", "0") == "1"
SERVER_REQUEST_ID = _env("EVOKE_SERVER_REQUEST_ID", "")
# Debug artefacts. Both default on (they are how a bad case gets diagnosed), but they are pure
#   overhead for a large sweep: per 5-chunk case DUMP_GEO writes 2 mp4s per chunk (~8 MB) and
#   SAVE_SEGMENTS one more per chunk (~5 MB), and the encoding is on the critical path -- of a
#   measured 144 s per case only ~50 s is the diffusion itself. Turn them off for a scored run and
#   back on to debug a specific case.
DUMP_GEO = _env("DUMP_GEO", "1")            # geo_debug/chunk_NNN_{warp,vis}.mp4
SAVE_SEGMENTS = _env("SAVE_SEGMENTS", "1")  # segments/segment_NNN_pred.mp4
# BG_POSTPROC=1 hands the postprocess (PNG dump + mp4 encode) to a detached CPU worker so the next
#   case's GPU work overlaps it. Off by default because it changes when the outputs appear: DONE_MARK
#   (geo_pred.mp4) is written BY the postprocess, so the bookkeeping pass has to drain the workers
#   first -- see _drain_bg_postprocess. BG_POSTPROC_MAX bounds workers in flight per OUT_ROOT.
BG_POSTPROC = _env("BG_POSTPROC", "0")
BG_POSTPROC_MAX = _env("BG_POSTPROC_MAX", "4")
if BG_POSTPROC == "1" and IN_PROCESS_BATCH != "1":
    # The per-case path checks DONE_MARK immediately after each child exits, and the whole point of
    #   backgrounding is that the marker is not there yet -- it would report every case FAIL and rmdir
    #   the output that is still being written. Draining after each case instead would remove the
    #   overlap that makes it worth doing. Only the batch path defers the check to the end.
    sys.exit("[ERROR] BG_POSTPROC=1 requires IN_PROCESS_BATCH=1 (the per-case path checks DONE_MARK "
             "before the detached worker has written it).")
WARP_MODE = _env("WARP_MODE", "fixed_mem")
NOISE_CENTER = _env("NOISE_CENTER", "1")

# -- warp recipe (mirrors training config geometric_state / validation) --
RENDER_MODE = _env("RENDER_MODE", "backward_zbuf")   # cloud_warp.render_mode
BW_FILL_ITERS = _env("BW_FILL_ITERS", "12")          # cloud_warp.bw_fill_iters=12
ZBUF_KSIZE = _env("ZBUF_KSIZE", "3"); ZBUF_FILL = _env("ZBUF_FILL", "4")
WARP_SIGMA_MAX = _env("WARP_SIGMA_MAX", "0.333")     # = geometric_state.warp_noise_sigma_max
WARP_LAG = _env("WARP_LAG", "0")
POSE_TYPE = _env("POSE_TYPE", "vipe")   # pose annotation flavour recorded in the jsonl
DA3_SRC = _env("DA3_SRC", os.environ.get("EVOKE_DA3_SRC", ""))   # empty -> let the da3_cloud module resolve its default
DA3_WEIGHTS = _env("DA3_WEIGHTS", "models/DA3"); DA3_PROCESS_RES = _env("DA3_PROCESS_RES", "644")
# -- depth estimator behind the cloud warp (mirrors training cloud_warp.backend) --
#   Only DEPTH_BACKEND normally needs setting; the ViGeo knobs already default to the validated recipe.
DEPTH_BACKEND = _env("DEPTH_BACKEND", "vigeo")        # vigeo | da3  <- cloud_warp.backend
VIGEO_SRC = _env("VIGEO_SRC", os.environ.get("EVOKE_VIGEO_SRC", ""))       # empty -> vendored evoke/third_party/vigeo
VIGEO_WEIGHTS = _env("VIGEO_WEIGHTS", "models/ViGeo1.1")
VIGEO_MODE = _env("VIGEO_MODE", "chunk"); VIGEO_SCALE_MODE = _env("VIGEO_SCALE_MODE", "auto")
# auto (DEFAULT) = let the engine pick per sample type: depth_median for i2v, anchor for v2v
#   (infer_single._resolve_vigeo_scale_mode). This used to be a hard-coded "anchor", which OVERRODE the
#   engine default for every mode -- and anchor is unsolvable for i2v: it needs the depth scale as a ratio
#   against camera motion measured in the frames, but i2v chunk 0 has no real frames, so on a
#   single-direction track (push_in/pull_out/move_*) the ratio collapses (measured 6.1e-3 here),
#   the cloud lands ~165x too far out, every chunk renders cov=1.000 (a
#   still image) and the camera is dead for the whole rollout. Verified A/B: anchor -> 0.000%/frame
#   zoom, depth_median -> 0.20-0.45%.
# per_window | anchor (both solve depth scale from the camera baseline) | depth_median | fixed
#   (baseline-free). A pose track with zero commanded translation (pure pan/tilt) has no baseline at all,
#   so anchor/per_window skip every window and the warp stays black -- use depth_median there.
VIGEO_SCALE_VALUE = _env("VIGEO_SCALE_VALUE", "0.0")            # required (>0) iff SCALE_MODE=fixed
# SCALE_MODE=depth_median only, i.e. i2v. "How deep the world is" in pose units: after scaling, the
#   scene's median depth IS this number, so it converts a commanded translation into parallax. 1.0 is the
#   bare unit definition and was the default until it was calibrated -- at 1.0 a sekai 5 s track commands
#   ~1.8 scene-depths per chunk, so the camera walks out of the geometry, the warp collapses to holes and
#   the model re-invents the scene, which looks like a sudden acceleration even though the commanded
#   motion is flat. Over the 6 worst of 100 i2v cases, min rollout warp coverage 0.163 median / 0.020
#   worst at 1.0 vs 0.886 / 0.571 at 5, with the steepest 0.5 s window dropping from 4.21x the clip
#   median to 1.79x while mean speed falls only 20 %. v2v solves ~9.9 for the same quantity from real
#   camera motion, so 10 is the physically measured value and safer on worst-case coverage; 5 keeps twice
#   the parallax and is the preferred look. Set 1.0 explicitly to reproduce a pre-calibration run.
VIGEO_DEPTH_MEDIAN_TARGET = _env("VIGEO_DEPTH_MEDIAN_TARGET", "5")
VIGEO_ANCHOR_WINDOWS = _env("VIGEO_ANCHOR_WINDOWS", "4")
VIGEO_CACHE_KEEP_FRAMES = _env("VIGEO_CACHE_KEEP_FRAMES", "6")
VIGEO_INTR_SOURCE = _env("VIGEO_INTR_SOURCE", "gt")   # gt (as da3) | vigeo (ablation)
RESTRICT = _env("RESTRICT", "1")
# Additive Plücker camera embedding on the warp+noise tokens. Legacy development knob: every shipped
#   model is trained with geometric_state.geo_warp_plucker_enabled=false, so this defaults to off and
#   no launcher sets it. The engine flag is kept for older plucker-trained checkpoints (PLUCKER=1).
#   NOTE: off here only means "do not force it on" -- for a merged checkpoint the constructor still
#   honours that ckpt's own config.json; overriding one that says true needs the engine's
#   --geo_warp_plucker_disabled.
PLUCKER = _env("PLUCKER", "0")
# Joystick/energy-ring HUD burned into geo_pred.mp4 + the per-chunk pred segments. auto = on for every
#   pose-carrying mode (v2v/i2v/segment), off for t2v; set JOYSTICK_HUD=off for a clean pred video.
JOYSTICK_HUD = _env("JOYSTICK_HUD", "auto")
STAGE0_ONLY = _env("GEO_WARP_STAGE0_ONLY", "0")   # 1 = inject warp into the coarse pyramid stage only (warp_stage0_only=true)
# i2v chunk-0 reference-image warp (default on in infer_single). 0 = disable -> chunk 0 falls back to the
# black warp / static first chunk. Only affects i2v; v2v/t2v ignore it.
CHUNK0_REF_WARP = _env("GEO_CHUNK0_REF_WARP", "1")
# i2v chunk-0 amplitude, as a p90 pixel budget for the commanded parallax. Empty -> the engine's own
# calibrated default (da3_cloud.CHUNK0_TARGET_DISPARITY_PX_DEFAULT, whose comment carries the calibration
# table); 0 -> off, chunk 0 then keeps whatever amplitude the reference frame's depth median happens to
# give, which is what made it run ~6x the flow of its own later chunks. See README.md "i2v chunk-0
# camera amplitude".
CHUNK0_TARGET_DISP_PX = _env("GEO_CHUNK0_TARGET_DISP_PX", "")
ZBUF_DESPECKLE = _env("ZBUF_DESPECKLE", "1")       # 0 = no salt-and-pepper cleanup (matches configs that leave zbuf_despeckle unset)
# persistent = correct for this ckpt: its per-chunk first latent is CONTINUOUS-frame distributed (NOT
# I-frame), so continuous cross-chunk decode is flicker-free (0.98-1.13x |dframe| at every boundary on
# real latents). default/default_warm0 decode each chunk head as an I-frame -> ~4.5x flicker at every
# boundary here; only use them for a ckpt verified to emit I-frame-distributed chunk heads.
VAE_DECODE_TYPE = _env("VAE_DECODE_TYPE", "persistent")  # default/persistent/default_warm0
# Very long rollouts (>10min): 1 = stream output; the pipeline stops accumulating full-clip
#   pixels and the final video is stitched from segments. 60min@384x640 as one tensor would be
#   254GB on GPU (fp32) / 64GB on CPU (uint8), i.e. guaranteed OOM without this.
STREAM_LONG = _env("STREAM_LONG", "0")
# DA3 point-cloud sliding window (gid = pixel frame index; 0 = unbounded). Required for very long
#   rollouts: the cloud otherwise grows monotonically (12 dense depth frames per chunk, hundreds of
#   GB over 2400 chunks). The window must be much larger than lag*stride.
GEO_HIST_MAX = _env("GEO_HIST_MAX_FRAMES", "0")
# Artifact used for done/skip detection. geo_pred.mp4 is the only output every mode produces:
#   the 4-panel comparison needs a GT reference video (v2v only), and even there the combine step
#   can fail on its own (postprocess_viz swallows the exception) -- keying on it would report a
#   perfectly generated case as FAIL and regenerate it on every resume. VIZ_MARK is reported only.
DONE_MARK = "geo_pred.mp4"
VIZ_MARK = "gt_vs_pred_cam_viz.mp4"
# Per-chunk camera motion clamp: real sekai trajectories move far enough per chunk that warp and
#   rollout degrade without a cap. deg = rotation cap (degrees), trans = translation cap;
#   lower them (e.g. 6/2) if the result still shakes.
CAP_MAX_DEG = _env("CAP_MAX_DEG", "10"); CAP_MAX_TRANS = _env("CAP_MAX_TRANS", "3")
CAP_MODE = _env("CAP_MODE", "clamp"); POSE_SMOOTH_WIN = _env("POSE_SMOOTH_WIN", "5")
# How to extend a pose track shorter than num_frames: relative_replay = replay relative motion
#   (default, keeps moving); clamp = freeze on the last frame
POSE_EXTEND_MODE = _env("POSE_EXTEND_MODE", "relative_replay")

NEG = _env("NEGATIVE_PROMPT", "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background")


def resolve_under(root, rel):
    """Resolve a jsonl path against the dataset root, falling back to repo-relative.

    The full sekai sweep stores paths relative to VROOT / AROOT (a mounted dataset), while the bundled
    demo (examples/v2v/cases.jsonl) stores them relative to the repo. Trying the dataset root
    first keeps the sweep unchanged; the fallback lets the demo run without setting any roots.
    Neither hit -> return the dataset-root form so the "missing" message points where it looked.
    """
    joined = os.path.join(root, rel)
    if os.path.isfile(joined):
        return joined
    return rel if os.path.isfile(rel) else joined


def load_prompt(prompt_path):
    """Read a caption json: overall.full_prompt (preferred), else short_prompt / description."""
    try:
        cj = json.load(open(prompt_path))
        ov = cj.get("overall", {}) if isinstance(cj, dict) else {}
        for k in ("full_prompt", "short_prompt", "description"):
            v = ov.get(k)
            if isinstance(v, str) and len(v) > 10:
                return v
    except Exception as e:
        print(f"[warn] failed to read caption {prompt_path}: {e}", flush=True)
    return "A first-person walk through an outdoor scene with continuous forward camera motion."


def main():
    if NOISE_CENTER == "1" and WARP_MODE != "fixed_mem":
        sys.exit(f"[ERROR] --warp_rope_noise_center_align requires WARP_MODE=fixed_mem (got {WARP_MODE})")
    if not TRANSFORMER_PATH:
        sys.exit("[ERROR] TRANSFORMER_PATH is required (the parent of the transformer/ dir). "
                 "Run one of the scripts/inference/infer_*.sh launchers, which set it for you.")
    if not (Path(TRANSFORMER_PATH) / "transformer").is_dir():
        sys.exit(f"[ERROR] transformer not found: {Path(TRANSFORMER_PATH) / 'transformer'}\n"
                 f"        TRANSFORMER_PATH must be the PARENT of the transformer/ dir -- the weights are\n"
                 f"        loaded with from_pretrained(TRANSFORMER_PATH, subfolder=\"transformer\").\n"
                 f"        Got TRANSFORMER_PATH={TRANSFORMER_PATH!r}.")
    # Full-clip pixels are accumulated on GPU in fp32 (~2.95 MB/frame at 384x640) unless streaming,
    #   so a long rollout OOMs deep into the run instead of failing now. 7200 frames ~= 5min ~= 21 GB.
    if int(NUM_FRAMES) > 7200 and STREAM_LONG != "1":
        _gb = int(NUM_FRAMES) * 2.95 / 1024
        sys.exit(f"[ERROR] NUM_FRAMES={NUM_FRAMES} would accumulate the whole clip on GPU (~{_gb:.0f} GB fp32).\n"
                 f"        Set STREAM_LONG=1 to decode per chunk and stitch the final mp4 from segments/\n"
                 f"        (requires VAE_DECODE_TYPE=persistent, the launcher default). For hour-scale\n"
                 f"        rollouts also consider GEO_HIST_MAX_FRAMES -- see scripts/inference/README.md.")
    if STREAM_LONG == "1" and VAE_DECODE_TYPE != "persistent":
        sys.exit(f"[ERROR] STREAM_LONG=1 requires VAE_DECODE_TYPE=persistent (got {VAE_DECODE_TYPE!r}); "
                 f"the full video is stitched from per-chunk segments.")

    recs = [json.loads(l) for l in open(JSONL) if l.strip()]
    if MAX_CASES > 0:
        recs = recs[:MAX_CASES]
    mine = [(i, r) for i, r in enumerate(recs) if i % NSHARD == SHARD]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[{MODE} shard {SHARD}/{NSHARD}] total={len(recs)} mine={len(mine)} frames={NUM_FRAMES} "
          f"warp={'on' if GEO_ON else 'OFF'} transformer={TRANSFORMER_PATH} out={OUT_ROOT}", flush=True)
    # Which checkpoint produced this directory is not recoverable from the videos, and OUT_ROOT is
    #   routinely reused across checkpoints -- record it next to the results. Shard 0 writes it; the
    #   other shards run the same settings.
    if SHARD == 0:
        (OUT_ROOT / "run_info.json").write_text(json.dumps({
            "transformer_path": TRANSFORMER_PATH, "base_ckpt": BASE_CKPT, "mode": MODE, "jsonl": JSONL,
            "geometric_state": GEO_ON,
            "num_frames": NUM_FRAMES, "num_chunks": NUM_CHUNKS or None, "height": HEIGHT, "width": WIDTH,
            # seed_default: a jsonl row may carry its own "seed", which wins. Rows that do not get this.
            "fps": FPS, "seed_default": SEED,
            "seed_per_case": sum(1 for r in recs if r.get("seed") is not None),
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": NUM_INFERENCE_STEPS, "stage2_steps": STAGE2_STEPS if IS_STAGE2 == "1" else None,
            "use_dmd": USE_DMD, "amplify_first_chunk": AMPLIFY_FIRST_CHUNK,
            # resolved, not the literal "auto": run_info.json is read months later to answer "what recipe
            #   produced these videos", and the scale mode is the knob that decides whether the camera
            #   moved at all, so it must not require re-deriving the auto rule to interpret.
            "vigeo_scale_mode": (VIGEO_SCALE_MODE if VIGEO_SCALE_MODE != "auto"
                                 else ("anchor" if MODE == "v2v" else "depth_median")),
            "vigeo_scale_mode_requested": VIGEO_SCALE_MODE,
            "vigeo_scale_value": VIGEO_SCALE_VALUE,
            "vigeo_depth_median_target": VIGEO_DEPTH_MEDIAN_TARGET,
            # NOT resolved to a number here, unlike scale_mode above: the default lives in
            # da3_cloud.CHUNK0_TARGET_DISPARITY_PX_DEFAULT, and reading it would mean importing torch into
            # this launcher (kept dependency-free on purpose: json/os/subprocess/sys only) or restating the
            # number here, which would silently drift from the engine. The resolved value is printed per
            # case as "[geo-infer] chunk0_target_disparity_px = ..." in _logs/<case>.log.
            "chunk0_target_disparity_px": CHUNK0_TARGET_DISP_PX or "engine-default",
            "cap_max_deg": CAP_MAX_DEG, "cap_max_trans": CAP_MAX_TRANS, "pose_smooth_win": POSE_SMOOTH_WIN,
            "ref_video_sec": REF_VIDEO_SEC if MODE == "v2v" else "0.0", "start_seconds": START_SECONDS,
            "warp_sigma_max": WARP_SIGMA_MAX, "warp_stage0_only": STAGE0_ONLY, "restrict_self_attn": RESTRICT,
            "vae_decode_type": VAE_DECODE_TYPE, "stream_long": STREAM_LONG,
            "joystick_hud": JOYSTICK_HUD,
        }, indent=2) + "\n")

    _rst = ["--restrict_self_attn", "--use_kv_cache"] if RESTRICT == "1" else []
    _plk = ["--geo_warp_plucker_enabled"] if PLUCKER == "1" else []
    _nc = ["--warp_rope_noise_center_align"] if NOISE_CENTER == "1" else []
    done = skip = fail = skip_missing = 0
    pending: list = []        # IN_PROCESS_BATCH=1: {"argv","name","log"} per case, run in one child
    pending_meta: list = []   # (name, out_dir, log_path), for the DONE/FAIL pass after that child exits
    for n, (i, r) in enumerate(mine):
        # -- resolve one record; the three modes read different fields --
        #    v2v: video_path / pose_path / prompt_path, resolved under VROOT / AROOT, else repo-relative
        #    i2v: image_path / pose_path / prompt_path, all relative to the repo (examples/*/cases.jsonl)
        #    t2v: prompt only; no pose, no warp (the CLI rejects warp+t2v outright)
        video = image = pose = None
        if MODE == "v2v":
            # Prefer the row's own name (i2v/t2v already do); fall back to the video stem, which is what
            #   the sekai sweep relies on -- its rows carry no "name" and the stem is the clip id.
            name = r.get("name") or Path(r["video_path"]).stem
            video = resolve_under(VROOT, r["video_path"])
            pose = resolve_under(AROOT, r["pose_path"])
            prompt = load_prompt(resolve_under(AROOT, r["prompt_path"]))
            p_fps = str(int(round(float(r.get("video_fps", 30)))))
            # sekai rows carry no pose_source_resolution and were estimated at 720p
            src_h, src_w = [str(int(x)) for x in r.get("pose_source_resolution", [720, 1280])]
        else:
            name = r.get("name") or Path(r.get("image_path", f"case_{i:04d}")).parent.name
            prompt = (Path(r["prompt_path"]).read_text().strip() if r.get("prompt_path")
                      else r.get("prompt", ""))
            p_fps = str(int(r.get("pose_fps", 24)))
            src_h, src_w = [str(int(x)) for x in r.get("pose_source_resolution", [480, 832])]
            if MODE == "i2v":
                image = r["image_path"]
                # WARP=off needs no pose at all; rows for that path may omit "pose_path".
                pose = r["pose_path"] if GEO_ON else None
                if GEO_ON and not pose:
                    sys.exit(f"[ERROR] {name}: i2v with warp on needs \"pose_path\" in the jsonl row. "
                             f"Run with WARP=off to generate from image + prompt only.")
        # Segment prompts (optional): a row may carry
        #   "segment_prompts": [{"start_sec": 0, "prompt": "..."}, {"start_sec": 12, "prompt": "..."}]
        #   or a "segment_prompts_path" pointing at a JSON file with the same shape.
        #   Empty -> the single `prompt` is used for every chunk (baseline).
        _sched = r.get("segment_prompts")
        if _sched is None and r.get("segment_prompts_path"):
            _sched = json.loads(Path(r["segment_prompts_path"]).read_text())
        # Optional per-case "seed". Needed by benchmarks that sample several videos per prompt from the
        # same conditioning: VBench's asks for a fresh random seed per sample, and
        #   with the run-level SEED those samples come out byte-identical. `seed` is a per-case arg on
        #   the infer_single side, so varying it does NOT rebuild the pipeline.
        _seed = str(int(r["seed"])) if r.get("seed") is not None else SEED
        # Optional "event_chunks": [2, 4] -- 0-indexed chunks that drop warp (also: static camera, and
        #   the chunk is skipped from the frame bank). Paired with a schedule this is the "drop warp on
        #   the first chunk of each new segment" recipe; see scripts/inference/README.md for when it
        #   helps and what it costs. Per-case rather than an env var so one jsonl can mix cases with
        #   different switch points. event_prompt is deliberately NOT exposed: it would override the
        #   schedule on every event chunk with one text, which is wrong as soon as there are 2 switches.
        _events = ",".join(str(int(x)) for x in (r.get("event_chunks") or []))
        if _events and MODE == "t2v":
            print(f"[warn] {name}: event_chunks ignored, t2v has no warp to drop", flush=True)
            _events = ""
        out_dir = OUT_ROOT / name
        _required = [(video, "video"), (image, "image"), (pose, "pose")]
        _missing = [(pth, tag) for pth, tag in _required if pth and not os.path.isfile(pth)]
        if _missing:
            print(f"[SKIP] {name}: missing {_missing[0][1]} {_missing[0][0]}", flush=True)
            skip += 1; skip_missing += 1; continue
        if (out_dir / DONE_MARK).is_file():
            print(f"[SKIP] {name}: exists", flush=True); skip += 1; continue
        out_dir.mkdir(parents=True, exist_ok=True)

        argv = [
            # sys.executable, not "python": the child must be the same interpreter as this driver,
            #   otherwise a PATH python without torch/cv2 fails once per case.
            sys.executable, "scripts/inference/infer_single.py",
            "--ckpt_path", BASE_CKPT, "--transformer_path", TRANSFORMER_PATH,
            *(["--is_enable_stage2", "--stage2_num_stages", STAGE2_NUM_STAGES,
               "--stage2_steps", *STAGE2_STEPS,
               "--stage2_stage_range", *STAGE2_STAGE_RANGE,
               "--stage2_warp_compression_mode", WARP_MODE] if IS_STAGE2 == "1" else []),
            *(["--use_dmd"] if USE_DMD == "1" else []),
            *(["--is_amplify_first_chunk"] if AMPLIFY_FIRST_CHUNK == "1" else []),
            "--height", HEIGHT, "--width", WIDTH, "--num_frames", NUM_FRAMES, "--fps", FPS,
            "--num_inference_steps", NUM_INFERENCE_STEPS, "--guidance_scale", GUIDANCE_SCALE, "--seed", _seed,
            "--vae_decode_type", VAE_DECODE_TYPE,
            "--start_seconds", START_SECONDS,
            # i2v/t2v have no reference video, so no GT prefix and no 4-panel comparison
            "--ref_seconds", REF_VIDEO_SEC if MODE == "v2v" else "0.0",
            "--no_raw_sink_frames",
            # t2v has no source pixels, and the CLI rejects --use_geometric_state with t2v.
            #   WARP=off takes the same branch for i2v/v2v -- see GEO_ON.
            *(["--use_geometric_state",
               "--lingbot_pose_path", pose, "--lingbot_pose_source_fps", p_fps,
               "--lingbot_pose_source_resolution", src_h, src_w, "--lingbot_pose_type", POSE_TYPE,
               "--visibility_aware_noise"] if GEO_ON else []),
            *(["--ref_video_for_viz", video] if MODE == "v2v" else []),
            "--warp_noise_sigma_invisible", "1.0", "--warp_noise_sigma_min", "0.0", "--warp_noise_sigma_max", WARP_SIGMA_MAX,
            *(["--visible_token_threshold", "0.5",
               "--prefix_idx_mode", "zero", "--warp_rope_mode", "overlap_noise",
               "--warp_lag_chunks", WARP_LAG,
               "--geo_recon_backend", "da3", "--geo_cloud_update_n", "12",
               *(["--geo_da3_src", DA3_SRC] if DA3_SRC else []),
               "--geo_da3_weights", DA3_WEIGHTS, "--geo_da3_process_res", DA3_PROCESS_RES,
               "--geo_depth_backend", DEPTH_BACKEND,
               *(["--geo_vigeo_weights", VIGEO_WEIGHTS,
                  "--geo_vigeo_mode", VIGEO_MODE, "--geo_vigeo_scale_mode", VIGEO_SCALE_MODE,
                  "--geo_vigeo_scale_value", VIGEO_SCALE_VALUE,
                  "--geo_vigeo_depth_median_target", VIGEO_DEPTH_MEDIAN_TARGET,
                  "--geo_vigeo_anchor_windows", VIGEO_ANCHOR_WINDOWS,
                  "--geo_vigeo_cache_keep_frames", VIGEO_CACHE_KEEP_FRAMES,
                  "--geo_vigeo_intr_source", VIGEO_INTR_SOURCE,
                  *(["--geo_vigeo_src", VIGEO_SRC] if VIGEO_SRC else [])]
                 if DEPTH_BACKEND == "vigeo" else []),
               "--geo_da3_render_mode", RENDER_MODE, "--geo_bw_fill_iters", BW_FILL_ITERS,
               *(["--geo_warp_stage0_only"] if STAGE0_ONLY == "1" else []),
               *(["--no_geo_chunk0_ref_warp"] if CHUNK0_REF_WARP == "0" else []),
               *(["--geo_chunk0_target_disparity_px", CHUNK0_TARGET_DISP_PX]
                 if CHUNK0_TARGET_DISP_PX else []),
               *_plk, *_nc] if GEO_ON else []),
            # training configs keep geo_invisible_history_noise=False, so --invisible_history_noise
            #   is intentionally not passed here (matches validation).
            # Segment output + 4 panels (gt/warp/vis/pred): segments/segment_NNN_{gt_vs_pred,pred}.mp4
            #   (only supported by persistent decode; skipped otherwise to avoid a warning per case)
            *(["--save_chunk_segments"]
              if VAE_DECODE_TYPE == "persistent" and SAVE_SEGMENTS == "1" else []),
            *(["--stream_long_video"] if STREAM_LONG == "1" else []),
            *(["--geo_hist_max_frames", GEO_HIST_MAX] if int(GEO_HIST_MAX) > 0 else []),
            *(["--dump_geo_intermediates"] if DUMP_GEO == "1" else []),
            *(["--bg_postprocess", "--bg_postprocess_max", BG_POSTPROC_MAX]
              if BG_POSTPROC == "1" else []),
            "--joystick_hud", JOYSTICK_HUD,
            *(["--prompt_schedule", json.dumps(_sched, ensure_ascii=False)] if _sched else []),
            *(["--event_chunks", _events] if _events else []),
            "--sample_type", MODE, "--prompt", prompt, "--negative_prompt", NEG,
            "--image_noise_sigma_min", "0.0", "--image_noise_sigma_max", "0.0",
            *(["--video_path", video,
               "--video_noise_sigma_min", "0.0", "--video_noise_sigma_max", "0.0"] if MODE == "v2v" else []),
            *(["--image_path", image] if MODE == "i2v" else []),
            "--output_folder", str(out_dir),
        ]
        if GEO_ON and RENDER_MODE == "backward_zbuf" and ZBUF_DESPECKLE == "1":
            argv += ["--geo_zbuf_despeckle", "--geo_zbuf_despeckle_ksize", ZBUF_KSIZE, "--geo_zbuf_despeckle_fill_iters", ZBUF_FILL]
        argv += _rst
        if MODE != "t2v":   # camera-motion caps and pose extension only exist when a pose track is fed
            if float(CAP_MAX_DEG) > 0: argv += ["--max_deg_per_chunk", CAP_MAX_DEG]
            if float(CAP_MAX_TRANS) > 0: argv += ["--max_trans_per_chunk", CAP_MAX_TRANS]
            if CAP_MAX_DEG != "0" or CAP_MAX_TRANS != "0": argv += ["--cap_mode", CAP_MODE]
            if float(POSE_SMOOTH_WIN) > 1: argv += ["--pose_smooth_win", POSE_SMOOTH_WIN]
            argv += ["--pose_extend_mode", POSE_EXTEND_MODE]

        _log_dir = OUT_ROOT / "_logs"; _log_dir.mkdir(parents=True, exist_ok=True)

        if IN_PROCESS_BATCH == "1":
            # One interpreter for the whole shard instead of one per case. Building the pipeline is
            #   ~131 s of the measured 251 s per-case wall time for this ckpt (54 GB transformer + 22 GB
            #   text encoder read over a network filesystem) and it is identical for every case in the
            #   shard, so paying it per case was over half the sweep. Collect the argv here; the batch
            #   runner below hands them to infer_single.run_argv_batch, which keeps the per-case logs
            #   (it redirects stdout per case) and leaves the DONE/FAIL bookkeeping to us, on the same
            #   DONE_MARK check, after the child exits.
            pending.append({"argv": argv[2:], "name": name,
                            "log": str(_log_dir / f"{name}.log")})
            pending_meta.append((name, out_dir, _log_dir / f"{name}.log"))
            continue

        print(f"\n==== [shard {SHARD}] {n+1}/{len(mine)} idx{i} {name} "
              f"[{TRANSFORMER_PATH}] ====", flush=True)
        env = dict(os.environ); env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
        # The chunk-level progress bar is live terminal UI, so it is only worth drawing when someone is
        #   watching: with several shards nothing reaches a terminal and the engine's plain per-chunk
        #   prints are the better record. Gated by an env var because the same pipeline code runs
        #   training validation, where a bar per rank per chunk would flood the log.
        env["EVOKE_INFER_PROGRESS"] = "1" if VERBOSE else "0"
        # The vendored DA3 logger prints three INFO lines per chunk (its own env-var log level, not
        #   python logging). They are depth-model timings nobody reads, and they interleave with the bar.
        env["DA3_LOG_LEVEL"] = os.environ.get(
            "DA3_LOG_LEVEL", "INFO" if _env("EVOKE_INFER_DEBUG", "0") == "1" else "WARN")
        log_dir = OUT_ROOT / "_logs"; log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"   # kept outside out_dir so it survives infer clearing output_folder
        if VERBOSE:
            # Forward raw bytes as they arrive: tqdm redraws with \r and no newline, so line buffering
            #   would hold the bar back until the chunk finished. bufsize=0 keeps read() one syscall.
            proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=0)
            with open(log_path, "wb") as lg:
                tail = b""   # partial line; a \r-redrawn bar frame lives here until a newline arrives
                while True:
                    buf = os.read(proc.stdout.fileno(), 65536)
                    if not buf:
                        break
                    # Terminal gets the stream verbatim, bar redraws included.
                    sys.stdout.buffer.write(buf); sys.stdout.buffer.flush()
                    # The log gets completed lines only, each reduced to what survived the last \r.
                    #   A progress bar redraws with \r and no newline, so its intermediate frames are
                    #   overwritten and never reach the file -- what lands is the text a reader wants.
                    tail += buf
                    if b"\n" in tail:
                        *lines, tail = tail.split(b"\n")
                        lg.write(b"".join(l.rsplit(b"\r", 1)[-1] + b"\n" for l in lines)); lg.flush()
                if tail.rsplit(b"\r", 1)[-1]:
                    lg.write(tail.rsplit(b"\r", 1)[-1] + b"\n")
            proc.stdout.close()
            rc = proc.wait()
        else:
            with open(log_path, "w") as lg:
                rc = subprocess.run(argv, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        # Done/skip detection keys on geo_pred.mp4 (see DONE_MARK): it is the one artifact every mode
        #   produces. The 4-panel comparison is reported but NOT required -- postprocess_viz swallows
        #   errors from the combine step, so keying on it would mark a perfectly generated case FAIL
        #   and regenerate it from scratch on every resume.
        if rc == 0 and (out_dir / DONE_MARK).is_file():
            _extra = "" if (out_dir / VIZ_MARK).is_file() or MODE != "v2v" else f" (no {VIZ_MARK})"
            print(f"[DONE] {name}{_extra}", flush=True); done += 1
        else:
            print(f"[FAIL] {name} rc={rc} - see {log_path}", flush=True); fail += 1
            # out_dir was created before the child ran, so a crash leaves an empty directory behind that
            #   looks like a finished case in a listing. The log lives outside it and is kept.
            try:
                if not any(out_dir.iterdir()):
                    out_dir.rmdir()
            except OSError:
                pass

    # ── in-process batch: one interpreter for the whole shard, pipeline built once ──
    if IN_PROCESS_BATCH == "1" and pending:
        batch_file = OUT_ROOT / f"_argv_batch_shard{SHARD}.jsonl"
        batch_file.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in pending))
        env = dict(os.environ); env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
        env["EVOKE_INFER_PROGRESS"] = "1" if VERBOSE else "0"
        env["DA3_LOG_LEVEL"] = os.environ.get(
            "DA3_LOG_LEVEL", "INFO" if _env("EVOKE_INFER_DEBUG", "0") == "1" else "WARN")
        # Thread caps. torch/OpenMP/OpenCV each size their pools from the machine's core count, so with
        #   NSHARD shards every one of them wants all 64 cores -> ~8x oversubscription on the CPU-side
        #   work (VAE decode post-processing, PNG deflate, mp4 encode) that dominates the non-diffusion
        #   half of a case. Give each shard its fair share instead. Only set when the caller has not,
        #   so an explicit OMP_NUM_THREADS still wins.
        # numexpr hard-caps its pool at 64; advertising the host's full 192 cores makes it reject the
        # setting at first job import and prints a misleading error into the persistent-worker log.
        _share = min(64, max(1, (os.cpu_count() or 8) // max(1, NSHARD)))
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "CV_NUM_THREADS"):
            env.setdefault(_v, str(_share))
        env.setdefault("EVOKE_CPU_THREADS", str(_share))
        shard_log = OUT_ROOT / "_logs" / f"_shard{SHARD}_batch.log"
        if ARGV_SERVER_DIR:
            server_dir = Path(ARGV_SERVER_DIR).resolve()
            requests_dir = server_dir / "requests"
            responses_dir = server_dir / "responses"
            requests_dir.mkdir(parents=True, exist_ok=True)
            responses_dir.mkdir(parents=True, exist_ok=True)
            state_path = server_dir / "state.json"

            def _atomic_json(path: Path, payload: dict) -> None:
                temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
                temporary.replace(path)

            if SERVER_PRELOAD:
                template_path = server_dir / "template.json"
                _atomic_json(template_path, pending[0])
                print(f"\n==== [shard {SHARD}] PRELOADING persistent pipeline "
                      f"[{TRANSFORMER_PATH}] ====\n     queue: {server_dir}", flush=True)
                # This deliberately never returns during normal operation: the UI owns this launcher
                # as its long-lived GPU worker and terminates the process group on shutdown.
                rc_batch = subprocess.run(
                    [sys.executable, "scripts/inference/infer_single.py", "--argv_server",
                     str(template_path), str(server_dir)],
                    env=env).returncode
            else:
                if len(pending) != 1:
                    sys.exit("[ERROR] EVOKE argv server submissions must contain exactly one case")
                request_id = SERVER_REQUEST_ID or f"shard{SHARD}-{os.getpid()}-{int(time.time() * 1000)}"
                request_path = requests_dir / f"{request_id}.json"
                response_path = responses_dir / f"{request_id}.json"
                response_path.unlink(missing_ok=True)
                _atomic_json(request_path, pending[0])
                print(f"\n==== [shard {SHARD}] submitted {pending[0]['name']} to persistent worker "
                      f"request={request_id} ====", flush=True)
                last_notice = 0.0
                while not response_path.is_file():
                    now = time.time()
                    if now - last_notice >= 10:
                        phase = "starting"
                        worker_pid = None
                        try:
                            state = json.loads(state_path.read_text(encoding="utf-8"))
                            phase = state.get("phase", phase)
                            worker_pid = state.get("pid")
                        except (OSError, json.JSONDecodeError):
                            pass
                        if worker_pid:
                            try:
                                os.kill(int(worker_pid), 0)
                            except (ProcessLookupError, ValueError):
                                sys.exit(f"[ERROR] persistent worker exited while request {request_id} was pending")
                            except PermissionError:
                                pass
                        print(f"[worker] request={request_id} phase={phase}", flush=True)
                        last_notice = now
                    time.sleep(0.5)
                response = json.loads(response_path.read_text(encoding="utf-8"))
                rc_batch = int(response.get("returnCode", 1))
                print(f"[worker] request={request_id} finished rc={rc_batch} "
                      f"elapsed={float(response.get('elapsed', 0)):.1f}s", flush=True)
        else:
            print(f"\n==== [shard {SHARD}] {len(pending)} case(s) in ONE process "
                  f"[{TRANSFORMER_PATH}] ====\n     argv batch: {batch_file}", flush=True)
            # The child prints one progress line per case (the engine's own output goes to the per-case
            # logs it redirects to), so a plain tee is enough here -- no \r bar to preserve.
            with open(shard_log, "w") as lg:
                rc_batch = subprocess.run(
                    [sys.executable, "scripts/inference/infer_single.py", "--argv_jsonl", str(batch_file)],
                    env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        # With BG_POSTPROC the child returns before its detached workers have written DONE_MARK, so the
        #   bookkeeping below would mark good cases FAIL and rmdir their (still-being-filled) output.
        if BG_POSTPROC == "1":
            _drain_bg_postprocess(OUT_ROOT)
        # Bookkeeping is per case and on the same marker as the one-process-per-case path, so a
        #   partially failed batch is still reported case by case (and is resumable on a re-run).
        for name, out_dir, log_path in pending_meta:
            if (out_dir / DONE_MARK).is_file():
                _extra = "" if (out_dir / VIZ_MARK).is_file() or MODE != "v2v" else f" (no {VIZ_MARK})"
                print(f"[DONE] {name}{_extra}", flush=True); done += 1
            else:
                print(f"[FAIL] {name} - see {log_path} (batch rc={rc_batch}, "
                      f"shard log {shard_log})", flush=True); fail += 1
                try:
                    if not any(out_dir.iterdir()):
                        out_dir.rmdir()
                except OSError:
                    pass

    print(f"\n[shard {SHARD}/{NSHARD}] FINISHED done={done} skip={skip} fail={fail}", flush=True)
    # Nothing resolved at all -> almost always unset/wrong dataset roots. Exiting 0 here would print
    #   "DONE failed_shards=0" from the launcher without having produced a single video.
    if mine and skip_missing == len(mine):
        sys.exit(f"[ERROR] all {len(mine)} cases skipped: no input file resolved.\n"
                 f"        jsonl paths are tried under the dataset roots first, then repo-relative.\n"
                 f"        VROOT={VROOT!r} AROOT={AROOT!r} JSONL={JSONL!r}\n"
                 f"        Set VROOT / AROOT for a mounted dataset, or use a jsonl with repo-relative "
                 f"paths (examples/*/cases.jsonl).")
    # Non-zero exit on per-case failures, so the launcher's failed_shards count is meaningful:
    #   otherwise a shard where every case crashed still reports "DONE failed_shards=0".
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
