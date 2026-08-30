"""geometric state inference entry point using the in-repo EvokePipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# True when the pipeline draws its chunk progress bar: per-chunk chatter is folded into that bar's
#   summary line instead of being printed alongside it. EVOKE_INFER_DEBUG=1 restores every print.
_PROGRESS_UI = (os.environ.get("EVOKE_INFER_PROGRESS", "0") == "1"
                and os.environ.get("EVOKE_INFER_DEBUG", "0") != "1")
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]   # scripts/inference/<this> -> repo root
if str(REPO_ROOT) not in sys.path:                # so `import evoke` works when run directly
    sys.path.insert(0, str(REPO_ROOT))

# EVOKE_CPU_THREADS: set by infer_batch to core_count // n_shards. The OMP_* env vars it also sets are
#   read at library init, but torch and OpenCV size their own pools independently of them, so cap those
#   here too. Unset -> leave every default alone (single-run behaviour is unchanged).
if os.environ.get("EVOKE_CPU_THREADS"):
    _n_cpu = max(1, int(os.environ["EVOKE_CPU_THREADS"]))
    torch.set_num_threads(_n_cpu)
    cv2.setNumThreads(_n_cpu)


# Single source of truth for the chunk-0 disparity default: the engine owns it (the pipeline falls back
# to the same constant when the attribute is absent, e.g. training validation), the CLI only surfaces it.
# Import is safe here -- da3_cloud's module scope is numpy/torch/pathlib only, no DA3/ViGeo weight load.
from evoke.modules.geometric_state.da3_cloud import (                                   # noqa: E402
    CHUNK0_TARGET_DISPARITY_PX_DEFAULT as _CHUNK0_TARGET_DISPARITY_PX_DEFAULT)

_PARSER = None   # set by parse_args; lets checks read a flag's default rather than duplicating it


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GEO inference (in-repo pipe)")
    # === Model paths ===
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="merged ckpt dir with transformer/vae/scheduler/text_encoder/tokenizer subdirs")
    p.add_argument("--transformer_path", type=str, default=None,
                   help="optional: load transformer from here instead of ckpt_path (e.g. a merged stage2 base); "
                        "vae/scheduler/text_encoder still come from ckpt_path")
    p.add_argument("--lora_path", type=str, default=None,
                   help="optional: main LoRA adapter (.safetensors)")
    p.add_argument("--geo_lora_path", type=str, default=None,
                   help="optional: GEO LoRA adapter (.safetensors)")
    p.add_argument("--partial_path", type=str, default=None,
                   help="optional: transformer_partial.pth with memory conv weights; camera_ctrl.safetensors in same dir is auto-loaded")

    # === Generation ===
    p.add_argument("--sample_type", type=str, default="i2v", choices=["t2v", "i2v", "v2v"],
                   help="generation mode; GEO requires i2v or v2v")
    p.add_argument("--prompt", type=str, required=True)
    # CFG negative prompt (actually used).
    p.add_argument("--negative_prompt", type=str, default=(
        "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, "
        "static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "three legs, many people in the background, walking backwards, messy background"
    ))
    p.add_argument("--image_path", type=str, default=None,
                   help="required for i2v/v2v + GEO (Pi3X source pixel)")
    p.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    p.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    p.add_argument("--video_path", type=str, default=None,
                   help="required for v2v (ref video conditioning)")
    p.add_argument("--video_noise_sigma_min", type=float, default=0.111)
    p.add_argument("--video_noise_sigma_max", type=float, default=0.135)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=165, help="pixel frame count; 1 chunk=33; 165=5 chunks")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    # === Stage 2 (NaViT pyramid) ===
    p.add_argument("--is_enable_stage2", action="store_true", default=False,
                   help="enable stage2 NaViT pyramid inference (coarse->fine across stages)")
    p.add_argument("--stage2_num_stages", type=int, default=3)
    p.add_argument("--stage2_steps", type=int, nargs="+", default=[3, 3, 3],
                   help="per-stage pyramid inference steps (len == stage2_num_stages)")
    p.add_argument("--use_dmd", action="store_true", default=False,
                   help="DMD timestep schedule (set it for a DMD-distilled ckpt, i.e. one trained with "
                        "is_train_dmd=true). At stage2_steps=[1,1,1] it is algebraically identical to the "
                        "default path for every chunk EXCEPT the first when paired with "
                        "--is_amplify_first_chunk. Must match the ckpt's training config")
    p.add_argument("--is_amplify_first_chunk", action="store_true", default=False,
                   help="give the FIRST chunk 2 sampling steps per pyramid stage instead of 1 "
                        "(pipeline_evoke.py). Requires --use_dmd. Set it iff the ckpt was "
                        "trained with is_amplify_first_chunk=true -- chunk 0 has no warp history, so it "
                        "is where the recipe matters most and where its error propagates furthest")
    p.add_argument("--stage2_stage_range", type=float, nargs="+", default=[0, 1 / 3, 2 / 3, 1],
                   help="pyramid sigma-range boundaries (len == stage2_num_stages + 1). "
                        "MUST match training stage2_stage_range, else the per-stage renoise (alpha/beta) is "
                        "mis-calibrated and the coarse->full upsample block artifact survives (grid pattern). "
                        "3-stage: [0,1/3,2/3,1]; 2-stage pyr2lvl: [0,2/3,1].")
    p.add_argument("--stage2_warp_compression_mode", type=str, default="fixed_mem",
                   choices=["fixed_mem", "synchronized"],
                   help="how warp interacts with the pyramid: synchronized (follows noise per-stage) or fixed_mem (legacy)")
    p.add_argument("--vae_decode_type", type=str, default="persistent",
                   choices=["default", "persistent", "default_warm0"],
                   help="GEO chunk-level decode. 'persistent' (DEFAULT) = carry the decoder feat cache across "
                        "chunks and decode every chunk's latents as continuous frames (cache restored across the "
                        "interleaved warp vae.encode). This ckpt generates each chunk's first latent in the "
                        "CONTINUOUS-frame distribution (NOT the I-frame/prefix-anchor distribution), so continuous "
                        "decode is what matches it -> no boundary flicker (measured 0.98-1.13x |dframe| at every "
                        "chunk boundary on real DiT latents). chunk0 is warm-cache decoded (warm_repeat copies of "
                        "the image latent kill the cold-start over-saturated I-frame; the warm-up frames are "
                        "trimmed from the TAIL so the real content is kept -> f33 0.98x instead of 2.32x for a "
                        "head-trim). 'default'/'default_warm0' decode each chunk FRESH with first_chunk=True, i.e. "
                        "they decode every chunk's first latent as an I-frame; correct ONLY for a ckpt whose chunk "
                        "heads really are I-frame-distributed -- for this continuous-head ckpt they MIS-decode "
                        "every chunk head and produce a ~4.5x |dframe| FLICKER at every boundary. Do not switch "
                        "without re-running the real-latent decode check for the target ckpt. persistent is also "
                        "required by --save_chunk_segments / --stream_long_video. Length: persistent 36*chunks-3, "
                        "default/default_warm0 33*chunks.")
    p.add_argument("--use_raw_sink_frames", action="store_true", default=True,
                   help="align with training ckpt; default True")
    p.add_argument("--no_raw_sink_frames", dest="use_raw_sink_frames", action="store_false",
                   help="disable raw sink: transformer receives no sink/nearby_sink latents")
    p.add_argument("--enable_cam_control", action="store_true",
                   help="enable camera control; overrides transformer config to init cam module")
    p.add_argument("--cam_rank", type=int, default=128, help="cam control rank, match training yaml")
    p.add_argument("--save_chunk_segments", action="store_true",
                   help="requires vae_decode_type=persistent; dump per-chunk segment mp4 after each decode")
    p.add_argument("--dump_geo_intermediates", action="store_true",
                   help="dump GEO debug outputs (warp, visibility mask, keyframes) to <output_folder>/geo_debug/")
    # Streaming output for very long videos (>10min, hour scale): the pipeline stops accumulating
    #   full-clip pixels and the final video is stitched from segments. Off = byte-identical behaviour.
    p.add_argument("--stream_long_video", action="store_true",
                   help="stream mode for very long rollouts: pipeline keeps NO full-length pixel tensor "
                        "(frames=None) and the final mp4s are concatenated from segments/ (constant memory). "
                        "Requires --vae_decode_type persistent + --save_chunk_segments.")
    # batch overlap: hand the CPU-bound pred-mp4 + GT|Pred HUD encode to a detached worker so
    # the NEXT sample's GPU inference overlaps this sample's encode (see postprocess_viz.py).
    p.add_argument("--bg_postprocess", action="store_true",
                   help="dump pred frames + spawn a detached CPU-only viz/encode worker, then exit "
                        "immediately (batch overlap). Without this the encode runs inline (blocking).")
    p.add_argument("--bg_postprocess_max", type=int, default=4,
                   help="max concurrent background postprocess workers before the GPU process waits")
    p.add_argument("--fps", type=int, default=24, help="output mp4 frame rate")
    p.add_argument("--start_seconds", type=float, default=0.0,
                   help="time-axis offset for source pose and GT video")
    p.add_argument("--ref_seconds", type=float, default=0.0,
                   help="v2v ref video duration in seconds; 0 for i2v/t2v")

    # === GEO ===
    p.add_argument("--use_geometric_state", action="store_true",
                   help="enable GEO; requires --lingbot_pose_path")
    p.add_argument("--geo_disable_prev_short", action="store_true", default=False,
                   help="diagnostic: skip prev_short token -> short tier reverts to [prefix | warp]")
    p.add_argument("--geo_drop_warp", action="store_true", default=False,
                   help="diagnostic: drop ALL warp tokens (force warp visibility=0) -> short tier = [prefix | prev_short]; cuts the cross-chunk warp feedback loop")
    p.add_argument("--geo_warp_vis_cap", type=float, default=0.0,
                   help="adaptive warp visibility cap (primary knob, 0=off): where the per-frame visible warp fraction v exceeds cap, thin it down with a spatially uniform ordered dither (keep=cap/v); v<=cap is left alone, so how much is removed follows visibility -- slow sekai walks (v~1) get cut hard, high-motion clips are spared")
    p.add_argument("--geo_warp_patch_drop_ratio", type=float, default=0.0,
                   help="inference-side per-patch warp drop (fixed ratio, random; control arm, mutually exclusive with --geo_warp_vis_cap which wins): zero warp visibility per grid x grid latent cell with uniform Bernoulli (0=off)")
    p.add_argument("--geo_warp_warm_encode", action="store_true", default=False,
                   help="warp encode warm-pad: prepend vae_t copies of the first frame then drop the first latent, so the leading warp latent follows the continuous-frame distribution instead of an I-frame. The DiT copies warp -> the predicted first frame is continuous too -> flicker-free one-shot/persistent decode.")
    p.add_argument("--geo_oneshot_output_decode", action="store_true", default=False,
                   help="diagnostic/workaround: the rollout still decodes per chunk to feed the warp source, but the output video is re-decoded in one shot from all latents (truly continuous in time, no per-chunk seam or flicker). Memory hungry.")
    p.add_argument("--geo_warp_stage0_only", action="store_true", default=False,
                   help="GEO reference alignment: inject warp only into the coarsest pyramid stage (stage0); for i_s>0 the short tier degrades to [prefix|prev_short] (warp segment dropped). off = legacy behaviour (warp injected into every stage).")
    p.add_argument("--geo_chunk0_ref_warp", action="store_true", default=True,
                   help="i2v only: build chunk 0's warp from the reference image (monocular depth + depth_median scale, "
                        "the same recipe training gives chunk 0) so the first chunk follows the camera instead of sitting "
                        "static. Chunk 0 has no generated history, so without this its pool is empty and its warp is black "
                        "-> no camera signal (warp is this model's only camera channel). Needs the vigeo backend with "
                        "--geo_vigeo_scale_mode depth_median/fixed (the auto default gives i2v exactly that); combining it "
                        "with anchor/per_window is rejected up front rather than degraded to a blank warp. Default on; adds one "
                        "ViGeo forward on the ref image at chunk 0 (so chunk 0 is no longer bit-reproducible). No effect on v2v/t2v.")
    p.add_argument("--no_geo_chunk0_ref_warp", dest="geo_chunk0_ref_warp", action="store_false",
                   help="disable the i2v chunk-0 reference-image warp (chunk 0 falls back to the black warp / static behaviour).")
    p.add_argument("--geo_chunk0_target_disparity_px", type=float,
                   default=_CHUNK0_TARGET_DISPARITY_PX_DEFAULT,
                   help="i2v chunk-0 amplitude control, in warp-render pixels (0 = off = legacy). chunk 0's "
                        "depth scale comes from ONE frame (median depth -> --geo_vigeo_depth_median_target), "
                        "and median-pinning does not pin the near field, so a close-up reference and a "
                        "landscape reference get very different chunk-0 parallax from the same pose track "
                        "(measured chunk 0 running 5.95x the flow of its own later chunks, worst case 17x). "
                        "This instead rescales chunk-0's monocular depth so that at chunk 0's largest "
                        "commanded translation, 90 %% of pixels have moved no further than this many pixels "
                        "-- solved in closed form from the depth map, the commanded poses and K (rotation "
                        "excluded: it does not scale with depth). chunk 0 only; every later chunk re-anchors "
                        "on its own multi-frame window.")
    p.add_argument("--geo_warp_patch_drop_grid", type=int, default=4,
                   help="block size for warp drop (latent cells; the vis_cap path defaults to 1 = finest, preserving low-frequency geometry, the patch_drop path to 4)")
    # FrameBank config
    p.add_argument("--geo_score", type=str, default="v1", choices=["v1", "v2", "v3"],
                   help="FrameBank scoring: v1 (dir-0.1*dist), v2 (multi-view+facing+gaussian), v3 (footprint IoU)")
    p.add_argument("--geo_nearby_k", type=int, default=0,
                   help="number of time-recent frames to include without scoring; 0 = metric-only")
    p.add_argument("--geo_select_k", type=int, default=5,
                   help="number of frames to select by metric from bank")
    p.add_argument("--geo_v3_depth", type=float, default=5.0,
                   help="v3 canonical depth in meters")
    p.add_argument("--geo_v3_fov_deg", type=float, default=60.0,
                   help="v3 camera FOV in degrees")
    p.add_argument("--geo_bank_max", type=int, default=0,
                   help="FrameBank size cap (FIFO evict); 0 = unlimited")
    p.add_argument("--geo_init_k", type=int, default=10,
                   help="v2v initial bank seeding limit (source + ref frames)")

    # === Plan A short-tier noise ===
    p.add_argument("--short_tier_noise_enabled", action="store_true", default=False,
                   help="add sigma noise to short-tier prefix/prev_short to balance attention trust")
    p.add_argument("--short_tier_sigma_min", type=float, default=0.2,
                   help="sigma lower bound for short-tier noise")
    p.add_argument("--short_tier_sigma_max", type=float, default=0.6,
                   help="sigma upper bound for short-tier noise")
    p.add_argument("--short_tier_targets", type=str, default="prefix,prev_short",
                   help="comma-separated noise targets; empty string disables")
    p.add_argument("--short_tier_sigma_lock_per_rollout", action="store_true", default=False,
                   help="lock one sigma per rollout for temporal consistency across chunks")

    # === Plan 06 vsnoise: spatial visibility-aware noise on warp_latents ===
    p.add_argument("--visibility_aware_noise", action="store_true", default=False,
                   help="apply spatial sigma noise to warp_latents based on visibility mask")
    p.add_argument("--warp_noise_sigma_invisible", type=float, default=0.8,
                   help="sigma for invisible regions")
    p.add_argument("--warp_noise_sigma_min", type=float, default=0.111,
                   help="sigma lower bound for visible regions")
    p.add_argument("--warp_noise_sigma_max", type=float, default=0.135,
                   help="sigma upper bound for visible regions")
    p.add_argument("--visible_token_threshold", type=float, default=0.1,
                   help="visibility threshold below which tokens are dropped from attention")
    p.add_argument("--no_rope_alignment", action="store_true", default=False,
                   help="continuation mode: warp idx != target idx (target shifted by latent_window_size)")
    p.add_argument("--prefix_idx_mode", type=str, default="zero", choices=["zero", "adjacent"],
                   help="prefix positional index mode: zero or adjacent")
    p.add_argument("--warp_rope_mode", type=str, default="overlap_noise",
                   choices=["overlap_noise", "before_prev_short", "before_prev_mid"],
                   help="warp RoPE slot (Plan 16); before_prev_short/before_prev_mid require --no_rope_alignment + --prefix_idx_mode=zero")
    p.add_argument("--warp_rope_noise_center_align", action="store_true", default=False,
                   help="fixed_mem: center coarse pyramid-stage NOISE rope into the full-res warp frame "
                        "(warp/history rope native -> restrict_self_attn/cache OK). Must match the training ckpt.")
    p.add_argument("--warp_keep_clean_anchor", action="store_true", default=False,
                   help="overwrite warp[0] with the clean previous-chunk last frame (v20); must match the training ckpt")
    p.add_argument("--invisible_history_noise", action="store_true", default=False,
                   help="inject sigma_invisible noise into zeros-padding history frames (i2v/t2v); mirrors training")
    p.add_argument("--geo_warp_plucker_enabled", action="store_true", default=False,
                   help="additive Plücker on warp+noise tokens (align_true_plk*); builds patch_embedding_wancamctrl "
                        "so the LoRA's plucker weights load. Requires --lingbot_pose_path. Must match the training ckpt.")
    p.add_argument("--geo_warp_plucker_disabled", action="store_true", default=False,
                   help="FORCE plk OFF even if the (merged) ckpt's config.json has geo_warp_plucker_enabled=true. "
                        "Overrides the constructor via transformer_additional_kwargs=False so patch_embedding_wancamctrl "
                        "is not built and its baked weights are ignored (A/B plk-on-vs-off on a plk-trained ckpt).")
    p.add_argument("--warp_lag_chunks", type=int, default=0,
                   help="lag (chunks) before a decoded frame fuses into the persistent warp cloud; 0 = synchronous")
    # === Per-chunk skill/VFX event control (GEO) ===
    # event chunks: drop warp + freeze camera (static plucker) + skip frame_bank add + use --event_prompt.
    # Empty (default) -> no event chunks -> rollout is byte-for-byte identical to a normal run.
    p.add_argument("--event_chunks", type=str, default="",
                   help="comma-separated 0-indexed chunk indices that are skill/VFX 'event' chunks "
                        "(e.g. '3,4,5'). Empty = no events (default; baseline unchanged).")
    p.add_argument("--event_prompt", type=str, default="",
                   help="skill text prompt used on event chunks (encoded once; non-event chunks keep --prompt).")
    # === Segment prompts: switch the text condition partway through a rollout ===
    #   This model is trained with segment prompts (data caption_key=segments +
    #   use_full_rollout_interleave), so switching prompt mid-rollout is in-distribution.
    #   Unlike --event_chunks this only swaps the text embedding: warp, camera trajectory and the
    #   frame bank are untouched.
    p.add_argument("--prompt_schedule", type=str, default="",
                   help="segment prompts: path to a JSON file (or inline JSON) shaped "
                        "[{\"start_sec\": 0, \"prompt\": \"...\"}, {\"start_sec\": 12, \"prompt\": \"...\"}]. "
                        "\"start_chunk\" may be used instead of \"start_sec\". Each entry stays in force "
                        "until the next one; chunks before the first entry use --prompt. "
                        "One chunk = latent_window_size*4 = 36 frames (1.5s @24fps).")
    # === DA3 known-trajectory point-cloud warp backend (evoke.modules.geometric_state.da3_cloud) ===
    # Known GT trajectory + DA3 for depth only -> persistent point cloud -> batched render (no pose estimation, no drift).
    p.add_argument("--geo_recon_backend", type=str, default="da3", choices=["da3"],
                   help="warp reconstruction backend: da3 (GT trajectory + depth-only estimation + persistent cloud). The only backend.")
    p.add_argument("--geo_cloud_update_n", type=int, default=16,
                   help="da3: how many frames per chunk enter the cloud (>=3; 16 is ~700ms, under 1s, so lag1 works)")
    p.add_argument("--geo_cloud_voxel", type=float, default=0.0,
                   help="da3: upper bound on the cloud voxel downsample size (0=off; required for long rollouts)")
    p.add_argument("--geo_cloud_splat_radius", type=int, default=2,
                   help="da3: render splat radius (density vs. time)")
    p.add_argument("--geo_da3_src", type=str, default=None,
                   help="da3: path to the DepthAnything3 source tree (configurable, never hardcode; None -> EVOKE_DA3_SRC or the module default)")
    p.add_argument("--geo_da3_weights", type=str, default=None,
                   help="da3: path to the DepthAnything3 weight snapshot (None -> EVOKE_DA3_WEIGHTS or the module default)")
    p.add_argument("--geo_da3_process_res", type=int, default=504,
                   help="processing resolution (shared by both depth backends, for same-resolution A/B)")
    # === Depth estimator behind the cloud/FrameBank pipeline; mirrors training cloud_warp.backend ===
    # Orthogonal to --geo_da3_render_mode, which selects the renderer.
    p.add_argument("--geo_depth_backend", type=str, default="vigeo", choices=["da3", "vigeo"],
                   help="depth estimator: vigeo=ViGeo (default; no pose/intrinsics input, needs the "
                        "models/ViGeo1.1 weights) | da3=pose-conditioned DepthAnything3")
    p.add_argument("--geo_vigeo_src", type=str, default=None,
                   help="vigeo: ViGeo source tree (None -> EVOKE_VIGEO_SRC or the vendored evoke/third_party/vigeo)")
    p.add_argument("--geo_vigeo_weights", type=str, default=None,
                   help="vigeo: directory holding vigeo.pt (None -> EVOKE_VIGEO_WEIGHTS or models/ViGeo1.1)")
    p.add_argument("--geo_vigeo_num_tokens", type=int, default=0,
                   help="vigeo: token budget (0 -> derive the resolution from --geo_da3_process_res; ViGeo's native default is 1369)")
    p.add_argument("--geo_vigeo_mode", type=str, default="chunk", choices=["offline", "chunk", "online"],
                   help="vigeo: chunk keeps the kv-cache across ingest windows so one stream shares one "
                        "coordinate frame and one scale (validated); offline treats each window independently")
    p.add_argument("--geo_vigeo_chunk_size", type=int, default=16,
                   help="vigeo: internal chunk length (mode=chunk/online)")
    p.add_argument("--geo_vigeo_intr_source", type=str, default="gt", choices=["gt", "vigeo"],
                   help="vigeo: unprojection intrinsics. gt=GT K rescaled to the output resolution (same as da3) | "
                        "vigeo=ViGeo's self-estimated focal (ablation)")
    p.add_argument("--geo_vigeo_conf_transform", type=str, default="exp", choices=["exp", "none"],
                   help="vigeo: ViGeo conf is raw logits (can be negative); exp maps it positive, leaving percentile gates unchanged")
    p.add_argument("--geo_vigeo_scale_mode", type=str, default="auto",
                   choices=["auto", "per_window", "anchor", "depth_median", "fixed"],
                   help="vigeo depth scale: auto (DEFAULT) = depth_median for i2v, anchor for v2v -- see "
                        "_resolve_vigeo_scale_mode for why the right answer is decided by the sample type | "
                        "anchor=lock the median Umeyama scale of the first N windows (validated for v2v) | "
                        "per_window=solve independently each window | "
                        "depth_median=no Umeyama at all, normalise the first window's median depth to "
                        "--geo_vigeo_depth_median_target and lock it (required for trajectories with zero "
                        "commanded translation, e.g. a pure pan: Umeyama has no baseline there and every "
                        "window is skipped, leaving the warp black) | fixed=use "
                        "--geo_vigeo_scale_value verbatim")
    p.add_argument("--geo_vigeo_scale_value", type=float, default=0.0,
                   help="vigeo: the scale for --geo_vigeo_scale_mode=fixed (depth is divided by it). "
                        "Required (>0) in that mode, ignored otherwise")
    p.add_argument("--geo_vigeo_depth_median_target", type=float, default=5.0,
                   help="vigeo: median cloud depth after scaling, for --geo_vigeo_scale_mode=depth_median "
                        "(i.e. i2v). This is what sets the units of the pose track: translating by this "
                        "much moves the camera one median-scene-depth, so it IS 'how deep the world is'. "
                        "1.0 is the bare unit definition and was the default until it was calibrated: at "
                        "1.0 a sekai 5 s track commands ~1.8 scene-depths per chunk, so the camera leaves "
                        "the geometry, the warp collapses to holes and the model re-invents the scene "
                        "(measured over the 6 worst of 100 i2v cases: min rollout warp coverage 0.163 "
                        "median / 0.020 worst, and the steepest 0.5 s window ran 4.21x the clip median). "
                        "At 5 those become 0.886 / 0.571 and 1.79x while mean speed falls only 20 %%, i.e. "
                        "this removes the resets rather than slowing the camera. v2v solves ~9.9 for the "
                        "same quantity from real camera motion; 10 is safer on worst-case coverage, 5 "
                        "gives twice the parallax and was the preferred look. Unused by v2v/t2v (v2v "
                        "resolves to scale_mode=anchor, t2v has no warp)")
    p.add_argument("--geo_vigeo_anchor_windows", type=int, default=4,
                   help="vigeo: how many leading ingest windows anchor the scale (4 = the v2v GT seed section)")
    p.add_argument("--geo_vigeo_cache_keep_frames", type=int, default=6,
                   help="vigeo: streaming kv-cache context frames kept per global block (folded into the token budget)")
    p.add_argument("--geo_vigeo_total_budget", type=int, default=0,
                   help="vigeo: kv-cache token budget; 0 = derive from --geo_vigeo_cache_keep_frames (recommended). "
                        "A per-block share at or below one frame's token count silently disables ViGeo's eviction")
    p.add_argument("--geo_da3_render_mode", type=str, default="multisrc",
                   choices=["multisrc", "backward", "backward_zbuf", "recall"],
                   help="da3 render mode: multisrc=forward-splat fusion over several sources | backward=backward warp (single main source + recall hole filling + fill_iters dilation) | "
                        "backward_zbuf=per-pixel multi-source z-buffer fusion (no dilation, holes stay invalid; steadier at inference, no chunk-boundary edge jumps) | recall=legacy cloud")
    p.add_argument("--geo_bw_fill_iters", type=int, default=12,
                   help="da3 backward-warp hole-closing iterations (small interior holes); only used by render_mode=backward (backward_zbuf does not dilate and ignores it)")
    # backward_zbuf hybrid salt-and-pepper cleanup (off by default; same renderer and parameters as training)
    p.add_argument("--geo_zbuf_despeckle", action="store_true",
                   help="da3 backward_zbuf per-frame hybrid despeckle (open->close to clear boundary specks and fill fully enclosed pixels); fixes z-buffer salt-and-pepper poisoning, matches training")
    p.add_argument("--geo_zbuf_despeckle_ksize", type=int, default=3,
                   help="despeckle morphology kernel size (open->close); default 3x3")
    p.add_argument("--geo_zbuf_despeckle_fill_iters", type=int, default=4,
                   help="despeckle interior-pixel neighbourhood mean-fill iterations; default 4")

    # === [cloud hygiene] pre-ingest saturation clipping + dropping low-confidence / flat (textureless) DA3 points (off = legacy) ===
    p.add_argument("--geo_cloud_hygiene", action="store_true", default=False,
                   help="[master switch] cloud hygiene: saturation-clip ingested frames and keep flat / low-confidence points out of the cloud (depth set to 0 -> not rendered, leaving a hole for the model to fill). Off by default.")
    p.add_argument("--geo_hygiene_sat_max", type=float, default=1.0,
                   help="[optional, default 1.0 = no clipping] hard HSV saturation ceiling (hue-agnostic, not aimed at any one colour; preserves V and hue, compresses S only). Set below 1 only for the extreme case where an over-saturated colour gets amplified on-policy; normally leave colours alone and fix the root cause with depth gating.")
    p.add_argument("--geo_hygiene_conf_abs", type=float, default=0.0,
                   help="[primary gate] absolute DA3 confidence threshold: drop points with conf below it (degenerate flood regions collapse to ~1-3 and get culled, real content sits high at ~11-14 and survives). Default 0=off; ~5 recommended. Requires DA3 to emit conf.")
    p.add_argument("--geo_hygiene_flat_std", type=float, default=0.0,
                   help="[optional, default 0=off, use with care] flat-region gate: drop points whose local grey-level std falls below this. It also hits genuinely flat regions DA3 handles well (a real blue road scores high conf), so keep it as a fallback for when conf is unavailable")
    p.add_argument("--geo_hygiene_flat_win", type=int, default=7,
                   help="local texture window for the flat-region gate (odd); default 7")
    p.add_argument("--geo_hygiene_conf_pct", type=float, default=0.0,
                   help="[weak] DA3 confidence percentile gate (per-frame, relative; when a whole frame collapses the threshold collapses with it and flood survives, which is why conf_abs is preferred); <=0 disables")

    # === [method 1] ingest consistency gate (the principled fix; independent of colour, shape and scale. Off by default) ===
    p.add_argument("--geo_consist_gate", action="store_true", default=False,
                   help="[method 1 master switch] ingest consistency gate: project the existing cloud into the new view to get a predicted depth, and treat pixels whose new DA3 depth deviates by more than tau as hallucination conflicting with known geometry (a wall, a blue flood, any colour) -> not ingested. Holes pass through. Off by default.")
    p.add_argument("--geo_consist_tau", type=float, default=0.15,
                   help="[method 1] relative depth tolerance |d-d_cloud|/d_cloud; smaller is stricter, typically 0.1-0.2 (default 0.15)")
    p.add_argument("--geo_consist_ref_frames", type=int, default=24,
                   help="[method 1] how many of the most recent frames provide the reference geometry (0=all, default 24)")
    p.add_argument("--geo_consist_min_ref", type=int, default=2000,
                   help="[method 1] skip the gate when fewer reference points than this are available (cold-start guard, default 2000)")
    # [variant 1] confidence-adaptive tolerance (removes the magic fixed tau; needs --geo_consist_gate)
    p.add_argument("--geo_consist_adaptive", action="store_true", default=False,
                   help="[variant 1] confidence-adaptive tolerance: tau rises per pixel with DA3 conf (strict where conf is low to block poisoning, loose where it is high to avoid despeckling real content), removing the magic fixed tau. Off by default.")
    p.add_argument("--geo_consist_tau_lo", type=float, default=0.20,
                   help="[variant 1] lower relative tolerance, used as conf -> 0 (the validated safe floor, default 0.20; lo=0.10 over-deletes and collapses coverage later in the rollout)")
    p.add_argument("--geo_consist_tau_hi", type=float, default=0.30,
                   help="[variant 1] upper relative tolerance, used at high conf (>=conf_ref); loose, default 0.30")
    p.add_argument("--geo_consist_conf_ref", type=float, default=12.0,
                   help="[variant 1] DA3 conf value at which the tolerance saturates at tau_hi (real content is ~11-14, default 12.0)")
    # hole probation (closes the hallucination fast lane; needs --geo_consist_gate)
    p.add_argument("--geo_consist_probation", type=int, default=0,
                   help="[P0] hole probation: pixels with no cloud reference (holes) no longer pass straight into the cloud but wait one window and are re-checked geometrically against the next window raw DA3 depth (consistent -> admitted / conflicting -> dropped / not visible -> admitted so they cannot starve). 0=off (default = legacy pass-through), 1=on. Needs consist_gate.")
    p.add_argument("--geo_consist_probation_frac", type=float, default=0.0,
                   help="[P0 v2] >0: only hold pixels in the interior of large holes (local hole fraction over a win x win window above this); narrow strips and small holes pass straight through, cutting the v1 coverage tax. 0 = hold everything (v1). Typical 0.6")
    p.add_argument("--geo_consist_probation_win", type=int, default=25,
                   help="[P0 v2] window side length for the local hole fraction (odd, default 25)")
    p.add_argument("--geo_consist_scale_align", type=int, default=0,
                   help="[P1a] per-window ingest scale alignment: correct the whole window depth by median(d/d_cloud) before gating (clipped to [0.75,1.33]), curing the compounding per-window Umeyama scale drift (measured 1.05-1.18). 0=off, 1=on. Needs consist_gate.")
    # colour-statistics re-anchoring (independent switch, does not need consist_gate; treats over-saturated / over-sharpened seeds)
    p.add_argument("--geo_color_anchor", type=int, default=0,
                   help="[P1b] colour-statistics re-anchoring: at the DA3 ingest entry, linearly re-anchor each channel of the decoded frame to the reference statistics (the first ref_windows windows = GT prime). Affects only the DA3 input and the cloud RGB, never the model output. 0=off, 1=on. Independent of consist_gate.")
    p.add_argument("--geo_color_anchor_alpha", type=float, default=0.5,
                   help="[P1b] correction blend factor: x_out = alpha*x' + (1-alpha)*x. 1=full correction, 0=untouched. Default 0.5.")
    p.add_argument("--geo_color_anchor_ref_windows", type=int, default=4,
                   help="[P1b] reference period: accumulate channel mean/std over this many initial ingests (for v2v the cold start is the GT prime frames) and leave them uncorrected. Default 4.")
    p.add_argument("--geo_self_reanchor", type=int, default=0,
                   help="[self re-anchor] on divergence (current-window scale_ratio or rejection rate over threshold), hard re-anchor the cloud to its own best past contiguous window (a legitimate oracle: only past self-generated information). 0=off, 1=on. Needs consist_gate.")
    p.add_argument("--geo_self_reanchor_scale_thr", type=float, default=1.5,
                   help="[self re-anchor] trigger: current-window median(d/d_cloud) outside [1/thr, thr]")
    p.add_argument("--geo_self_reanchor_rej_thr", type=float, default=0.8,
                   help="[self re-anchor] trigger: rejection rate over the checkable region exceeds this")
    p.add_argument("--geo_self_reanchor_min_gap", type=int, default=8,
                   help="[self re-anchor] minimum gap between two re-anchors (in ingests; anti-oscillation)")
    p.add_argument("--geo_self_reanchor_keep_windows", type=int, default=3,
                   help="[self re-anchor] keep the anchor window plus this many-1 contiguous windows before it (geometric continuity)")
    p.add_argument("--geo_self_reanchor_lookback", type=int, default=20,
                   help="[self re-anchor] how far back to look for anchor candidates (in ingest windows)")
    p.add_argument("--geo_self_reanchor_anchor_min_conf", type=float, default=0.0,
                   help="[self re-anchor v3] minimum conf for an anchor candidate (0=off); with no acceptable anchor, skip re-anchoring (keep the current cloud) rather than anchor to garbage")
    p.add_argument("--geo_self_reanchor_pin_prime", type=int, default=0,
                   help="[self re-anchor v4] pin the prime anchor: the first pin_windows ingests (the GT prefix) are kept forever, and with no healthy recent anchor we fall back to re-anchoring onto it (resetting to near-free generation)")
    p.add_argument("--geo_self_reanchor_pin_windows", type=int, default=4,
                   help="[self re-anchor v4] how many leading ingests count as prime")

    # === [method 2] bounded / periodically re-anchored history (lets old poisoning age out; independent switch, off by default) ===
    p.add_argument("--geo_hist_max_frames", type=int, default=0,
                   help="[method 2] sliding window: keep only this many recent frames in the cloud (in gid units, 0=off). Must exceed lag*stride or the warp comes back empty. 720 (~30s @24fps) is typical")
    p.add_argument("--geo_reanchor_every", type=int, default=0,
                   help="[method 2] hard re-anchor every N ingests (shrink to the most recent keep frames); 0=off")
    p.add_argument("--geo_reanchor_keep_frames", type=int, default=0,
                   help="[method 2] how many recent frames a hard re-anchor keeps (goes with reanchor_every; 0 disables re-anchoring)")

    # === [method 3] free-space carving (the real correction: deleting old points; independent switch, off by default) ===
    p.add_argument("--geo_carve", action="store_true", default=False,
                   help="[method 3 master switch] free-space carving: use the new frame depth to see through the old cloud and delete floating old points that fall in free space (closer than the newly observed surface by more than margin). Clears floating poison and the ghosts of objects that have moved away. Off by default.")
    p.add_argument("--geo_carve_margin", type=float, default=0.10,
                   help="[method 3] how much closer (relatively) an old point must be than the new observed surface to count as free space and be deleted; larger is more conservative (default 0.10)")
    p.add_argument("--geo_carve_ref_frames", type=int, default=24,
                   help="[method 3] how many past frames at most to carve back through per step (cost control, 0=all; default 24)")
    p.add_argument("--geo_carve_min_views", type=int, default=1,
                   help="[variant 2] multi-view voting: an old point must be seen through by at least this many new frames before deletion (1 = delete on any single frame / legacy OR behaviour; 3-4 recommended)")
    p.add_argument("--geo_carve_strike_windows", type=int, default=1,
                   help="cross-window strike: a pixel must satisfy the deletion criterion in this many CONSECUTIVE ingest windows before it is really deleted, and any break resets the count (the 12 frames of one window share a DA3 forward pass and a scale, so their votes are correlated and cannot suppress a window-level systematic error). 1=off (default = delete within the window).")

    # === Camera control (lingbot pose) ===
    p.add_argument("--lingbot_pose_path", type=str, default=None,
                   help="pose npz with cam_c2w + intrinsics; required for GEO")
    p.add_argument("--lingbot_pose_source_fps", type=int, default=30)
    p.add_argument("--lingbot_pose_source_resolution", type=int, nargs=2, default=[1080, 1920],
                   help="source video resolution [H, W] for pose npz")
    p.add_argument("--lingbot_pose_type", type=str, default="vipe", choices=["vipe", "raw"])
    p.add_argument("--lingbot_fallback_default_intrinsic", action="store_true", default=False,
                   help="when the pose npz has no intrinsics/intrinsic/K key ("
                        "which stores only data/inds), use a default normalized intrinsic. "
                        "Default OFF → sekai/vipe path unchanged.")
    # Per-chunk trajectory cap (limits camera motion so warp stays in-distribution). 0 = off.
    p.add_argument("--max_deg_per_chunk", type=float, default=0.0,
                   help="per-chunk rotation cap in degrees; 0 = off. (GEO demo worst ~17)")
    p.add_argument("--max_trans_per_chunk", type=float, default=0.0,
                   help="per-chunk translation cap (same coord as input c2w); 0 = off. (sekai p95 ~3.3, user ~5.5)")
    p.add_argument("--cap_mode", type=str, default="clamp", choices=["clamp", "resample"],
                   help="clamp = clip per-chunk delta in place (frames unchanged); resample = SLERP-subdivide (frames grow)")
    p.add_argument("--pose_smooth_win", type=int, default=0,
                   help="inference-side camera control smoothing window (odd); 0/1=off. Applies a Gaussian low pass to the translation and a quaternion low pass to the rotation of the target-segment c2w, "
                        "suppressing high-frequency vipe jitter. Inference only, never used in training.")
    p.add_argument("--pose_extend_mode", type=str, default="clamp", choices=["clamp", "relative_replay"],
                   help="how to extend a source pose track shorter than num_frames: clamp = pin to the last frame (camera freezes); "
                        "relative_replay = replay the same relative motion (the camera keeps following the original pattern, no teleport and no freeze).")

    # === restrict_self_attn + KV-cache ===
    # NOTE: both default to False -> byte-equivalent to the fully bidirectional, cache-less behaviour. They must be enabled as a pair: the cache only takes effect under restrict_self_attn.
    # A stage2 cache checkpoint (trained with restrict_self_attn=true) MUST be run with --restrict_self_attn to match the training attention;
    #   --use_kv_cache additionally reuses the history K/V across denoising steps for speed. Compatible with the navit pyramid and
    #   warp_rope_noise_center_align (warp/history rope stay native, so restrict/cache remain valid). If the model was not trained with restrict, the quality is simply wrong.
    p.add_argument("--restrict_self_attn", action="store_true", default=False,
                   help="on: history (including warp) only self-attends and never attends to noise -> becomes timestep-independent and reusable by the KV cache. "
                        "Default False = fully bidirectional. A checkpoint trained with restrict MUST enable this, otherwise it is mismatched with training and the quality suffers.")
    p.add_argument("--use_kv_cache", action="store_true", default=False,
                   help="on: compute the history K/V and their attention output once for the whole denoising run and reuse it (requires --restrict_self_attn). "
                        "Default False = recomputed every step.")
    p.add_argument("--no_dynamic_shifting", action="store_true", default=False,
                   help="disable the use_dynamic_shifting passed to the stage1 pipe (default True). Training validation uses False "
                        "(validation_config.use_dynamic_shifting=false), so add this flag when aligning frame by frame against val.")

    # === Output / viz ===
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--ref_video_for_viz", type=str, default=None,
                   help="optional: GT reference video for GT|Pred side-by-side with joystick HUD")
    # The joystick HUD ("energy rings": Move bottom-left, Rotate bottom-right) used to be reachable
    #   only through the GT|Pred comparison, so i2v / segment runs never got one even though they
    #   carry the pose that drives it. It is driven by the pose alone, hence auto = on whenever a
    #   pose was given. Burned into geo_pred.mp4 and the per-chunk pred segments; the GT|Pred panels
    #   draw their own and are unaffected.
    p.add_argument("--joystick_hud", type=str, default="auto", choices=["auto", "on", "off", "both"],
                   help="joystick/energy-ring HUD on the pred video: auto = on when --lingbot_pose_path "
                        "is set (i2v/v2v/segment), off for t2v; on = force (no-op without a pose); off = never; "
                        "both = write BOTH a clean geo_pred.mp4 and an overlaid geo_pred_hud.mp4 from the "
                        "same generation (one inference pass, two encodes)")

    # Kept so later checks can ask the parser for a flag's default instead of restating the literal
    # (see the ViGeo conf-unit guard in main).
    global _PARSER
    _PARSER = p
    args = p.parse_args(argv)
    _resolve_vigeo_scale_mode(args)
    return args


def _resolve_vigeo_scale_mode(args) -> None:
    """Turn --geo_vigeo_scale_mode=auto into a concrete mode, in place, from the sample type.

    `anchor` / `per_window` solve the depth scale as a RATIO: c = (camera motion ViGeo reads out of the
    frames) / (camera motion that was commanded), and depth /= c. That ratio is only meaningful when the
    frames it is measured on show real camera motion, which is exactly what differs between the modes:

      v2v  chunk 0's window is REF-VIDEO frames. Real footage, real parallax, so the ratio is a genuine
           measurement and locking its median (`anchor`) is the validated recipe. Keep it.
      i2v  chunk 0 has no real frames at all -- its pool is empty, so it is generated from image+text with
           NO warp, and the very first window ViGeo measures is that self-generated content. On a
           single-direction track (push_in / pull_out / move_*) a chunk-0 that came out near-static makes
           ViGeo correctly report ~no parallax, the ratio collapses toward 0, depth /= ~1e-4 puts the cloud
           thousands of times too far away, and the warp becomes a still image -- which the model then
           faithfully continues, keeping every later window static too. Self-locking: measured c=2.7e-4 and
           a dead warp for the whole rollout. depth_median never divides
           by that ratio -- it normalises the median depth of the map itself -- so it cannot collapse, and
           it is also the only mode the i2v chunk-0 reference warp can run under (it needs a scale from a
           SINGLE frame, and a ratio needs >=3 posed ones).

    t2v cannot reach here (the CLI rejects --use_geometric_state with t2v).
    """
    if str(getattr(args, "geo_vigeo_scale_mode", "auto")) != "auto":
        return                                    # explicit choice wins, including an explicit "anchor"
    _st = str(getattr(args, "sample_type", "i2v"))
    args.geo_vigeo_scale_mode = "anchor" if _st == "v2v" else "depth_median"


def _check_prereqs(args):
    print(f"[geo-infer] repo root: {REPO_ROOT}", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("[geo-infer] FATAL: no CUDA")

    # Depth-backend assets, checked before anything loads: the estimator loads lazily inside a
    # per-window try, so a bad path would otherwise be reported as a degenerate window and the run
    # would finish "successfully" with an all-black warp.
    if args.use_geometric_state:
        from evoke.modules.geometric_state.depth_backend import check_assets
        try:
            check_assets(args.geo_depth_backend, weights=args.geo_vigeo_weights, src=args.geo_vigeo_src)
        except FileNotFoundError as _e:
            raise SystemExit(f"[geo-infer] FATAL: {_e}")

        # The i2v chunk-0 warp needs a depth scale solvable from ONE frame, which only the baseline-free
        # modes give. It used to degrade to a blank chunk-0 warp on its own, reported as
        # "[da3-single-src-mono] WARN reference-frame depth failed (NotImplementedError ...)" -- which reads
        # as a bad reference image while actually meaning "this scale_mode cannot do it". The flag then did
        # nothing at all, so --no_geo_chunk0_ref_warp changed no pixel and looked exonerated while the real
        # culprit (scale_mode) went unexamined. Fail instead, and fail HERE rather than in build_pipe: this
        # is a pure argument check, so a misconfigured sweep should cost a second, not a 54 GB weight load
        # per case. With scale_mode=auto the combination is unreachable, so reaching it means an explicit
        # --geo_vigeo_scale_mode contradicting an explicitly-on chunk-0 warp.
        if str(args.geo_depth_backend) == "vigeo" and str(args.sample_type) == "i2v" \
                and bool(args.geo_chunk0_ref_warp) \
                and str(args.geo_vigeo_scale_mode) not in ("depth_median", "fixed"):
            raise SystemExit(
                f"[geo-infer] FATAL: --geo_chunk0_ref_warp (default on for i2v) needs a baseline-free "
                f"depth scale, but --geo_vigeo_scale_mode={args.geo_vigeo_scale_mode}. Chunk 0's warp is "
                f"built from the single reference image, while anchor/per_window solve their scale from "
                f">=3 posed frames that chunk 0 does not have. Either use --geo_vigeo_scale_mode "
                f"depth_median (or drop the flag entirely for the auto default, which picks it for i2v), "
                f"or pass --no_geo_chunk0_ref_warp to deliberately accept a static chunk 0.")
    cap = torch.cuda.get_device_properties(0)
    print(f"[geo-infer] gpu: {cap.name}, total mem: {cap.total_memory / 1024**3:.1f} GB", flush=True)

    # validate sample_type vs image/video path
    # GEO v2v may omit --image_path: the source frame is auto-extracted from the ref video first frame (train-val aligned).
    if args.use_geometric_state and args.sample_type in ("i2v", "v2v"):
        _has_image = bool(args.image_path and Path(args.image_path).is_file())
        _can_extract = (args.sample_type == "v2v" and bool(args.video_path and Path(args.video_path).is_file()))
        if not _has_image and not _can_extract:
            raise SystemExit(f"[geo-infer] FATAL: sample_type={args.sample_type} + use_geometric_state needs --image_path or (v2v) --video_path")
    elif args.sample_type == "i2v" and not (args.image_path and Path(args.image_path).is_file()):
        # warp-free v2v uses --video_path (validated below) and does not force image_path;
        # the pipeline forbids image+video together (non-warp), so v2v only gets video.
        raise SystemExit(f"[geo-infer] FATAL: sample_type={args.sample_type} needs --image_path")
    if args.sample_type == "v2v" and not (args.video_path and Path(args.video_path).is_file()):
        raise SystemExit(f"[geo-infer] FATAL: sample_type=v2v needs --video_path")
    if args.use_geometric_state and args.sample_type == "t2v":
        raise SystemExit(f"[geo-infer] FATAL: --use_geometric_state cannot be combined with sample_type=t2v (the warp backend needs source pixels)")

    if args.use_geometric_state:
        # warp backend = DA3 (known GT trajectory + depth-only estimation); weight directory: --geo_da3_weights / EVOKE_DA3_WEIGHTS.
        if not args.lingbot_pose_path or not Path(args.lingbot_pose_path).is_file():
            raise SystemExit(f"[geo-infer] FATAL: --use_geometric_state requires --lingbot_pose_path")

    if args.stream_long_video:
        # In streaming mode the full video can only be stitched from per-chunk segments, so both
        # preconditions must hold now -- otherwise the run finishes after N hours with nothing to stitch.
        if args.vae_decode_type != "persistent":
            raise SystemExit(f"[geo-infer] FATAL: --stream_long_video requires --vae_decode_type persistent "
                             f"(got {args.vae_decode_type})")
        if not args.save_chunk_segments:
            raise SystemExit(f"[geo-infer] FATAL: --stream_long_video requires --save_chunk_segments "
                             f"(the full video is stitched from segments/)")
        if args.bg_postprocess:
            raise SystemExit(f"[geo-infer] FATAL: --stream_long_video and --bg_postprocess are mutually exclusive "
                             f"(streaming has no full frame array to hand to a background worker)")


def _load_lingbot_pose(args):
    """Load, resample, and slice pose data to cover ref + generation window.

    Returns:
        (Ks, c2ws): Ks [4] intrinsics, c2ws [F_pix, 4, 4] camera-to-world (cpu fp32, no batch dim).
    """
    from evoke.utils.ev_validation import load_pose_for_v2v   # REPO_ROOT is on sys.path (module top)
    # compute required pose length to cover ref + generation frames
    _W = 9                                                         # latent_window_size
    _vae_stride_t = 4
    _window_pix = (_W - 1) * _vae_stride_t + 1                     # 33
    _num_secs = max(1, (int(args.num_frames) + _window_pix - 1) // _window_pix)
    _ref_pix = max(0, int(round(float(args.ref_seconds) * 24)))    # ref_seconds * target_fps
    _ref_lat = ((_ref_pix - 1) // _vae_stride_t + 1) if _ref_pix > 0 else 0
    _needed_lat = _ref_lat + _num_secs * _W
    _pose_num_target_frames = (_needed_lat - 1) * _vae_stride_t + 1
    _Ks, c2ws = load_pose_for_v2v(
        args.lingbot_pose_path,
        target_height=args.height, target_width=args.width,
        source_resolution=tuple(args.lingbot_pose_source_resolution),
        pose_type=args.lingbot_pose_type,
        num_target_frames=_pose_num_target_frames,
        target_fps=24,
        source_fps=int(args.lingbot_pose_source_fps),
        start_seconds=float(args.start_seconds),
        fallback_default_intrinsic=bool(getattr(args, "lingbot_fallback_default_intrinsic", False)),
        pose_extend_mode=str(getattr(args, "pose_extend_mode", "clamp")),
    )
    return _Ks, c2ws


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_worker_slot(workers_dir: Path, max_workers: int, poll: float = 2.0) -> None:
    """Block until fewer than max_workers detached postprocess workers are in flight.

    Each live worker owns a <name>.lock file containing its pid. Markers whose pid is dead
    (crashed worker that never cleaned up) are pruned so the slot is reclaimed.
    """
    while True:
        live = 0
        for m in list(workers_dir.glob("*.lock")):
            try:
                pid = int((m.read_text().strip() or "0"))
            except (ValueError, OSError):
                pid = 0
            if pid > 0 and _pid_alive(pid):
                live += 1
            elif pid > 0:  # only prune once a real pid was written and is now dead
                try:
                    m.unlink()
                except OSError:
                    pass
        if live < max_workers:
            return
        print(f"[geo-infer] [bg] {live} postprocess worker(s) in flight (max {max_workers}), waiting...",
              flush=True)
        time.sleep(poll)


def _spawn_bg_postprocess(output_dir: Path, video_np, params: dict, max_workers: int = 4) -> None:
    """Dump pred frames + params and launch a detached CPU-only worker, then return so the
    GPU process can exit and let the next sample's inference overlap this encode."""
    workers_dir = output_dir.parent / ".pp_workers"   # shared per batch (OUT_ROOT/.pp_workers)
    workers_dir.mkdir(parents=True, exist_ok=True)
    _wait_for_worker_slot(workers_dir, max_workers)

    marker = workers_dir / f"{output_dir.name}.lock"
    params = dict(params)
    params["_worker_marker"] = str(marker)

    _t = time.time()
    npy = output_dir / ".pred_frames.npy"
    np.save(npy, video_np)
    (output_dir / ".postproc_params.json").write_text(json.dumps(params))
    print(f"[geo-infer] [bg] dumped pred frames {tuple(video_np.shape)} -> {npy} ({time.time() - _t:.1f}s)",
          flush=True)

    worker = str(Path(__file__).resolve().parent / "postprocess_viz.py")
    log_f = open(output_dir / "postproc.log", "w")
    proc = subprocess.Popen(
        [sys.executable, worker, str(output_dir)],
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,                          # survive the GPU process exit
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # CPU-only: never grab a GPU
    )
    marker.write_text(str(proc.pid))                     # now visible to the concurrency guard
    print(f"[geo-infer] [bg] spawned detached postprocess worker pid={proc.pid} "
          f"-> {output_dir}/postproc.log", flush=True)



# ---------------------------------------- pipeline construction (cached) ------------------------------
# The setup below is ~50 s of CUDA init and, on a network filesystem, minutes of weight reads (54 GB
# transformer + 22 GB text encoder for this ckpt). infer_batch.py used to spawn one process per case, so a
# 2000-case sweep paid it 2000 times -- 131 s of the 251 s per-case wall time. Split out so a batch driver can
# build it ONCE and loop over cases in-process (see run_argv_batch / --argv_jsonl).
#
# The reuse is correct because everything the pipeline mutates per generation is per-call state, not instance
# state: `_geo_da3_setup` builds a fresh FrameBank + depth estimator inside __call__, `set_timesteps` resets the
# scheduler's step index and history per stage, and `_fuse_geo_lora` / `_unfuse_geo_lora` are idempotent and
# symmetric. What is genuinely per-case (output_dir, `_geo_dump_dir`) is re-set by main() after this returns.
_PIPE_CACHE: dict = {}

# args that vary per case and provably do not affect anything built here (they are all consumed after the
# boundary). Keying the cache on "everything else" fails safe: a new setup-affecting arg that nobody adds to
# this list simply busts the cache (slow but correct), whereas an allowlist of setup args would silently reuse a
# stale pipeline.
#
# `seed` is here so a benchmark that needs a different seed per sample (VBench samples N videos per prompt) does
# not rebuild 54 GB of transformer weights per video. The seed reaches generation through the torch Generator
# built per call in main(); its one setup-time use, `warp_patch_drop_seed`, is re-set per case in main().
#
# This list must hold argparse DESTS, not jsonl field names. "segment_prompts_path" is the jsonl field, which
# infer_batch.py turns into `--prompt_schedule <json>` whose dest is `prompt_schedule`; while that dest was
# missing here, every multi-segment case re-read the whole transformer, and single-segment cases pass no
# schedule at all, so a small smoke test does not catch it. The same applies to `ref_video_for_viz` (v2v passes
# a per-case video) and the two pose-source args (a mixed-resolution jsonl would vary them). All of these are
# consumed in main() only -- never inside build_pipe.
_PER_CASE_ARGS = frozenset({
    "output_folder", "image_path", "video_path", "prompt", "negative_prompt",
    "lingbot_pose_path", "num_frames", "num_chunks", "event_prompt", "event_chunks",
    "segment_prompts_path", "start_seconds", "case_name", "seed",
    "prompt_schedule", "ref_video_for_viz",
    "lingbot_pose_source_fps", "lingbot_pose_source_resolution",
    # i2v and v2v share identical weights/pipeline construction. These only select
    # and prepare per-call conditioning; keeping them in the cache key made the
    # persistent UI re-read 54 GB of weights on the first director-mode branch.
    "sample_type", "ref_seconds",
    "image_noise_sigma_min", "image_noise_sigma_max",
    "video_noise_sigma_min", "video_noise_sigma_max",
    "geo_vigeo_scale_mode",
})


def _pipe_cache_key(args):
    def _h(v):
        return tuple(v) if isinstance(v, (list, tuple)) else v
    return tuple(sorted((k, _h(v)) for k, v in vars(args).items() if k not in _PER_CASE_ARGS))


def build_pipe(args):
    """Build (or return the cached) pipeline for `args`. Returns (pipe, transformer, vae, device)."""
    key = _pipe_cache_key(args)
    hit = _PIPE_CACHE.get(key)
    if hit is not None:
        print("[geo-infer] reusing the already-loaded pipeline (weights are not re-read)", flush=True)
        return hit
    from diffusers import AutoencoderKLWan
    from evoke.modules.transformer_evoke import EvokeTransformer3DModel
    from evoke.pipelines.pipeline_evoke import EvokePipeline
    # use diffusers-fork scheduler to accept mu kwarg
    from evoke.diffusers_version.scheduling_evoke_diffusers import EvokeScheduler

    # output_dir is created in main(): it is per-case, and on a cache hit this function returns above,
    # so leaving the mkdir here would silently skip every later case's directory.
    print(f"[geo-infer] GEO on:  {args.use_geometric_state}", flush=True)

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    # load models
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_dir():
        raise SystemExit(f"[geo-infer] FATAL: ckpt dir missing: {ckpt_path}")
    print(f"[geo-infer] ckpt:    {ckpt_path}", flush=True)

    _from_pretrained_kwargs = dict(
        subfolder="transformer", torch_dtype=weight_dtype, low_cpu_mem_usage=True,
    )
    _additional_kwargs = {}
    if args.enable_cam_control:
        # override config so transformer __init__ initialises the cam submodule
        _additional_kwargs.update({
            "enable_cam_control": True,
            "cam_rank": int(args.cam_rank),
        })
        print(f"[geo-infer] override enable_cam_control=True, cam_rank={args.cam_rank}", flush=True)
    # Plücker (plk): build patch_embedding_wancamctrl so the main LoRA's plucker weights
    # (transformer.patch_embedding_wancamctrl.lora_A/B) attach on load; pipeline then feeds
    # per-section cam_plucker_emb to the transformer. Auto-detected from the LoRA so the right
    # ckpt just works (plk in LoRA -> on; absent -> off). --geo_warp_plucker_enabled forces on.
    _plk_on = bool(args.geo_warp_plucker_enabled)
    if args.geo_warp_plucker_disabled:
        # FORCE off: override the merged ckpt's config.json (geo_warp_plucker_enabled=true) so the
        # constructor gate (transformer.geo_warp_plucker_enabled) is False → plk never applied.
        _plk_on = False
        _additional_kwargs["geo_warp_plucker_enabled"] = False
        print("[geo-infer] geo_warp_plucker_enabled=False (FORCED off via --geo_warp_plucker_disabled)", flush=True)
    elif not _plk_on and args.lora_path:
        try:
            from safetensors import safe_open
            _lp = Path(args.lora_path)
            if not _lp.is_file():
                _lp = _lp / "pytorch_lora_weights.safetensors"
            if _lp.is_file():
                with safe_open(str(_lp), framework="pt") as _f:
                    _plk_on = any("patch_embedding_wancamctrl" in k for k in _f.keys())
        except Exception as e:
            print(f"[geo-infer] plk auto-detect skipped: {e}", flush=True)
    if _plk_on:
        _additional_kwargs["geo_warp_plucker_enabled"] = True
        _src = "forced" if args.geo_warp_plucker_enabled else "auto-detected from LoRA"
        print(f"[geo-infer] geo_warp_plucker_enabled=True ({_src}; additive Plücker on warp+noise)", flush=True)
    if _additional_kwargs:
        _from_pretrained_kwargs["transformer_additional_kwargs"] = _additional_kwargs
    # Transformer may come from a separate dir (e.g. a merged stage2 base); other components from ckpt_path.
    _transformer_src = args.transformer_path if args.transformer_path else str(ckpt_path)
    print(f"[geo-infer] transformer: {_transformer_src}", flush=True)
    transformer = EvokeTransformer3DModel.from_pretrained(_transformer_src, **_from_pretrained_kwargs)
    transformer.config.use_raw_sink_frames = bool(args.use_raw_sink_frames)
    transformer.use_raw_sink_frames = bool(args.use_raw_sink_frames)
    print(f"[geo-infer] use_raw_sink_frames = {transformer.config.use_raw_sink_frames}", flush=True)
    print(f"[geo-infer] enable_cam_control  = {getattr(transformer, 'enable_cam_control', False)}", flush=True)

    # plk: the additive-Plücker submodules (patch_embedding_wancamctrl + c2ws_hidden_states_layer{1,2})
    # are built fresh in transformer.__init__ and are absent from the base ckpt, so under
    # low_cpu_mem_usage=True their frozen zero-init base weights stay on the `meta` device. Only their
    # lora_A/B are trained (in the LoRA file); the base is a no-op zero Linear. Materialize + zero them
    # now — before LoRA load / pipe.to(device) — otherwise `.to(device)` raises "Cannot copy out of
    # meta tensor; no data!". (enable_cam_control restores these via the partial loader, so it's plk-only.)
    if _plk_on:
        import torch.nn as _nn
        for _m in ("patch_embedding_wancamctrl", "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2"):
            _mod = getattr(transformer, _m, None)
            if _mod is not None and any(p.is_meta for p in _mod.parameters()):
                _mod.to_empty(device="cpu")
                _nn.init.zeros_(_mod.weight)
                _nn.init.zeros_(_mod.bias)
                _mod.to(dtype=weight_dtype)  # match the rest of the transformer (else fp32 vs bf16 matmul error)
                print(f"[geo-infer] plk: materialized meta base {_m} -> zeros ({weight_dtype})", flush=True)

    vae = AutoencoderKLWan.from_pretrained(str(ckpt_path), subfolder="vae", torch_dtype=torch.float32)
    if args.is_enable_stage2:
        # Stage2 needs a pyramid scheduler (stages=N). ckpt_path's scheduler is stage1 (stages=1,
        # use_dynamic_shifting=true), which makes set_timesteps run an internal mu-based time_shift that
        # collides with stage2_sample's manual shift (mu=None -> crash). Build the pyramid scheduler with
        # use_dynamic_shifting=False (the per-stage shift is applied manually inside stage2_sample).
        scheduler = EvokeScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            stages=int(args.stage2_num_stages),
            stage_range=list(args.stage2_stage_range),
            gamma=1 / 3,
            scheduler_type="unipc",
            use_dynamic_shifting=False,
            time_shift_type="exponential",
        )
        print(f"[geo-infer] stage2 pyramid scheduler: stages={args.stage2_num_stages}, "
              f"stage_range={list(args.stage2_stage_range)}, "
              f"use_dynamic_shifting=False (manual per-stage shift in stage2_sample)", flush=True)
        assert len(args.stage2_stage_range) == int(args.stage2_num_stages) + 1, (
            f"stage2_stage_range (len={len(args.stage2_stage_range)}) must be stage2_num_stages+1 "
            f"(={int(args.stage2_num_stages) + 1}); mismatch mis-calibrates per-stage renoise → block artifact.")
    else:
        scheduler = EvokeScheduler.from_pretrained(str(ckpt_path), subfolder="scheduler")
    pipe = EvokePipeline.from_pretrained(
        str(ckpt_path), transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=weight_dtype,
    )

    # inject short-tier noise config
    _st_targets = [t.strip() for t in str(args.short_tier_targets or "").split(",") if t.strip()]
    pipe._short_tier_noise_cfg = {
        "enabled": bool(args.short_tier_noise_enabled),
        "sigma_min": float(args.short_tier_sigma_min),
        "sigma_max": float(args.short_tier_sigma_max),
        "target_tiers": _st_targets,
        "apply_at_inference": True,
        "sigma_lock_per_rollout": bool(args.short_tier_sigma_lock_per_rollout),
    }
    print(f"[geo-infer] short_tier_noise cfg: {pipe._short_tier_noise_cfg}", flush=True)

    # warp_rope_mode (Plan 16) cross-validation: before_prev_* require no rope align + zero prefix.
    if str(args.warp_rope_mode) in ("before_prev_short", "before_prev_mid"):
        assert bool(args.no_rope_alignment), f"--warp_rope_mode={args.warp_rope_mode} requires --no_rope_alignment."
        assert str(args.prefix_idx_mode) == "zero", f"--warp_rope_mode={args.warp_rope_mode} requires --prefix_idx_mode=zero."

    if str(args.geo_depth_backend) == "vigeo":
        # ViGeo conf is raw logits mapped through exp(); DA3 conf is positive on a different scale, so
        # ABSOLUTE thresholds calibrated for DA3 silently over- or under-cull here (measured equivalents
        # of DA3's 2.5 are ~0.17 real / ~0.084 game footage). Percentile gates are invariant under a
        # monotone transform and are therefore fine. Defaults come from the parser, so changing a flag's
        # default cannot turn this guard into a rejection of the default config.
        for _k in ("geo_hygiene_conf_abs", "geo_consist_conf_ref", "geo_self_reanchor_anchor_min_conf"):
            _v, _default = float(getattr(args, _k)), float(_PARSER.get_default(_k))
            assert _v == _default, (
                f"--{_k}={_v} is calibrated in DA3 conf units and is not valid under "
                f"--geo_depth_backend=vigeo. Use the percentile gates (--geo_hygiene_conf_pct) instead, "
                f"or re-measure the ViGeo conf histogram and pass a re-derived value.")
        assert str(args.geo_vigeo_scale_mode) not in ("anchor", "depth_median") \
            or str(args.geo_vigeo_mode) != "offline", (
            f"--geo_vigeo_scale_mode={args.geo_vigeo_scale_mode} locks one scale for the whole stream and "
            "so requires --geo_vigeo_mode chunk|online: with independent windows there is no stream to "
            "anchor, and the lock would freeze one window's scale for unrelated windows. "
            "(--geo_vigeo_scale_mode=fixed is safe under offline: it is a constant.)")

    # inject spatial visibility-aware noise config
    pipe._geo_vsnoise_cfg = {
        "enabled": bool(args.visibility_aware_noise),
        "sigma_invisible": float(args.warp_noise_sigma_invisible),
        "sigma_min": float(args.warp_noise_sigma_min),
        "sigma_max": float(args.warp_noise_sigma_max),
        "visible_token_threshold": float(args.visible_token_threshold),
        "geo_warp_stage0_only": bool(args.geo_warp_stage0_only),  # reference alignment: inject warp only at stage0, drop it for i_s>0
        "geo_warp_warm_encode": bool(args.geo_warp_warm_encode),  # make the leading warp latent follow the continuous-frame distribution
        "geo_oneshot_output_decode": bool(args.geo_oneshot_output_decode),  # diagnostic: re-decode the output from all latents at once
        "geo_drop_warp": bool(args.geo_drop_warp),   # diagnostic: force warp visibility=0 -> all warp tokens filtered out
        "warp_vis_cap": float(args.geo_warp_vis_cap),                    # primary: adaptive warp visibility cap (0=off)
        "warp_patch_drop_ratio": float(args.geo_warp_patch_drop_ratio),  # control: fixed-ratio per-patch drop (0=off)
        "warp_patch_drop_grid": int(args.geo_warp_patch_drop_grid),      # drop block size (latent cells)
        "warp_patch_drop_seed": int(args.seed),                          # reproducibility: drop generator seed = run seed
        "rope_alignment": not bool(args.no_rope_alignment),
        "prefix_idx_mode": str(args.prefix_idx_mode),
        "warp_rope_mode": str(args.warp_rope_mode),
        "warp_keep_clean_anchor": bool(args.warp_keep_clean_anchor),
        "invisible_history_noise": bool(args.invisible_history_noise),
        "warp_lag_chunks": int(args.warp_lag_chunks),
        "warp_rope_noise_center_align": bool(args.warp_rope_noise_center_align),
        # DA3 known-trajectory cloud backend (read by the pipeline)
        "recon_backend": str(args.geo_recon_backend),
        "cloud_update_n": int(args.geo_cloud_update_n),
        "cloud_voxel": float(args.geo_cloud_voxel),
        "cloud_splat_radius": int(args.geo_cloud_splat_radius),
        # Depth estimator selection + the active backend's paths. da3_weights / da3_src carry whichever
        # backend is active, matching what resolve_cloud_warp produces on the training side.
        "depth_backend": str(args.geo_depth_backend),
        "da3_src": (args.geo_vigeo_src if args.geo_depth_backend == "vigeo" else args.geo_da3_src),
        "da3_weights": (args.geo_vigeo_weights if args.geo_depth_backend == "vigeo" else args.geo_da3_weights),
        "da3_process_res": int(args.geo_da3_process_res),
        "vigeo_mode": str(args.geo_vigeo_mode),
        "vigeo_chunk_size": int(args.geo_vigeo_chunk_size),
        "vigeo_intr_source": str(args.geo_vigeo_intr_source),
        "vigeo_conf_transform": str(args.geo_vigeo_conf_transform),
        "vigeo_scale_mode": str(args.geo_vigeo_scale_mode),
        "vigeo_scale_value": float(args.geo_vigeo_scale_value),
        "vigeo_depth_median_target": float(args.geo_vigeo_depth_median_target),
        "vigeo_anchor_windows": int(args.geo_vigeo_anchor_windows),
        "vigeo_cache_keep_frames": int(args.geo_vigeo_cache_keep_frames),
        "vigeo_total_budget": int(args.geo_vigeo_total_budget),
        "vigeo_num_tokens": (int(args.geo_vigeo_num_tokens) or None),
        "render_mode": str(args.geo_da3_render_mode),   # backward / backward_zbuf / multisrc / recall (matches the training render_mode)
        "bw_fill_iters": int(args.geo_bw_fill_iters),
        # backward_zbuf hybrid despeckle (off by default; read via cfg.get, same names as the training cloud_warp block)
        "zbuf_despeckle": bool(args.geo_zbuf_despeckle),
        "zbuf_despeckle_ksize": int(args.geo_zbuf_despeckle_ksize),
        "zbuf_despeckle_fill_iters": int(args.geo_zbuf_despeckle_fill_iters),
        # [cloud hygiene] off by default; read by the pipeline and passed to DA3FrameBank
        "cloud_hygiene": bool(args.geo_cloud_hygiene),
        "hygiene_conf_abs": float(args.geo_hygiene_conf_abs),   # primary gate
        "hygiene_sat_max": float(args.geo_hygiene_sat_max),
        "hygiene_flat_std": float(args.geo_hygiene_flat_std),
        "hygiene_flat_win": int(args.geo_hygiene_flat_win),
        "hygiene_conf_pct": float(args.geo_hygiene_conf_pct),
        # [method 1] ingest consistency gate (read by the pipeline)
        "consist_gate": bool(args.geo_consist_gate),
        "consist_tau": float(args.geo_consist_tau),
        "consist_ref_frames": int(args.geo_consist_ref_frames),
        "consist_min_ref": int(args.geo_consist_min_ref),
        # [variant 1] confidence-adaptive tolerance
        "consist_adaptive": bool(args.geo_consist_adaptive),
        "consist_tau_lo": float(args.geo_consist_tau_lo),
        "consist_tau_hi": float(args.geo_consist_tau_hi),
        "consist_conf_ref": float(args.geo_consist_conf_ref),
        # hole probation (0=off=legacy pass-through)
        "consist_probation": int(args.geo_consist_probation),
        "consist_probation_frac": float(args.geo_consist_probation_frac),   # only hold large-hole interiors
        "consist_probation_win": int(args.geo_consist_probation_win),
        # per-window ingest scale alignment
        "consist_scale_align": bool(args.geo_consist_scale_align),
        # colour-statistics re-anchoring (independent of consist_gate)
        "color_anchor": bool(args.geo_color_anchor),
        "color_anchor_alpha": float(args.geo_color_anchor_alpha),
        "color_anchor_ref_windows": int(args.geo_color_anchor_ref_windows),
        # [self re-anchor] divergence trigger -> re-anchor to the best past contiguous window
        "self_reanchor": bool(args.geo_self_reanchor),
        "self_reanchor_scale_thr": float(args.geo_self_reanchor_scale_thr),
        "self_reanchor_rej_thr": float(args.geo_self_reanchor_rej_thr),
        "self_reanchor_min_gap": int(args.geo_self_reanchor_min_gap),
        "self_reanchor_keep_windows": int(args.geo_self_reanchor_keep_windows),
        "self_reanchor_lookback": int(args.geo_self_reanchor_lookback),
        "self_reanchor_anchor_min_conf": float(args.geo_self_reanchor_anchor_min_conf),
        "self_reanchor_pin_prime": bool(args.geo_self_reanchor_pin_prime),
        "self_reanchor_pin_windows": int(args.geo_self_reanchor_pin_windows),
        # [method 2] bounded / periodically re-anchored history
        "hist_max_frames": int(args.geo_hist_max_frames),
        "reanchor_every": int(args.geo_reanchor_every),
        "reanchor_keep_frames": int(args.geo_reanchor_keep_frames),
        # [method 3] free-space carving
        "carve": bool(args.geo_carve),
        "carve_margin": float(args.geo_carve_margin),
        "carve_ref_frames": int(args.geo_carve_ref_frames),
        "carve_min_views": int(args.geo_carve_min_views),   # [variant 2] multi-view vote threshold
        "carve_strike_windows": int(args.geo_carve_strike_windows),   # cross-window strike (1=off)
    }
    print(f"[geo-infer] vsnoise cfg: {pipe._geo_vsnoise_cfg}", flush=True)
    # i2v chunk-0 reference-image warp (read in _geo_render_chunk_da3). Default on; only fires for i2v
    # chunk 0 with an empty pool, so v2v/t2v and every later chunk are unaffected.
    pipe._geo_chunk0_ref_warp = bool(args.geo_chunk0_ref_warp)
    pipe._geo_chunk0_target_disparity_px = float(args.geo_chunk0_target_disparity_px)
    print(f"[geo-infer] chunk0_ref_warp = {pipe._geo_chunk0_ref_warp} "
          f"(i2v chunk-0 warp from reference image; needs vigeo depth_median/fixed)", flush=True)
    print(f"[geo-infer] chunk0_target_disparity_px = {pipe._geo_chunk0_target_disparity_px:g} "
          f"({'off -> pure depth_median scale' if pipe._geo_chunk0_target_disparity_px <= 0 else 'p90 pixel budget for chunk-0 parallax'})",
          flush=True)

    # load optional main LoRA adapter
    if args.lora_path:
        lora_path = Path(args.lora_path)
        if lora_path.is_file() and lora_path.suffix == ".safetensors":
            _lora_dir, _lora_name = str(lora_path.parent), lora_path.name
        else:
            _lora_dir, _lora_name = str(lora_path), "pytorch_lora_weights.safetensors"
        pipe.load_lora_weights(_lora_dir, weight_name=_lora_name, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])
        print(f"[geo-infer] main LoRA loaded: {args.lora_path}", flush=True)

    # load optional partial weights (memory convs + camera_ctrl.safetensors from same dir)
    if args.partial_path:
        from argparse import Namespace
        from evoke.utils.utils_base import load_extra_components
        # build minimal fake args required by load_extra_components gates
        _fake_args = Namespace(
            training_config=Namespace(
                is_enable_stage1=True,
                restrict_self_attn=False,
                is_amplify_history=False,
                is_use_gan=False,
            ),
            model_config=Namespace(
                camera_control=Namespace(
                    enabled=bool(getattr(transformer, "enable_cam_control", False)),
                    cam_ctrl_layers=None,      # None = load all layers
                    strict_camera_ckpt=True,   # raise on missing/extra keys
                ),
            ),
        )
        # GEO: warp_residual_mlp is attached to the transformer by train_evoke (not in __init__), so it is
        # missing here -> partial.pth's warp_residual_mlp.* keys would be silently skipped by strict=False,
        # i.e. training (through the MLP) and inference (skipping it) would diverge. Rebuild it from the saved
        # tensor shapes and load, so a backward-warp checkpoint with an MLP runs faithfully (nomlp checkpoints
        # lack these keys and skip this automatically).
        try:
            import torch.nn as _nn
            _psd = torch.load(args.partial_path, map_location="cpu")
            _psd = _psd.get("state_dict", _psd) if isinstance(_psd, dict) else _psd
            _mlp_keys = {k: v for k, v in _psd.items() if k.startswith("warp_residual_mlp.")}
            if _mlp_keys and getattr(transformer, "warp_residual_mlp", None) is None:
                _hid, _in = _mlp_keys["warp_residual_mlp.0.weight"].shape    # [hidden, inner]
                _mlp = _nn.Sequential(_nn.Linear(_in, _hid), _nn.GELU(), _nn.Linear(_hid, _in))
                _mlp.load_state_dict({k.replace("warp_residual_mlp.", ""): v for k, v in _mlp_keys.items()}, strict=True)
                transformer.warp_residual_mlp = _mlp.to(dtype=weight_dtype)   # pipe.to(device) moves it to GPU later
                print(f"[geo-infer] warp_residual_mlp attached + loaded ({_in}->{_hid}->{_in}, {len(_mlp_keys)} keys) -- train/infer aligned", flush=True)
        except Exception as _e:
            print(f"[geo-infer] warp_residual_mlp attach skipped: {_e}", flush=True)
        load_extra_components(_fake_args, transformer, args.partial_path)
        print(f"[geo-infer] partial weights loaded: {args.partial_path}", flush=True)

    # === restrict_self_attn: set the flag on the config and on every block.attn1 (the processor reads attn.restrict_self_attn) ===
    # restrict_lora stays False -> no extra q/k/v_loras needed, a merged checkpoint loads cleanly. The cache is enabled via pipe(use_kv_cache=) and only works under restrict.
    # A stage2 cache checkpoint is trained with restrict_self_attn=true, so inference must enable this to match training (the base config.json has restrict=false; it is flipped on here at runtime).
    if args.restrict_self_attn:
        transformer.config.restrict_self_attn = True
        _n_set = 0
        for _blk in transformer.blocks:
            if hasattr(_blk, "attn1"):
                _blk.attn1.restrict_self_attn = True
                _n_set += 1
        print(f"[geo-infer] restrict_self_attn=ON -> set on {_n_set} block.attn1 modules (use_kv_cache={args.use_kv_cache}).", flush=True)
        if not args.use_kv_cache:
            print("[geo-infer] [WARN] restrict is on but use_kv_cache is off -> only the history->noise attention block is saved, so the speedup is small (quality is unaffected).", flush=True)
    elif args.use_kv_cache:
        print("[geo-infer] [WARN] use_kv_cache is on but restrict_self_attn is off -> the cache does nothing (it only works under restrict).", flush=True)

    pipe = pipe.to(device)

    # Drop the per-chunk sampling bar: it covers ~2s, is rebuilt once per chunk, and being a \r redraw it
    #   is noise both on the terminal (against the chunk bar) and in the log file. The chunk-level bar
    #   plus its per-chunk summary line carry the same information. EVOKE_INFER_DEBUG=1 brings it back.
    if os.environ.get("EVOKE_INFER_DEBUG", "0") != "1":
        pipe.set_progress_bar_config(disable=True)

    _PIPE_CACHE.clear()      # one pipeline per process: a second key would double 28 GB of VRAM
    _PIPE_CACHE[key] = (pipe, transformer, vae, device)
    return pipe, transformer, vae, device

def main(argv=None):
    args = parse_args(argv)
    _check_prereqs(args)

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[geo-infer] output:  {output_dir}", flush=True)

    progress_path = output_dir / "progress.json"
    progress_chunks = max(1, (int(args.num_frames) + 32) // 33)

    def _write_progress(*, phase: str, chunks: int = progress_chunks, chunk_index: int = 0,
                        stage: int = 0, step: int = 0, total_steps: int = 0, **_unused) -> None:
        payload = {
            "phase": phase,
            "chunks": int(chunks),
            "chunkIndex": int(chunk_index),
            "stage": int(stage),
            "step": int(step),
            "totalSteps": int(total_steps),
            "updatedAt": time.time(),
        }
        temporary = progress_path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(progress_path)

    _write_progress(phase="preparing")

    pipe, transformer, vae, device = build_pipe(args)
    pipe._evoke_progress_callback = _write_progress
    pipe._evoke_offload_text_encoder = True

    # `seed` is a per-case arg (see _PER_CASE_ARGS), so a cached pipeline still carries the seed of
    #   whichever case built it. Only one field is affected -- the patch-drop generator, which is inert
    #   unless --geo_warp_patch_drop_ratio > 0 -- but re-set it here so the cached and the freshly-built
    #   pipeline behave identically for every case. Everything else the seed touches (the torch
    #   Generator below) is created per call.
    if getattr(pipe, "_geo_vsnoise_cfg", None) is not None:
        pipe._geo_vsnoise_cfg["warp_patch_drop_seed"] = int(args.seed)
        # ``auto`` resolves to depth_median for i2v and anchor for v2v. It is a
        # rollout policy consumed by the fresh per-call FrameBank, not a weight
        # construction setting, so switching Director modes must not rebuild the
        # 54 GB transformer.
        pipe._geo_vsnoise_cfg["vigeo_scale_mode"] = str(args.geo_vigeo_scale_mode)

    # enable GEO intermediate dumps if requested (pipeline reads _geo_dump_dir attr)
    if args.dump_geo_intermediates:
        _dump_dir = output_dir / "geo_debug"
        _dump_dir.mkdir(parents=True, exist_ok=True)
        pipe._geo_dump_dir = str(_dump_dir)
        print(f"[geo-infer] GEO intermediates dump → {_dump_dir}/", flush=True)

    # prepare inputs
    image_pil = None
    if args.image_path:
        image_pil = Image.open(args.image_path).convert("RGB").resize(
            (int(args.width), int(args.height)), Image.LANCZOS
        )
        print(f"[geo-infer] source image: {args.image_path}  size={image_pil.size}", flush=True)

    video_input = None
    if args.video_path:
        # load ref video with start_seconds offset and source→target fps resample
        from evoke.utils.ev_validation import load_ref_video_for_v2v
        video_input = load_ref_video_for_v2v(
            args.video_path,
            height=int(args.height),
            width=int(args.width),
            seconds=float(args.ref_seconds),
            target_fps=24,
            source_fps=int(args.lingbot_pose_source_fps),
            start_seconds=float(args.start_seconds),
        )                                                       # [T, 3, H, W] in [-1, 1], fp32 CPU
        print(
            f"[geo-infer] ref video (v2v): {args.video_path}  "
            f"sliced [{float(args.start_seconds):.1f}s, "
            f"{float(args.start_seconds) + float(args.ref_seconds):.1f}s] "
            f"@ {int(args.lingbot_pose_source_fps)}→24fps "
            f"→ shape={tuple(video_input.shape)} (T, C, H, W) [-1, 1]",
            flush=True,
        )

    # train-val aligned: v2v + GEO without --image_path uses the ref video first frame as source pixel.
    if image_pil is None and video_input is not None and args.use_geometric_state:
        image_pil = video_input[0:1].clone()   # [1, 3, H, W] fp32 [-1, 1]
        print("[geo-infer] source image auto-extracted from ref video first frame (v2v + GEO)", flush=True)

    lingbot_c2ws = None
    lingbot_Ks = None
    if args.lingbot_pose_path:
        lingbot_Ks, lingbot_c2ws = _load_lingbot_pose(args)
        print(f"[geo-infer] lingbot_c2ws: shape={tuple(lingbot_c2ws.shape)} Ks={tuple(lingbot_Ks.shape)} "
              f"from {args.lingbot_pose_path}", flush=True)

        # Pose control smoothing: low-pass the target-segment c2w to suppress high-frequency vipe jitter (before the cap, so the raw trajectory is de-jittered).
        if args.pose_smooth_win and args.pose_smooth_win > 1:
            from evoke.utils.trajectory_capping import smooth_c2w_trajectory
            _ref_pix_for_sm = max(0, int(round(float(args.ref_seconds) * 24)))
            lingbot_c2ws, _sm_info = smooth_c2w_trajectory(
                lingbot_c2ws, ref_pix=_ref_pix_for_sm, win=int(args.pose_smooth_win), verbose=True,
            )

        # Per-chunk trajectory cap: limit camera rotation/translation so the warp stays in-distribution.
        if args.max_deg_per_chunk > 0 or args.max_trans_per_chunk > 0:
            from evoke.utils.trajectory_capping import (
                analyze_per_chunk, cap_target_per_chunk_rotation, clamp_target_per_chunk_motion,
            )
            # The engine's chunks are not uniform: chunk 0 decodes 1 + 4*(W-1) = 33 pixel frames, every
            #   later chunk decodes 4*W = 36 (W = latent_window_size = 9). Capping on a uniform 33 grid
            #   drifts 3 frames per chunk against the real boundaries, so the window being clamped soon
            #   straddles two real chunks and the accumulated lag gets released in the last short window
            #   at several times the nominal speed -- which is the opposite of a cap.
            _W_pix, _W_pix0 = 36, 33
            _ref_pix_for_cap = max(0, int(round(float(args.ref_seconds) * 24)))
            _c2w_np = (lingbot_c2ws.detach().cpu().numpy()
                       if isinstance(lingbot_c2ws, torch.Tensor) else np.asarray(lingbot_c2ws))
            _c2w_tgt = _c2w_np[0, _ref_pix_for_cap:] if _c2w_np.ndim == 4 else _c2w_np[_ref_pix_for_cap:]
            _st = analyze_per_chunk(_c2w_tgt, frames_per_chunk=_W_pix, first_chunk_frames=_W_pix0)
            print(f"[traj-stat] target-only ({len(_c2w_tgt)} frames, {_st['n_chunks']} chunks): "
                  f"rot mean={_st['rot_mean']:.2f}° max={_st['rot_max']:.2f}° p95={_st['rot_p95']:.2f}°, "
                  f"trans mean={_st['trans_mean']:.2f}", flush=True)
            if args.cap_mode == "clamp":
                lingbot_c2ws, _n_new, _info = clamp_target_per_chunk_motion(
                    c2ws=lingbot_c2ws, ref_pix=_ref_pix_for_cap, frames_per_chunk=_W_pix,
                    first_chunk_frames=_W_pix0,
                    max_deg=float(args.max_deg_per_chunk), max_trans=float(args.max_trans_per_chunk), verbose=True,
                )
            else:
                # resample mode subdivides chunks and emits K * frames_per_chunk frames, so it only works
                #   on a uniform grid -- keep it on 33 rather than silently changing its frame arithmetic.
                lingbot_c2ws, _n_new, _info = cap_target_per_chunk_rotation(
                    c2ws=lingbot_c2ws, ref_pix=_ref_pix_for_cap, frames_per_chunk=_W_pix0,
                    max_deg=float(args.max_deg_per_chunk), max_trans=float(args.max_trans_per_chunk), verbose=True,
                )
                if _info.get("n_subdivided", 0) > 0:
                    _old = int(args.num_frames); args.num_frames = int(_n_new)
                    print(f"[traj-cap] num_frames auto-updated: {_old} -> {args.num_frames}", flush=True)

    # Joystick/energy-ring HUD on the pred video. Driven by the pose alone (no GT needed), so `auto`
    #   turns it on for every pose-carrying run -- i2v, v2v and segment -- and leaves t2v untouched.
    _hud_on = (args.joystick_hud == "on"
               or (args.joystick_hud == "auto" and lingbot_c2ws is not None))
    if args.joystick_hud == "on" and lingbot_c2ws is None:
        print("[geo-infer] [WARN] --joystick_hud=on but no pose was given -> nothing to draw, HUD off", flush=True)
        _hud_on = False
    # `both` = one generation, two encodes: a clean geo_pred.mp4 and an overlaid geo_pred_hud.mp4.
    #   _hud_on stays False so nothing burns the HUD in upstream (segments stay clean, and the pred
    #   array handed to the postprocess is the model's own pixels); the second file is written in
    #   postprocess_viz._write_pred_outputs, which draws on a copy.
    _hud_dual = (args.joystick_hud == "both" and lingbot_c2ws is not None)
    if args.joystick_hud == "both" and lingbot_c2ws is None:
        print("[geo-infer] [WARN] --joystick_hud=both but no pose was given -> nothing to draw, "
              "writing the plain geo_pred.mp4 only", flush=True)
    # The streaming path stitches geo_pred.mp4 from per-chunk segments and never reaches the
    #   postprocess writer, so it cannot produce the second file. Fail loudly rather than return a
    #   run that is quietly missing half of what was asked for.
    if _hud_dual and args.stream_long_video:
        raise SystemExit("[geo-infer] FATAL: --joystick_hud=both is not supported with "
                         "--stream_long_video (the streamed mp4 is stitched from segments, so the "
                         "clean/overlaid pair cannot be split). Use --joystick_hud on|off there.")

    # monkey-patch _decode_chunk_persistent_cache to dump per-chunk segment mp4s
    if args.save_chunk_segments:
        if args.vae_decode_type != "persistent":
            print(f"[geo-infer] [WARN] --save_chunk_segments only supports vae_decode_type=persistent (got {args.vae_decode_type}), skipping", flush=True)
        else:
            from evoke.utils.ev_validation import _load_gt_video_rgb_for_viz, add_joystick_overlay_from_c2ws
            segments_dir = output_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)

            _has_gt = bool(args.ref_video_for_viz and args.lingbot_pose_path)
            # The HUD only needs the pose, so it is NOT gated on the GT video: _c2ws_full is kept
            #   whenever a pose exists, and the GT-only work below stays behind _has_gt.
            _c2ws_full = lingbot_c2ws if (_has_gt or (_hud_on and lingbot_c2ws is not None)) else None
            _global_move_scale = None
            _global_rot_scale = None
            if _has_gt or (_hud_on and lingbot_c2ws is not None):
                # compute global scale once so all chunks share the same normalisation.
                # When the trajectory cap is active, use cap as joystick full-deflection (±1 == cap maxed);
                # uncapped dims fall back to clip p90 auto.
                from evoke.utils.ev_validation import resolve_joystick_scale_with_cap
                _global_move_scale, _global_rot_scale = resolve_joystick_scale_with_cap(
                    lingbot_c2ws,
                    cap_max_deg=float(args.max_deg_per_chunk),
                    cap_max_trans=float(args.max_trans_per_chunk),
                    frames_per_chunk=33,
                )
                if _has_gt:
                    print(f"[geo-infer] [save_chunk_segments] enabled (GT sliced on demand, not preloaded -> fast startup).", flush=True)
                    print(f"[geo-infer] [save_chunk_segments]   GT video: {args.ref_video_for_viz}", flush=True)
                    print(f"[geo-infer] [save_chunk_segments]   reads the matching pixel range per chunk (~33-36 frames/chunk, 1-2s)", flush=True)
                else:
                    print(f"[geo-infer] [save_chunk_segments] enabled, no GT video -> pred-only segments (no side-by-side)", flush=True)
                print(f"[geo-infer] [save_chunk_segments]   joystick HUD: {'ON' if _hud_on else 'off'} "
                      f"(--joystick_hud={args.joystick_hud})", flush=True)
                print(f"[geo-infer] [save_chunk_segments]   global move_scale={_global_move_scale:.4f}, "
                      f"rot_scale={_global_rot_scale:.4f} (whole-clip percentile, avoids inter-chunk jitter)", flush=True)
            else:
                print(f"[geo-infer] [save_chunk_segments] no GT/pose -> dumping pred-only segments (no side-by-side, no energy ring)", flush=True)

            # track chunk index and cumulative pixel offset across decode calls
                # gt_dry: the GT video only covers the source clip (~60s), so the tail of a long rollout (2min/60min) has no GT to compare against.
                #   After _GT_DRY_MAX consecutive chunks fail to slice a GT frame, stop slicing for good (otherwise every chunk
                #   re-decodes the source video for nothing -- hours of pure waste over 2400 chunks) and dump pred-only segments.
            _state = {"chunk_idx": 0, "pixel_offset": 0, "gt_dry": 0}
            _GT_DRY_MAX = 2

            import types
            _orig_decode_chunk = pipe._decode_chunk_persistent_cache

            def _patched_decode_chunk(_self, z_chunk, is_first_chunk, latents_mean, latents_std, vae_dtype,
                                      warm_latents=None, warm_repeat=0, **decode_kwargs):
                out = _orig_decode_chunk(z_chunk, is_first_chunk, latents_mean, latents_std, vae_dtype,
                                         warm_latents=warm_latents, warm_repeat=warm_repeat, **decode_kwargs)
                # out: [B, 3, T_pix, H, W] fp32 [-1, 1]
                try:
                    chunk_idx = _state["chunk_idx"]
                    pix_off = _state["pixel_offset"]
                    T_pix = int(out.shape[2])

                    # convert pred to uint8 RGB [T, H, W, 3]
                    pred_seg_np = (out[0].float().permute(1, 2, 3, 0).clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()
                    pred_seg_u8 = (pred_seg_np * 255.0).round().clip(0, 255).astype(np.uint8)
                    h_seg, w_seg = pred_seg_u8.shape[1], pred_seg_u8.shape[2]

                    # write pred-only segment. With the HUD on, draw it on a COPY: the GT|Pred panels
                    #   below are built from pred_seg_u8 and add their own overlay, so mutating it
                    #   here would double-draw on the 4-panel video.
                    _seg_out = pred_seg_u8
                    if _hud_on and _c2ws_full is not None:
                        # pred pixel 0 == first frame after the ref prefix (ref_seconds=0 for i2v/segment)
                        _rp = int(round(float(args.ref_seconds) * int(args.fps)))
                        _s = _rp + pix_off
                        _e = min(_s + T_pix, int(_c2ws_full.shape[0]))
                        if _e > _s:
                            _hud_frames = add_joystick_overlay_from_c2ws(
                                list(pred_seg_u8[: _e - _s]), _c2ws_full[_s:_e],
                                move_scale=_global_move_scale, rot_scale=_global_rot_scale,
                                label_left="Move", label_right="Rot",
                            )
                            # the pose can run out before the video does (replayed/extended tails):
                            #   keep the un-overlaid remainder rather than truncating the segment
                            _seg_out = list(_hud_frames) + list(pred_seg_u8[_e - _s:])
                    seg_pred_mp4 = segments_dir / f"segment_{chunk_idx:03d}_pred.mp4"
                    _w = cv2.VideoWriter(str(seg_pred_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(args.fps), (int(w_seg), int(h_seg)))
                    for f_rgb in _seg_out:
                        _w.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
                    _w.release()

                    if _has_gt and _state["gt_dry"] < _GT_DRY_MAX:
                        # v2v: pred pixel offset 0 corresponds to first frame after ref, offset GT slice by ref_pix
                        _ref_pix_viz = int(round(float(args.ref_seconds) * int(args.fps)))
                        # slice c2w for this chunk
                        _max_pose = int(_c2ws_full.shape[0])
                        _end = min(_ref_pix_viz + pix_off + T_pix, _max_pose)
                        _start = _ref_pix_viz + pix_off
                        if _end > _start:
                            c2w_slice = _c2ws_full[_start:_end]  # [n, 4, 4]
                            # load only the GT frames for this chunk (avoids preloading all frames)
                            _gt_err = False
                            try:
                                _gt_t0 = time.time()
                                gt_slice = _load_gt_video_rgb_for_viz(
                                    args.ref_video_for_viz,
                                    height=int(args.height), width=int(args.width),
                                    target_frame_indices=list(range(_start, _end)),
                                    target_fps=int(args.fps),
                                    source_fps=int(args.lingbot_pose_source_fps),
                                    start_seconds=float(args.start_seconds),
                                )
                                _gt_elapsed = time.time() - _gt_t0
                            except Exception as _ge:
                                print(f"[geo-infer] [chunk {chunk_idx:03d}] GT slice failed: {type(_ge).__name__}: {_ge}", flush=True)
                                gt_slice = []
                                _gt_elapsed = 0.0
                                _gt_err = True     # an error is not the same as GT running out -> do not count it as dry (the next chunk retries)
                            pred_slice = list(pred_seg_u8[: len(gt_slice)])
                            if len(gt_slice) > 0:
                                # overlay joystick HUD using pre-computed global scale
                                gt_cam = add_joystick_overlay_from_c2ws(
                                    gt_slice, c2w_slice,
                                    move_scale=_global_move_scale, rot_scale=_global_rot_scale,
                                    label_left="GT Move", label_right="GT Rot",
                                )
                                pred_cam = add_joystick_overlay_from_c2ws(
                                    pred_slice, c2w_slice,
                                    move_scale=_global_move_scale, rot_scale=_global_rot_scale,
                                    label_left="Pred Move", label_right="Pred Rot",
                                )
                                # append warp + warp_mask columns -> 4-way concat: GT | Pred | Warp | WarpMask
                                def _read_mp4_rgb(_p):
                                    _fr = []
                                    if _p.is_file():
                                        _c = cv2.VideoCapture(str(_p))
                                        while True:
                                            _ret, _bf = _c.read()
                                            if not _ret:
                                                break
                                            _fr.append(cv2.cvtColor(_bf, cv2.COLOR_BGR2RGB))
                                        _c.release()
                                    return _fr
                                _warp_frames = _read_mp4_rgb(output_dir / "geo_debug" / f"chunk_{chunk_idx:03d}_warp.mp4")
                                _vis_frames = _read_mp4_rgb(output_dir / "geo_debug" / f"chunk_{chunk_idx:03d}_vis.mp4")
                                # Tolerant: build 4-way (GT|Pred|Warp|WarpMask) whenever warp/vis frames exist.
                                # warp/vis are per-chunk render frames (e.g. 33) and may be shorter than the
                                # decoded segment (e.g. 36) -> index-clamp to the last warp frame for the tail
                                # instead of dropping to 2 columns.
                                _has_warp = len(_warp_frames) > 0
                                _has_vis = len(_vis_frames) > 0
                                _num_cols = 2 + (1 if _has_warp else 0) + (1 if _has_vis else 0)

                                def _fit_seg(_f):
                                    if _f.shape[0] != h_seg or _f.shape[1] != w_seg:
                                        _f = cv2.resize(_f, (int(w_seg), int(h_seg)))
                                    return _f

                                seg_sbs_mp4 = segments_dir / f"segment_{chunk_idx:03d}_gt_vs_pred.mp4"
                                _w2 = cv2.VideoWriter(str(seg_sbs_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                                      float(args.fps), (int(w_seg * _num_cols), int(h_seg)))
                                for i, (gtf, prf) in enumerate(zip(gt_cam, pred_cam)):
                                    _parts = [gtf, prf]
                                    if _has_warp:
                                        _parts.append(_fit_seg(_warp_frames[min(i, len(_warp_frames) - 1)]))
                                    if _has_vis:
                                        _parts.append(_fit_seg(_vis_frames[min(i, len(_vis_frames) - 1)]))
                                    row = np.concatenate(_parts, axis=1)
                                    _w2.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
                                _w2.release()
                                _state["gt_dry"] = 0          # got GT -> reset the dry counter
                                _warp_tag = (" + Warp" if _has_warp else "") + (" + WarpMask" if _has_vis else "")
                                # Under the chunk progress bar this repeats what the per-chunk summary
                                #   line already says, and being a plain print it lands on the bar.
                                if not _PROGRESS_UI:
                                    print(f"[geo-infer] [chunk {chunk_idx:03d}] dumped: {seg_pred_mp4.name} + "
                                          f"{seg_sbs_mp4.name} (GT|Pred{_warp_tag}, {T_pix} frames @ pix {pix_off}..{_end}, GT slice {_gt_elapsed:.2f}s)",
                                          flush=True)
                            else:
                                if not _gt_err:
                                    _state["gt_dry"] += 1      # the slice returned 0 frames = GT length exhausted
                                print(f"[geo-infer] [chunk {chunk_idx:03d}] {seg_pred_mp4.name} (no GT, pred only)", flush=True)
                        else:
                            _state["gt_dry"] += 1
                            print(f"[geo-infer] [chunk {chunk_idx:03d}] pose segment out of range (pix_off={pix_off}>{_max_pose}), pred only", flush=True)
                        if _state["gt_dry"] == _GT_DRY_MAX:
                            print(f"[geo-infer] [chunk {chunk_idx:03d}] GT failed to slice for {_GT_DRY_MAX} consecutive segments "
                                  f"(source video exhausted) -> later chunks dump pred-only segments and stop reading GT.", flush=True)
                    else:
                        print(f"[geo-infer] [chunk {chunk_idx:03d}] {seg_pred_mp4.name} ({T_pix} frames)", flush=True)

                    _state["chunk_idx"] = chunk_idx + 1
                    _state["pixel_offset"] = pix_off + T_pix
                except Exception as _e:
                    print(f"[geo-infer] [WARN] chunk segment dump failed: {type(_e).__name__}: {_e}", flush=True)
                return out

            pipe._decode_chunk_persistent_cache = types.MethodType(_patched_decode_chunk, pipe)
            print(f"[geo-infer] [save_chunk_segments] hook installed; dumping after every chunk decode -> {segments_dir}/segment_*", flush=True)

    # run pipeline
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    print(f"[geo-infer] ===== pipe inference sample_type={args.sample_type} "
          f"(num_frames={args.num_frames}, steps={args.num_inference_steps}) =====", flush=True)
    # Parse per-chunk event chunk indices ("" -> []); empty list keeps the rollout path unchanged.
    _event_chunks = [int(x) for x in str(args.event_chunks).split(",") if x.strip() != ""]
    # -- segment prompts: {chunk_index: text}; empty dict => baseline path --
    _chunk_prompts = {}
    if str(args.prompt_schedule).strip():
        _spec = str(args.prompt_schedule).strip()
        # Discriminate on the leading character, not on is_file(): infer_batch.py always passes the
        #   schedule inline, and stat() on a string longer than NAME_MAX (255) raises ENAMETOOLONG
        #   before json.loads ever runs -- which is every realistic schedule.
        _raw = (json.loads(_spec) if _spec[0] in "[{"
                else json.loads(Path(_spec).read_text()))
        _stride = 9 * 4          # latent_window_size * vae temporal factor = 36 frames per chunk
        for _e in _raw:
            if "start_chunk" in _e:
                _ck = int(_e["start_chunk"])
            else:
                _ck = int(float(_e["start_sec"]) * float(args.fps) / _stride)
            _chunk_prompts[_ck] = str(_e["prompt"])
        print(f"[geo-infer] prompt schedule: "
              + ", ".join(f"chunk{c}(~{c * _stride / float(args.fps):.1f}s)" for c in sorted(_chunk_prompts)),
              flush=True)
    if _event_chunks:
        print(f"[geo-infer] event chunks (drop warp + static cam + skip bank): {_event_chunks} "
              f"event_prompt={args.event_prompt!r}", flush=True)
    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=int(args.height), width=int(args.width),
        num_frames=int(args.num_frames),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        image=image_pil,  # None for t2v
        image_noise_sigma_min=float(args.image_noise_sigma_min),
        image_noise_sigma_max=float(args.image_noise_sigma_max),
        video=video_input,  # v2v conditioning
        video_noise_sigma_min=float(args.video_noise_sigma_min),
        video_noise_sigma_max=float(args.video_noise_sigma_max),
        use_dynamic_shifting=not bool(args.no_dynamic_shifting),
        time_shift_type="exponential",
        is_keep_x0=True,
        history_sizes=[16, 2, 1],
        is_enable_stage2=bool(args.is_enable_stage2),
        vae_decode_type=str(args.vae_decode_type),
        output_type="np",
        return_dict=True,
        generator=generator,
        use_kv_cache=bool(args.use_kv_cache),   # default False -> EvokePipeline leaves the cache off; pair with --restrict_self_attn
        stream_output=bool(args.stream_long_video),   # True -> do not accumulate full-clip pixels, return frames=None
    )
    if args.is_enable_stage2:
        pipe_kwargs["stage2_num_stages"] = int(args.stage2_num_stages)
        pipe_kwargs["stage2_num_inference_steps_list"] = [int(x) for x in args.stage2_steps]
        print(f"[geo-infer] stage2 pyramid: {args.stage2_num_stages} stages, steps={args.stage2_steps}, "
              f"warp_mode={args.stage2_warp_compression_mode}", flush=True)
    # DMD scheduling. These two were previously never passed, so the pipeline fell back to
    #   use_dmd=False / is_amplify_first_chunk=False (pipeline_evoke.py) regardless of how the
    #   checkpoint was distilled. For a DMD ckpt trained with is_amplify_first_chunk=true the two paths
    #   differ on exactly one thing (pipeline_evoke.py): the FIRST chunk gets
    #   `steps*2 + 1` timesteps, i.e. 2 sampling steps per pyramid stage instead of 1. Everywhere else
    #   the two branches are algebraically identical at steps=[1,1,1]. Chunk 0 is the anchor the whole
    #   rollout is built on (every later chunk's prev_short and warp source come from it), so
    #   under-denoising it propagates. Training/validation passes both (train_evoke.py).
    pipe_kwargs["use_dmd"] = bool(args.use_dmd)
    pipe_kwargs["is_amplify_first_chunk"] = bool(args.is_amplify_first_chunk)
    if args.is_amplify_first_chunk and not args.use_dmd:
        # is_amplify_first_chunk is only read inside the `if use_dmd:` branch, so this combination
        #   silently does nothing -- fail instead of pretending the first chunk was amplified.
        sys.exit("[ERROR] --is_amplify_first_chunk has no effect without --use_dmd "
                 "(pipeline_evoke.py reads it only inside the use_dmd branch).")
    if args.use_dmd:
        print(f"[geo-infer] DMD scheduling: use_dmd=True "
              f"amplify_first_chunk={bool(args.is_amplify_first_chunk)} "
              f"-> first chunk {'2' if args.is_amplify_first_chunk else '1'} step(s)/stage, "
              f"later chunks 1 step/stage", flush=True)
    # Carry the warp compression mode (and match training's visibility threshold) via attention_kwargs.
    pipe_kwargs["attention_kwargs"] = {
        "stage2_warp_compression_mode": str(args.stage2_warp_compression_mode),
        "history_visible_token_threshold": 0.1,
    }
    if args.warp_rope_noise_center_align:
        assert str(args.stage2_warp_compression_mode) == "fixed_mem", (
            "--warp_rope_noise_center_align requires --stage2_warp_compression_mode fixed_mem."
        )
        pipe_kwargs["attention_kwargs"]["warp_rope_noise_center_align"] = True
    # Segment prompts are text-embedding only and the rollout chunk loop is not warp-gated
    #   (pipeline_evoke.py `for k in _chunk_iter`), so the schedule must reach the pipe on the
    #   warp-off path too -- nesting it under use_geometric_state made --prompt_schedule a silent
    #   no-op there. No-op when the schedule is empty.
    pipe_kwargs["chunk_prompts"] = _chunk_prompts
    if args.use_geometric_state:
        import math as _math
        pipe_kwargs["use_geometric_state"] = True
        pipe_kwargs["geo_disable_prev_short"] = bool(args.geo_disable_prev_short)
        pipe_kwargs["geo_lora_path"] = args.geo_lora_path
        # FrameBank retrieve config
        pipe_kwargs["geo_score"] = str(args.geo_score)
        pipe_kwargs["geo_nearby_k"] = int(args.geo_nearby_k)
        pipe_kwargs["geo_select_k"] = int(args.geo_select_k)
        # legacy compat: pipe still accepts geo_top_k
        pipe_kwargs["geo_top_k"] = int(args.geo_select_k)
        pipe_kwargs["geo_bank_max"] = int(args.geo_bank_max) if args.geo_bank_max > 0 else None
        pipe_kwargs["geo_init_k"] = int(args.geo_init_k)
        if args.geo_score == "v3":
            pipe_kwargs["geo_score_kwargs"] = {
                "depth": float(args.geo_v3_depth),
                "fov_rad": _math.radians(float(args.geo_v3_fov_deg)),
            }
        else:
            pipe_kwargs["geo_score_kwargs"] = {}
        pipe_kwargs["lingbot_c2ws"] = lingbot_c2ws
        # Ks must be provided together with c2ws for Plücker construction
        pipe_kwargs["lingbot_Ks"] = lingbot_Ks
        # Per-chunk skill/VFX event control (no-op when event_chunks is empty).
        pipe_kwargs["event_chunks"] = _event_chunks
        pipe_kwargs["event_prompt"] = str(args.event_prompt)
    elif lingbot_c2ws is not None:
        # baseline with cam control (when transformer.enable_cam_control=True)
        pipe_kwargs["lingbot_c2ws"] = lingbot_c2ws
        pipe_kwargs["lingbot_Ks"] = lingbot_Ks

    # The persistent worker parks the text encoder on CPU between requests and after prompt encoding;
    # bring it back only for the short embedding phase. This frees several GiB before ViGeo/VAE peak.
    pipe.text_encoder.to(device)
    out = pipe(**pipe_kwargs)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[geo-infer] ===== pipe done: {elapsed:.1f}s, peak VRAM {peak:.1f} GB =====", flush=True)

    # Streaming (very long video): the pipeline holds no full frame array, so the complete video is stitched from per-chunk segments (constant memory).
    if args.stream_long_video:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from postprocess_viz import assemble_from_segments
        _asm = assemble_from_segments(output_dir, fps=int(args.fps))
        if _asm.get("pred") is None:
            raise SystemExit(f"[geo-infer] FATAL: streaming assembly produced no geo_pred.mp4 "
                             f"({_asm.get('n_pred_seg', 0)} pred segments) -- see the [assemble] log above")
        if _asm.get("sbs") is None:
            print(f"[geo-infer] [WARN] no gt_vs_pred_cam_viz.mp4 (0 GT segments: the GT source video is shorter than this rollout, "
                  f"or every GT slice failed) -- the plain pred video was still written.", flush=True)
        print(f"[geo-infer] DONE (stream mode).", flush=True)
        return

    # collect pred frames as uint8 RGB
    video_np = out.frames[0]
    if video_np.dtype != np.uint8:
        video_np = (np.clip(video_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    print(f"[geo-infer] output: shape={video_np.shape}, range=[{video_np.min()}, {video_np.max()}], mean={video_np.mean():.1f}", flush=True)

    # Post-inference encode (pred mp4 + sample frames + optional GT|Pred joystick HUD).
    # CPU-bound and GPU-idle, so with --bg_postprocess we hand it to a detached worker and
    # exit immediately, letting the next sample's GPU inference overlap this encode.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from postprocess_viz import run_postprocess

    _pp_params = {
        "fps": int(args.fps),
        "ref_video_for_viz": args.ref_video_for_viz,
        "lingbot_pose_path": args.lingbot_pose_path,
        "height": int(args.height), "width": int(args.width),
        "num_frames": int(args.num_frames),
        "ref_seconds": float(args.ref_seconds), "start_seconds": float(args.start_seconds),
        "lingbot_pose_source_fps": int(args.lingbot_pose_source_fps),
        "lingbot_pose_source_resolution": list(args.lingbot_pose_source_resolution),
        "lingbot_pose_type": args.lingbot_pose_type,
        "lingbot_fallback_default_intrinsic": bool(getattr(args, "lingbot_fallback_default_intrinsic", False)),  # passed to cam_viz; fixes cam_viz silently failing when a pose npz has no intrinsics
        "use_geometric_state": bool(args.use_geometric_state),
    }
    # Joystick HUD for geo_pred.mp4 + sample frames. The c2ws go to disk rather than into the params
    #   dict: --bg_postprocess json-dumps params, and a long rollout's trajectory is far too big for
    #   that. Scales are recomputed here so the HUD does not depend on --save_chunk_segments.
    #   (The STREAM_LONG path never reaches here -- it stitches geo_pred.mp4 from the per-chunk
    #   segments, which already have the HUD burned in.)
    if (_hud_on or _hud_dual) and lingbot_c2ws is not None:
        from evoke.utils.ev_validation import resolve_joystick_scale_with_cap
        _ms, _rs = resolve_joystick_scale_with_cap(
            lingbot_c2ws,
            cap_max_deg=float(args.max_deg_per_chunk),
            cap_max_trans=float(args.max_trans_per_chunk),
            frames_per_chunk=33,
        )
        _c2ws_npy = output_dir / ".joystick_c2ws.npy"
        np.save(_c2ws_npy, np.asarray(lingbot_c2ws))
        _pp_params.update({
            "joystick_c2ws_npy": str(_c2ws_npy),
            "joystick_move_scale": float(_ms), "joystick_rot_scale": float(_rs),
            "joystick_ref_pix": int(round(float(args.ref_seconds) * int(args.fps))),
            # dual -> the writer emits both files; absent/False keeps the historical single-file
            # behaviour (overlaid when the HUD is on, clean when it is off).
            "joystick_dual": bool(_hud_dual),
        })

    if args.bg_postprocess:
        _spawn_bg_postprocess(output_dir, video_np, _pp_params, max_workers=int(args.bg_postprocess_max))
        print(f"[geo-infer] DONE (encode running in background, see {output_dir}/postproc.log).", flush=True)
    else:
        run_postprocess(output_dir, video_np, _pp_params)
        print(f"[geo-infer] DONE.", flush=True)


def _acquire_load_slot(root: Path, poll: float = 3.0):
    """Block until fewer than EVOKE_MAX_CONCURRENT_LOADS workers are building a pipeline.

    Same live-pid-marker scheme as _wait_for_worker_slot: a slot is a <pid>.lock under
    <root>/.load_slots, and a marker whose pid is gone is pruned so a crashed worker cannot wedge the
    queue. Unset or <=0 disables the throttle and returns None.
    """
    try:
        cap = int(os.environ.get("EVOKE_MAX_CONCURRENT_LOADS", "0") or 0)
    except ValueError:
        cap = 0
    if cap <= 0:
        return None
    slots = root / ".load_slots"
    slots.mkdir(parents=True, exist_ok=True)
    mine = slots / f"{os.getpid()}.lock"
    while True:
        live = 0
        for m in list(slots.glob("*.lock")):
            if m == mine:
                continue
            try:
                pid = int(m.stem)
            except ValueError:
                pid = 0
            if pid > 0 and _pid_alive(pid):
                live += 1
            elif pid > 0:
                try:
                    m.unlink()
                except OSError:
                    pass
        if live < cap:
            mine.write_text(str(os.getpid()))
            print(f"[geo-infer] load slot acquired ({live + 1}/{cap} building)", flush=True)
            return mine
        print(f"[geo-infer] waiting for a load slot ({live}/{cap} shards building a pipeline) ...",
              flush=True)
        time.sleep(poll)


def _release_load_slot(marker) -> None:
    try:
        marker.unlink()
    except OSError:
        pass


def run_argv_batch(path: str) -> int:
    """Run every case in `path` in THIS process. One JSON object per line:

        {"argv": [...], "name": "<case>", "log": "<path>"|null}

    Why: infer_batch.py used to spawn one interpreter per case, so the pipeline was rebuilt every
    time -- measured 131 s of a 251 s per-case wall time for this ckpt (54 GB transformer + 22 GB text
    encoder read over a network filesystem). build_pipe() caches on everything except the per-case
    args, so the second case onwards reuses the loaded weights and pays only the generation itself.

    `log` keeps the per-case log files the previous one-process-per-case layout produced: this process
    redirects stdout/stderr into it for the duration of that case, so `_logs/<case>.log` still holds
    exactly that case's engine output (which is what makes a single bad case debuggable afterwards),
    while the shard's own stdout gets one progress line per case.

    A failing case must not take the rest of the shard with it (a 2000-case sweep would lose hours),
    so each case is wrapped: the traceback goes to that case's log, the case is counted, and the loop
    continues. The exit code is non-zero iff at least one case failed, which the shell driver checks.
    """
    import contextlib
    import gc
    import traceback

    with open(path) as f:
        jobs = [json.loads(l) for l in f if l.strip()]
    print(f"[geo-infer] argv batch: {len(jobs)} case(s) in one process "
          f"(pipeline built once, then reused)", flush=True)
    # Throttle how many shards build their pipeline at the same time. Building one streams the whole
    #   checkpoint through host RAM (54 GB transformer + 22 GB text encoder for post_distill), so with
    #   two shards per GPU all 16 loading at once SIGKILLs the workers mid-load -- the symptom is a
    #   per-case log that stops right after the "transformer:" line with no traceback. The throttle
    #   belongs here rather than on the shard launch: a shard whose cases are all already done never
    #   loads anything and must not be delayed. Slot is held only across the first case, by which point
    #   build_pipe has returned and the resident footprint has settled.
    _slot = _acquire_load_slot(Path(path).parent) if len(jobs) else None
    failed = []
    for i, job in enumerate(jobs, 1):
        argv, name, log = job["argv"], job.get("name", f"case{i}"), job.get("log")
        t0 = time.time()
        print(f"[geo-infer] [{i}/{len(jobs)}] {name} ...", flush=True)
        try:
            if log:
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                lg = open(log, "w")
                ctx = contextlib.ExitStack()
                ctx.enter_context(lg)
                ctx.enter_context(contextlib.redirect_stdout(lg))
                ctx.enter_context(contextlib.redirect_stderr(lg))
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                try:
                    main(argv)
                except SystemExit as e:            # argparse / an explicit sys.exit inside main
                    if e.code not in (0, None):
                        raise RuntimeError(f"SystemExit({e.code})")
        except BaseException as e:                 # keep going; the shard has other cases to do
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"[geo-infer] [{i}/{len(jobs)}] {name} FAILED "
                  f"({type(e).__name__}) -- see {log}", flush=True)
            if log:
                with open(log, "a") as lg:
                    traceback.print_exc(file=lg)
            else:
                traceback.print_exc()
        else:
            print(f"[geo-infer] [{i}/{len(jobs)}] {name} done in {time.time() - t0:.0f}s", flush=True)
        finally:
            # Release the per-case peak (decoded pixels, warp dumps, the frame bank) before the next
            # case allocates its own. The pipeline weights are held by _PIPE_CACHE and stay resident.
            gc.collect()
            torch.cuda.empty_cache()
            if _slot is not None:      # first case done => the pipeline is built; let the next shard load
                _release_load_slot(_slot)
                _slot = None
    ok = len(jobs) - len(failed)
    print(f"\n[geo-infer] argv batch done: {ok} ok, {len(failed)} failed"
          + (f" -> {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


def run_argv_server(template_path: str, queue_path: str) -> int:
    """Keep one cached pipeline alive and execute file-queued argv jobs serially.

    ``template_path`` contains one batch-style JSON object whose argv defines the setup-affecting
    model recipe. The template is parsed and loaded immediately, but is never generated. Producers
    atomically place one JSON object per request under ``queue_path/requests``; this worker writes a
    matching response under ``queue_path/responses``. Per-request stdout/stderr still goes to the
    ``log`` path carried by the request, matching ``run_argv_batch``.
    """
    import contextlib
    import gc
    import traceback

    queue = Path(queue_path)
    requests = queue / "requests"
    responses = queue / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    state_path = queue / "state.json"

    def write_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def set_state(phase: str, message: str, request_id: str | None = None) -> None:
        write_json(state_path, {
            "phase": phase,
            "message": message,
            "requestId": request_id,
            "pid": os.getpid(),
            "updatedAt": time.time(),
        })

    try:
        template = json.loads(Path(template_path).read_text(encoding="utf-8"))
        template_args = parse_args(template["argv"])
        _check_prereqs(template_args)
        set_state("loading", "正在预加载 Post-distill 管线")
        pipe, _, _, _ = build_pipe(template_args)
        pipe._evoke_offload_text_encoder = True
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    except BaseException as error:
        set_state("error", f"预加载失败: {type(error).__name__}: {error}")
        traceback.print_exc()
        return 1

    set_state("ready", "Post-distill 管线已加载并常驻 GPU")
    print(f"[geo-infer] argv server READY pid={os.getpid()} queue={queue}", flush=True)
    while True:
        pending = sorted(requests.glob("*.json"), key=lambda path: path.stat().st_mtime)
        if not pending:
            time.sleep(0.25)
            continue
        request_path = pending[0]
        request_id = request_path.stem
        claimed = queue / f"running-{request_id}.json"
        try:
            request_path.replace(claimed)
        except FileNotFoundError:
            continue
        started = time.time()
        rc = 0
        error_text = None
        log = None
        cached_pipe = next(iter(_PIPE_CACHE.values()))[0] if _PIPE_CACHE else None
        original_decode = (
            cached_pipe._decode_chunk_persistent_cache if cached_pipe is not None else None
        )
        try:
            job = json.loads(claimed.read_text(encoding="utf-8"))
            argv, log = job["argv"], job.get("log")
            set_state("running", f"正在生成任务 {request_id}", request_id)
            if log:
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                log_file = open(log, "w", encoding="utf-8")
                stack = contextlib.ExitStack()
                stack.enter_context(log_file)
                stack.enter_context(contextlib.redirect_stdout(log_file))
                stack.enter_context(contextlib.redirect_stderr(log_file))
            else:
                stack = contextlib.nullcontext()
            with stack:
                try:
                    main(argv)
                except SystemExit as error:
                    if error.code not in (0, None):
                        raise RuntimeError(f"SystemExit({error.code})")
        except BaseException as error:
            rc = 1
            error_text = f"{type(error).__name__}: {error}"
            if log:
                with open(log, "a", encoding="utf-8") as log_file:
                    traceback.print_exc(file=log_file)
            else:
                traceback.print_exc()
        finally:
            # save_chunk_segments installs a per-case closure on the shared pipe. Restore the method
            # before accepting another request so chunk indices, output paths and reference frames
            # cannot leak across jobs. Also clear causal VAE/KV state when an exception bypassed the
            # normal end-of-rollout cleanup (notably CUDA OOM during decode).
            if cached_pipe is not None:
                if original_decode is not None:
                    cached_pipe._decode_chunk_persistent_cache = original_decode
                try:
                    cached_pipe.vae.clear_cache()
                except Exception:
                    pass
                try:
                    cached_pipe.transformer.clear_kv_cache()
                except Exception:
                    pass
                cached_pipe._current_timestep = None
                cached_pipe._evoke_progress_callback = None
            gc.collect()
            torch.cuda.empty_cache()
            response = {
                "requestId": request_id,
                "returnCode": rc,
                "error": error_text,
                "elapsed": time.time() - started,
            }
            write_json(responses / f"{request_id}.json", response)
            try:
                claimed.unlink()
            except OSError:
                pass
            set_state("ready", "Post-distill 管线已加载并常驻 GPU")


if __name__ == "__main__":
    # --argv_jsonl is dispatched here rather than through argparse: its payload is a list of argv
    # lists for main(), so it cannot share a command line with the per-case flags.
    if len(sys.argv) == 3 and sys.argv[1] == "--argv_jsonl":
        sys.exit(run_argv_batch(sys.argv[2]))
    if len(sys.argv) == 4 and sys.argv[1] == "--argv_server":
        sys.exit(run_argv_server(sys.argv[2], sys.argv[3]))
    main()
