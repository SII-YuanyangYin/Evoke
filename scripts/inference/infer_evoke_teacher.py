#!/usr/bin/env python3
"""Minimal sampling example for the EvokeTeacher dual-expert model (models/evoke/evoke_teacher).

The teacher normally only ever runs as a frozen *scorer* inside stage-3 DMD: the training engine calls
`EvokeTeacherScoreWrapper.forward` once per step and reads the v-prediction. That single-step forward is
also all a sampler needs, so this script wraps it in a flow-match Euler loop and decodes the result --
no extra model code, and nothing from the teacher's own training framework.

    text prompt ─► UMT5 ─┐
    reference image ─► VAE ─► build_i2v_y ─► set_condition
                          └─► for sigma in schedule:
                                  v = wrapper(x_t, sigma*1000, prompt_emb)      # routes high/low expert
                                  x = x + v * (sigma_next - sigma)
                              └─► VAE decode ─► mp4

Scope, and why it is called an example:

  * **nocam, non-SP, single process.** `wrapper._forward_core` is a port of exactly one path of the
    teacher's `model_fn_wan_video` -- the nocam / non-SP one. Camera conditioning and sequence-parallel
    inference are not reachable from here.
  * **Not numerically validated against the teacher's own sampler.** The conventions below (sigma
    schedule, shift, v-prediction sign, expert boundary) are taken from the training path, where they
    were verified for a *single* scoring step. A sampling loop compounds any mismatch over N steps, so
    before trusting the output for anything but a smoke check, A/B one prompt+seed against the teacher's
    own inference and compare frames, not vibes.
  * **Memory.** Both experts are 14B; at bf16 that is ~56 GB of weights before activations. Use
    `--single_expert high|low` (routes every step to one expert -- wrong for half the schedule, fine to
    check the plumbing) or `--offload` to keep only the routed expert resident.

Sparse-attention compatibility:

  * The currently shipped EvokeTeacher weights were trained with `chunk_size=9` and
    `num_select_frames=1` (cs9/select1). The cs8/select4 values in `loader.py` are fallback values for
    an older configuration.
  * These settings do not change parameter shapes, so loading a checkpoint cannot detect a mismatch.
    Keep the effective-config log emitted after model construction in inference logs and check it when
    using weights trained with a different sparse-attention configuration.

Example:

    bash scripts/inference/infer_evoke_teacher.sh

    python scripts/inference/infer_evoke_teacher.py \\
        --prompt "a drone shot flying over a snowy mountain village at sunrise" \\
        --image_path examples/i2v/image.jpg \\
        --num_frames 33 --num_inference_steps 50 --offload \\
        --output output/evoke_teacher/i2v.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from diffusers import AutoencoderKLWan                                    # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel                  # noqa: E402

from evoke.modules.evoke_teacher.wrapper import EvokeTeacherScoreWrapper, build_i2v_y  # noqa: E402
from evoke.utils.utils_base import encode_prompt                          # noqa: E402

# The negative prompt every EVOKE path uses; see evoke/modules/geometric_state/defaults.py.
NEGATIVE_PROMPT = (
    "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, "
    "color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal "
    "inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, three legs, many people in the background, walking backwards, messy background"
)

SHIPPED_CHUNK_SIZE = 9
SHIPPED_NUM_SELECT_FRAMES = 1


def wan_sigmas(num_inference_steps: int, shift: float) -> torch.Tensor:
    """The teacher's flow-match schedule: linspace(1,0) over N+1 points, drop the tail, then shift.

    Mirrors the `Wan` template of the teacher's flow-match scheduler (sigma_min=0, sigma_max=1,
    shift=5). `timestep` handed to the model is sigma*1000.
    """
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def load_text_stack(base: str, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype)
    return tokenizer, text_encoder.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_reference(vae, image_path, height, width, num_frames, device):
    """[cond frame | zero frames] -> VAE -> normalised latent, the layout build_i2v_y expects.

    Returns (cond_latent_norm [1,16,T_lat,h,w], latents_mean, latents_std) so the caller can reuse the
    same constants when decoding.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device)
    latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device)

    video = torch.zeros(1, 3, num_frames, height, width, device=device, dtype=torch.float32)
    if image_path:
        img = Image.open(image_path).convert("RGB").resize((width, height), Image.BICUBIC)
        frame = TF.to_tensor(img).to(device) * 2.0 - 1.0                  # [3,H,W] in [-1,1]
        video[:, :, 0] = frame                                            # frame 0 conditions, rest stay zero
    cond_latent = vae.encode(video).latent_dist.mode()                    # [1,16,T_lat,h,w]
    # the training side normalises with (z - mean) / std before build_i2v_y; keep that identical
    cond_latent_norm = (cond_latent - latents_mean) / latents_std
    return cond_latent_norm, latents_mean, latents_std


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_dir", default="models/evoke/evoke_teacher",
                   help="parent of high_noise/ and low_noise/")
    p.add_argument("--base", default="models/evoke-base", help="supplies vae / tokenizer / text_encoder")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--image_path", default=None, help="i2v reference frame; omit for t2v")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num_frames", type=int, default=33, help="pixel frames; latent T = (n-1)//4 + 1")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--shift", type=float, default=5.0, help="flow-match shift (Wan default)")
    p.add_argument("--guidance_scale", type=float, default=5.0, help="1.0 disables CFG")
    p.add_argument("--boundary", type=float, default=0.9,
                   help="expert switch: t >= boundary*1000 -> high-noise expert")
    p.add_argument("--chunk_size", type=int, default=SHIPPED_CHUNK_SIZE,
                   help="sparse-attention chunk size (shipped weights: 9; loader fallback 8 is legacy)")
    p.add_argument("--num_select_frames", type=int, default=SHIPPED_NUM_SELECT_FRAMES,
                   help="remote frames selected by sparse attention (shipped weights: 1; loader fallback 4 is legacy)")
    p.add_argument("--single_expert", choices=["high", "low"], default=None,
                   help="load one expert only; plumbing check, not a valid sample")
    p.add_argument("--offload", action="store_true", help="keep only the routed expert on GPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--output", default="output/evoke_teacher/i2v.mp4")
    return p.parse_args(argv)


def build_teacher_wrapper(args, device, dtype):
    """Build the wrapper and report sparse settings from the constructed model itself."""
    wrapper = EvokeTeacherScoreWrapper(
        high_dir=os.path.join(args.teacher_dir, "high_noise"),
        low_dir=os.path.join(args.teacher_dir, "low_noise"),
        boundary=args.boundary,
        model_cfg_overrides={
            "chunk_size": args.chunk_size,
            "num_select_frames": args.num_select_frames,
        },
        torch_dtype=dtype,
        single_expert=args.single_expert,
    ).to(device).eval()

    # Read the values from a live block rather than echoing the CLI. Since chunk_size and
    # num_select_frames do not affect parameter shapes, successful weight loading is not evidence
    # that these construction-time settings match the checkpoint's training configuration.
    expert = wrapper.dit_low if wrapper.dit_low is not None else wrapper.dit_high
    block = expert.blocks[0]
    print(
        "[teacher] sparse EFFECTIVE: "
        f"chunk_size={block.chunk_size} "
        f"num_select_frames={block.num_select_frames} "
        f"num_nearby_frames={block.num_nearby_frames} "
        f"overlap_size={block.overlap_size} "
        f"per_frame_tokens={block.per_frame_tokens} "
        f"select_scales={list(block.select_scales)}",
        flush=True,
    )
    return wrapper


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    if device.type != "cuda":
        print("[teacher] WARNING: no CUDA device -- a 14B forward on CPU is not going to finish.", flush=True)

    print(f"[teacher] loading vae / text encoder from {args.base}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    tokenizer, text_encoder = load_text_stack(args.base, device, dtype)

    prompt_embeds, _ = encode_prompt(tokenizer, text_encoder, args.prompt, device=device, dtype=dtype)
    do_cfg = args.guidance_scale != 1.0
    negative_embeds = None
    if do_cfg:
        negative_embeds, _ = encode_prompt(tokenizer, text_encoder, args.negative_prompt,
                                           device=device, dtype=dtype)
    del text_encoder
    torch.cuda.empty_cache() if device.type == "cuda" else None

    cond_latent_norm, latents_mean, latents_std = encode_reference(
        vae, args.image_path, args.height, args.width, args.num_frames, device)
    num_cond_px = 1 if args.image_path else 0
    y = build_i2v_y(cond_latent_norm.to(dtype), num_cond_px_frames=num_cond_px)

    print(f"[teacher] building experts from {args.teacher_dir}", flush=True)
    wrapper = build_teacher_wrapper(args, device, dtype)
    wrapper._per_expert_offload = bool(args.offload and args.single_expert is None)
    wrapper.set_condition(y)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(cond_latent_norm.shape, generator=generator, device=device, dtype=torch.float32)

    sigmas = wan_sigmas(args.num_inference_steps, args.shift).to(device)
    print(f"[teacher] {args.num_inference_steps} steps, shift={args.shift}, "
          f"boundary={args.boundary} (t>={args.boundary * 1000:.0f} -> high), cfg={args.guidance_scale}",
          flush=True)

    for i, sigma in enumerate(sigmas):
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.zeros_like(sigma)
        timestep = (sigma * 1000.0).reshape(1)
        x = latents.to(dtype)

        v = wrapper(hidden_states=x, timestep=timestep, encoder_hidden_states=prompt_embeds)[0].float()
        if do_cfg:
            v_uncond = wrapper(hidden_states=x, timestep=timestep,
                               encoder_hidden_states=negative_embeds)[0].float()
            v = v_uncond + args.guidance_scale * (v - v_uncond)

        # x_t = (1-sigma)x0 + sigma*eps and the model predicts v = eps - x0, so dx/dsigma = v.
        latents = latents + v * (sigma_next - sigma)
        if i % 10 == 0 or i == len(sigmas) - 1:
            expert = "high" if float(timestep) >= wrapper.boundary_t else "low"
            print(f"[teacher] step {i + 1}/{len(sigmas)} sigma={float(sigma):.4f} expert={expert}", flush=True)

    print("[teacher] decoding", flush=True)
    frames = vae.decode(latents * latents_std + latents_mean).sample     # undo the normalisation
    frames = ((frames.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    frames = frames[0].permute(1, 2, 3, 0).cpu().numpy()                  # [T,H,W,3]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(args.output, list(frames), fps=args.fps, quality=8)
    except ImportError:
        import cv2
        vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (frames.shape[2], frames.shape[1]))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
    print(f"[teacher] wrote {args.output}  ({frames.shape[0]} frames)", flush=True)


if __name__ == "__main__":
    main()
