"""Runtime hook connecting the UI LightTAE decoder to EVOKE chunk decoding."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch


def install() -> None:
    from evoke.pipelines.pipeline_evoke import EvokePipeline

    if getattr(EvokePipeline, "_evoke_ui_lighttae_installed", False):
        return

    def ensure_decoder(pipe, device: torch.device):
        decoder = getattr(pipe, "_evoke_ui_lighttae_decoder", None)
        if decoder is None:
            from lighttae_decoder import LightTAEWan21Decoder

            weights = Path(os.environ["EVOKE_UI_LIGHTTAE_WEIGHTS"])
            decoder = LightTAEWan21Decoder(weights).to(device=device, dtype=torch.bfloat16).eval()
            pipe._evoke_ui_lighttae_decoder = decoder
            print(f"[ui-lighttae] loaded Wan2.1 decoder: {weights}", flush=True)
        return decoder

    original_to = EvokePipeline.to

    def move_to(self, *args, **kwargs):
        result = original_to(self, *args, **kwargs)
        device = self._execution_device
        if device.type == "cuda" and not getattr(self, "_evoke_ui_lighttae_warmed", False):
            decoder = ensure_decoder(self, device)
            started = time.perf_counter()
            dummy = torch.zeros((1, 10, 16, 48, 80), device=device, dtype=torch.bfloat16)
            decoder(dummy)
            torch.cuda.synchronize(device)
            del dummy
            torch.cuda.empty_cache()
            self._evoke_ui_lighttae_warmed = True
            print(f"[ui-lighttae] CUDA warmup complete in {time.perf_counter() - started:.3f}s", flush=True)
        return result

    def decode(
        self,
        z_chunk: torch.Tensor,
        is_first_chunk: bool,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        vae_dtype: torch.dtype,
        warm_latents: torch.Tensor | None = None,
        warm_repeat: int = 0,
        warm_as_prior: bool = False,
    ) -> torch.Tensor:
        del vae_dtype, warm_repeat
        started = time.perf_counter()
        decoder = ensure_decoder(self, z_chunk.device)

        # DiT latents are normalized. LightTAE was distilled in the raw Wan 2.1
        # VAE space, matching the official decoder's unscale operation.
        raw = z_chunk.to(dtype=torch.bfloat16) / latents_std.to(dtype=torch.bfloat16) + latents_mean.to(dtype=torch.bfloat16)
        previous = getattr(self, "_evoke_ui_lighttae_previous", None)
        if is_first_chunk:
            previous = warm_latents[:, :, -1:] if warm_as_prior and warm_latents is not None else None
        if previous is not None:
            previous_raw = previous.to(dtype=torch.bfloat16) / latents_std.to(dtype=torch.bfloat16) + latents_mean.to(dtype=torch.bfloat16)
            raw_input = torch.cat([previous_raw, raw], dim=2)
        else:
            raw_input = raw

        decoded = decoder(raw_input.transpose(1, 2)).transpose(1, 2).float() * 2 - 1
        # One prepended causal latent gives 37 pixels after LightTAE's trim. Drop
        # its first pixel so every UI chunk remains exactly 36 frames at 24 FPS.
        if previous is not None and decoded.shape[2] > 36:
            decoded = decoded[:, :, -36:]
        self._evoke_ui_lighttae_previous = z_chunk[:, :, -1:].detach()
        self._geo_persist_feat_map = None
        print(
            f"[ui-lighttae] decoded latent={tuple(z_chunk.shape)} -> pixels={tuple(decoded.shape)} "
            f"in {time.perf_counter() - started:.3f}s",
            flush=True,
        )
        return decoded.clamp_(-1, 1)

    EvokePipeline.to = move_to
    EvokePipeline._decode_chunk_persistent_cache = decode
    EvokePipeline._evoke_ui_lighttae_installed = True
    print("[ui-lighttae] UI fast decoder hook installed", flush=True)
