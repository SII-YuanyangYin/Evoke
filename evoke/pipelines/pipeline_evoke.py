# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
# Copyright 2026 The Evoke Team. All rights reserved. (modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import math
from enum import Enum
from itertools import accumulate
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import regex as re
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from ..modules.transformer_evoke import EvokeTransformer3DModel
from ..scheduler.scheduling_evoke import EvokeScheduler
from ..utils.utils_base import AdaptiveAntiDrifting, apply_schedule_shift
from ..utils.utils_evoke_post import add_noise, convert_flow_pred_to_x0
from .pipeline_output import EvokePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Process-local cloud-warp depth estimator cache for the inference/validation path, mirroring
# online_materialize._DEPTH_ESTIMATORS on the training side (same key, same reset-at-stream-boundary
# contract). Keyed so two different backends or recipes never silently share one instance; see
# _geo_da3_setup for why reuse is safe and what has to be reset.
_GEO_DEPTH_ESTIMATORS: dict = {}

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, EvokePipeline

        >>> model_id = "models/evoke-base"  # local weights dir, see README
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = EvokePipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background"

        >>> output = pipe(
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     height=384,
...     width=640,
...     num_frames=132,
...     guidance_scale=5.0,
... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=24)
        ```
"""


@torch.amp.autocast("cuda", dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class VAEDecodeType(str, Enum):
    DEFAULT = "default"
    DEFAULT_BATCH = "default_batch"
    LONG = "long"
    # Maintains vae._feat_map sliding window across chunks; memory-efficient vs "long".
    PERSISTENT = "persistent"


class EvokePipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video / image-to-video / video-to-video generation using Evoke.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`EvokeTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler | EvokeScheduler,
        transformer: EvokeTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # Duplicate embeddings for each generation per prompt.
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        negative_prompt_attention_mask = None
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_image_latents(
        self,
        image: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        fake_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        if latents is None:
            image = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            latents = (latents - latents_mean) * latents_std
        if fake_latents is None:
            fake_video = image.repeat(1, 1, 33, 1, 1).to(device=device, dtype=self.vae.dtype)
            fake_latents_full = self.vae.encode(fake_video).latent_dist.sample(generator=generator)
            fake_latents_full = (fake_latents_full - latents_mean) * latents_std
            fake_latents = fake_latents_full[:, :, -1:, :, :]
        return latents.to(device=device, dtype=dtype), fake_latents.to(device=device, dtype=dtype)

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        latent_window_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        video = video.to(device=device, dtype=self.vae.dtype)
        if latents is None:
            num_frames = video.shape[2]
            min_frames = (latent_window_size - 1) * 4 + 1
            num_chunks = num_frames // min_frames
            if num_chunks == 0:
                raise ValueError(
                    f"Video must have at least {min_frames} frames "
                    f"(got {num_frames} frames). "
                    f"Required: (latent_window_size - 1) * 4 + 1 = ({latent_window_size} - 1) * 4 + 1 = {min_frames}"
                )
            total_valid_frames = num_chunks * min_frames
            start_frame = num_frames - total_valid_frames

            first_frame = video[:, :, 0:1, :, :]
            first_frame_latent = self.vae.encode(first_frame).latent_dist.sample(generator=generator)
            first_frame_latent = (first_frame_latent - latents_mean) * latents_std

            latents_chunks = []
            for i in range(num_chunks - 1, -1, -1):
                chunk_start = start_frame + i * min_frames
                chunk_end = chunk_start + min_frames
                video_chunk = video[:, :, chunk_start:chunk_end, :, :]
                chunk_latents = self.vae.encode(video_chunk).latent_dist.sample(generator=generator)
                chunk_latents = (chunk_latents - latents_mean) * latents_std
                latents_chunks.insert(0, chunk_latents)
            latents = torch.cat(latents_chunks, dim=2)
        return first_frame_latent.to(device=device, dtype=dtype), latents.to(device=device, dtype=dtype)

    def _decode_chunk_persistent_cache(
        self,
        z_chunk: torch.Tensor,
        is_first_chunk: bool,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        vae_dtype: torch.dtype,
        warm_latents: Optional[torch.Tensor] = None,
        warm_repeat: int = 0,
        warm_as_prior: bool = False,
    ) -> torch.Tensor:
        """Per-chunk VAE decode that preserves vae._feat_map across chunks. Caller must clear_cache() after the loop.

        First-frame cold-start fix: on the first chunk the causal-VAE feat cache is empty, so the very first
        decoded pixel is systematically over-saturated/warm regardless of latent content (the I-frame slot).
        When ``warm_latents`` is given (e.g. the clean image/prefix latent), we prepend ``warm_repeat`` copies
        of it to warm the cache, decode the real chunk as temporal frames, then drop the warm-up pixels — so the
        chunk's first frame decodes with a warm cache. No-op when warm_latents is None / warm_repeat<=0.
        """
        import os as _os
        _sd = _os.environ.get("EVOKE_SAVE_DECODE_LATENTS")
        if _sd:
            _idx = getattr(self, "_decode_dump_idx", 0)
            torch.save(z_chunk.detach().float().cpu(), f"{_sd}/decode_chunk{_idx}_first{int(is_first_chunk)}.pt")
            print(f"[DECODE-DUMP] chunk{_idx} first={is_first_chunk} shape={tuple(z_chunk.shape)} "
                  f"warm={None if warm_latents is None else tuple(warm_latents.shape)}", flush=True)
            self._decode_dump_idx = _idx + 1
        if is_first_chunk:
            self.vae.clear_cache()
            self._geo_persist_feat_map = None
        elif getattr(self, "_geo_persist_feat_map", None) is not None:
            # The interleaved warp vae.encode() (built each chunk before this decode) calls clear_cache(),
            # which ALSO wipes the DECODER feat cache (_feat_map) — not just the encoder one. That breaks
            # cross-chunk persistent decode: every chunk then decodes cold -> a seam/flicker at every chunk
            # boundary. Restore the decoder cache we saved right after the previous chunk's decode.
            self.vae._feat_map = self._geo_persist_feat_map
            self.vae._conv_idx = [0]

        # warm_as_prior gates the two decode methods, split by whether a GENUINE prior frame exists:
        #   i2v / v2v -> True -> warm the cache with the real prior (i2v input-image latent / v2v ref-video tail), so
        #     chunk0's first latent decodes as a CONTINUATION: correct colour, no I-frame cast, no static blur.
        #   t2v       -> False -> warm_n=0 -> chunk0 decodes as a fresh clip (I-frame slot). t2v has no real prior
        #     frame, and self-warming with real0's own copy feeds the causal decoder an identical pair, which only
        #     blurs the first frames (~40% softer on f1-2). Clip-start is therefore the t2v method: sharp, at the cost
        #     of a subtle first-frame colour cast inherent to zero-shot t2v.
        warm_n = int(warm_repeat) if (is_first_chunk and warm_latents is not None and warm_as_prior) else 0
        # First-chunk cache warm-up. The first VAE decode after clear_cache() has an empty causal cache, so decoding
        # chunk0's first latent directly makes it the I-frame slot -> a colour-shifted first pixel frame (on i2v, a
        # cool/green cast on frame0 that converges by ~frame2). Prime the cache with the GENUINE prior frame(s) the
        # caller passes as warm_latents (v2v ref tail, or the i2v input-image latent) so chunk0's first latent decodes
        # as a *continuation*. Use the real lead-in AS-IS -- never static-repeat a single latent, since a constant
        # sequence is what the causal temporal conv low-passes into a hazy first frame. Take as many real lead-in
        # frames as are available, capped at warm_n.
        if warm_n > 0:
            warm_n = min(warm_n, int(warm_latents.shape[2]))
        if warm_n > 0:
            warm = warm_latents[:, :, -warm_n:].to(z_chunk.dtype).to(z_chunk.device)   # genuine lead-in, no static repeat
            z_chunk = torch.cat([warm, z_chunk], dim=2)

        z = z_chunk.to(vae_dtype) / latents_std + latents_mean
        x = self.vae.post_quant_conv(z)

        out_list = []
        for i in range(x.shape[2]):
            self.vae._conv_idx = [0]
            first_frame_flag = is_first_chunk and i == 0
            slice_out = self.vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self.vae._feat_map,
                feat_idx=self.vae._conv_idx,
                first_chunk=first_frame_flag,
            )
            out_list.append(slice_out)

        out = torch.cat(out_list, dim=2)

        # Save the decoder feat cache so the NEXT chunk can restore it: the warp vae.encode() between
        # chunks calls clear_cache() and would otherwise wipe this, forcing every chunk to decode cold.
        self._geo_persist_feat_map = self.vae._feat_map

        if getattr(self.vae.config, "patch_size", None) is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            out = unpatchify(out, patch_size=self.vae.config.patch_size)

        if warm_n > 0:
            drop = 1 + 4 * (warm_n - 1)                 # pixel frames produced by the warm-up lead-in (i0=1px, rest=4px)
            out = out[:, :, drop:]                      # keep every real-latent frame (each latent -> 4 px, continuous ckpt)

        return torch.clamp(out, min=-1.0, max=1.0)

    def interpolate_prompt_embeds(
        self,
        prompt_embeds_1: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        interpolation_steps: int = 4,
    ):
        x = torch.lerp(
            prompt_embeds_1,
            prompt_embeds_2,
            torch.linspace(0, 1, steps=interpolation_steps).unsqueeze(1).unsqueeze(2).to(prompt_embeds_1),
        )
        interpolated_prompt_embeds = list(x.chunk(interpolation_steps, dim=0))
        return interpolated_prompt_embeds

    def sample_block_noise(
        self,
        batch_size,
        channel,
        num_frames,
        height,
        width,
        patch_size: tuple[int, ...] = (1, 2, 2),
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ):
        # A fixed-seed generator is required for reproducibility; fallback creates a non-deterministic one.
        if generator is None:
            generator = torch.Generator(device=device)
        elif isinstance(generator, list):
            generator = generator[0]

        gamma = self.scheduler.config.gamma
        _, ph, pw = patch_size
        block_size = ph * pw

        cov = (
            torch.eye(block_size, device=device) * (1 + gamma)
            - torch.ones(block_size, block_size, device=device) * gamma
        )
        cov += torch.eye(block_size, device=device) * 1e-8
        cov = cov.float()  # Upcast to fp32; Cholesky is unreliable in fp16/bf16.

        L = torch.linalg.cholesky(cov)
        block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)
        z = torch.randn(block_number, block_size, generator=generator, device=generator.device).to(device=device)
        noise = z @ L.T

        noise = noise.view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
        noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)

        return noise

    # ---------------- geometric state helpers ----------------

    def _geo_init_state(self, source_image_pixel: torch.Tensor, bank_max_size: Optional[int] = None) -> dict:
        """Initialize per-call GEO state with a FrameBank (warp render goes through DA3, see _geo_da3_setup)."""
        from evoke.modules.geometric_state.frame_bank import FrameBank

        fb = FrameBank(max_size=bank_max_size)

        return {
            "source_image_pixel": source_image_pixel.detach().cpu(),
            "frame_bank": fb,
            "prev_chunk_last_frame": None,    # pixel [1,3,H,W] used as source for the next chunk
            "warp_history_prefix_latent": None,  # cached prefix latent for the first frame
        }

    # ---------------- DA3 FrameBank recall-based warp (inference/validation;) ----------------
    def _geo_da3_setup(self, geo_state, cfg, lingbot_Ks, height, width, pix_stride, device):
        """Init DA3FrameBank + estimator + render-resolution intrinsics (adaptively rescaled from lingbot_Ks)."""
        from evoke.modules.geometric_state.da3_cloud import DA3FrameBank
        from evoke.modules.geometric_state.depth_backend import build_estimator
        from evoke.utils.train_config import vigeo_opts_from_cfg
        import numpy as _np
        kn = lingbot_Ks.detach().cpu().numpy().reshape(-1)[:4]      # [fx,fy,cx,cy] @ pose source resolution
        fx, fy, cx, cy = float(kn[0]), float(kn[1]), float(kn[2]), float(kn[3])
        src_w = max(2.0 * cx, 1.0); src_h = max(2.0 * cy, 1.0)     # principal point ~= centre -> back out source resolution, rescale to render
        sx = float(width) / src_w; sy = float(height) / src_h
        K_pix = _np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], _np.float32)
        pr = int(cfg.get("da3_process_res", 644))
        # depth_backend picks the estimator ('da3' | 'vigeo'); everything below is shared.
        # The estimator is cached per process -- as the training side already does
        # (online_materialize._get_da3_estimator, same key) -- rather than rebuilt per generation: ViGeo's weights are
        # 4.8 GB and re-reading them per case costs 9.6 TB of reads over a 2000-case sweep, a large share of the
        # per-case wall time. `reset_stream` is what makes a shared instance correct -- it drops the kv-cache, the
        # locked depth scale, the anchor list and the window counter, i.e. everything that must not cross a video
        # boundary. The FrameBank below is still built fresh, so the point cloud does not cross cases either.
        from evoke.modules.geometric_state.depth_backend import reset_stream as _depth_reset_stream
        _backend = str(cfg.get("depth_backend", "da3"))
        _est_key = (_backend.lower(), pr, str(cfg.get("da3_weights") or ""),
                    str(cfg.get("da3_src") or ""),
                    tuple(sorted((k, str(v)) for k, v in vigeo_opts_from_cfg(cfg).items())))
        _est = _GEO_DEPTH_ESTIMATORS.get(_est_key)
        if _est is None:
            # One recipe per process: a second entry would keep a second copy of the weights resident
            # on the same device. A changed recipe means a new run, so dropping the old one is right.
            _GEO_DEPTH_ESTIMATORS.clear()
            _est = build_estimator(
                _backend, device, pr, weights=(cfg.get("da3_weights") or None),
                src=(cfg.get("da3_src") or None), vigeo_opts=vigeo_opts_from_cfg(cfg))
            _GEO_DEPTH_ESTIMATORS[_est_key] = _est
        else:
            _depth_reset_stream(_est)
        geo_state["da3_est"] = _est
        geo_state["da3_bank"] = DA3FrameBank(
            device=device, conf_percentile=float(cfg.get("conf_percentile", 30.0)),
            cloud_hygiene=bool(cfg.get("cloud_hygiene", False)),
            hygiene_sat_max=float(cfg.get("hygiene_sat_max", 1.0)),
            hygiene_flat_std=float(cfg.get("hygiene_flat_std", 0.0)),
            hygiene_flat_win=int(cfg.get("hygiene_flat_win", 7)),
            hygiene_conf_pct=float(cfg.get("hygiene_conf_pct", 0.0)),
            hygiene_conf_abs=float(cfg.get("hygiene_conf_abs", 0.0)),
            # [method 1] ingest consistency gate
            consist_gate=bool(cfg.get("consist_gate", False)),
            consist_tau=float(cfg.get("consist_tau", 0.15)),
            consist_ref_frames=int(cfg.get("consist_ref_frames", 24)),
            consist_min_ref=int(cfg.get("consist_min_ref", 2000)),
            # [variant 1] confidence-adaptive tolerance (removes the magic fixed tau)
            consist_adaptive=bool(cfg.get("consist_adaptive", False)),
            consist_tau_lo=float(cfg.get("consist_tau_lo", 0.20)),
            consist_tau_hi=float(cfg.get("consist_tau_hi", 0.30)),
            consist_conf_ref=float(cfg.get("consist_conf_ref", 12.0)),
            # hole probation (0=off=holes pass straight through, legacy; only takes effect with consist_gate on)
            consist_probation=int(cfg.get("consist_probation", 0)),
            consist_probation_frac=float(cfg.get("consist_probation_frac", 0.0)),   # >0 = only hold pixels inside large holes
            consist_probation_win=int(cfg.get("consist_probation_win", 25)),
            # per-window ingest scale alignment (off by default; cures the compounding per-window Umeyama scale drift, measured 1.05-1.18)
            consist_scale_align=bool(cfg.get("consist_scale_align", False)),
            # colour-statistics re-anchoring (off by default, independent of consist_gate; cures over-saturated/over-sharpened seeds)
            color_anchor=bool(cfg.get("color_anchor", False)),
            color_anchor_alpha=float(cfg.get("color_anchor_alpha", 0.5)),
            color_anchor_ref_windows=int(cfg.get("color_anchor_ref_windows", 4)),
            # [self re-anchor] on divergence -> hard re-anchor the cloud to its own best past contiguous window (legitimate oracle)
            self_reanchor=bool(cfg.get("self_reanchor", False)),
            self_reanchor_scale_thr=float(cfg.get("self_reanchor_scale_thr", 1.5)),
            self_reanchor_rej_thr=float(cfg.get("self_reanchor_rej_thr", 0.8)),
            self_reanchor_min_gap=int(cfg.get("self_reanchor_min_gap", 8)),
            self_reanchor_keep_windows=int(cfg.get("self_reanchor_keep_windows", 3)),
            self_reanchor_lookback=int(cfg.get("self_reanchor_lookback", 20)),
            self_reanchor_anchor_min_conf=float(cfg.get("self_reanchor_anchor_min_conf", 0.0)),  # [v3] anchor quality floor
            self_reanchor_pin_prime=bool(cfg.get("self_reanchor_pin_prime", False)),    # [v4] pin the prime anchor
            self_reanchor_pin_windows=int(cfg.get("self_reanchor_pin_windows", 4)),
            # [method 2] bounded / periodically re-anchored history
            hist_max_frames=int(cfg.get("hist_max_frames", 0)),
            reanchor_every=int(cfg.get("reanchor_every", 0)),
            reanchor_keep_frames=int(cfg.get("reanchor_keep_frames", 0)),
            # [method 3] free-space carving
            carve=bool(cfg.get("carve", False)),
            carve_margin=float(cfg.get("carve_margin", 0.10)),
            carve_ref_frames=int(cfg.get("carve_ref_frames", 24)),
            carve_min_views=int(cfg.get("carve_min_views", 1)),   # [variant 2] multi-view vote threshold
            # cross-window strike (1=off=delete within the window, legacy; needs N consecutive windows of confirmation before a real delete)
            carve_strike_windows=int(cfg.get("carve_strike_windows", 1)),
        )
        if bool(cfg.get("color_anchor", False)):   # independent of consist_gate
            print(f"[GEO-da3] [P1b] colour-statistics re-anchoring ON: channel mean/std accumulated over the first {cfg.get('color_anchor_ref_windows', 4)} "
                  f"windows (GT prime) is the reference; every later window is linearly matched back to it then blended with alpha={cfg.get('color_anchor_alpha', 0.5)} "
                  f"(affects only the DA3 input + cloud RGB, never the model output; past information only)", flush=True)
        if bool(cfg.get("cloud_hygiene", False)):
            print(f"[GEO-da3] cloud_hygiene ON: conf_abs={cfg.get('hygiene_conf_abs', 0.0)}(master gate,>0=on) "
                  f"sat_max={cfg.get('hygiene_sat_max', 1.0)}(>=1=off) flat_std={cfg.get('hygiene_flat_std', 0.0)}(0=off) "
                  f"conf_pct={cfg.get('hygiene_conf_pct', 0.0)}", flush=True)
        if bool(cfg.get("consist_gate", False)):
            if bool(cfg.get("consist_adaptive", False)):
                print(f"[GEO-da3] [method 1+variant 1] consistency gate ON (confidence-adaptive tolerance): "
                      f"tau_lo={cfg.get('consist_tau_lo', 0.20)}→tau_hi={cfg.get('consist_tau_hi', 0.30)} "
                      f"conf_ref={cfg.get('consist_conf_ref', 12.0)} (strict where conf is low / loose where it is high) "
                      f"ref_frames={cfg.get('consist_ref_frames', 24)} min_ref={cfg.get('consist_min_ref', 2000)}", flush=True)
            else:
                print(f"[GEO-da3] [method 1] ingest consistency gate ON: tau={cfg.get('consist_tau', 0.15)}(fixed) "
                      f"ref_frames={cfg.get('consist_ref_frames', 24)} min_ref={cfg.get('consist_min_ref', 2000)}", flush=True)
            if int(cfg.get("consist_probation", 0)) > 0:
                _pf = float(cfg.get("consist_probation_frac", 0.0))
                print(f"[GEO-da3] [P0] hole probation ON: pixels with no reference wait one round, geometry re-check next window (tau={cfg.get('consist_tau', 0.15)})"
                      f" then admitted/dropped; fully invisible = admitted (benefit of the doubt)"
                      + (f"; [v2 narrowed] only hold inside large holes (win={cfg.get('consist_probation_win', 25)} hole frac>{_pf})" if _pf > 0 else ""),
                      flush=True)
            if bool(cfg.get("consist_scale_align", False)):
                print(f"[GEO-da3] [P1a] per-window ingest scale alignment ON: correct the whole window depth by median(d/d_cloud) before gating "
                      f"(clip[0.75,1.33], overlap<2000px or absurd ratio -> skip)", flush=True)
            if bool(cfg.get("self_reanchor", False)):
                print(f"[GEO-da3] [self re-anchor] ON: divergence trigger (scale outside [1/{cfg.get('self_reanchor_scale_thr', 1.5)},"
                      f"{cfg.get('self_reanchor_scale_thr', 1.5)}] or rej>{cfg.get('self_reanchor_rej_thr', 0.8)}) -> "
                      f"re-anchor to the best past contiguous {cfg.get('self_reanchor_keep_windows', 3)} windows "
                      f"(min_gap={cfg.get('self_reanchor_min_gap', 8)} lookback={cfg.get('self_reanchor_lookback', 20)})", flush=True)
        if int(cfg.get("hist_max_frames", 0)) > 0 or int(cfg.get("reanchor_every", 0)) > 0:
            print(f"[GEO-da3] [method 2] bounded / re-anchored history ON: hist_max_frames={cfg.get('hist_max_frames', 0)}(0=off) "
                  f"reanchor_every={cfg.get('reanchor_every', 0)}(0=off) keep={cfg.get('reanchor_keep_frames', 0)}", flush=True)
        if bool(cfg.get("carve", False)):
            print(f"[GEO-da3] [method 3{'+variant 2' if int(cfg.get('carve_min_views', 1)) > 1 else ''}] free-space carving ON: "
                  f"margin={cfg.get('carve_margin', 0.10)} ref_frames={cfg.get('carve_ref_frames', 24)} "
                  f"min_views={cfg.get('carve_min_views', 1)}(1=delete on any single frame/OR; >=3=multi-view voting)", flush=True)
            if int(cfg.get("carve_strike_windows", 1)) > 1:
                print(f"[GEO-da3] carve cross-window strike ON: needs {cfg.get('carve_strike_windows', 1)} consecutive ingest windows"
                      f" of confirmation (votes>=min_views) before a real delete; any break resets the count (cures the systematic over-deletion from correlated same-window votes)", flush=True)
        geo_state["da3_K_pix"] = K_pix
        geo_state["da3_pix_stride"] = int(pix_stride)
        geo_state["da3_ingest_n"] = int(cfg.get("cloud_update_n", cfg.get("update_frames_per_chunk", 12)) or 12)
        geo_state["da3_recall_k"] = int(cfg.get("recall_k", 12)); geo_state["da3_n_nearby"] = int(cfg.get("n_nearby", 4))
        geo_state["da3_n_tframe"] = int(cfg.get("n_tframe", 6)); geo_state["da3_grid_div"] = int(cfg.get("recall_grid_div", 8))
        geo_state["da3_mask_pts"] = int(cfg.get("recall_mask_pts", 8000))
        geo_state["da3_lag"] = int(cfg.get("warp_lag_chunks", 1))
        geo_state["da3_splat"] = int(cfg.get("cloud_splat_radius", 2))
        # multi-source priority fusion params (aligned with training build_multisrc_warp;)
        geo_state["da3_nsrc"] = int(cfg.get("nsrc", 8))
        geo_state["da3_nearby"] = int(cfg.get("nearby_window", 16))
        geo_state["da3_ms_splat"] = int(cfg.get("multisrc_splat", 1))
        geo_state["da3_dens_thresh"] = float(cfg.get("dens_thresh", 0.45))
        geo_state["da3_dens_win"] = int(cfg.get("dens_win", 7))
        geo_state["da3_recall_min_cov"] = float(cfg.get("recall_min_cov", 0.5))
        geo_state["da3_recall_margin"] = float(cfg.get("recall_margin", 0.15))
        # render_mode switch (multisrc default / backward=single main source backward warp / backward_zbuf=per-pixel multi-source z-buffer)
        geo_state["da3_render_mode"] = str(cfg.get("render_mode", "multisrc"))
        geo_state["da3_bw_fill_iters"] = int(cfg.get("bw_fill_iters", 12))
        # backward_zbuf blended despeckle (off by default; same renderer and same params as training build_backward_warp)
        geo_state["da3_zbuf_despeckle"] = bool(cfg.get("zbuf_despeckle", False))
        geo_state["da3_zbuf_despeckle_ksize"] = int(cfg.get("zbuf_despeckle_ksize", 3))
        geo_state["da3_zbuf_despeckle_fill_iters"] = int(cfg.get("zbuf_despeckle_fill_iters", 4))
        print(f"[GEO-da3] setup: depth_backend={_backend} process_res={pr} "
              f"K_render~[fx{fx*sx:.1f} fy{fy*sy:.1f} cx{cx*sx:.1f} cy{cy*sy:.1f}] "
              f"(src~{int(src_w)}x{int(src_h)}->{width}x{height}) ingest_n={geo_state['da3_ingest_n']} "
              f"recall_k={geo_state['da3_recall_k']} lag={geo_state['da3_lag']}", flush=True)

    def _geo_da3_ingest(self, geo_state, frames_local, global_pix_start, c2w_full, device):
        """Ingest a contiguous pixel-frame span [3,Nloc,H,W] in[-1,1] (ref seed or an already generated chunk) by sampling ingest_n frames.
        gid = global pixel index (used to cut the recall pool by chunk lag); degenerate motion is skipped by the try/except inside bank.ingest."""
        bank = geo_state.get("da3_bank")
        if bank is None:
            return
        Nloc = int(frames_local.shape[1]); n = int(geo_state["da3_ingest_n"])
        if Nloc < 3:
            return
        loc = torch.unique(torch.linspace(0, Nloc - 1, n).round().long().clamp(0, Nloc - 1))
        if loc.numel() < 3:
            return
        gids = [int(global_pix_start) + int(i) for i in loc.tolist()]
        pose_idx = torch.tensor([min(g, c2w_full.shape[1] - 1) for g in gids], device=c2w_full.device)
        c2w = c2w_full[0, pose_idx].float().cpu().numpy()
        frames01 = (frames_local[:, loc].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()  # [N,H,W,3]
        bank.ingest(geo_state["da3_est"], frames01, c2w, geo_state["da3_K_pix"], gids)

    def _geo_render_chunk_da3(self, geo_state, chunk_idx, camera_poses_pix_window, height, width,
                              device, anchor_c2w=None, source_frame_pix=None, **kwargs):
        """DA3 recall-based warp render (replaces Pi3X). pool = bank frames with chunk <= k-1-lag -> recall -> render to the target GT poses."""
        # multi-source priority fusion (the same _render_multisrc as training build_multisrc_warp, keeping
        # train/infer consistent). Replaces the old recall-cloud render_recalled.
        from evoke.modules.geometric_state.da3_cloud import _render_multisrc
        cfg = getattr(self, "_geo_vsnoise_cfg", None) or {}
        bank = geo_state["da3_bank"]; K_pix = geo_state["da3_K_pix"]
        stride = int(geo_state["da3_pix_stride"]); lag = int(geo_state["da3_lag"])
        tgt = camera_poses_pix_window
        if tgt.ndim == 4:
            tgt = tgt[0]
        tgt = tgt.to(device=device, dtype=torch.float32)            # [F,4,4]
        # gid = absolute pixel index (seed/generated frames); lag: exclude the most recent lag chunks in the bank (based on max gid).
        _keys = list(bank.frames.keys())
        if _keys:
            _cutoff = max(_keys) - lag * stride
            pool = [g for g in _keys if g <= _cutoff]
        else:
            pool = []
        store = {g: bank.frames[g] for g in pool}
        render_mode = str(geo_state.get("da3_render_mode", "multisrc"))
        # i2v chunk-0 reference-image warp. Chunk 0's pool is empty (no generated history yet), so the normal recall
        # render below returns an all-black warp and the model gets no camera signal -- a near-static first chunk,
        # since warp is this model's only camera channel. Training gives chunk 0 a single-source warp from its own
        # first frame; reproduce that here from the reference image. i2v only (v2v seeds its bank from the ref video),
        # only when the pool really is empty, gated by the default-on _geo_chunk0_ref_warp. Any failure inside falls
        # back to a blank warp, i.e. the previous behaviour.
        _chunk0_ref = (
            int(chunk_idx) == 0 and not pool
            and source_frame_pix is not None
            and bool(geo_state.get("da3_is_i2v", False))
            and bool(getattr(self, "_geo_chunk0_ref_warp", True))
        )
        if _chunk0_ref:
            from evoke.modules.geometric_state.da3_cloud import build_single_source_warp_mono
            _ref = source_frame_pix[0] if source_frame_pix.ndim == 4 else source_frame_pix   # [3,H,W]
            _ref_pose = tgt[0]                                # chunk-0 first target pose = the reference view
            from evoke.modules.geometric_state.da3_cloud import CHUNK0_TARGET_DISPARITY_PX_DEFAULT
            warp_video, visibility_mask = build_single_source_warp_mono(
                geo_state["da3_est"], _ref.to(device), _ref_pose, K_pix, tgt,
                height=int(height), width=int(width),
                splat_radius=int(geo_state.get("da3_ms_splat", 1)), device=device,
                chunk0_target_disparity_px=float(getattr(
                    self, "_geo_chunk0_target_disparity_px", CHUNK0_TARGET_DISPARITY_PX_DEFAULT)))
            render_mode = "i2v_chunk0_ref"
        elif render_mode in ("backward", "backward_zbuf"):
            # render_mode picks the backward render core: backward (default, single main source) / backward_zbuf (per-pixel multi-source z-buffer).
            from evoke.modules.geometric_state.da3_cloud import _render_backward, _render_backward_multisrc_zbuf
            _render_bw = _render_backward_multisrc_zbuf if render_mode == "backward_zbuf" else _render_backward
            # the zbuf despeckle params are only accepted by the backward_zbuf path (same function and params as training build_backward_warp) -> only added to kwargs there.
            _zbuf_extra = dict(
                zbuf_despeckle=bool(geo_state.get("da3_zbuf_despeckle", False)),
                zbuf_despeckle_ksize=int(geo_state.get("da3_zbuf_despeckle_ksize", 3)),
                zbuf_despeckle_fill_iters=int(geo_state.get("da3_zbuf_despeckle_fill_iters", 4)),
            ) if render_mode == "backward_zbuf" else {}
            warp_video, visibility_mask = _render_bw(
                store, pool, tgt, K_pix, int(height), int(width),
                nearby=int(geo_state.get("da3_nearby", 16)),
                fill_iters=int(geo_state.get("da3_bw_fill_iters", 12)),
                recall_min_cov=float(geo_state.get("da3_recall_min_cov", 0.5)),
                recall_margin=float(geo_state.get("da3_recall_margin", 0.15)), device=device, **_zbuf_extra)
        else:
            warp_video, visibility_mask = _render_multisrc(
                store, pool, tgt, K_pix, int(height), int(width),
                nsrc=int(geo_state.get("da3_nsrc", 8)), nearby=int(geo_state.get("da3_nearby", 16)),
                splat_radius=int(geo_state.get("da3_ms_splat", 1)),
                dens_thresh=float(geo_state.get("da3_dens_thresh", 0.45)), dens_win=int(geo_state.get("da3_dens_win", 7)),
                recall_min_cov=float(geo_state.get("da3_recall_min_cov", 0.5)),
                recall_margin=float(geo_state.get("da3_recall_margin", 0.15)), device=device)
        warp_video = warp_video.to(device=device, dtype=torch.float32)
        visibility_mask = visibility_mask.to(device=device, dtype=torch.float32)
        # warp_keep_clean_anchor: overwrite warp[0] with the previous chunk's clean last frame (same as the Pi3X/persistent branch)
        if bool(cfg.get("warp_keep_clean_anchor", False)):
            _anc = geo_state.get("prev_chunk_last_frame", None)
            if _anc is None:
                _anc = source_frame_pix
            if _anc is not None:
                _anc = _anc[0:1] if _anc.ndim == 4 else _anc.unsqueeze(0)
                if tuple(_anc.shape[-2:]) != tuple(warp_video.shape[-2:]):
                    _anc = torch.nn.functional.interpolate(_anc, size=tuple(warp_video.shape[-2:]),
                                                           mode="bilinear", align_corners=False)
                warp_video[:, :, 0] = _anc.to(device=device, dtype=warp_video.dtype).clamp(-1, 1)
                visibility_mask[:, :, 0] = 1.0
        # Debug dump (same as the Pi3X branch): writes chunk_NNN_warp.mp4 + chunk_NNN_vis.mp4 -> fed to combine_gt_pred_with_cam_viz for the 4-column layout.
        _dump_dir = getattr(self, "_geo_dump_dir", None)
        if _dump_dir is not None:
            try:
                import os, cv2, numpy as np
                os.makedirs(_dump_dir, exist_ok=True)
                _warp = warp_video[0].permute(1, 2, 3, 0).clamp(-1, 1).cpu().numpy()
                _warp_u8 = ((_warp * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                _h, _w = _warp_u8.shape[1], _warp_u8.shape[2]
                _wp = os.path.join(_dump_dir, f"chunk_{int(chunk_idx):03d}_warp.mp4")
                _ww = cv2.VideoWriter(_wp, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (int(_w), int(_h)))
                for f_rgb in _warp_u8:
                    _ww.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
                _ww.release()
                # show the blocky patch mask (= the one the latent conditioning actually uses, VAE8x + patch2 -> 16px blocks), not the speckly per-pixel keep
                _vt = visibility_mask[0, 0].clamp(0, 1)                       # [T,H,W]
                _gh = max(1, (int(_h) // 8) // 2); _gw = max(1, (int(_w) // 8) // 2)
                _vb = torch.nn.functional.adaptive_avg_pool2d(_vt.unsqueeze(1), (_gh, _gw))
                _vb = (_vb > 0.5).float()
                _vb = torch.nn.functional.interpolate(_vb, size=(int(_h), int(_w)), mode="nearest")[:, 0]
                _vis_u8 = (_vb.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                _vp = os.path.join(_dump_dir, f"chunk_{int(chunk_idx):03d}_vis.mp4")
                _vw = cv2.VideoWriter(_vp, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (int(_w), int(_h)), isColor=False)
                for f_gray in _vis_u8:
                    _vw.write(f_gray)
                _vw.release()
                if not getattr(self, "_geo_quiet_chunk_logs", False):
                    print(f"[GEO-da3-dump] chunk {int(chunk_idx):03d}: warp + vis → {_dump_dir}", flush=True)
            except Exception as _e:
                print(f"[GEO-da3-dump] WARN dump failed ({type(_e).__name__}: {_e})", flush=True)
        # Published for the per-chunk progress line; the raw print stays for training validation and
        #   for EVOKE_INFER_DEBUG=1, where there is no bar to interleave with.
        self._geo_last_render_stats = {"pool": len(pool), "render_mode": render_mode,
                                       "cov": float(visibility_mask[0, 0].mean())}
        if not getattr(self, "_geo_quiet_chunk_logs", False):
            print(f"[GEO-da3] chunk {int(chunk_idx):03d}: pool={len(pool)} ({render_mode}) "
                  f"cov={float(visibility_mask[0,0].mean()):.3f} warp={tuple(warp_video.shape)}", flush=True)
        return warp_video, visibility_mask

    def _geo_render_chunk(
        self,
        geo_state: dict,
        chunk_idx: int,
        source_frame_pix: torch.Tensor,
        camera_poses_pix_window: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        anchor_c2w: Optional[torch.Tensor] = None,  # anchor pose for this chunk
        top_k: int = 8,                             # retrieve up to K frames from FrameBank
        nearby_k: int = 0,                          # take N most-recent frames by pixel_idx
        select_k: Optional[int] = None,              # score-based selection count; None=top_k
        metric: str = "v1",                          # FrameBank score: v1/v2/v3
        metric_kwargs: Optional[dict] = None,        # per-metric kwargs
    ):
        """Render warp_video and visibility_mask for the current chunk (DA3 backend)."""
        # Prepend anchor pose so the renderer emits one extra frame; the first output frame is dropped.
        assert anchor_c2w is not None, (
            "plan B requires anchor_c2w to be passed explicitly (instead of falling back to camera_poses[0]). "
            "chunk 0 anchor = source pose; chunk>=1 anchor = previous chunk's last-frame pose."
        )
        # DA3 FrameBank recall-based warp (recon_backend=da3): replaces Pi3X, feeds validation/inference the same warp distribution as training.
        if str((getattr(self, "_geo_vsnoise_cfg", None) or {}).get("recon_backend", "da3")) == "da3" \
                and geo_state.get("da3_bank") is not None:
            return self._geo_render_chunk_da3(
                geo_state=geo_state, chunk_idx=chunk_idx,
                camera_poses_pix_window=camera_poses_pix_window,
                height=height, width=width, device=device,
                anchor_c2w=anchor_c2w, source_frame_pix=source_frame_pix)
        # the Pi3X path (legacy keyframe recall + persistent cloud) has been removed: DA3 is the only warp backend.
        raise RuntimeError(
            "[GEO] warp render requires the DA3 backend: recon_backend=da3 and an initialized da3_bank "
            "(see _geo_da3_setup). The Pi3X reconstruction path has been deleted."
        )

    def _geo_encode_warp_to_latents(
        self,
        warp_video: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        latent_window_size: int,
        device: torch.device,
        generator,
        sigma_min: float = 0.111,
        sigma_max: float = 0.135,
        visibility_aware_noise: bool = False,
        sigma_invisible: float = 0.8,
        visibility_mask_pix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """VAE-encode warp_video to latents and add sigma noise (uniform or visibility-aware)."""
        _warm_enc = bool((getattr(self, "_geo_vsnoise_cfg", None) or {}).get("geo_warp_warm_encode", False))
        if _warm_enc:
            # Warm-pad encode: prepend vae_t copies of the warp's first frame so its FIRST latent is
            # encoded as a CONTINUATION frame (not the I-frame / VAE-first-frame distribution), then drop
            # that warm I-frame latent. The warp's first latent thus carries the continuation distribution;
            # since the (minimal attn1 LoRA) DiT copies the warp, pred[0] also becomes continuation-
            # distributed, which lets one-shot/persistent decode run without the per-chunk I-frame flicker.
            _vt = int(self.vae_scale_factor_temporal)
            _min_frames = (int(latent_window_size) - 1) * _vt + 1
            _vid = warp_video[:, :, -_min_frames:].to(device=device, dtype=self.vae.dtype)
            _warm = _vid[:, :, :1].repeat(1, 1, _vt, 1, 1)
            _padded = torch.cat([_warm, _vid], dim=2)   # (_min_frames + _vt) frames -> (latent_window+1) latents
            self.vae.clear_cache()
            _lat = self.vae.encode(_padded).latent_dist.sample(generator=generator)
            self.vae.clear_cache()
            _lat = (_lat - latents_mean) * latents_std
            warp_latents = _lat[:, :, 1:, :, :].to(device=device, dtype=torch.float32)  # drop warm I-frame latent
            assert int(warp_latents.shape[2]) == int(latent_window_size), (
                f"warm-encode warp latents {warp_latents.shape[2]} != window {latent_window_size}"
            )
        else:
            _, warp_latents = self.prepare_video_latents(
                warp_video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                latent_window_size=latent_window_size,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        chunk_frames = int(warp_latents.shape[2])
        rand_generator = generator[0] if isinstance(generator, list) else generator
        frame_sigmas = (
            torch.rand(chunk_frames, device=device, generator=rand_generator) * (sigma_max - sigma_min) + sigma_min
        ).to(dtype=warp_latents.dtype)

        if visibility_aware_noise and visibility_mask_pix is not None:
            # Spatial-sigma path: pool pixel mask to latent domain and broadcast per-pixel sigma.
            assert 0.0 < float(sigma_invisible) <= 1.0, (
                f"sigma_invisible must be in (0, 1.0], got {sigma_invisible}"
            )
            H_lat = int(warp_latents.shape[3])
            W_lat = int(warp_latents.shape[4])
            # Downsample pixel-domain mask to latent domain — BLOCKY (coverage-style patch mask),
            # reusing the same function as training so train/val/infer match, no more per-pixel speckle.
            from evoke.dataset.online_materialize import _geo_resize_visibility_to_latent as _blocky_vis
            mask_lat = _blocky_vis(
                visibility_mask_pix.to(device=device, dtype=torch.float32),
                chunk_frames, H_lat, W_lat, vae_t_stride=int(self.vae_scale_factor_temporal),
            ).to(dtype=warp_latents.dtype)
            sigma_visible_5d = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
            sigmas = mask_lat * sigma_visible_5d + (1.0 - mask_lat) * float(sigma_invisible)
            noise = randn_tensor(warp_latents.shape, generator=generator, device=device, dtype=warp_latents.dtype)
            return sigmas * noise + (1.0 - sigmas) * warp_latents
        else:
            # Frame-uniform sigma (default path).
            frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
            return (
                frame_sigmas
                * randn_tensor(warp_latents.shape, generator=generator, device=device, dtype=warp_latents.dtype)
                + (1 - frame_sigmas) * warp_latents
            )

    def _geo_maybe_noise_invisible_history(self, latents_history_long, latents_history_mid, video_latents):
        """v4: replace all-zero (padding) long/mid history frames with sigma_invisible noise.

        Only active for i2v/t2v (video_latents is None) when invisible_history_noise is set; mirrors training.
        """
        _cfg = getattr(self, "_geo_vsnoise_cfg", None) or {}
        if (not bool(_cfg.get("invisible_history_noise", False))) or video_latents is not None:
            return latents_history_long, latents_history_mid
        _sig = float(_cfg.get("sigma_invisible", 0.8))
        if latents_history_long is not None:
            _z = (latents_history_long == 0).all(dim=(0, 1, 3, 4), keepdim=True)
            if bool(_z.any()):
                latents_history_long = torch.where(_z, _sig * torch.randn_like(latents_history_long), latents_history_long)
        if latents_history_mid is not None:
            _z = (latents_history_mid == 0).all(dim=(0, 1, 3, 4), keepdim=True)
            if bool(_z.any()):
                latents_history_mid = torch.where(_z, _sig * torch.randn_like(latents_history_mid), latents_history_mid)
        return latents_history_long, latents_history_mid

    def _geo_even_threshold_map(self, Tn: int, H: int, W: int, g: int, device: torch.device) -> torch.Tensor:
        """Spatially uniform ordered-dither threshold map [Tn,H,W] in (0,1): an 8x8 Bayer matrix tiled to (H/g,W/g) then upscaled by g,
        with a per-frame phase roll -> the cells kept by keep = (thr < keep_frac) spread out uniformly in space (no random clumping)
        and the pattern shifts over time instead of staying fixed. g controls the thinning cell granularity (g=1 is finest = per latent cell, preserving low-frequency geometry as much as possible)."""
        base = getattr(self, "_geo_bayer8", None)
        if base is None or base.device != device:
            n = 8
            m = torch.zeros(1, 1, device=device)
            size = 1
            while size < n:
                m = torch.cat([torch.cat([4 * m, 4 * m + 2], dim=1),
                               torch.cat([4 * m + 3, 4 * m + 1], dim=1)], dim=0)
                size *= 2
            base = (m + 0.5) / (n * n)        # [8,8] in (0,1), 64 levels
            self._geo_bayer8 = base
        n = base.shape[0]
        gh, gw = (H + g - 1) // g, (W + g - 1) // g
        tile = base.repeat((gh + n - 1) // n, (gw + n - 1) // n)[:gh, :gw]   # [gh,gw]
        out = torch.empty(Tn, gh, gw, device=device)
        for t in range(Tn):
            out[t] = torch.roll(tile, shifts=((t * 3) % n, (t * 5) % n), dims=(0, 1))
        return out.repeat_interleave(g, -2).repeat_interleave(g, -1)[:, :H, :W]  # [Tn,H,W]

    def _geo_build_short_visibility_mask(
        self,
        visibility_mask_pix: torch.Tensor,
        num_lat_per_chunk: int,
        has_prev_short: bool,
        H_lat: int,
        W_lat: int,
        device: torch.device,
        vae_t_stride: int,
    ) -> torch.Tensor:
        """Build short-tier latent-domain visibility mask from pixel-domain mask (BLOCKY patch mask)."""
        from evoke.dataset.online_materialize import _geo_resize_visibility_to_latent as _blocky_vis
        sampled = _blocky_vis(
            visibility_mask_pix.to(device=device, dtype=torch.float32),
            num_lat_per_chunk, H_lat, W_lat, vae_t_stride=int(vae_t_stride),
        )

        # diagnostic (--geo_drop_warp): force the warp segment visibility to all 0 -> the downstream token filter drops every warp token ->
        # the short tier degrades to [prefix | prev_short] (prefix/prev_short are ones, kept). Cuts the cross-chunk warp feedback.
        _vs_cfg = getattr(self, "_geo_vsnoise_cfg", None) or {}
        if bool(_vs_cfg.get("geo_drop_warp", False)):
            sampled = torch.zeros_like(sampled)

        # inference-side adaptive warp visibility cap (--geo_warp_vis_cap, the primary mechanism): measure the visible warp fraction v per latent frame,
        # and where v>cap thin the visible warp down to cap with a spatially uniform ordered-dither (Bayer) pattern (keep_frac=cap/v); v<=cap is left alone.
        # motivation: warp is both the camera control signal and an appearance reference, and on slow straight walks v~1 -> the model copies warp frame by frame (frozen frames). Fine uniform thinning
        # keeps the low-frequency geometry / camera control and removes the high-frequency appearance copy surface -> DiT synthesizes detail/motion itself; how much is removed is decided entirely by v.
        _cap = float(_vs_cfg.get("warp_vis_cap", 0.0))
        _Tn = sampled.shape[2] if sampled.dim() == 5 else 0
        if 0.0 < _cap < 1.0 and sampled.numel() > 0 and _Tn > 0:
            _g = max(1, int(_vs_cfg.get("warp_patch_drop_grid", 1)))
            _vis = sampled.mean(dim=(0, 1, 3, 4)).clamp_min(1e-6)                  # [Tn] visible fraction per frame
            _keep_frac = (_cap / _vis).clamp(max=1.0)                             # [Tn] v<=cap -> 1 (keep all)
            _thr = self._geo_even_threshold_map(_Tn, H_lat, W_lat, _g, sampled.device)  # [Tn,H,W] in (0,1)
            _keep = (_thr < _keep_frac.view(_Tn, 1, 1)).to(sampled.dtype)          # 1=keep
            sampled = sampled * _keep.view(1, 1, _Tn, H_lat, W_lat)
            # thinned warp frames get no extra handling beyond the keep-clean / cache reuse semantics; only visibility changes -> downstream token filtering stays identical.

        # inference-side per-patch warp drop (--geo_warp_patch_drop_ratio>0, fixed-ratio/random control arm): zero the warp visibility
        # per grid x grid latent-cell block with a uniform Bernoulli (like training per_patch). Mutually exclusive with vis_cap, which wins.
        _pdr = float(_vs_cfg.get("warp_patch_drop_ratio", 0.0))
        if not (0.0 < _cap < 1.0) and _pdr > 0.0 and sampled.numel() > 0:
            _g = max(1, int(_vs_cfg.get("warp_patch_drop_grid", 4)))
            _gen = getattr(self, "_geo_patchdrop_gen", None)
            if _gen is None:
                _gen = torch.Generator(device=sampled.device).manual_seed(
                    int(_vs_cfg.get("warp_patch_drop_seed", 0))
                )
                self._geo_patchdrop_gen = _gen
            _Tn = sampled.shape[2]
            _gh, _gw = (H_lat + _g - 1) // _g, (W_lat + _g - 1) // _g
            _pd = torch.rand(_Tn, _gh, _gw, generator=_gen, device=sampled.device) < _pdr
            _pd = _pd.repeat_interleave(_g, -2).repeat_interleave(_g, -1)[:, :H_lat, :W_lat]  # [Tn,H_lat,W_lat]
            sampled = sampled * (~_pd).to(sampled.dtype).view(1, 1, _Tn, H_lat, W_lat)

        # Layout [prefix | warp | prev_short]: prefix(ones) then warp(sampled) then prev_short(ones, trailing).
        prefix_vis = torch.ones(1, 1, 1, H_lat, W_lat, device=device, dtype=torch.float32)
        parts = [prefix_vis, sampled]
        if has_prev_short:
            prev_short_vis = torch.ones(1, 1, 1, H_lat, W_lat, device=device, dtype=torch.float32)
            parts.append(prev_short_vis)
        return torch.cat(parts, dim=2)

    @staticmethod
    def _geo_quantize_8bit(frame: torch.Tensor) -> torch.Tensor:
        """Simulate mp4 8-bit quantization for keyframe storage."""
        frame01 = (frame.detach().float().cpu() / 2.0 + 0.5).clamp(0.0, 1.0)
        frame01 = (frame01 * 255.0).round() / 255.0
        return frame01 * 2.0 - 1.0

    # ---------------- GEO LoRA fuse/unfuse helpers ----------------

    _GEO_ADAPTER_NAME = "geo"

    def _geo_has_loaded_adapters(self) -> bool:
        transformer = getattr(self, "transformer", None)
        peft_config = getattr(transformer, "peft_config", None)
        if isinstance(peft_config, dict) and len(peft_config) > 0:
            return True
        return getattr(self, "_geo_loaded_lora_path", None) is not None

    def _geo_lora_is_fused(self) -> bool:
        transformer = getattr(self, "transformer", None)
        if transformer is None:
            return False
        for module in transformer.modules():
            if bool(getattr(module, "merged", False)):
                return True
        return False

    def _set_geo_lora_enabled(self, enabled: bool) -> None:
        transformer = getattr(self, "transformer", None)
        if transformer is None or not self._geo_has_loaded_adapters():
            return
        if not enabled:
            # Unfuse only (do not call disable_adapters, which would also disable training adapters).
            self._unfuse_geo_lora()
            return
        if hasattr(transformer, "enable_adapters"):
            transformer.enable_adapters()

    def _fuse_geo_lora(self) -> bool:
        transformer = getattr(self, "transformer", None)
        if transformer is None or not self._geo_has_loaded_adapters():
            return False
        if self._geo_lora_is_fused():
            return True
        self._set_geo_lora_enabled(True)
        adapter_names = [self._GEO_ADAPTER_NAME]
        if hasattr(self, "fuse_lora"):
            try:
                self.fuse_lora(components=["transformer"], adapter_names=adapter_names)
            except TypeError:
                self.fuse_lora(adapter_names=adapter_names)
            return self._geo_lora_is_fused()
        if hasattr(transformer, "fuse_lora"):
            transformer.fuse_lora(adapter_names=adapter_names)
            return self._geo_lora_is_fused()
        return False

    def _unfuse_geo_lora(self) -> None:
        if not self._geo_lora_is_fused():
            return
        transformer = getattr(self, "transformer", None)
        if hasattr(self, "unfuse_lora"):
            try:
                self.unfuse_lora(components=["transformer"])
            except TypeError:
                self.unfuse_lora()
            return
        if transformer is not None and hasattr(transformer, "unfuse_lora"):
            transformer.unfuse_lora()

    def _configure_geo_lora(self, lora_path) -> bool:
        """Load or switch GEO LoRA adapter; disables it when lora_path is None."""
        if lora_path is None or str(lora_path).strip() == "":
            self._set_geo_lora_enabled(False)
            return False
        loaded_path = getattr(self, "_geo_loaded_lora_path", None)
        if loaded_path != lora_path:
            if loaded_path is not None and hasattr(self, "delete_adapters"):
                try:
                    self.delete_adapters(self._GEO_ADAPTER_NAME)
                except Exception:
                    pass
            self.load_lora_weights(str(lora_path), adapter_name=self._GEO_ADAPTER_NAME)
            self._geo_loaded_lora_path = lora_path
        if hasattr(self, "set_adapters"):
            self.set_adapters([self._GEO_ADAPTER_NAME], adapter_weights=[1.0])
        self._set_geo_lora_enabled(True)
        return True

    def stage1_sample(
        self,
        latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        sink_latents: torch.Tensor = None,
        nearby_sink_latents: torch.Tensor = None,
        nearby_sink_indices: torch.Tensor = None,  # explicit sink indices for GEO + raw_sink mode
        # GEO visibility masks; None = baseline.
        history_visible_mask_short: Optional[torch.Tensor] = None,
        history_visible_mask_mid: Optional[torch.Tensor] = None,
        history_visible_mask_long: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        generator: Optional[torch.Generator] = None,
        # Camera plucker embedding for the current section; None skips cam control path.
        cam_plucker_emb: Optional[torch.Tensor] = None,
        # ------------ CFG Zero ------------
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        # -------------- DMD --------------
        use_dmd: bool = False,
        dmd_sigmas: torch.Tensor = None,
        dmd_timesteps: torch.Tensor = None,
        is_amplify_first_chunk: bool = False,
        # ------------ Callback ------------
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        batch_size = latents.shape[0]

        _sink = sink_latents.to(transformer_dtype) if sink_latents is not None else None
        _nearby_sink = nearby_sink_latents.to(transformer_dtype) if nearby_sink_latents is not None else None

        for i, t in enumerate(timesteps):
            is_first_step = i == 0

            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0])

            latent_model_input = latents.to(transformer_dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(transformer_dtype),
                    latents_history_mid=latents_history_mid.to(transformer_dtype),
                    latents_history_long=latents_history_long.to(transformer_dtype),
                    sink_latents=_sink,
                    nearby_sink_latents=_nearby_sink,
                    nearby_sink_indices=nearby_sink_indices,
                    history_visible_mask_short=history_visible_mask_short,
                    history_visible_mask_mid=history_visible_mask_mid,
                    history_visible_mask_long=history_visible_mask_long,
                    cam_plucker_emb=cam_plucker_emb,
                    is_first_denoising_step=is_first_step,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance and not use_dmd:
                with self.transformer.cache_context("uncond"):
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        sink_latents=_sink,
                        nearby_sink_latents=_nearby_sink,
                        nearby_sink_indices=nearby_sink_indices,
                        history_visible_mask_short=history_visible_mask_short,
                        history_visible_mask_mid=history_visible_mask_mid,
                        history_visible_mask_long=history_visible_mask_long,
                        cam_plucker_emb=cam_plucker_emb,
                        is_first_denoising_step=is_first_step,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if use_cfg_zero_star:
                    noise_pred_text = noise_pred
                    positive_flat = noise_pred_text.view(batch_size, -1)
                    negative_flat = noise_uncond.view(batch_size, -1)

                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                    alpha = alpha.to(noise_pred_text.dtype)

                    if (i <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred_text * 0.0
                    else:
                        noise_pred = noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                else:
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            if use_dmd:
                pred_image_or_video = convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=latent_model_input,
                    timestep=t * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                    sigmas=dmd_sigmas,
                    timesteps=dmd_timesteps,
                )
                if i < len(timesteps) - 1:
                    latents = add_noise(
                        pred_image_or_video,
                        randn_tensor(pred_image_or_video.shape, generator=generator, device=device),
                        timesteps[i + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                        sigmas=dmd_sigmas,
                        timesteps=dmd_timesteps,
                    )
                else:
                    latents = pred_image_or_video
            else:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        return latents

    def stage2_sample(
        self,
        latents: torch.Tensor = None,
        stage2_num_stages: int = None,
        stage2_num_inference_steps_list: List[int] = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        sink_latents: torch.Tensor = None,
        nearby_sink_latents: torch.Tensor = None,
        nearby_sink_indices: torch.Tensor = None,  # explicit sink indices for GEO + raw_sink mode
        # GEO visibility masks; None = baseline.
        history_visible_mask_short: Optional[torch.Tensor] = None,
        history_visible_mask_mid: Optional[torch.Tensor] = None,
        history_visible_mask_long: Optional[torch.Tensor] = None,
        # GEO LoRA: fused only at pyramid stage 0, unfused at other stages.
        geo_lora_active: bool = False,
        # Geo: inject warp into the short tier ONLY at pyramid stage 0 (matches the Geo
        # reference, which builds the warp-bearing short history only at the coarsest stage and
        # conditions finer stages on [prefix|prev_short]). At i_s>0 the warp segment is stripped.
        geo_warp_stage0_only: bool = False,
        attention_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        scheduler_type: str = "unipc",  # unipc, euler
        use_dynamic_shifting: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "linear",
        generator: torch.Generator | list[torch.Generator] | None = None,
        # Per-stage cam plucker is recomputed from c2ws_chunk + Ks at each pyramid stage resolution.
        cam_lingbot_Ks_chunk: Optional[torch.Tensor] = None,           # [B, 4]
        cam_lingbot_c2ws_chunk: Optional[torch.Tensor] = None,         # [B, F_pix_chunk, 4, 4]
        cam_base_height_pix: Optional[int] = None,
        cam_base_width_pix: Optional[int] = None,
        cam_pc_resolution_strategy: str = "scale_ks",
        # ------------ CFG Zero ------------
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        # -------------- DMD --------------
        use_dmd: bool = False,
        is_amplify_first_chunk: bool = False,
        # ------------ Callback ------------
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        num_frames, height, width = (
            latents.shape[-3],
            latents.shape[-2],
            latents.shape[-1],
        )
        latents = rearrange(latents, "b c t h w -> (b t) c h w")
        for _ in range(stage2_num_stages - 1):
            height //= 2
            width //= 2
            latents = (
                F.interpolate(
                    latents,
                    size=(height, width),
                    mode="bilinear",
                )
                * 2
            )
        latents = rearrange(latents, "(b t) c h w -> b c t h w", t=num_frames)

        batch_size = latents.shape[0]
        import time as _time  # [difftime] per-step DiT timing below
        # Per-step timing is debug output: it fires 3x per chunk and, being newline-terminated while a
        #   progress bar is not, it lands in the middle of the bar. Opt in with EVOKE_INFER_DEBUG=1.
        import os as _os_dt
        _difftime_log = _os_dt.environ.get("EVOKE_INFER_DEBUG", "0") == "1"
        if use_dmd:
            start_point_list = [latents]

        i = 0
        for i_s in range(stage2_num_stages):
            # Fuse GEO LoRA only at stage 0; unfuse at higher pyramid stages.
            if geo_lora_active and i_s == 0:
                if not self._fuse_geo_lora():
                    # Fallback to enable_adapters if fuse fails (no LoRA loaded).
                    self._set_geo_lora_enabled(True)
            else:
                self._unfuse_geo_lora()
            if use_dmd:
                if is_amplify_first_chunk:
                    self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=device)
                else:
                    self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=device)
                self.scheduler.timesteps = self.scheduler.timesteps[:-1]
                self.scheduler.sigmas = torch.cat([self.scheduler.sigmas[:-2], self.scheduler.sigmas[-1:]])
            else:
                self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s], i_s, device=device)

            if i_s > 0:
                height *= 2
                width *= 2
                num_frames = latents.shape[2]
                latents = rearrange(latents, "b c t h w -> (b t) c h w")
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                latents = rearrange(latents, "(b t) c h w -> b c t h w", t=num_frames)
                # Fix the stage
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]  # the original coeff of signal
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, channel, num_frames, height, width = latents.shape
                noise = self.sample_block_noise(
                    batch_size,
                    channel,
                    num_frames,
                    height,
                    width,
                    self.transformer.config.patch_size,
                    device,
                    generator,
                )
                noise = noise.to(device=device, dtype=transformer_dtype)
                latents = alpha * latents + beta * noise  # To fix the block artifact

                if use_dmd:
                    start_point_list.append(latents)

            if use_dynamic_shifting:
                temp_sigmas = apply_schedule_shift(
                    self.scheduler.sigmas,
                    latents,
                    base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                    max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                    base_shift=self.scheduler.config.get("base_shift", 0.5),
                    max_shift=self.scheduler.config.get("max_shift", 1.15),
                    time_shift_type=time_shift_type,
                )
                temp_timesteps = self.scheduler.timesteps_per_stage[i_s].min() + temp_sigmas[:-1] * (
                    self.scheduler.timesteps_per_stage[i_s].max() - self.scheduler.timesteps_per_stage[i_s].min()
                )

                self.scheduler.sigmas = temp_sigmas
                self.scheduler.timesteps = temp_timesteps

            timesteps = self.scheduler.timesteps

            _sink_s2 = sink_latents.to(transformer_dtype) if sink_latents is not None else None
            _nearby_sink_s2 = nearby_sink_latents.to(transformer_dtype) if nearby_sink_latents is not None else None

            # Recompute plucker embedding at current stage resolution (shared c2ws/Ks, rescaled grid).
            cam_plucker_emb_stage = None
            if (
                cam_lingbot_Ks_chunk is not None
                and cam_lingbot_c2ws_chunk is not None
                and getattr(self.transformer, "enable_cam_control", False)
            ):
                from evoke.modules.camera_control import prepare_cam_plucker_emb
                _H_lat_stage, _W_lat_stage = latents.shape[-2], latents.shape[-1]
                _H_pix_stage = _H_lat_stage * self.vae_scale_factor_spatial
                _W_pix_stage = _W_lat_stage * self.vae_scale_factor_spatial
                _base_h = cam_base_height_pix or _H_pix_stage   # default to current stage resolution
                _base_w = cam_base_width_pix or _W_pix_stage
                _Ks_f = cam_lingbot_Ks_chunk.to(device=device, dtype=torch.float32)
                _c2ws_f = cam_lingbot_c2ws_chunk.to(device=device, dtype=torch.float32)
                cam_plucker_emb_stage = prepare_cam_plucker_emb(
                    Ks=_Ks_f,
                    c2ws_window=_c2ws_f,
                    H_pix=_H_pix_stage, W_pix=_W_pix_stage,
                    base_h_pix=_base_h, base_w_pix=_base_w,
                    vae_stride_t=self.vae_scale_factor_temporal,
                    vae_stride_h=self.vae_scale_factor_spatial,
                    vae_stride_w=self.vae_scale_factor_spatial,
                    strategy=cam_pc_resolution_strategy,
                ).to(dtype=transformer_dtype)

            # Geo stage0-only warp: at finer stages (i_s>0) strip the warp segment from the short
            # tier so the (LoRA-unfused) base model conditions on [prefix|prev_short] only — the
            # warp-bearing short history is used solely at the coarsest stage, mirroring the Geo
            # reference (_build_pyramid_base_histories called only at i_s==0). Default off.
            _cur_lat_short = latents_history_short
            _cur_idx_short = indices_latents_history_short
            _cur_mask_short = history_visible_mask_short
            _cur_attn_kwargs = attention_kwargs
            _wf = int((attention_kwargs or {}).get("geo_warp_frames", 0) or 0)
            if geo_warp_stage0_only and i_s > 0 and _wf > 0 and latents_history_short is not None:
                # short-tier layout along temporal dim: [prefix(1) | warp(_wf) | prev_short(Sp)]
                _cur_lat_short = torch.cat(
                    [latents_history_short[:, :, :1], latents_history_short[:, :, 1 + _wf:]], dim=2
                )
                if indices_latents_history_short is not None:
                    if indices_latents_history_short.dim() == 1:
                        _cur_idx_short = torch.cat(
                            [indices_latents_history_short[:1], indices_latents_history_short[1 + _wf:]], dim=0
                        )
                    else:
                        _cur_idx_short = torch.cat(
                            [indices_latents_history_short[:, :1], indices_latents_history_short[:, 1 + _wf:]], dim=1
                        )
                if history_visible_mask_short is not None:
                    _cur_mask_short = torch.cat(
                        [history_visible_mask_short[:, :, :1], history_visible_mask_short[:, :, 1 + _wf:]], dim=2
                    )
                _cur_attn_kwargs = dict(attention_kwargs or {})
                _cur_attn_kwargs["geo_warp_frames"] = 0

            for idx, t in enumerate(timesteps):
                is_first_step = i_s == 0 and idx == 0

                timestep = t.expand(latents.shape[0]).to(torch.int64)

                with self.transformer.cache_context("cond"):
                    # [difftime] CUDA-synced pure-diffusion (DiT forward) timing per denoising step.
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _difft0 = _time.time()
                    noise_pred = self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=_cur_attn_kwargs,
                        return_dict=False,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=_cur_idx_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=_cur_lat_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        sink_latents=_sink_s2,
                        nearby_sink_latents=_nearby_sink_s2,
                        nearby_sink_indices=nearby_sink_indices,
                        history_visible_mask_short=_cur_mask_short,
                        history_visible_mask_mid=history_visible_mask_mid,
                        history_visible_mask_long=history_visible_mask_long,
                        cam_plucker_emb=cam_plucker_emb_stage,
                        is_first_denoising_step=is_first_step,
                    )[0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if _difftime_log:
                        print(f"[difftime] i_s={i_s} idx={idx} H={latents.shape[-2]} W={latents.shape[-1]} "
                              f"dit_forward={_time.time() - _difft0:.3f}s", flush=True)

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("cond_uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=_cur_attn_kwargs,
                            return_dict=False,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=_cur_idx_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=_cur_lat_short.to(transformer_dtype),
                            latents_history_mid=latents_history_mid.to(transformer_dtype),
                            latents_history_long=latents_history_long.to(transformer_dtype),
                            sink_latents=_sink_s2,
                            nearby_sink_latents=_nearby_sink_s2,
                            nearby_sink_indices=nearby_sink_indices,
                            history_visible_mask_short=_cur_mask_short,
                            history_visible_mask_mid=history_visible_mask_mid,
                            history_visible_mask_long=history_visible_mask_long,
                            cam_plucker_emb=cam_plucker_emb_stage,
                            is_first_denoising_step=is_first_step,
                        )[0]

                    if use_cfg_zero_star:
                        noise_pred_text = noise_pred
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_uncond.view(batch_size, -1)

                        alpha = optimized_scale(positive_flat, negative_flat)
                        alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                        alpha = alpha.to(noise_pred_text.dtype)

                        if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                            noise_pred = noise_pred_text * 0.0
                        else:
                            noise_pred = noise_uncond * alpha + guidance_scale * (
                                noise_pred_text - noise_uncond * alpha
                            )
                    else:
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                if use_dmd:
                    pred_image_or_video = convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=latents,
                        timestep=timestep,
                        sigmas=self.scheduler.sigmas,
                        timesteps=self.scheduler.timesteps,
                    )
                    if idx < len(timesteps) - 1:
                        latents = add_noise(
                            pred_image_or_video,
                            start_point_list[i_s],
                            timesteps[idx + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                            sigmas=self.scheduler.sigmas,
                            timesteps=self.scheduler.timesteps,
                        )
                    else:
                        latents = pred_image_or_video
                else:
                    if scheduler_type == "unipc":
                        latents = self.scheduler.step_unipc(noise_pred.float(), t, latents, return_dict=False)[0]
                    else:
                        latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

                i += 1
                _step_observer = getattr(self, "_evoke_progress_callback", None)
                if callable(_step_observer):
                    try:
                        _step_observer(
                            phase="denoising",
                            chunks=int(getattr(self, "_evoke_progress_chunks", 1)),
                            chunk_index=int(getattr(self, "_evoke_progress_chunk_index", 0)),
                            stage=i_s + 1,
                            step=i,
                            total_steps=sum(stage2_num_inference_steps_list) * (
                                2 if use_dmd and is_amplify_first_chunk else 1
                            ),
                        )
                    except Exception:
                        pass

        # Unfuse GEO LoRA at stage2 exit to prevent leaking into subsequent calls.
        if geo_lora_active:
            self._unfuse_geo_lora()
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 384,
        width: int = 640,
        num_frames: int = 73,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # ------------ I2V ------------
        image: Optional[PipelineImageInput] = None,
        image_latents: Optional[torch.Tensor] = None,
        fake_image_latents: Optional[torch.Tensor] = None,
        add_noise_to_image_latents: bool = True,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        # ------------ V2V ------------
        video: Optional[PipelineImageInput] = None,
        video_latents: Optional[torch.Tensor] = None,
        add_noise_to_video_latents: bool = True,
        video_noise_sigma_min: float = 0.111,
        video_noise_sigma_max: float = 0.135,
        # ------------ Camera control (lingbot-style AdaLN, optional) ------------
        # lingbot_Ks: [B,4] or [4] camera intrinsics [fx,fy,cx,cy]; lingbot_c2ws: frame-wise c2w poses.
        # Skipped when None or transformer has no cam control.
        lingbot_Ks: Optional[torch.Tensor] = None,
        lingbot_c2ws: Optional[torch.Tensor] = None,
        cam_pc_resolution_strategy: str = "scale_ks",   # "scale_ks" | "resample"
        cam_base_height_pix: Optional[int] = None,       # None defaults to pipeline `height`
        cam_base_width_pix: Optional[int] = None,        # None defaults to pipeline `width`
        # ------------ Interactive ------------
        use_interpolate_prompt: bool = False,
        interpolate_time_list: list = [7, 7, 7],
        interpolation_steps: int = 3,
        # ------------ Stage 1 ------------
        history_sizes: list = [16, 2, 1],
        latent_window_size: int = 9,
        use_dynamic_shifting: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "linear",
        is_keep_x0: bool = True,
        # ------------ Stage 2 ------------
        is_enable_stage2: bool = False,
        stage2_num_stages: int = 3,
        stage2_num_inference_steps_list: list = [10, 10, 10],
        scheduler_type: str = "unipc",  # unipc, euler
        # ------------ CFG Zero ------------
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        # ------------ DMD ------------
        use_dmd: bool = False,
        is_skip_first_section: bool = False,
        is_amplify_first_chunk: bool = False,
        # ------------ Adaptive Anti-Drifting ------------
        use_adaptive_anti_drifting: bool = False,
        anti_drift_rho_mu: float = 0.9,
        anti_drift_rho_sigma: float = 0.9,
        anti_drift_delta_mu: float = 0.15,
        anti_drift_delta_sigma: float = 0.15,
        anti_drift_corruption_strength: float = 0.1,
        # ------------ pose-addressed geometric state ------------
        # When enabled, overrides short tier with [prefix, warp_latents, prev_short] (prev_short closest to noise).
        use_geometric_state: bool = False,
        # Diagnostic: skip the prev_short token -> short tier reverts to [prefix, warp] (geo_prev_short_frames=0).
        geo_disable_prev_short: bool = False,
        # Path to GEO LoRA weights; None disables LoRA.
        geo_lora_path: Optional[str] = None,
        # Number of keyframes retrieved from FrameBank per chunk for Pi3X.
        geo_top_k: int = 8,
        # FrameBank retrieval config: nearby_k = most-recent frames; select_k = score-based; score = metric.
        geo_nearby_k: int = 0,
        geo_select_k: Optional[int] = None,
        geo_score: str = "v1",
        geo_score_kwargs: Optional[dict] = None,
        # FrameBank size cap and v2v seeding limit.
        geo_bank_max: Optional[int] = None,
        geo_init_k: int = 10,
        # Per-chunk skill/VFX event control. event_chunks = 0-indexed rollout chunks marked as "event":
        # for those chunks warp is dropped, the camera is frozen (static plucker), the chunk's frames are
        # NOT added to the frame bank (warp retrieval source), and event_prompt (if non-empty) is used.
        # event_chunks=None/[] => no event chunks => rollout path is byte-for-byte identical to baseline.
        event_chunks: Optional[list] = None,
        event_prompt: str = "",
        # Per-chunk prompt schedule (segment prompts). {chunk_index: prompt_text}; a chunk keeps the
        # global `prompt` until an entry takes over, and the entry stays in force until the next one.
        # This is the inference-side counterpart of training's segment prompts
        # (data caption_key=segments + use_full_rollout_interleave). Unlike event_chunks it ONLY
        # swaps the text embedding: warp, camera and the frame bank are untouched.
        # None/{} => every chunk uses `prompt` => rollout is byte-for-byte identical to baseline.
        chunk_prompts: Optional[dict] = None,
        # ------------ other ------------
        use_kv_cache: bool = False,
        vae_decode_type: VAEDecodeType = "default",  # "default", "default_batch", "long", "persistent"
        # streaming output for ultra-long rollouts: when True the in-chunk decode does not accumulate history_video and
        #   frames=None is returned -- the caller stitches the full video from the per-chunk segment mp4s (see
        #   scripts/inference/infer_single.py --stream_long_video). history_video is [1,3,T,H,W] fp32, which at
        #   T=86397 (60min@24fps, 384x640) is 254GB on the GPU, with postprocess_video's numpy conversion the same
        #   order on the CPU. Only vae_decode_type="persistent"/"default" support it; the other paths decode the whole
        #   latent sequence in one shot. Default False = the original behaviour.
        stream_output: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`EvokePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~EvokePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`EvokePipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if image is not None and video is not None:
            # image + video coexist only in GEO mode (image = Pi3X source; video = v2v conditioning).
            if not use_geometric_state:
                raise ValueError(
                    "image and video cannot be provided simultaneously "
                    "(set use_geometric_state=True to allow both: image as the Pi3X source, video as v2v conditioning)"
                )

        if use_kv_cache:
            self.transformer.enable_kv_cache()

        if use_interpolate_prompt:
            assert num_videos_per_prompt == 1, f"num_videos_per_prompt must be 1, got {num_videos_per_prompt}"
            assert isinstance(prompt, list), "prompt must be a list"
            assert len(prompt) == len(interpolate_time_list), (
                f"Length mismatch: {len(prompt)} vs {len(interpolate_time_list)}"
            )
            assert min(interpolate_time_list) > interpolation_steps, (
                f"Minimum value {min(interpolate_time_list)} must be greater than {interpolation_steps}"
            )
            interpolate_interval_idx = None
            interpolate_embeds = None
            interpolate_cumulative_list = list(accumulate(interpolate_time_list))

        anti_drifting_helper = None
        if use_adaptive_anti_drifting:
            anti_drifting_helper = AdaptiveAntiDrifting(
                rho_mu=anti_drift_rho_mu,
                rho_sigma=anti_drift_rho_sigma,
                delta_mu=anti_drift_delta_mu,
                delta_sigma=anti_drift_delta_sigma,
                device=self._execution_device,
                dtype=torch.float32,
            )

        history_sizes = sorted(history_sizes, reverse=True)  # descending order

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Validate inputs.
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        vae_dtype = self.vae.dtype

        # 2. Resolve batch size.
        if use_interpolate_prompt or (prompt is not None and isinstance(prompt, str)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode prompts.
        all_prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        )

        transformer_dtype = self.transformer.dtype
        all_prompt_embeds = all_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            if use_interpolate_prompt:
                negative_prompt_embeds = negative_prompt_embeds[0].unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # ── Per-chunk skill/VFX event control: cache the event prompt embeds once (mirrors the encode above).
        # _event_set empty => is_event_k always False below => every event hook reduces to the baseline path.
        _event_set = set(event_chunks or [])
        _event_prompt_embeds = None
        if _event_set and event_prompt:
            _ep_embeds, _, _, _ = self.encode_prompt(
                prompt=event_prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            _event_prompt_embeds = _ep_embeds.to(transformer_dtype)

        # ── Per-chunk prompt schedule: encode each distinct segment prompt once, then index by chunk.
        #   Built as {chunk_index: embeds}; the loop below picks the entry in force at that chunk.
        _chunk_prompt_embeds = {}
        if chunk_prompts:
            _uniq = {}
            for _ck, _ctext in sorted(chunk_prompts.items(), key=lambda kv: int(kv[0])):
                _ctext = str(_ctext)
                if _ctext not in _uniq:
                    _ce, _, _, _ = self.encode_prompt(
                        prompt=_ctext,
                        negative_prompt=negative_prompt,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        num_videos_per_prompt=num_videos_per_prompt,
                        prompt_embeds=None,
                        negative_prompt_embeds=None,
                        max_sequence_length=max_sequence_length,
                        device=device,
                    )
                    _uniq[_ctext] = _ce.to(transformer_dtype)
                _chunk_prompt_embeds[int(_ck)] = _uniq[_ctext]
            print(f"[prompt-schedule] {len(_chunk_prompt_embeds)} switch point(s) at chunks "
                  f"{sorted(_chunk_prompt_embeds)}, {len(_uniq)} distinct prompt(s)", flush=True)

        # The text encoder is not used again after all prompt variants have been embedded. The
        # Director worker keeps the pipeline alive between jobs, so leaving it on CUDA needlessly
        # competes with ViGeo + the VAE decoder at their peak. Opt-in: the regular inference path is
        # unchanged. infer_single moves it back before the next call.
        if bool(getattr(self, "_evoke_offload_text_encoder", False)):
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            print("[geo-infer] text encoder offloaded after prompt encoding", flush=True)

        # 4. Prepare image latents.
        _v2v_decode_warm_latents = None
        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image_latents, fake_image_latents = self.prepare_image_latents(
                image,
                latents_mean=latents_mean,
                latents_std=latents_std,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=image_latents,
                fake_latents=fake_image_latents,
            )

        if image is not None and image_latents is not None:
            _v2v_decode_warm_latents = image_latents.clone()

        if image_latents is not None and add_noise_to_image_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            fake_image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min)
                + video_noise_sigma_min
            )
            fake_image_latents = (
                fake_image_noise_sigma * randn_tensor(fake_image_latents.shape, generator=generator, device=device)
                + (1 - fake_image_noise_sigma) * fake_image_latents
            )

        if video is not None:
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            image_latents, video_latents = self.prepare_video_latents(
                video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                latent_window_size=latent_window_size,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=video_latents,
            )
            # In v2v + GEO mode, use the last video frame latent as chunk-0 prefix for temporal continuity.
            if use_geometric_state and video_latents is not None and video_latents.shape[2] > 0:
                image_latents = video_latents[:, :, -1:, :, :].clone()

        if video_latents is not None and video_latents.shape[2] > 0:
            _warm_n_max = int(getattr(self, "_geo_first_chunk_decode_warm", 4))
            _v2v_decode_warm_latents = video_latents[:, :, -_warm_n_max:, :, :].clone()

        # Two decode methods for chunk0's first frame, split by whether a GENUINE prior frame exists:
        #   i2v / v2v -> a real prior is available (i2v input-image latent / v2v ref-video tail) -> warm the chunk0 VAE
        #     decode cache with it so chunk0's first latent decodes as a CONTINUATION: correct colour, no cast, no blur.
        #   t2v -> no prior frame exists (a true clip start) -> decode chunk0 as a fresh clip (I-frame slot); there is
        #     nothing real to continue from, and warming with the self latent only casts and blurs it.
        _warm_is_real_prior = (image is not None) or (video is not None)

        if video_latents is not None and add_noise_to_video_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )

            noisy_latents_chunks = []
            num_latent_chunks = video_latents.shape[2] // latent_window_size
            for i in range(num_latent_chunks):
                chunk_start = i * latent_window_size
                chunk_end = chunk_start + latent_window_size
                latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]

                chunk_frames = latent_chunk.shape[2]
                frame_sigmas = (
                    torch.rand(chunk_frames, device=device, generator=generator)
                    * (video_noise_sigma_max - video_noise_sigma_min)
                    + video_noise_sigma_min
                )
                frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)

                noisy_chunk = (
                    frame_sigmas * randn_tensor(latent_chunk.shape, generator=generator, device=device)
                    + (1 - frame_sigmas) * latent_chunk
                )
                noisy_latents_chunks.append(noisy_chunk)
            video_latents = torch.cat(noisy_latents_chunks, dim=2)

        # 5. Prepare latent variables.
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
        num_latent_sections = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        history_video = None
        total_generated_latent_frames = 0
        if stream_output and vae_decode_type not in ("persistent", "default", "default_warm0"):
            raise ValueError(
                f"stream_output=True only supports in-chunk decode vae_decode_type "
                f"('persistent'/'default'/'default_warm0'), got '{vae_decode_type}' "
                f"(one-shot whole-sequence decode is incompatible with streaming)."
            )

        # Latent-domain v2v offset used for history indexing and FrameBank seeding gates.
        cam_video_lat_offset = video_latents.shape[2] if video_latents is not None else 0

        # Pixel-domain v2v offset from raw video shape (not latent shape, which differs due to chunk encoding).
        geo_ref_pose_offset_pix = (
            int(video.shape[2])
            if (video is not None and isinstance(video, torch.Tensor) and video.ndim == 5)
            else 0
        )
        if geo_ref_pose_offset_pix > 0 or cam_video_lat_offset > 0:
            print(
                f"[pipe] V2V offsets: ref_video_frames={geo_ref_pose_offset_pix}, "
                f"video_latents_T={int(video_latents.shape[2]) if video_latents is not None else 0}, "
                f"cam_video_lat_offset={int(cam_video_lat_offset)}, "
                f"geo_ref_pose_offset_pix={geo_ref_pose_offset_pix}",
                flush=True,
            )

        # 5.5 Camera control: move full-sequence c2ws/Ks to device; slice per-section window in main loop.
        cam_Ks_dev = None         # full segment Ks on device
        cam_c2ws_dev = None       # full segment absolute c2ws on device
        cam_base_h_used = None
        cam_base_w_used = None
        # Populate the per-section Plucker pose buffers when EITHER the camera_control AdaLN path OR the GEO
        # additive Plucker path is on. Both consume the same lingbot c2ws/Ks (the chunk's TARGET poses, which
        # also drive the warp render) and feed the per-window cam_plucker_emb into the transformer forward.
        if (
            lingbot_Ks is not None
            and lingbot_c2ws is not None
            and (
                getattr(self.transformer, "enable_cam_control", False)
                or getattr(self.transformer, "geo_warp_plucker_enabled", False)
            )
        ):
            # Normalize input shapes to batched form.
            _Ks = lingbot_Ks.unsqueeze(0) if lingbot_Ks.dim() == 1 else lingbot_Ks
            _c2ws = lingbot_c2ws.unsqueeze(0) if lingbot_c2ws.dim() == 3 else lingbot_c2ws
            _Ks = _Ks.to(device=device, dtype=torch.float32)
            _c2ws = _c2ws.to(device=device, dtype=torch.float32)
            cam_Ks_dev = _Ks
            cam_c2ws_dev = _c2ws
            cam_base_h_used = cam_base_height_pix or height
            cam_base_w_used = cam_base_width_pix or width

        # GEO state init: maintains a separate geo_c2ws_dev buffer independent of cam_c2ws_dev.
        geo_state = None
        geo_lora_active = False
        geo_c2ws_dev = None
        if use_geometric_state:
            assert lingbot_c2ws is not None, "use_geometric_state=True requires lingbot_c2ws (the target poses the Pi3X render needs)"
            assert image is not None, (
                "use_geometric_state=True requires image (PIL/np/Tensor) as the first chunk source. "
                "image_latents-only input is not supported yet (Pi3X needs the source pixels)."
            )
            # Prepare independent c2ws buffer (decoupled from cam_c2ws_dev).
            _geo_c2ws_raw = lingbot_c2ws.unsqueeze(0) if lingbot_c2ws.dim() == 3 else lingbot_c2ws
            geo_c2ws_dev = _geo_c2ws_raw.to(device=device, dtype=torch.float32)
            # Use already-preprocessed image tensor directly to avoid double-normalization.
            assert isinstance(image, torch.Tensor) and image.ndim == 4, (
                f"GEO expects image to have been preprocessed into a [B,3,H,W] tensor upstream, got {type(image)} shape {getattr(image, 'shape', None)}"
            )
            _geo_source_pix = image  # [B, 3, H, W] in [-1, 1]
            geo_state = self._geo_init_state(_geo_source_pix, bank_max_size=geo_bank_max)
            geo_state["prev_chunk_last_decoded_latent"] = None
            # Load or switch GEO LoRA; None/empty disables it.
            geo_lora_active = self._configure_geo_lora(geo_lora_path)

            # Seed FrameBank: source frame always added; v2v also seeds from ref at 36-stride.
            _geo_init_max = int(geo_init_k) if geo_init_k is not None and int(geo_init_k) > 0 else 10
            _geo_pix_stride_seed = latent_window_size * self.vae_scale_factor_temporal  # 36
            # chunk-0 anchor = most recent ref frame to minimize the anchor-to-target gap.
            _is_v2v_geo = (
                video is not None
                and isinstance(video, torch.Tensor)
                and video.ndim == 5
                and int(cam_video_lat_offset) > 0
            )
            if _is_v2v_geo:
                # Use pixel-domain offset (actual ref video frames) for anchor index.
                _ref_pix_end = geo_ref_pose_offset_pix
                _v2v_anchor_pix_idx = max(0, _ref_pix_end - 1)
                # Override source_image_pixel with the last ref frame for Pi3X geometry estimation.
                _geo_anchor_image = video[0, :, -1].contiguous()  # [3, H, W] in [-1, 1]
                geo_state["source_image_pixel"] = _geo_anchor_image.unsqueeze(0).detach().cpu()
                geo_state["v2v_chunk0_anchor_pix_idx"] = int(_v2v_anchor_pix_idx)
                # Clamp pose index defensively to avoid IndexError on misaligned inputs.
                _v2v_anchor_pose_idx = min(_v2v_anchor_pix_idx, geo_c2ws_dev.shape[1] - 1)
                _geo_source_pose_seed = geo_c2ws_dev[0, _v2v_anchor_pose_idx].detach().cpu().float()
                _geo_source_frame_seed = _geo_anchor_image.detach().float().cpu()
                _geo_source_pixel_idx_seed = int(_v2v_anchor_pix_idx)
            else:
                # I2V/T2V: anchor is the user-supplied image at pixel index 0.
                _geo_source_pose_seed = geo_c2ws_dev[0, 0].detach().cpu().float()
                _geo_source_frame_seed = _geo_source_pix[0].detach().float().cpu()
                _geo_source_pixel_idx_seed = 0

            # Always add source frame as chunk-0 anchor.
            geo_state["frame_bank"].add(
                frame=_geo_source_frame_seed,
                c2w=_geo_source_pose_seed,
                chunk_idx=0,
                pixel_idx=_geo_source_pixel_idx_seed,
            )
            # Seed additional v2v ref frames at 36-stride (up to _geo_init_max - 1 extra frames).
            if _is_v2v_geo:
                _video_pix_len = int(video.shape[2])
                _seed_max_extra = _geo_init_max - 1  # anchor already occupies 1 slot
                _seed_pixel_idxs = list(range(
                    0,
                    min(_video_pix_len, _ref_pix_end),
                    _geo_pix_stride_seed,
                ))[:_seed_max_extra]
                for _pi in _seed_pixel_idxs:
                    if _pi == _geo_source_pixel_idx_seed:
                        continue   # skip if duplicate with anchor
                    _seed_pix = video[0, :, _pi].detach().float().cpu()  # [3, H, W] in [-1, 1]
                    _seed_pose_idx = min(_pi, geo_c2ws_dev.shape[1] - 1)
                    _seed_pose = geo_c2ws_dev[0, _seed_pose_idx].detach().cpu().float()
                    geo_state["frame_bank"].add(
                        frame=_seed_pix,
                        c2w=_seed_pose,
                        chunk_idx=0,
                        pixel_idx=_pi,
                    )
                if len(_seed_pixel_idxs) > 0:
                    print(
                        f"[GEO] v2v init FrameBank seeded: anchor=ref last frame (pix_idx={_geo_source_pixel_idx_seed}) "
                        f"+ {len(_seed_pixel_idxs)} ref-section candidates (36-stride from idx 0), "
                        f"total bank = {len(geo_state['frame_bank'])} / max {_geo_init_max}.",
                        flush=True,
                    )

            # -- DA3 recall backend: build bank/est/K + seed the ref frames (coexists with the Pi3X frame_bank; active when recon_backend=da3) --
            _da3_cfg = getattr(self, "_geo_vsnoise_cfg", None) or {}
            if str(_da3_cfg.get("recon_backend", "da3")) == "da3" and lingbot_Ks is not None:
                _da3_stride = latent_window_size * self.vae_scale_factor_temporal   # 36
                self._geo_da3_setup(geo_state, _da3_cfg, lingbot_Ks, int(height), int(width), _da3_stride, device)
                # i2v (not v2v) is the only mode whose chunk 0 has no warp source at all -- v2v seeds the
                # bank from the ref video above, so its chunk 0 already has a pool. GEO forbids t2v
                # (infer_single.py), so `not _is_v2v_geo` is exactly i2v. Gates the chunk-0 reference-image
                # warp in _geo_render_chunk_da3.
                geo_state["da3_is_i2v"] = (not _is_v2v_geo)
                if _is_v2v_geo and video is not None:        # v2v: ingest the ref video chunk by chunk as seed (i2v/t2v accumulate from generated frames)
                    _vlen = int(video.shape[2])
                    for _cs in range(0, _vlen, _da3_stride):
                        self._geo_da3_ingest(geo_state, video[0, :, _cs:min(_cs + _da3_stride, _vlen)],
                                             _cs, geo_c2ws_dev, device)
                    print(f"[GEO-da3] v2v seed: ref {_vlen} frames -> bank {geo_state['da3_bank'].num_frames} frames", flush=True)

        if not is_keep_x0:
            history_sizes[-1] = history_sizes[-1] + 1
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents, fake_image_latents], dim=2)
            total_generated_latent_frames += 1
        if video_latents is not None:
            history_frames = history_latents.shape[2]
            video_frames = video_latents.shape[2]
            if video_frames < history_frames:
                keep_frames = history_frames - video_frames
                history_latents = torch.cat([history_latents[:, :, :keep_frames, :, :], video_latents], dim=2)
            else:
                history_latents = video_latents
            total_generated_latent_frames += video_latents.shape[2]

        # 6. Main denoising loop.
        if use_interpolate_prompt:
            if num_latent_sections < max(interpolate_cumulative_list):
                num_latent_sections = sum(interpolate_cumulative_list)
                print(f"Update num_latent_sections to: {num_latent_sections}")

        # Reset per-rollout short-tier noise state.
        self._short_tier_print_count = 0
        self._short_tier_rollout_sigma = None
        _st_cfg_pre = getattr(self, "_short_tier_noise_cfg", None) or {}
        if (
            use_geometric_state     # only active in GEO path
            and bool(_st_cfg_pre.get("enabled", False))
            and bool(_st_cfg_pre.get("apply_at_inference", True))
            and bool(_st_cfg_pre.get("sigma_lock_per_rollout", False))
        ):
            _st_gen = generator[0] if isinstance(generator, list) else generator
            _st_min = float(_st_cfg_pre.get("sigma_min", 0.2))
            _st_max = float(_st_cfg_pre.get("sigma_max", 0.6))
            self._short_tier_rollout_sigma = float(
                torch.rand(1, device=device, generator=_st_gen).item() * (_st_max - _st_min) + _st_min
            )
            print(f"[ShortTier-Noise INFER] sigma_lock_per_rollout: sigma={self._short_tier_rollout_sigma:.3f}", flush=True)

        # -- rollout progress, two levels: the bar counts chunks, and each chunk reports its own two
        #    phases (warp render + diffusion) as a single line written through the bar. Opt-in via env
        #    var, because this pipeline also runs training validation where per-chunk output on every
        #    rank floods the log. EVOKE_INFER_DEBUG=1 keeps the raw per-chunk prints instead.
        import os as _os      # not a module-level import in this file; the methods import it locally
        import time as _ptime
        _prog_on = _os.environ.get("EVOKE_INFER_PROGRESS", "0") == "1"
        _prog_debug = _os.environ.get("EVOKE_INFER_DEBUG", "0") == "1"
        # Read by _geo_render_chunk_da3: its own per-chunk prints are folded into the summary line.
        self._geo_quiet_chunk_logs = _prog_on and not _prog_debug
        _chunk_iter = range(num_latent_sections)
        _prog_bar = None
        if _prog_on:
            import sys as _psys
            from tqdm.auto import tqdm as _tqdm
            # file=stdout on purpose: tqdm draws to stderr but tqdm.write() goes to stdout, and once the
            #   two are merged (2>&1) the bar can no longer erase itself before a line is written, so
            #   every summary line ends up with a bar frame glued to its front.
            _prog_bar = _tqdm(range(num_latent_sections), total=num_latent_sections,
                              desc="chunks", unit="chunk", dynamic_ncols=True, file=_psys.stdout)
            _chunk_iter = _prog_bar

        # Optional lightweight observer used by the Director UI. It receives metadata only (never
        # tensors), so enabling it cannot retain GPU memory or alter sampling. Other callers leave the
        # attribute unset and take the exact historical path.
        _evoke_progress = getattr(self, "_evoke_progress_callback", None)

        def _emit_progress(phase, **payload):
            if callable(_evoke_progress):
                try:
                    _evoke_progress(phase=phase, chunks=num_latent_sections, **payload)
                except Exception as _progress_error:
                    if _prog_debug:
                        print(f"[progress] observer failed: {_progress_error}", flush=True)

        for k in _chunk_iter:
            self._evoke_progress_chunk_index = k
            self._evoke_progress_chunks = num_latent_sections
            _t_chunk0 = _ptime.time()
            _t_warp_s = None      # None = phase did not run for this chunk (t2v, or warp disabled)
            _t_diff_s = None
            self._geo_last_render_stats = None   # else an event chunk reports the previous chunk's warp
            _emit_progress("geometry", chunk_index=k, stage=0, step=0, total_steps=num_inference_steps)
            if use_interpolate_prompt:
                assert num_latent_sections >= max(interpolate_cumulative_list)

                current_interval_idx = 0
                for idx, cumulative_val in enumerate(interpolate_cumulative_list):
                    if k < cumulative_val:
                        current_interval_idx = idx
                        break

                if current_interval_idx == 0:
                    prompt_embeds = all_prompt_embeds[0].unsqueeze(0)
                else:
                    interval_start = interpolate_cumulative_list[current_interval_idx - 1]
                    position_in_interval = k - interval_start

                    if position_in_interval < interpolation_steps:
                        if interpolate_embeds is None or interpolate_interval_idx != current_interval_idx:
                            interpolate_embeds = self.interpolate_prompt_embeds(
                                prompt_embeds_1=all_prompt_embeds[current_interval_idx - 1].unsqueeze(0),
                                prompt_embeds_2=all_prompt_embeds[current_interval_idx].unsqueeze(0),
                                interpolation_steps=interpolation_steps,
                            )
                            interpolate_interval_idx = current_interval_idx

                        prompt_embeds = interpolate_embeds[position_in_interval]
                    else:
                        prompt_embeds = all_prompt_embeds[current_interval_idx].unsqueeze(0)
            else:
                prompt_embeds = all_prompt_embeds

            # Per-chunk skill/VFX event flag. False for every chunk when event_chunks is empty/unset,
            # so all event hooks below collapse to the original code path (baseline byte-for-byte).
            is_event_k = (k in _event_set)
            # EVENT = a camera-control gap (A6'): skill/VFX frames cannot be DA3-estimated -> they do not enter the frame bank, and they do NOT advance the camera trajectory cursor.
            #   after the skill, poses are indexed by the "real (non-event) chunk count" _geo_pose_k so the trajectory resumes from the pre-skill pose; otherwise the target poses skip
            #   a whole stretch of trajectory and desync from the lagging bank -> warp goes fully black after the skill (the user-reported bug).
            #   with event_chunks empty, _geo_pose_k == k -> the rollout path is byte-for-byte unchanged.
            _geo_pose_k = k - sum(1 for _e in _event_set if _e < k)
            # SEGMENT PROMPT: the schedule entry in force at chunk k (last switch point <= k).
            #   Text embedding only -- warp / camera / frame bank stay on the baseline path.
            if _chunk_prompt_embeds:
                _sw = [_c for _c in _chunk_prompt_embeds if _c <= k]
                if _sw:
                    prompt_embeds = _chunk_prompt_embeds[max(_sw)]
            # EVENT PROMPT: swap in the cached event-prompt embeds for this chunk only (takes priority).
            if is_event_k and _event_prompt_embeds is not None:
                prompt_embeds = _event_prompt_embeds

            is_first_section = k == 0
            is_second_section = k == 1
            # Whether to use raw sink frames (x0 + memory last-frame raw view).
            _use_raw_sink_frames = bool(getattr(self.transformer.config, "use_raw_sink_frames", False))
            sink_latents = None
            nearby_sink_latents = None
            if is_keep_x0:
                if is_first_section:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                        device=device,
                        dtype=torch.float32,
                    )
                    if fake_image_latents is not None:
                        history_latents_first_section = torch.cat(
                            [history_latents_first_section, fake_image_latents], dim=2
                        )
                    if video_latents is not None:
                        history_frames = history_latents_first_section.shape[2]
                        video_frames = video_latents.shape[2]
                        if video_frames < history_frames:
                            keep_frames = history_frames - video_frames
                            history_latents_first_section = torch.cat(
                                [history_latents_first_section[:, :, :keep_frames, :, :], video_latents], dim=2
                            )
                        else:
                            history_latents_first_section = video_latents

                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    if _use_raw_sink_frames:
                        # prefix handled by sink rope; short tier contains only the 1x segment.
                        indices_latents_history_short = indices_latents_history_1x
                    else:
                        indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    latents_history_long, latents_history_mid = self._geo_maybe_noise_invisible_history(
                        latents_history_long, latents_history_mid, video_latents,
                    )
                    if image_latents is not None:
                        latents_prefix = image_latents
                    if _use_raw_sink_frames:
                        # x0 as first sink; last memory frame as nearby sink.
                        sink_latents = latents_prefix
                        latents_history_short = latents_history_1x
                        nearby_sink_latents = latents_history_short[:, :, -1:, :, :].clone()
                    else:
                        latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    if _use_raw_sink_frames:
                        indices_latents_history_short = indices_latents_history_1x
                    else:
                        indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix = image_latents
                    # NOTE (event/skill limitation): event-chunk frames still leak into mid/long here via the
                    # rolling history_latents cache; A6 (skip frame_bank) only stops event frames from polluting
                    # the WARP retrieval source, not the latent history. Mid/long exclusion is intentionally NOT done.
                    latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_long, latents_history_mid = self._geo_maybe_noise_invisible_history(
                        latents_history_long, latents_history_mid, video_latents,
                    )
                    if _use_raw_sink_frames:
                        sink_latents = latents_prefix
                        latents_history_short = latents_history_1x
                        nearby_sink_latents = latents_history_short[:, :, -1:, :, :].clone()
                    else:
                        latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                indices = torch.arange(0, sum([*history_sizes, latent_window_size]))
                (
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_short,
                    indices_hidden_states,
                ) = indices.split([*history_sizes, latent_window_size], dim=0)
                # NOTE (event/skill limitation): event-chunk frames still leak into mid/long via the rolling
                # history_latents cache; A6 only prevents warp-source pollution, not latent-history pollution.
                latents_history_long, latents_history_mid, latents_history_short = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)
                if _use_raw_sink_frames:
                    # No x0 source; skip first sink but still provide nearby sink from short last frame.
                    nearby_sink_latents = latents_history_short[:, :, -1:, :, :].clone()

            # GEO short-tier override; visibility masks default to None (baseline).
            history_visible_mask_short = None
            history_visible_mask_mid = None
            history_visible_mask_long = None
            # nearby_sink_indices explicitly set in GEO+raw_sink path (default=None uses transformer fallback).
            nearby_sink_indices = None
            _call_attention_kwargs = attention_kwargs
            if use_geometric_state:
                assert geo_state is not None
                assert is_keep_x0, "use_geometric_state requires is_keep_x0=True (it depends on the image prefix)"

                # Pixel stride = 36 (9 latents * 4), window length = 33.
                _geo_pix_stride = latent_window_size * self.vae_scale_factor_temporal       # 36
                _geo_window_pix = window_num_frames                                          # 33
                _geo_pix_start = geo_ref_pose_offset_pix + _geo_pose_k * _geo_pix_stride   # _geo_pose_k: events do not advance the cursor -> resume after the skill
                _geo_pix_count = _geo_window_pix                                              # 33 (always)
                _geo_pix_end = _geo_pix_start + _geo_pix_count

                if is_first_section and geo_ref_pose_offset_pix > 0:
                    print(
                        f"[pipe] GEO chunk0 anchor={max(0, geo_ref_pose_offset_pix - 1)} "
                        f"target=[{_geo_pix_start}:{_geo_pix_end}) pose_len={geo_c2ws_dev.shape[1]}",
                        flush=True,
                    )

                # Use geo_c2ws_dev (not cam_c2ws_dev, which requires full camera control setup).
                _geo_pose_len = geo_c2ws_dev.shape[1]
                if _geo_pix_start >= _geo_pose_len:
                    _geo_pose_window = geo_c2ws_dev[0, -1:].expand(_geo_pix_count, -1, -1)
                elif _geo_pix_end > _geo_pose_len:
                    _geo_valid = _geo_pose_len - _geo_pix_start
                    _geo_tail = geo_c2ws_dev[0, -1:].expand(_geo_pix_count - _geo_valid, -1, -1)
                    _geo_pose_window = torch.cat([geo_c2ws_dev[0, _geo_pix_start:_geo_pose_len], _geo_tail], dim=0)
                else:
                    _geo_pose_window = geo_c2ws_dev[0, _geo_pix_start:_geo_pix_end]

                if is_first_section:
                    # chunk-0 source: use last ref frame (v2v) or user image (i2v/t2v).
                    if video is not None and isinstance(video, torch.Tensor) and video.ndim == 5 and int(cam_video_lat_offset) > 0:
                        _geo_source = video[:, :, -1].to(device=device, dtype=torch.float32)  # [B, 3, H, W]
                        # Anchor pose = last ref frame (pixel idx = ref_pix - 1).
                        _geo_anchor_pix_idx = max(0, geo_ref_pose_offset_pix - 1)
                        _geo_anchor_c2w = geo_c2ws_dev[0, min(_geo_anchor_pix_idx, geo_c2ws_dev.shape[1] - 1)]
                    else:
                        _geo_source = geo_state["source_image_pixel"].to(device=device, dtype=torch.float32)
                        _geo_anchor_c2w = geo_c2ws_dev[0, 0]
                else:
                    _geo_source = geo_state["prev_chunk_last_frame"].to(device=device, dtype=torch.float32)
                    # Anchor = last decoded pixel of previous chunk (causal VAE: (k-1)*36 + 32).
                    _geo_anchor_pix_idx = max(
                        0,
                        geo_ref_pose_offset_pix + (_geo_pose_k - 1) * _geo_pix_stride + (_geo_window_pix - 1),
                    )
                    _geo_anchor_c2w = geo_c2ws_dev[0, min(_geo_anchor_pix_idx, geo_c2ws_dev.shape[1] - 1)]

                # EVENT/SKILL static camera (A4): freeze the chunk's pose window to the anchor pose so the
                # plucker (added unconditionally to noise tokens at transformer_evoke.py:~1479) is STATIC during
                # the skill ("camera not moving"). Non-event chunks keep their original per-frame _geo_pose_window.
                if is_event_k:
                    _geo_pose_window = _geo_anchor_c2w.unsqueeze(0).expand(_geo_pix_count, -1, -1)

                _H_lat = height // self.vae_scale_factor_spatial
                _W_lat = width // self.vae_scale_factor_spatial

                # Spatial visibility-aware sigma; config injected via self._geo_vsnoise_cfg.
                _vsnoise_cfg = getattr(self, "_geo_vsnoise_cfg", None) or {}
                _vsnoise_enabled = bool(_vsnoise_cfg.get("enabled", False))
                _vsnoise_sigma_invisible = float(_vsnoise_cfg.get("sigma_invisible", 0.8))
                _vsnoise_sigma_min = float(_vsnoise_cfg.get("sigma_min", 0.111))
                _vsnoise_sigma_max = float(_vsnoise_cfg.get("sigma_max", 0.135))

                _t_warp0 = _ptime.time()   # phase 1 of the chunk: warp render + encode to latents
                if is_event_k:
                    # EVENT/SKILL drop warp (A5): skip the warp render compute and feed a ZERO warp tier with an
                    # all-INVISIBLE mask (matches training online_materialize.py:~789). Downstream
                    # filter_history_tokens_by_mask removes the warp tokens, collapsing the short tier to
                    # [prefix | prev_short]. The static plucker from A4 still flows independently.
                    _geo_warp_video = torch.zeros(1, 3, latent_window_size * self.vae_scale_factor_temporal,
                                                  height, width, device=device, dtype=torch.float32)
                    _geo_visibility_mask = torch.zeros(1, 1, latent_window_size * self.vae_scale_factor_temporal,
                                                       _H_lat, _W_lat, device=device, dtype=torch.float32)
                    _geo_warp_latents = torch.zeros(
                        batch_size, num_channels_latents, latent_window_size, _H_lat, _W_lat,
                        device=device, dtype=torch.float32,
                    )
                else:
                    _geo_warp_video, _geo_visibility_mask = self._geo_render_chunk(
                        geo_state=geo_state, chunk_idx=k, source_frame_pix=_geo_source,
                        camera_poses_pix_window=_geo_pose_window,
                        height=height, width=width, device=device,
                        anchor_c2w=_geo_anchor_c2w,
                        top_k=int(geo_top_k),
                        nearby_k=int(geo_nearby_k),
                        select_k=int(geo_select_k) if geo_select_k is not None else None,
                        metric=str(geo_score),
                        metric_kwargs=geo_score_kwargs or {},
                    )
                    _geo_warp_latents = self._geo_encode_warp_to_latents(
                        warp_video=_geo_warp_video,
                        latents_mean=latents_mean, latents_std=latents_std,
                        latent_window_size=latent_window_size,
                        device=device, generator=generator,
                        sigma_min=_vsnoise_sigma_min,
                        sigma_max=_vsnoise_sigma_max,
                        visibility_aware_noise=_vsnoise_enabled,
                        sigma_invisible=_vsnoise_sigma_invisible,
                        visibility_mask_pix=_geo_visibility_mask if _vsnoise_enabled else None,
                    )

                _t_warp_s = _ptime.time() - _t_warp0

                # rope_alignment / warp_rope_mode control warp vs target RoPE idx assignment.
                _vsnoise_cfg_for_rope = getattr(self, "_geo_vsnoise_cfg", None) or {}
                _rope_align = bool(_vsnoise_cfg_for_rope.get("rope_alignment", True))
                _prefix_idx_mode = str(_vsnoise_cfg_for_rope.get("prefix_idx_mode", "zero"))
                _warp_rope_mode = str(_vsnoise_cfg_for_rope.get("warp_rope_mode", "overlap_noise"))

                if _warp_rope_mode == "before_prev_short":
                    # Plan 16: warp takes the prev_short slot [s..s+W-1]; prev_short -> noise[0]-1; noise += W; prefix stays 0.
                    assert not _rope_align, "warp_rope_mode='before_prev_short' requires rope_alignment=False."
                    assert _prefix_idx_mode == "zero", "warp_rope_mode='before_prev_short' requires prefix_idx_mode='zero'."
                    assert int(indices_latents_history_1x.numel()) == 1, (
                        "warp_rope_mode='before_prev_short' only supports history_sizes[-1]==1 "
                        f"(got 1x tier numel={int(indices_latents_history_1x.numel())})."
                    )
                    _prev_short_idx_old = int(indices_latents_history_1x.flatten()[0].item())
                    _W_p16 = int(latent_window_size)
                    _geo_warp_indices = torch.arange(
                        _prev_short_idx_old, _prev_short_idx_old + _W_p16,
                        dtype=indices_hidden_states.dtype, device=indices_hidden_states.device,
                    )
                    # prev_short pushed to noise[0]-1; noise shifted +W.
                    indices_latents_history_1x = torch.tensor(
                        [_prev_short_idx_old + _W_p16],
                        dtype=indices_latents_history_1x.dtype, device=indices_latents_history_1x.device,
                    )
                    indices_hidden_states = indices_hidden_states + _W_p16
                elif _warp_rope_mode == "before_prev_mid":
                    raise NotImplementedError(
                        "warp_rope_mode='before_prev_mid' (Plan 22) is not ported to this codebase; "
                        "use 'overlap_noise' or 'before_prev_short'."
                    )
                elif _rope_align:
                    # warp idx = target idx (overlap)
                    _geo_warp_indices = indices_hidden_states.flatten()
                else:
                    # warp takes target idx position; target shifts by W
                    _geo_warp_indices = indices_hidden_states.flatten().clone()
                    _W_offset = int(latent_window_size)
                    indices_hidden_states = indices_hidden_states + _W_offset

                # prefix_idx_mode: "zero" (default) or "adjacent".
                if _prefix_idx_mode == "adjacent":
                    # prefix idx = noise[0] - 1 (clamped to 0).
                    _noise_first = int(indices_hidden_states.flatten()[0].item())
                    _prefix_idx_val = max(0, _noise_first - 1)
                    indices_prefix = torch.tensor(
                        [_prefix_idx_val], dtype=indices_prefix.dtype, device=indices_prefix.device
                    )

                # Short-tier noise config (injected via self._short_tier_noise_cfg).
                _short_noise_cfg = getattr(self, "_short_tier_noise_cfg", None) or {}
                _apply_short_noise = bool(_short_noise_cfg.get("enabled", False)) and bool(_short_noise_cfg.get("apply_at_inference", True))
                _short_noise_min = float(_short_noise_cfg.get("sigma_min", 0.2))
                _short_noise_max = float(_short_noise_cfg.get("sigma_max", 0.6))
                _short_noise_targets = set(_short_noise_cfg.get("target_tiers", []) or [])
                # Per-rollout locked sigma (if set); otherwise sampled independently per chunk.
                _rollout_locked_sigma = getattr(self, "_short_tier_rollout_sigma", None)

                def _corrupt_clean_latent(clean: torch.Tensor, tier_name: str) -> torch.Tensor:
                    """Add sigma noise to a clean latent if the tier is in target_tiers."""
                    if (not _apply_short_noise) or (not _short_noise_targets) or (tier_name not in _short_noise_targets):
                        return clean
                    _gen = generator[0] if isinstance(generator, list) else generator
                    if _rollout_locked_sigma is not None:
                        sigma_scalar = float(_rollout_locked_sigma)
                    else:
                        sigma_scalar = torch.rand(1, device=clean.device, generator=_gen).item() * (_short_noise_max - _short_noise_min) + _short_noise_min
                    sigmas = torch.full((1, 1, 1, 1, 1), sigma_scalar, device=clean.device, dtype=clean.dtype)
                    _noise = randn_tensor(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
                    out = sigmas * _noise + (1 - sigmas) * clean
                    # Rate-limited print (first 3 per rollout).
                    _pcount = getattr(self, "_short_tier_print_count", 0)
                    if _pcount < 3:
                        print(f"[ShortTier-Noise INFER] σ={sigma_scalar:.3f} applied to {tier_name} (lock={_rollout_locked_sigma is not None})", flush=True)
                        self._short_tier_print_count = _pcount + 1
                    return out

                # Save clean prefix copy before noise injection (used for raw_sink which should stay clean).
                _latents_prefix_clean = latents_prefix.clone()

                # short tier physical layout: [prefix | warp | prev_short] — matches RoPE order
                # (prefix < warp < prev_short < noise). prev_short is the frame CLOSEST to noise and goes LAST.
                _short_lat_parts = [_corrupt_clean_latent(latents_prefix, "prefix")]
                _short_idx_parts = [indices_prefix.flatten() if hasattr(indices_prefix, 'ndim') and indices_prefix.ndim > 1 else indices_prefix]
                # warp goes in the MIDDLE (after prefix, before prev_short).
                _short_lat_parts.append(_geo_warp_latents)
                _short_idx_parts.append(_geo_warp_indices)
                # prev_short (continuity anchor) goes LAST = closest to noise. ALWAYS present:
                #   chunk>=1   -> last decoded latent of the previous chunk
                #   v2v chunk0 -> last video latent (ref-video tail)
                #   i2v chunk0 -> ref image latent (= prefix content); the first generated chunk connects to it.
                # RoPE index is indices_latents_history_1x[-1:] (= noise[0]-1 after the before_prev_short shift).
                # geo_disable_prev_short (diagnostic) skips prev_short entirely -> short tier = [prefix | warp].
                _geo_has_prev_short = False
                if not geo_disable_prev_short:
                    if not is_first_section:
                        _geo_prev_short_lat = latents_history_1x[:, :, -1:, :, :].clone()
                    elif video_latents is not None and int(video_latents.shape[2]) > 0:
                        _geo_prev_short_lat = video_latents[:, :, -1:, :, :].clone()
                    else:
                        # i2v chunk0: prev_short = ref image latent (= prefix), continuity anchor for the first chunk.
                        _geo_prev_short_lat = latents_prefix[:, :, -1:, :, :].clone()
                    _short_lat_parts.append(_corrupt_clean_latent(_geo_prev_short_lat, "prev_short"))
                    _short_idx_parts.append(indices_latents_history_1x.flatten()[-1:])
                    _geo_has_prev_short = True

                latents_history_short = torch.cat(_short_lat_parts, dim=2)
                indices_latents_history_short = torch.cat(_short_idx_parts, dim=0)

                # Build short-tier visibility mask.
                history_visible_mask_short = self._geo_build_short_visibility_mask(
                    visibility_mask_pix=_geo_visibility_mask,
                    num_lat_per_chunk=latent_window_size,
                    has_prev_short=_geo_has_prev_short,
                    H_lat=_H_lat, W_lat=_W_lat,
                    device=device, vae_t_stride=self.vae_scale_factor_temporal,
                )

                # GEO + raw_sink: provide explicit sink content and indices.
                if _use_raw_sink_frames:
                    sink_latents = _latents_prefix_clean  # first sink (clean, idx=0 assigned in transformer)
                    if is_first_section:
                        nearby_sink_latents = _latents_prefix_clean.clone()
                        nearby_sink_indices = torch.zeros(1, 1, dtype=torch.long, device=device)
                    else:
                        # nearby_sink = last decoded latent from previous chunk (not warp tail).
                        assert geo_state["prev_chunk_last_decoded_latent"] is not None, (
                            "GEO + raw_sink chunk>=1 requires state[prev_chunk_last_decoded_latent], "
                            "which should have been stored after the previous chunk's decode. Check that vae_decode_type is default / persistent on the GEO path."
                        )
                        nearby_sink_latents = geo_state["prev_chunk_last_decoded_latent"].clone()
                        nearby_sink_indices = indices_latents_history_1x.flatten()[-1:].unsqueeze(0).to(device=device)

                # Visibility token threshold (default 0.1); configurable via _geo_vsnoise_cfg.
                _vsnoise_cfg_for_thr = getattr(self, "_geo_vsnoise_cfg", None) or {}
                _vis_token_threshold = float(_vsnoise_cfg_for_thr.get("visible_token_threshold", 0.1))
                _call_attention_kwargs = dict(attention_kwargs or {})
                _call_attention_kwargs["history_visible_token_threshold"] = _vis_token_threshold
                # Short tier layout [prefix | warp(W) | prev_short(1)]. warp_mlp (and synchronized per-stage
                # compression) apply ONLY to the middle W warp frames; prefix(leading 1) + prev_short(trailing 1)
                # = uncompressed anchor. prev_short is always present (i2v chunk0 -> ref image), so Sp=1 always.
                _call_attention_kwargs["geo_warp_frames"] = int(latent_window_size)
                _call_attention_kwargs["geo_prev_short_frames"] = 0 if geo_disable_prev_short else 1
                # warp_rope_noise_center_align: derive from the GEO cfg so in-training VAL applies it (train/val
                # consistency); a caller-passed attention_kwargs value wins (setdefault). Inference (CLI / infer
                # script) passes it directly in attention_kwargs.
                if bool(_vsnoise_cfg_for_thr.get("warp_rope_noise_center_align", False)):
                    _call_attention_kwargs.setdefault("warp_rope_noise_center_align", True)

            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            if not is_enable_stage2:
                self.scheduler.set_timesteps(num_inference_steps, mu=1, device=device)

                if use_dynamic_shifting:
                    sigmas = torch.linspace(
                        0.999, 0.0, steps=num_inference_steps + 1, dtype=torch.float32, device=device
                    )[:-1]
                    sigmas = apply_schedule_shift(
                        sigmas=sigmas,
                        noise=latents,
                        base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                        max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                        base_shift=self.scheduler.config.get("base_shift", 0.5),
                        max_shift=self.scheduler.config.get("max_shift", 1.15),
                        time_shift_type=time_shift_type,
                    )
                    timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)
                    timesteps = timesteps.to(device)
                    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
                    self.scheduler.timesteps = timesteps
                    self.scheduler.sigmas = sigmas

                timesteps = self.scheduler.timesteps

                dmd_sigmas = None
                dmd_timesteps = None
                if use_dmd:
                    dmd_sigmas = self.scheduler.sigmas.to(self.transformer.device)
                    dmd_timesteps = self.scheduler.timesteps.to(self.transformer.device)

                self._num_timesteps = len(timesteps)
            else:
                num_inference_steps = (
                    sum(stage2_num_inference_steps_list) * 2
                    if is_amplify_first_chunk and use_dmd and is_first_section
                    else sum(stage2_num_inference_steps_list)
                )

            _t_diff0 = _ptime.time()   # phase 2 of the chunk: the denoising steps themselves
            _emit_progress("denoising", chunk_index=k, stage=0, step=0, total_steps=num_inference_steps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                if is_enable_stage2:
                    # Slice per-chunk c2ws using pixel-domain offset; stage2 recomputes plucker per stage.
                    _cam_c2ws_chunk = None
                    _cam_Ks_chunk = None
                    if cam_c2ws_dev is not None:
                        _s_pix = geo_ref_pose_offset_pix + _geo_pose_k * latent_window_size * self.vae_scale_factor_temporal   # events do not advance the cursor: plucker uses the same convention as the GEO warp (resume after the skill)
                        _e_pix = _s_pix + (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
                        _window_pix_len = _e_pix - _s_pix  # = (W-1)*stride + 1
                        _pose_len = cam_c2ws_dev.shape[1]
                        if _s_pix >= _pose_len:
                            # Entire window out of range: clamp to last pose.
                            _cam_c2ws_chunk = cam_c2ws_dev[:, -1:].expand(-1, _window_pix_len, -1, -1)
                        elif _e_pix > _pose_len:
                            # Partial overlap: valid segment + repeated last pose to fill window.
                            _valid_n = _pose_len - _s_pix
                            _tail_n = _window_pix_len - _valid_n
                            _tail = cam_c2ws_dev[:, -1:].expand(-1, _tail_n, -1, -1)
                            _cam_c2ws_chunk = torch.cat([cam_c2ws_dev[:, _s_pix:_pose_len], _tail], dim=1)
                        else:
                            _cam_c2ws_chunk = cam_c2ws_dev[:, _s_pix:_e_pix]
                        # EVENT/SKILL static camera (A4): freeze the plucker cam poses to the window's first pose
                        # so the stage2 plucker is static during the skill. No-op for non-event chunks.
                        if is_event_k and _cam_c2ws_chunk is not None:
                            _cam_c2ws_chunk = _cam_c2ws_chunk[:, :1].expand(-1, _cam_c2ws_chunk.shape[1], -1, -1)
                        _cam_Ks_chunk = cam_Ks_dev
                    latents = self.stage2_sample(
                        latents=latents,
                        stage2_num_stages=stage2_num_stages,
                        stage2_num_inference_steps_list=stage2_num_inference_steps_list,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        sink_latents=sink_latents,
                        nearby_sink_latents=nearby_sink_latents,
                        nearby_sink_indices=nearby_sink_indices,
                        # GEO visibility filter
                        history_visible_mask_short=history_visible_mask_short,
                        history_visible_mask_mid=history_visible_mask_mid,
                        history_visible_mask_long=history_visible_mask_long,
                        # GEO LoRA
                        geo_lora_active=geo_lora_active,
                        # Geo: inject warp only at coarsest pyramid stage (reference-aligned). Off by default.
                        geo_warp_stage0_only=bool(
                            (getattr(self, "_geo_vsnoise_cfg", None) or {}).get("geo_warp_stage0_only", False)
                        ),
                        # Camera control
                        cam_lingbot_Ks_chunk=_cam_Ks_chunk,
                        cam_lingbot_c2ws_chunk=_cam_c2ws_chunk,
                        cam_base_height_pix=cam_base_h_used,
                        cam_base_width_pix=cam_base_w_used,
                        cam_pc_resolution_strategy=cam_pc_resolution_strategy,
                        attention_kwargs=_call_attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        scheduler_type=scheduler_type,
                        use_dynamic_shifting=use_dynamic_shifting,
                        time_shift_type=time_shift_type,
                        generator=generator,
                        # ------------ CFG Zero ------------
                        use_cfg_zero_star=use_cfg_zero_star,
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        # -------------- DMD --------------
                        use_dmd=use_dmd,
                        is_amplify_first_chunk=is_amplify_first_chunk and is_first_section,
                        # Callback
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )
                else:
                    # Slice per-section c2w window using pixel-domain offset; compute plucker per window.
                    _cam_emb_section = None
                    if cam_c2ws_dev is not None and cam_Ks_dev is not None:
                        _s_pix = geo_ref_pose_offset_pix + _geo_pose_k * latent_window_size * self.vae_scale_factor_temporal   # events do not advance the cursor: plucker uses the same convention as the GEO warp (resume after the skill)
                        _e_pix = _s_pix + (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
                        _window_pix_len = _e_pix - _s_pix
                        _pose_len = cam_c2ws_dev.shape[1]
                        if _s_pix >= _pose_len:
                            _cam_c2ws_window = cam_c2ws_dev[:, -1:].expand(-1, _window_pix_len, -1, -1)
                        elif _e_pix > _pose_len:
                            _valid_n = _pose_len - _s_pix
                            _tail_n = _window_pix_len - _valid_n
                            _tail = cam_c2ws_dev[:, -1:].expand(-1, _tail_n, -1, -1)
                            _cam_c2ws_window = torch.cat(
                                [cam_c2ws_dev[:, _s_pix:_pose_len], _tail], dim=1
                            )
                        else:
                            _cam_c2ws_window = cam_c2ws_dev[:, _s_pix:_e_pix]
                        # EVENT/SKILL static camera (A4): freeze the plucker cam poses to the window's first pose
                        # so the stage1 plucker is static during the skill. No-op for non-event chunks.
                        if is_event_k:
                            _cam_c2ws_window = _cam_c2ws_window[:, :1].expand(-1, _cam_c2ws_window.shape[1], -1, -1)
                        from evoke.modules.camera_control import prepare_cam_plucker_emb
                        _cam_emb_section = prepare_cam_plucker_emb(
                            Ks=cam_Ks_dev,
                            c2ws_window=_cam_c2ws_window,
                            H_pix=height, W_pix=width,
                            base_h_pix=cam_base_h_used, base_w_pix=cam_base_w_used,
                            vae_stride_t=self.vae_scale_factor_temporal,
                            vae_stride_h=self.vae_scale_factor_spatial,
                            vae_stride_w=self.vae_scale_factor_spatial,
                            strategy=cam_pc_resolution_strategy,
                        ).to(dtype=transformer_dtype)
                    latents = self.stage1_sample(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        timesteps=timesteps,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        sink_latents=sink_latents,
                        nearby_sink_latents=nearby_sink_latents,
                        nearby_sink_indices=nearby_sink_indices,
                        # GEO visibility filter
                        history_visible_mask_short=history_visible_mask_short,
                        history_visible_mask_mid=history_visible_mask_mid,
                        history_visible_mask_long=history_visible_mask_long,
                        cam_plucker_emb=_cam_emb_section,
                        attention_kwargs=_call_attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        generator=generator,
                        # ------------ CFG Zero ------------
                        use_cfg_zero_star=use_cfg_zero_star,
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        # -------------- DMD --------------
                        use_dmd=use_dmd,
                        dmd_sigmas=dmd_sigmas,
                        dmd_timesteps=dmd_timesteps,
                        is_amplify_first_chunk=is_amplify_first_chunk and is_first_section,
                        # Callback
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )

                _t_diff_s = _ptime.time() - _t_diff0

                _emit_progress(
                    "decoding", chunk_index=k, stage=stage2_num_stages if is_enable_stage2 else 1,
                    step=num_inference_steps, total_steps=num_inference_steps,
                )
                # Return fragmented denoiser workspace to CUDA before the VAE requests its largest
                # contiguous activation. This is especially important on 48 GiB cards where the two
                # phases otherwise miss each other by only a few dozen MiB.
                torch.cuda.empty_cache()

                if use_kv_cache:
                    self.transformer.clear_kv_cache()

                if use_adaptive_anti_drifting:
                    current_mean, current_var = anti_drifting_helper.compute_latent_statistics(latents)
                    anti_drifting_helper.update_global_statistics(current_mean, current_var)
                    has_drift = anti_drifting_helper.detect_drift(current_mean, current_var)

                    if has_drift and k < num_latent_sections - 1:
                        print(
                            f"Drift detected at chunk {k + 1}/{num_latent_sections}. Applying Frame-Aware Corruption."
                        )
                        latents = anti_drifting_helper.apply_frame_aware_corruption(
                            latents,
                            corruption_strength=anti_drift_corruption_strength,
                            generator=generator,
                        )

                if is_keep_x0 and (
                    (is_first_section and image_latents is None) or (is_skip_first_section and is_second_section)
                ):
                    image_latents = latents[:, :, 0:1, :, :]

                total_generated_latent_frames += latents.shape[2]
                history_latents = torch.cat([history_latents, latents], dim=2)
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
                index_slice = (
                    slice(None),
                    slice(None),
                    slice(-latent_window_size, None),
                )

                if vae_decode_type in ("default", "default_warm0"):
                    # default_warm0: chunk0 uses the warm-cache decode (avoids the cold-start over-saturated
                    # I-frame on the first shown frame), chunks k>=1 use plain fresh vae.decode (no cross-chunk
                    # cache carry -> no per-chunk-first-frame flicker). Best of persistent(chunk0) + default(k>=1).
                    if vae_decode_type == "default_warm0" and k == 0 and image_latents is not None:
                        current_video = self._decode_chunk_persistent_cache(
                            z_chunk=real_history_latents[index_slice],
                            is_first_chunk=True,
                            latents_mean=latents_mean,
                            latents_std=latents_std,
                            vae_dtype=vae_dtype,
                            warm_latents=_v2v_decode_warm_latents if _v2v_decode_warm_latents is not None else image_latents,
                            warm_repeat=getattr(self, "_geo_first_chunk_decode_warm", 4),
                            warm_as_prior=_warm_is_real_prior,
                        )
                        # Do NOT carry the cache to chunk1 (k>=1 uses plain vae.decode which clears it anyway).
                        self._geo_persist_feat_map = None
                    else:
                        current_latents = real_history_latents[index_slice].to(vae_dtype) / latents_std + latents_mean
                        current_video = self.vae.decode(current_latents, return_dict=False)[0]

                    if stream_output:
                        pass                       # streaming: use this chunk only (the downstream hook already wrote it out), no accumulation
                    elif history_video is None:
                        history_video = current_video
                    else:
                        history_video = torch.cat([history_video, current_video], dim=2)
                    _geo_chunk_decoded = current_video
                elif vae_decode_type == "persistent":
                    new_pixels = self._decode_chunk_persistent_cache(
                        z_chunk=real_history_latents[index_slice],
                        is_first_chunk=(k == 0),
                        latents_mean=latents_mean,
                        latents_std=latents_std,
                        vae_dtype=vae_dtype,
                        # First-chunk cold-start fix: warm the empty feat cache with the clean image/prefix
                        # latent so the first decoded frame isn't an over-saturated I-frame. No-op when
                        # image_latents is None (t2v) or k>0 (cache already warm). See EXP.
                        warm_latents=_v2v_decode_warm_latents if _v2v_decode_warm_latents is not None else image_latents,
                        warm_repeat=getattr(self, "_geo_first_chunk_decode_warm", 4),
                        warm_as_prior=_warm_is_real_prior,
                    )
                    if stream_output:
                        pass                       # streaming: same as above, history_video stays None
                    elif history_video is None:
                        history_video = new_pixels
                    else:
                        history_video = torch.cat([history_video, new_pixels], dim=2)
                    _geo_chunk_decoded = new_pixels
                else:
                    _geo_chunk_decoded = None

                # GEO post-chunk: update last frame, latent cache, and FrameBank.
                if use_geometric_state and geo_state is not None:
                    assert _geo_chunk_decoded is not None, (
                        f"use_geometric_state requires vae_decode_type='default' or 'persistent' (in-chunk decode), "
                        f"got '{vae_decode_type}'."
                    )
                    # Quantize last decoded frame for FrameBank storage.
                    _geo_boundary = self._geo_quantize_8bit(_geo_chunk_decoded[:, :, -1])  # [1,3,H,W]
                    geo_state["prev_chunk_last_frame"] = _geo_boundary

                    # Add N decoded frames to FrameBank using 36-stride pixel indices.
                    _geo_num_pix = int(_geo_chunk_decoded.shape[2])
                    _geo_pix_stride_add = latent_window_size * self.vae_scale_factor_temporal   # 36
                    _geo_chunk_pixel_start = geo_ref_pose_offset_pix + _geo_pose_k * _geo_pix_stride_add   # events do not advance the cursor (gid stays contiguous, no gaps)
                    _geo_add_n = int(getattr(self, "_geo_chunk_add_n", 1) or 1)
                    if _geo_add_n == 1:
                        _geo_frame_offsets = (_geo_num_pix - 1,)                                # last frame
                    elif _geo_add_n == 3:
                        _geo_frame_offsets = (0, _geo_num_pix // 2, _geo_num_pix - 1)           # first/mid/last
                    else:
                        # Uniformly distributed offsets for arbitrary N.
                        _geo_frame_offsets = tuple(
                            min(_geo_num_pix - 1, max(0, int(round(i * (_geo_num_pix - 1) / max(1, _geo_add_n - 1)))))
                            for i in range(_geo_add_n)
                        )
                    # EVENT/SKILL skip frame bank (A6): event-chunk (VFX) frames must NOT pollute the warp
                    # retrieval source used by later normal chunks, so skip the FrameBank.add offset-loop AND the
                    # DA3 recall ingest for event chunks. prev_chunk_last_frame / prev_chunk_last_decoded_latent
                    # are still updated below (immediate continuity anchor, not the bank).
                    if not is_event_k:
                        for offset in _geo_frame_offsets:
                            _geo_pixel_idx = _geo_chunk_pixel_start + offset
                            if geo_c2ws_dev is not None:
                                _geo_pose_idx = min(_geo_pixel_idx, geo_c2ws_dev.shape[1] - 1)
                                _geo_pose = geo_c2ws_dev[0, _geo_pose_idx]
                            else:
                                _geo_pose = torch.eye(4, device=device)
                            _geo_frame_b = self._geo_quantize_8bit(_geo_chunk_decoded[:, :, offset])
                            geo_state["frame_bank"].add(
                                frame=_geo_frame_b[0] if _geo_frame_b.ndim == 4 else _geo_frame_b,
                                c2w=_geo_pose,
                                chunk_idx=k,
                                pixel_idx=_geo_pixel_idx,
                            )
                    # Cache last latent for next chunk's nearby_sink (continuity, unchanged for event chunks).
                    geo_state["prev_chunk_last_decoded_latent"] = real_history_latents[:, :, -1:].clone()
                    # DA3 recall: ingest this chunk's decoded frames (gid=global pixel index) so later chunks can recall them.
                    # Skip for event chunks (A6): VFX frames must not enter the DA3 warp-recall bank either.
                    if (not is_event_k) and geo_state.get("da3_bank") is not None and geo_c2ws_dev is not None:
                        self._geo_da3_ingest(geo_state, _geo_chunk_decoded[0],
                                             int(_geo_chunk_pixel_start), geo_c2ws_dev, device)
                        # -- [oracle diagnostic] env EVOKE_ORACLE_REANCHOR_EVERY=N (>0): every N non-event chunks,
                        #    clear the DA3 cloud after this chunk's ingest and rebuild it from GT source video frames -> the next
                        #    chunk's warp renders from the GT cloud. Pure diagnostic cheating (real inference has no GT, never ships in a product recipe), quantifying the warp reconstruction quality ceiling:
                        # P-diagnostic row.
                        #    env unset/0 -> _orc_every=0 -> this block is a no-op, behaviour byte-for-byte unchanged. Helpers are at the end of the file.
                        _orc_every = _oracle_reanchor_every()
                        if _orc_every > 0:
                            _orc_bank = geo_state["da3_bank"]
                            _orc_cache = getattr(self, "_oracle_gt_cache", None)
                            if _orc_cache is None:
                                _orc_cache = self._oracle_gt_cache = {}
                            # GT availability bound = min(decodable 24fps target frames of the source video, pose tensor length);
                            # the relative_replay extrapolated tail (beyond the GT range) is skipped as out-of-range by _oracle_reanchor_maybe.
                            _orc_avail = _oracle_gt_avail_pix(_orc_cache, ref_video=video)
                            if _orc_avail is not None:
                                _orc_avail = min(int(_orc_avail), int(geo_c2ws_dev.shape[1]))
                            # the window length follows the v2v seed convention (= the ref section pixel frame count, usually 120); gid/stride/ingest
                            # all go through _geo_da3_ingest -> exactly the same indexing/scale convention as the seed, so render recall never notices the oracle.
                            _orc_win = int(geo_ref_pose_offset_pix) if int(geo_ref_pose_offset_pix) > 0 else 120

                            def _orc_evict():
                                _n = int(_orc_bank.num_frames)
                                if _orc_bank.pts:  # clearing = a hard re-anchor blood swap, avoiding GT/on-policy double-image geometry
                                    _orc_bank.evict_before(max(_orc_bank.pts.keys()) + 1)
                                return _n

                            def _orc_ingest(_fr, _a, _b, _ws):
                                # exactly like the v2v seed: feed 36-stride segments to _geo_da3_ingest (gid=absolute pixel index,
                                # pose taken from geo_c2ws_dev[gid] -- the same pose source as the seed/normal ingest, same umeyama scale convention)
                                self._geo_da3_ingest(geo_state, _fr[:, _a - _ws:_b - _ws],
                                                     _a, geo_c2ws_dev, device)

                            if _oracle_reanchor_maybe(
                                chunk_num=int(_geo_pose_k) + 1, every=_orc_every,
                                window_end=int(_geo_chunk_pixel_start) + int(_geo_num_pix),
                                window_pix=_orc_win, stride=int(_geo_pix_stride_add), gt_avail=_orc_avail,
                                fetch_fn=lambda _a, _b: _oracle_gt_fetch(
                                    _orc_cache, _a, _b, int(height), int(width), ref_video=video),
                                evict_fn=_orc_evict, ingest_fn=_orc_ingest,
                            ):
                                print(f"[oracle-reanchor] bank now {int(_orc_bank.num_frames)} GT frames "
                                      f"(the next chunk's warp renders from the GT cloud)", flush=True)

            # -- chunk k is done: one line for its phases, written through the bar so tqdm lifts the bar
            #    out of the way instead of letting the text land in the middle of it. Placed at the very
            #    end of the body so `total` is the same wall time the bar's s/chunk rate is built from;
            #    measuring right after the denoise loop would omit the VAE decode and the segment dump,
            #    which together outweigh both phases.
            if _prog_bar is not None:
                _t_total_s = _ptime.time() - _t_chunk0
                _st = getattr(self, "_geo_last_render_stats", None) or {}
                _warp_txt = "warp     -   " if _t_warp_s is None else f"warp {_t_warp_s:6.2f}s"
                if _st:
                    _warp_txt += f" (pool={_st.get('pool', '?'):>4} cov={_st.get('cov', float('nan')):.3f})"
                _rest_s = _t_total_s - (_t_warp_s or 0.0) - (_t_diff_s or 0.0)
                _prog_bar.write(
                    f"  chunk {k + 1:>3}/{num_latent_sections:<3} {_warp_txt}"
                    f"  |  diffusion {_t_diff_s:6.2f}s ({num_inference_steps} steps)"
                    f"  |  decode+dump {_rest_s:6.2f}s"
                    f"  |  total {_t_total_s:6.2f}s",
                    file=_psys.stdout)
            _emit_progress(
                "chunk_done", chunk_index=k, stage=stage2_num_stages if is_enable_stage2 else 1,
                step=num_inference_steps, total_steps=num_inference_steps,
            )

        self._current_timestep = None

        # Clear persistent VAE cache after chunk loop to prevent leaking into the next call.
        if vae_decode_type == "persistent":
            self.vae.clear_cache()

        if stream_output:
            # streaming: there is no accumulated history_video to post-process; the caller stitches the segments.
            print(f"[stream_output] history_video was not accumulated (frames=None); stitch the full video from the per-chunk "
                  f"segment mp4 files.", flush=True)
            video = None
        elif output_type != "latent":
            if use_geometric_state and bool(
                (getattr(self, "_geo_vsnoise_cfg", None) or {}).get("geo_oneshot_output_decode", False)
            ):
                # DIAGNOSTIC/fix: the per-chunk decode in the rollout above is only needed to feed the warp
                # SOURCE frame each chunk. For the OUTPUT video, re-decode the full accumulated latent
                # sequence in ONE shot = true temporal continuity (no per-chunk I-frame seam, no cross-chunk
                # cache-carry flicker). Tells us whether the boundary artifacts live in the decode mechanics
                # (one-shot clean) or the latents themselves (one-shot still seams). High VRAM on long rollouts.
                self.vae.clear_cache()
                full_latents = real_history_latents.to(vae_dtype) / latents_std + latents_mean
                history_video = self.vae.decode(full_latents, return_dict=False)[0]
            elif vae_decode_type == "long":
                # Decode all chunks at once for seamless temporal continuity (high VRAM usage).
                full_latents = real_history_latents.to(vae_dtype) / latents_std + latents_mean
                history_video = self.vae.decode(full_latents, return_dict=False)[0]

            elif vae_decode_type == "default_batch":
                total_latent_frames = real_history_latents.shape[2]
                batch_size = real_history_latents.shape[0]
                num_chunks = total_latent_frames // latent_window_size

                chunks = (
                    real_history_latents.reshape(
                        batch_size,
                        -1,
                        num_chunks,
                        latent_window_size,
                        real_history_latents.shape[-2],
                        real_history_latents.shape[-1],
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        batch_size * num_chunks,
                        -1,
                        latent_window_size,
                        real_history_latents.shape[-2],
                        real_history_latents.shape[-1],
                    )
                )

                chunks = chunks.to(vae_dtype) / latents_std + latents_mean
                batch_video = self.vae.decode(chunks, return_dict=False)[0]

                video_frames_per_chunk = batch_video.shape[2]
                history_video = (
                    batch_video.reshape(
                        batch_size,
                        num_chunks,
                        -1,
                        video_frames_per_chunk,
                        batch_video.shape[-2],
                        batch_video.shape[-1],
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        batch_size,
                        -1,
                        num_chunks * video_frames_per_chunk,
                        batch_video.shape[-2],
                        batch_video.shape[-1],
                    )
                )

            generated_frames = history_video.size(2)
            generated_frames = (
                generated_frames - 1
            ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            history_video = history_video[:, :, :generated_frames]
            if not _warm_is_real_prior and history_video.shape[2] > 1:
                history_video = history_video[:, :, 1:]
            video = self.video_processor.postprocess_video(history_video, output_type=output_type)
        else:
            video = real_history_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return EvokePipelineOutput(frames=video)


# ====================================================================================================
# [oracle diagnostic] EVOKE_ORACLE_REANCHOR_EVERY: periodically hard re-anchor the GEO DA3 cloud to GT source
#   video frames. This is a diagnostic cheat -- in real v2v inference only the conditioning prefix is GT. It
#   quantifies the ceiling of warp reconstruction quality: if warp/pred stay clean under the oracle, the
#   long-video degradation bottleneck is entirely in cloud reconstruction; if they still degrade, the model's
#   own drift dominates. Never ship it in a product recipe.
#   env (deliberately not wired to CLI/cfg):
#     EVOKE_ORACLE_REANCHOR_EVERY : int N; unset/0/invalid -> short-circuits, behaviour unchanged.
#     EVOKE_ORACLE_GT_VIDEO       : GT source video path (unset -> the pipeline's ref video tensor, which only
#                                   covers the ref section).
#     EVOKE_ORACLE_GT_FPS         : source fps (default 30; keep equal to --lingbot_pose_source_fps).
#     EVOKE_ORACLE_GT_START       : start_seconds (default 0.0; keep equal to --start_seconds).
#   trigger point: end of the chunk loop, after this chunk's on-policy ingest, so the next chunk's warp renders
#   from the GT cloud.
# ====================================================================================================
def _oracle_reanchor_every():
    """env EVOKE_ORACLE_REANCHOR_EVERY -> int N; unset/empty/invalid/negative -> 0 (= off, behaviour unchanged)."""
    import os as _os
    try:
        return max(0, int(_os.environ.get("EVOKE_ORACLE_REANCHOR_EVERY", "0") or "0"))
    except (TypeError, ValueError):
        return 0


def _oracle_gt_avail_pix(cache, ref_video=None):
    """Upper bound on the GT-available target (24fps) pixel frame count (= max re-anchorable gid + 1); unavailable -> None.
    if env EVOKE_ORACLE_GT_VIDEO is set -> open the source video and derive it (same fps resampling map as _load_test_clip, result cached);
    unset -> fall back to the length of the pipeline's own ref video tensor (covers the ref section only, hint printed once)."""
    import os as _os
    path = (_os.environ.get("EVOKE_ORACLE_GT_VIDEO", "") or "").strip()
    if not path:
        if ref_video is not None and isinstance(ref_video, torch.Tensor) and ref_video.ndim == 5:
            if not cache.get("_hint_ref_fallback"):
                cache["_hint_ref_fallback"] = True
                print(f"[oracle-reanchor] EVOKE_ORACLE_GT_VIDEO unset -> GT source falls back to the ref tensor "
                      f"(covers gid<{int(ref_video.shape[2])} only; every later trigger is skipped as out-of-range)", flush=True)
            return int(ref_video.shape[2])
        return None
    if "avail" not in cache:
        try:
            from video_reader import PyVideoReader
            src_fps = float(_os.environ.get("EVOKE_ORACLE_GT_FPS", "30") or 30.0)
            start_s = float(_os.environ.get("EVOKE_ORACLE_GT_START", "0") or 0.0)
            vr = PyVideoReader(path, threads=0)
            total_src = int(vr.get_shape()[0])
            start_src = int(round(start_s * src_fps))
            # max target frame count p_avail: for every p < p_avail, start_src + round(p*src_fps/24) < total_src
            # (byte-for-byte the same index mapping as ev_validation._load_test_clip).
            p = max(0, int((total_src - start_src) * 24.0 / src_fps) + 2)
            while p > 0 and start_src + round((p - 1) * src_fps / 24.0) >= total_src:
                p -= 1
            cache.update(vr=vr, total_src=total_src, src_fps=src_fps, start_src=start_src, avail=int(p))
            print(f"[oracle-reanchor] GT source opened: {path} total_src={total_src} "
                  f"src_fps={src_fps} start_src={start_src} → avail {p} target(24fps) frames", flush=True)
        except Exception as _e:
            print(f"[oracle-reanchor] WARN GT source open failed ({type(_e).__name__}: {_e}) "
                  f"-> every oracle trigger will be skipped", flush=True)
            cache["avail"] = None
    return cache.get("avail")


def _oracle_gt_fetch(cache, pix_a, pix_b, height, width, ref_video=None):
    """Read GT source frames at absolute pixel indices [pix_a, pix_b) (24fps target timeline = the FrameBank gid convention).
    env source: replicates the fps resampling + centre crop + resize of ev_validation._load_test_clip -> same distribution and
    geometry as the frames the v2v ref seed feeds into the bank (same K_pix convention). Returns [3,N,H,W] fp32 CPU in [-1,1] or None."""
    import os as _os
    path = (_os.environ.get("EVOKE_ORACLE_GT_VIDEO", "") or "").strip()
    if not path:
        if (ref_video is not None and isinstance(ref_video, torch.Tensor) and ref_video.ndim == 5
                and int(pix_b) <= int(ref_video.shape[2])):
            return ref_video[0, :, int(pix_a):int(pix_b)].detach().float().cpu()
        return None
    if cache.get("avail") is None or int(pix_b) > int(cache["avail"]):
        return None
    try:
        import torchvision as _tv
        vr = cache["vr"]; src_fps = cache["src_fps"]; start_src = cache["start_src"]
        idxs = [start_src + round(p * src_fps / 24.0) for p in range(int(pix_a), int(pix_b))]
        import numpy as _np
        fr = torch.from_numpy(_np.asarray(vr.get_batch(idxs))).float()   # [N,h,w,3] 0..255
        v = ((fr / 127.5) - 1.0).permute(0, 3, 1, 2)                      # [N,3,h,w] in [-1,1]
        _, _, h, w = v.shape                                              # centre crop: same as _load_test_clip
        target_ar = float(height) / float(width); src_ar = float(h) / float(w)
        if src_ar >= target_ar:
            nh = int(w * target_ar); top = (h - nh) // 2
            v = v[:, :, top:top + nh, :]
        else:
            nw = int(h / target_ar); left = (w - nw) // 2
            v = v[:, :, :, left:left + nw]
        v = _tv.transforms.functional.resize(v, (int(height), int(width)))
        return v.permute(1, 0, 2, 3).contiguous()                         # [3,N,H,W]
    except Exception as _e:
        print(f"[oracle-reanchor] WARN GT frame read failed ({type(_e).__name__}: {_e})", flush=True)
        return None


def _oracle_reanchor_maybe(chunk_num, every, window_end, window_pix, stride, gt_avail,
                           fetch_fn, evict_fn, ingest_fn):
    """[oracle diagnostic] pure control logic (no torch/bank dependency, so it can be unit-tested with CPU mocks): decides whether
    chunk number chunk_num (1-based, non-event) triggers a GT hard re-anchor at its end, and orchestrates evict -> segmented ingest in the v2v seed convention.
      only triggers when chunk_num % every == 0 (every<=0 -> never, fully disabled).
      window = [max(0, window_end-window_pix), window_end); window_end past gt_avail (the GT source video /
      pose available range, e.g. the relative_replay extrapolated tail) -> skip + log, leaving the existing cloud untouched.
      fetch_fn(a, b) -> frames|None (failure -> skip, no evict); evict_fn() -> clear the cloud and return the old frame count;
      ingest_fn(frames, seg_a, seg_b, win_a) -> ingest the [seg_a, seg_b) segment (gid=absolute pixel index).
    Returns whether a re-anchor was triggered."""
    every = int(every)
    if every <= 0 or int(chunk_num) % every != 0:
        return False
    ws = max(0, int(window_end) - int(window_pix)); we = int(window_end)
    if gt_avail is None or we > int(gt_avail):
        print(f"[oracle-reanchor] chunk={int(chunk_num)} SKIP: window [{ws}..{we - 1}] beyond the GT available range "
              f"(gt_avail={gt_avail}; the relative_replay tail has no GT) -> keeping the on-policy cloud", flush=True)
        return False
    frames = fetch_fn(ws, we)
    if frames is None:
        print(f"[oracle-reanchor] chunk={int(chunk_num)} SKIP: GT frame read failed [{ws}..{we - 1}] "
              f"-> keeping the on-policy cloud", flush=True)
        return False
    n_evicted = evict_fn()
    for seg_a in range(ws, we, max(1, int(stride))):
        ingest_fn(frames, seg_a, min(seg_a + max(1, int(stride)), we), ws)
    print(f"[oracle-reanchor] chunk={int(chunk_num)} evicted={n_evicted} "
          f"reingested GT frames [{ws}..{we - 1}]", flush=True)
    return True
