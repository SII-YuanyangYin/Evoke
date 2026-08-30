"""Minimal LightTAE Wan 2.1 decoder used by the local Director UI.

Architecture adapted from ModelTC/LightX2V (Apache-2.0):
https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/video_encoders/hf/tae.py

Only decoding is included. Keeping this adapter in ``ui/`` makes the fast VAE an
explicit UI runtime choice and leaves EVOKE's released inference configuration intact.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional


def _conv(channels_in: int, channels_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(channels_in, channels_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _MemBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(channels_in * 2, channels_out),
            nn.ReLU(inplace=True),
            _conv(channels_out, channels_out),
            nn.ReLU(inplace=True),
            _conv(channels_out, channels_out),
        )
        self.skip = nn.Conv2d(channels_in, channels_out, 1, bias=False) if channels_in != channels_out else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(torch.cat([value, previous], dim=1)) + self.skip(value))


class _TGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        frames, channels, height, width = value.shape
        return self.conv(value).reshape(frames * self.stride, channels, height, width)


def _apply_parallel(model: nn.Sequential, value: torch.Tensor) -> torch.Tensor:
    """Run LightTAE over [B,T,C,H,W], parallelizing frames within each layer."""
    if value.ndim != 5:
        raise ValueError(f"LightTAE expects BTCHW, got {tuple(value.shape)}")
    batch, frames, channels, height, width = value.shape
    value = value.reshape(batch * frames, channels, height, width)
    for block in model:
        if isinstance(block, _MemBlock):
            flat_frames, channels, height, width = value.shape
            frames = flat_frames // batch
            sequence = value.reshape(batch, frames, channels, height, width)
            previous = functional.pad(sequence, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :frames]
            value = block(value, previous.reshape_as(value))
        else:
            value = block(value)
    flat_frames, channels, height, width = value.shape
    return value.reshape(batch, flat_frames // batch, channels, height, width)


class LightTAEWan21Decoder(nn.Module):
    latent_channels = 16
    frames_to_trim = 3

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        features = [256, 128, 64, 64]
        activation = nn.ReLU(inplace=True)
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(self.latent_channels, features[0]),
            activation,
            _MemBlock(features[0], features[0]),
            _MemBlock(features[0], features[0]),
            _MemBlock(features[0], features[0]),
            nn.Upsample(scale_factor=2),
            _TGrow(features[0], 1),
            _conv(features[0], features[1], bias=False),
            _MemBlock(features[1], features[1]),
            _MemBlock(features[1], features[1]),
            _MemBlock(features[1], features[1]),
            nn.Upsample(scale_factor=2),
            _TGrow(features[1], 2),
            _conv(features[1], features[2], bias=False),
            _MemBlock(features[2], features[2]),
            _MemBlock(features[2], features[2]),
            _MemBlock(features[2], features[2]),
            nn.Upsample(scale_factor=2),
            _TGrow(features[2], 2),
            _conv(features[2], features[3], bias=False),
            activation,
            _conv(features[3], 3),
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        decoder_state = {key.removeprefix("decoder."): value for key, value in state.items() if key.startswith("decoder.")}
        self.decoder.load_state_dict(decoder_state, strict=True)

    @torch.inference_mode()
    def forward(self, latent_btchw: torch.Tensor) -> torch.Tensor:
        output = _apply_parallel(self.decoder, latent_btchw)
        return output.clamp_(0, 1)[:, self.frames_to_trim :]
