"""UI-only ViGeo preload and single-model reuse for the persistent worker."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch


_ESTIMATOR = None


def _write_state(
    message: str,
    *,
    ready: bool = False,
    timings: dict | None = None,
    weights: str | None = None,
) -> None:
    queue_root = Path(os.environ.get("EVOKE_ARGV_SERVER_DIR", ""))
    state_path = queue_root / "state.json"
    if queue_root and state_path.parent.is_dir():
        try:
            current = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        current.update({"phase": "loading", "message": message, "pid": os.getpid(), "updatedAt": time.time()})
        temporary = state_path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(state_path)
    if ready:
        marker_path = Path(os.environ["EVOKE_UI_VIGEO_READY_STATE"])
        payload = {
            "pid": os.getpid(),
            "weights": weights or os.environ.get("EVOKE_VIGEO_WEIGHTS", "models/ViGeo1.1"),
            "warmed": True,
            "updatedAt": time.time(),
            **(timings or {}),
        }
        temporary = marker_path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(marker_path)


def _configure(estimator, process_res: int, options: dict) -> None:
    estimator.process_res = int(process_res)
    converters = {
        "num_tokens": lambda value: int(value) if value else None,
        "mode": str,
        "chunk_size": int,
        "intr_source": str,
        "conf_transform": str,
        "scale_mode": str,
        "anchor_windows": int,
        "total_budget": int,
        "cache_keep_frames": int,
        "scale_value": float,
        "depth_median_target": float,
    }
    for key, convert in converters.items():
        value = options.get(key)
        if value is not None:
            setattr(estimator, key, convert(value))
    if estimator.mode not in {"offline", "chunk", "online"}:
        raise ValueError(f"invalid UI ViGeo mode: {estimator.mode}")
    if estimator.scale_mode not in {"per_window", "anchor", "depth_median", "fixed"}:
        raise ValueError(f"invalid UI ViGeo scale mode: {estimator.scale_mode}")
    estimator.reset_stream()


def install() -> None:
    from evoke.modules.geometric_state import depth_backend
    from evoke.modules.geometric_state.vigeo_cloud import ViGeoDepthEstimator, _VIGEO_SRC, _VIGEO_WEIGHTS
    from evoke.pipelines.pipeline_evoke import EvokePipeline

    if getattr(EvokePipeline, "_evoke_ui_vigeo_installed", False):
        return

    original_build_estimator = depth_backend.build_estimator

    def ensure_estimator(device: torch.device):
        global _ESTIMATOR
        if _ESTIMATOR is None:
            _ESTIMATOR = ViGeoDepthEstimator(
                device=device,
                process_res=644,
                src=_VIGEO_SRC,
                weights=_VIGEO_WEIGHTS,
                mode="chunk",
                chunk_size=16,
                intr_source="gt",
                conf_transform="exp",
                scale_mode="depth_median",
                anchor_windows=4,
                total_budget=0,
                cache_keep_frames=6,
                depth_median_target=5.0,
            )
        return _ESTIMATOR

    def build_estimator(backend, device, process_res, weights=None, src=None, vigeo_opts=None):
        if str(backend or "da3").lower() != "vigeo":
            return original_build_estimator(backend, device, process_res, weights, src, vigeo_opts)
        estimator = ensure_estimator(torch.device(device))
        requested_weights = Path(weights or _VIGEO_WEIGHTS).resolve()
        requested_src = Path(src or _VIGEO_SRC).resolve()
        if requested_weights != estimator.weights.resolve() or requested_src != estimator.src.resolve():
            return original_build_estimator(backend, device, process_res, weights, src, vigeo_opts)
        _configure(estimator, process_res, vigeo_opts or {})
        return estimator

    depth_backend.build_estimator = build_estimator

    original_to = EvokePipeline.to

    def move_to(self, *args, **kwargs):
        result = original_to(self, *args, **kwargs)
        device = self._execution_device
        if device.type != "cuda" or getattr(self, "_evoke_ui_vigeo_warmed", False):
            return result
        # The argv server offloads this immediately after build_pipe returns.
        # Do it a few lines earlier so ViGeo never overlaps the large text
        # encoder during preload on a 48 GB card.
        self._evoke_offload_text_encoder = True
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        _write_state("正在预加载 ViGeo 权重到 GPU")
        estimator = ensure_estimator(device)
        load_started = time.perf_counter()
        estimator._lazy()
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_started

        _write_state("正在执行 ViGeo CUDA warmup")
        warmup_started = time.perf_counter()
        with torch.inference_mode():
            warmup = estimator._infer_window(np.zeros((1, 384, 640, 3), dtype=np.float32))
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started
        del warmup
        estimator.reset_stream()
        torch.cuda.empty_cache()
        self._evoke_ui_vigeo_warmed = True
        _write_state(
            "ViGeo 已预加载并完成 CUDA warmup",
            ready=True,
            timings={"loadSeconds": round(load_seconds, 3), "warmupSeconds": round(warmup_seconds, 3)},
            weights=str(estimator.weights),
        )
        print(
            f"[ui-vigeo] preload complete: weights={load_seconds:.3f}s warmup={warmup_seconds:.3f}s",
            flush=True,
        )
        return result

    EvokePipeline.to = move_to
    EvokePipeline._evoke_ui_vigeo_installed = True
    print("[ui-vigeo] UI preload hook installed", flush=True)
