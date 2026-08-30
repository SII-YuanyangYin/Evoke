"""Standalone EVOKE Director UI backend.

The backend only prepares inputs and launches the repository's existing post-distill
launcher. No training or inference source files are imported or modified.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:  # Supports both ``python app.py`` and ``uvicorn ui.app:app``.
    from .trajectory import OUTPUT_FPS, POSE_FPS, SOURCE_HEIGHT, SOURCE_WIDTH, append_pose_npz, build_pose_npz, resample_pose_npz
except ImportError:
    from trajectory import OUTPUT_FPS, POSE_FPS, SOURCE_HEIGHT, SOURCE_WIDTH, append_pose_npz, build_pose_npz, resample_pose_npz


UI_ROOT = Path(__file__).resolve().parent
REPO_ROOT = UI_ROOT.parent
JOBS_ROOT = Path(os.environ.get("EVOKE_UI_JOBS", UI_ROOT / "jobs")).resolve()
PROJECTS_ROOT = Path(os.environ.get("EVOKE_UI_PROJECTS", UI_ROOT / "projects")).resolve()
STATIC_ROOT = UI_ROOT / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SEGMENT_RE = re.compile(r"segment_(\d+)_pred\.mp4$")
DEFAULT_CASE_ROOT = REPO_ROOT / "examples" / "segment_prompts"
DEFAULT_CASE_NAME = "meteor"
DEFAULT_REFERENCE = DEFAULT_CASE_ROOT / f"{DEFAULT_CASE_NAME}.jpg"
DEFAULT_SCHEDULE = DEFAULT_CASE_ROOT / f"schedule_{DEFAULT_CASE_NAME}.json"
DEFAULT_CASE_JSONL = DEFAULT_CASE_ROOT / f"cases_{DEFAULT_CASE_NAME}.jsonl"
DEFAULT_CHUNKS = 6
WORKER_ROOT = JOBS_ROOT / "_worker"
WORKER_STATE = WORKER_ROOT / "state.json"
VIGEO_READY_STATE = WORKER_ROOT / "vigeo_ready.json"
REVISION_SCHEMA_VERSION = 1
UI_CONFIG_PATH = UI_ROOT / "config.json"


def _load_ui_config() -> dict:
    try:
        return json.loads(UI_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


UI_CONFIG = _load_ui_config()


MODEL_PATH_DEFAULTS = {
    "base": ("models/evoke-base", ("BASE_CKPT",)),
    "transformer": ("models/evoke/stage3_post_distillation", ("TRANSFORMER_PATH",)),
    "vigeo": ("models/ViGeo1.1", ("VIGEO_WEIGHTS", "EVOKE_VIGEO_WEIGHTS")),
    "da3": ("models/DA3", ("DA3_WEIGHTS", "EVOKE_DA3_WEIGHTS")),
}


def _model_paths() -> dict[str, Path]:
    """Resolve model paths relative to the repository, with environment overrides."""
    configured = UI_CONFIG.get("models") or {}
    resolved: dict[str, Path] = {}
    for name, (default, environment_names) in MODEL_PATH_DEFAULTS.items():
        value = next((os.environ[key] for key in environment_names if os.environ.get(key)), None)
        value = value or configured.get(name) or default
        path = Path(str(value)).expanduser()
        resolved[name] = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    return resolved


def _apply_model_environment(env: dict[str, str]) -> None:
    """Give the launcher and imported geometry backends one consistent path set."""
    paths = _model_paths()
    env["BASE_CKPT"] = str(paths["base"])
    env["TRANSFORMER_PATH"] = str(paths["transformer"])
    env["VIGEO_WEIGHTS"] = str(paths["vigeo"])
    env["EVOKE_VIGEO_WEIGHTS"] = str(paths["vigeo"])
    env["DA3_WEIGHTS"] = str(paths["da3"])
    env["EVOKE_DA3_WEIGHTS"] = str(paths["da3"])


def _read_progress(output_root: Path, chunks: int, chunk_offset: int = 0, total_chunks: int | None = None) -> dict | None:
    try:
        raw = json.loads((output_root / "evoke_ui" / "progress.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    phase = str(raw.get("phase", "preparing"))
    local_chunk_index = max(0, min(chunks - 1, int(raw.get("chunkIndex", 0))))
    total_chunks = total_chunks or chunks
    chunk_index = local_chunk_index + chunk_offset
    step = max(0, int(raw.get("step", 0)))
    total_steps = max(1, int(raw.get("totalSteps", 3) or 3))
    within = {
        "preparing": 0.01,
        "geometry": 0.08,
        "denoising": 0.12 + 0.62 * min(1.0, step / total_steps),
        "decoding": 0.82,
        "chunk_done": 1.0,
    }.get(phase, 0.01)
    labels = {
        "preparing": f"CH {chunk_index + 1}/{total_chunks} · 准备输入",
        "geometry": f"CH {chunk_index + 1}/{total_chunks} · ViGeo / Frame Bank 几何准备",
        "denoising": f"CH {chunk_index + 1}/{total_chunks} · 去噪 {step}/{total_steps}",
        "decoding": f"CH {chunk_index + 1}/{total_chunks} · VAE 解码并写入预览",
        "chunk_done": f"CH {chunk_index + 1}/{total_chunks} · 预览已生成",
    }
    return {
        "phase": phase,
        "chunkIndex": chunk_index,
        "chunkNumber": chunk_index + 1,
        "stage": int(raw.get("stage", 0)),
        "step": step,
        "totalSteps": total_steps,
        "message": labels.get(phase, phase),
        "overall": min(1.0, (chunk_index + within) / total_chunks),
        "updatedAt": raw.get("updatedAt"),
    }


def _inner_log_tail(output_root: Path, limit: int = 80) -> list[str]:
    path = output_root / "_logs" / "evoke_ui.log"
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 256_000))
            text = handle.read().decode("utf-8", errors="replace").replace("\r", "\n")
        return [line for line in text.splitlines() if line.strip()][-limit:]
    except OSError:
        return []


@dataclass
class Job:
    id: str
    root: Path
    chunks: int
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    message: str = "Waiting for GPU"
    log_tail: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    process: asyncio.subprocess.Process | None = None
    mode: str = "i2v"
    run_chunks: int | None = None
    branch_from_chunk: int = 0
    project_id: str | None = None
    revision_id: str | None = None
    parent_revision_id: str | None = None
    engine_output: Path | None = None
    ref_video_sec: float = 0.0
    start_seconds: float = 0.0

    @property
    def inference_output(self) -> Path:
        return self.engine_output or (self.root / "output")

    @property
    def generated_chunks(self) -> int:
        return self.run_chunks if self.run_chunks is not None else self.chunks

    def snapshot(self) -> dict:
        detail = _read_progress(
            self.inference_output,
            self.generated_chunks,
            self.branch_from_chunk,
            self.chunks,
        )
        progress = len(self.segments) / self.chunks
        if detail and self.status in {"queued", "running", "cancelling"}:
            progress = max(progress, float(detail["overall"]))
        inner_log = _inner_log_tail(self.inference_output)
        browser_segments = [
            str(_ensure_browser_video(self.root / path, self.root).relative_to(self.root))
            for path in self.segments
        ]
        final_source = self.root / "output" / "evoke_ui" / "geo_pred.mp4"
        final_ready = final_source.is_file() and final_source.stat().st_size > 0
        full_ready = self.status == "complete" and len(browser_segments) == self.chunks
        return {
            "id": self.id,
            "status": self.status,
            "chunks": self.chunks,
            "completedChunks": len(self.segments),
            "progress": min(1.0, progress),
            "message": detail["message"] if detail and self.status == "running" else self.message,
            "progressDetail": detail,
            "segments": [f"api/jobs/{self.id}/files/{path}" for path in browser_segments],
            # FULL preview remains a client-side playlist.  The canonical file is
            # assembled lazily when download (or v2v continuation) actually needs it.
            "finalVideo": f"api/jobs/{self.id}/files/{final_source.relative_to(self.root)}" if final_ready else None,
            "downloadVideo": f"api/jobs/{self.id}/download" if full_ready else None,
            "fullPlayback": {
                "mode": "segments",
                "segmentCount": len(browser_segments),
            } if full_ready else None,
            "log": (self.log_tail[-30:] + inner_log)[-100:],
            "returnCode": self.return_code,
            "mode": self.mode,
            "projectId": self.project_id,
            "revisionId": self.revision_id,
            "parentRevisionId": self.parent_revision_id,
            "branchFromChunk": self.branch_from_chunk,
        }


JOBS: dict[str, Job] = {}
GPU_LOCK = asyncio.Lock()
FINALIZE_LOCKS: dict[str, asyncio.Lock] = {}
WORKER_PROCESS: subprocess.Popen | None = None
app = FastAPI(title="EVOKE Director", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _chunk_start_frame(chunk_index: int) -> int:
    """Canonical UI chunks are exactly 1.5 seconds: 36 frames at 24 FPS."""
    return max(0, chunk_index) * 36


def _next_revision_id(project_root: Path) -> str:
    existing = []
    for path in (project_root / "revisions").glob("r[0-9][0-9][0-9][0-9]"):
        try:
            existing.append(int(path.name[1:]))
        except ValueError:
            pass
    return f"r{max(existing, default=0) + 1:04d}"


def _revision_manifest(job: Job) -> dict:
    return {
        "schemaVersion": REVISION_SCHEMA_VERSION,
        "id": job.revision_id,
        "jobId": job.id,
        "projectId": job.project_id,
        "parentRevisionId": job.parent_revision_id,
        "mode": job.mode,
        "status": job.status,
        "message": job.message,
        "createdAt": job.created_at,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
        "returnCode": job.return_code,
        "chunkCount": job.chunks,
        "generatedChunkCount": job.generated_chunks,
        "branchFromChunk": job.branch_from_chunk,
        "referenceWindow": {
            "startSeconds": job.start_seconds,
            "durationSeconds": job.ref_video_sec,
        } if job.mode == "v2v" else None,
        "engineOutput": str(job.inference_output.relative_to(job.root)),
        "output": {
            "video": "output/evoke_ui/geo_pred.mp4",
            "segments": [f"output/evoke_ui/segments/segment_{i:03d}_pred.mp4" for i in range(job.chunks)],
        },
    }


def _persist_job(job: Job) -> None:
    if not job.project_id or not job.revision_id:
        return
    _write_json(job.root / "manifest.json", _revision_manifest(job))
    project_path = PROJECTS_ROOT / job.project_id / "project.json"
    try:
        project = json.loads(project_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        project = {"schemaVersion": REVISION_SCHEMA_VERSION, "id": job.project_id, "createdAt": job.created_at}
    project["activeRevisionId"] = job.revision_id
    project["updatedAt"] = time.time()
    _write_json(project_path, project)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _ensure_browser_video(source: Path, job_root: Path) -> Path:
    """Return an H.264 copy suitable for Chromium without changing canonical inference files."""
    if not source.is_file() or source.stat().st_size <= 0:
        return source
    evoke_root = job_root / "output" / "evoke_ui"
    try:
        relative = source.relative_to(evoke_root)
    except ValueError:
        return source
    destination = evoke_root / "browser" / relative
    if destination.is_file() and destination.stat().st_size > 0 \
            and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg produced no browser preview")
        temporary.replace(destination)
        return destination
    except (OSError, subprocess.SubprocessError, RuntimeError):
        temporary.unlink(missing_ok=True)
        return source


def _restore_jobs() -> None:
    """Restore completed/local revisions so browser refresh and UI restarts keep history."""
    for manifest_path in PROJECTS_ROOT.glob("*/revisions/*/manifest.json"):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = manifest_path.parent
            engine_rel = raw.get("engineOutput", "output")
            job = Job(
                id=str(raw["jobId"]),
                root=root,
                chunks=int(raw["chunkCount"]),
                status=str(raw.get("status", "complete")),
                created_at=float(raw.get("createdAt", root.stat().st_mtime)),
                started_at=raw.get("startedAt"),
                finished_at=raw.get("finishedAt"),
                return_code=raw.get("returnCode"),
                message=str(raw.get("message", "Local revision")),
                mode=str(raw.get("mode", "i2v")),
                run_chunks=int(raw.get("generatedChunkCount", raw["chunkCount"])),
                branch_from_chunk=int(raw.get("branchFromChunk", 0)),
                project_id=str(raw.get("projectId")),
                revision_id=str(raw.get("id")),
                parent_revision_id=raw.get("parentRevisionId"),
                engine_output=root / engine_rel,
                ref_video_sec=float((raw.get("referenceWindow") or {}).get("durationSeconds", 0)),
                start_seconds=float((raw.get("referenceWindow") or {}).get("startSeconds", 0)),
            )
            # An interrupted process cannot survive an app restart; retain its files but make state honest.
            if job.status in {"queued", "running", "cancelling"}:
                job.status, job.message = "failed", "UI restarted before this revision completed"
            _discover_segments(job)
            JOBS[job.id] = job
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue

    # Backward-compatible discovery for runs made before Project/Revision storage existed.
    for root in JOBS_ROOT.iterdir() if JOBS_ROOT.is_dir() else []:
        if not root.is_dir() or root.name.startswith("_") or root.name in JOBS:
            continue
        try:
            spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        complete = (root / "output" / "evoke_ui" / "geo_pred.mp4").is_file()
        job = Job(
            id=root.name,
            root=root,
            chunks=int(spec.get("chunks", 1)),
            status="complete" if complete else "failed",
            created_at=root.stat().st_mtime,
            finished_at=root.stat().st_mtime if complete else None,
            return_code=0 if complete else None,
            message="Imported legacy result" if complete else "Incomplete legacy result",
        )
        _discover_segments(job)
        JOBS[job.id] = job


def _read_worker_state() -> dict:
    try:
        return json.loads(WORKER_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _start_worker(*, clear_stale_requests: bool = True) -> None:
    """Start the one persistent GPU process; it preloads the pipeline before accepting jobs."""
    global WORKER_PROCESS
    if WORKER_PROCESS is not None and WORKER_PROCESS.poll() is None:
        return
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    (WORKER_ROOT / "requests").mkdir(exist_ok=True)
    (WORKER_ROOT / "responses").mkdir(exist_ok=True)
    # A request whose launcher disappeared during a previous UI shutdown must never be replayed.
    # During an in-app cancellation restart, keep requests queued behind the cancelled job: their
    # launchers are alive and the replacement worker should consume them after preloading.
    stale_requests = list(WORKER_ROOT.joinpath("requests").glob("*.json")) if clear_stale_requests else []
    for stale in [*stale_requests, *WORKER_ROOT.glob("running-*.json")]:
        stale.unlink(missing_ok=True)
    WORKER_STATE.unlink(missing_ok=True)
    VIGEO_READY_STATE.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MODE": "i2v",
            "JSONL": str(DEFAULT_CASE_JSONL),
            "NUM_CHUNKS": "1",
            "MAX_CASES": "1",
            "LOCAL_GPUS": "1",
            "OUT_ROOT": str(WORKER_ROOT / "bootstrap"),
            "JOYSTICK_HUD": "off",
            "SAVE_SEGMENTS": "1",
            "IN_PROCESS_BATCH": "1",
            "EVOKE_ARGV_SERVER_DIR": str(WORKER_ROOT),
            "EVOKE_SERVER_PRELOAD": "1",
        }
    )
    _apply_model_environment(env)
    geometry = UI_CONFIG.get("geometry") or {}
    if bool(geometry.get("preloadViGeo", True)):
        python_paths = [str(UI_ROOT / "runtime"), str(UI_ROOT), str(REPO_ROOT)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
        env["EVOKE_UI_PRELOAD_VIGEO"] = "1"
        env["EVOKE_UI_VIGEO_READY_STATE"] = str(VIGEO_READY_STATE)
    preview_vae = UI_CONFIG.get("previewVae") or {}
    preview_weights = UI_ROOT / str(preview_vae.get("weights", ""))
    if bool(preview_vae.get("enabled")) and preview_vae.get("modelType") == "taew2_1" and preview_weights.is_file():
        python_paths = [str(UI_ROOT / "runtime"), str(UI_ROOT), str(REPO_ROOT)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
        env["EVOKE_UI_FAST_VAE"] = "1"
        env["EVOKE_UI_LIGHTTAE_WEIGHTS"] = str(preview_weights)
    if os.environ.get("EVOKE_UI_PYTHON_BIN"):
        env["EVOKE_PYTHON_BIN"] = os.environ["EVOKE_UI_PYTHON_BIN"]
    log_path = WORKER_ROOT / "worker.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        WORKER_PROCESS = subprocess.Popen(
            ["bash", "scripts/inference/infer_post_distill.sh"],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _stop_worker() -> None:
    global WORKER_PROCESS
    process = WORKER_PROCESS
    WORKER_PROCESS = None
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


async def _restart_worker() -> None:
    await asyncio.to_thread(_stop_worker)
    await asyncio.sleep(0.5)
    _start_worker(clear_stale_requests=False)


@app.on_event("startup")
async def startup() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    _restore_jobs()
    _start_worker()


@app.on_event("shutdown")
async def shutdown() -> None:
    await asyncio.to_thread(_stop_worker)


def _weights_state() -> dict[str, bool]:
    paths = _model_paths()
    base_shards = list((paths["base"] / "text_encoder").glob("model-*-of-00005.safetensors"))
    post_shards = list(
        (paths["transformer"] / "transformer").glob(
            "diffusion_pytorch_model-*-of-00012.safetensors"
        )
    )
    vigeo = paths["vigeo"] / "vigeo.pt"
    active_downloads = any(
        path.is_dir() and next(path.rglob("*.aria2"), None) is not None
        for path in (paths["base"], paths["transformer"], paths["vigeo"])
    )
    return {
        # Size floors/control files prevent an in-progress download from looking ready merely because it exists.
        "postDistill": not active_downloads
        and len(post_shards) == 12
        and min((p.stat().st_size for p in post_shards), default=0) > 2_000_000_000,
        "base": not active_downloads
        and len(base_shards) == 5
        and min((p.stat().st_size for p in base_shards), default=0) > 2_000_000_000,
        "vigeo": not active_downloads and vigeo.is_file() and vigeo.stat().st_size > 4_000_000_000,
    }


def _preview_vae_state() -> dict:
    config = UI_CONFIG.get("previewVae") or {}
    weights = UI_ROOT / str(config.get("weights", ""))
    requested = UI_ROOT / str(config.get("requestedWeights", ""))
    enabled = bool(config.get("enabled"))
    return {
        "enabled": enabled,
        "ready": enabled and config.get("modelType") == "taew2_1" and weights.is_file() and weights.stat().st_size > 40_000_000,
        "backend": config.get("backend"),
        "modelType": config.get("modelType"),
        "weights": str(weights.relative_to(UI_ROOT)) if weights.is_relative_to(UI_ROOT) else str(weights),
        "requestedDownloaded": requested.is_file(),
        "requestedCompatibility": config.get("requestedCompatibility"),
    }


def _vigeo_preload_state() -> dict:
    config = UI_CONFIG.get("geometry") or {}
    enabled = bool(config.get("preloadViGeo", True))
    marker = {}
    try:
        marker = json.loads(VIGEO_READY_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    worker = _read_worker_state()
    ready = enabled and bool(marker) and marker.get("pid") == worker.get("pid")
    return {
        "enabled": enabled,
        "ready": ready,
        "warmed": bool(marker.get("warmed", False)) if ready else False,
        "weights": marker.get("weights", "models/ViGeo1.1"),
        "loadSeconds": marker.get("loadSeconds"),
        "warmupSeconds": marker.get("warmupSeconds"),
    }


def _gpu_state() -> dict:
    """Probe CUDA hardware without importing torch into the lightweight UI process."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        devices = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, name, total, free = (part.strip() for part in line.split(",", 3))
            devices.append(
                {
                    "index": int(index),
                    "name": name,
                    "totalMemoryMiB": int(total),
                    "freeMemoryMiB": int(free),
                }
            )
        primary = devices[0] if devices else None
        return {
            "available": bool(devices),
            "devices": devices,
            # The persistent/text-offloaded UI path peaks around 45.5 GiB in the
            # verified six-chunk run; a 47.4 GiB RTX 4090 is therefore supported.
            "capacityReady": bool(primary and primary["totalMemoryMiB"] >= 47_104),
            "recommendedMemoryMiB": 47_104,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "devices": [], "capacityReady": False, "error": str(error)}


def _runtime_state() -> dict[str, str | None]:
    worker = _read_worker_state()
    worker_phase = worker.get("phase")
    worker_message = worker.get("message")
    active = [job for job in JOBS.values() if job.status in {"queued", "running", "cancelling"}]
    if not active:
        if worker_phase in {"loading", "ready", "error"}:
            if worker_phase == "ready" and _vigeo_preload_state()["ready"]:
                preview_name = "LightTAE" if _preview_vae_state()["enabled"] else "官方 Wan VAE"
                worker_message = f"Post-distill · ViGeo · {preview_name} 已加载并常驻 GPU"
            return {"phase": worker_phase, "message": worker_message, "jobId": None}
        if WORKER_PROCESS is not None and WORKER_PROCESS.poll() is None:
            return {"phase": "loading", "message": "正在启动常驻模型服务", "jobId": None}
        return {"phase": "error", "message": "常驻模型服务未运行", "jobId": None}
    job = max(active, key=lambda item: item.created_at)
    if job.status == "queued":
        return {"phase": "queued", "message": "等待 GPU", "jobId": job.id}
    if job.status == "cancelling":
        return {"phase": "cancelling", "message": "正在停止任务", "jobId": job.id}
    _discover_segments(job)
    if job.segments:
        return {
            "phase": "generating",
            "message": f"模型已加载 · {len(job.segments)}/{job.chunks} chunks",
            "jobId": job.id,
        }
    inner_log = job.inference_output / "_logs" / "evoke_ui.log"
    try:
        with inner_log.open("rb") as handle:
            handle.seek(max(0, inner_log.stat().st_size - 256_000))
            tail = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        tail = ""
    if "===== pipe inference" in tail:
        return {"phase": "generating", "message": "模型已加载 · 正在生成", "jobId": job.id}
    if worker_phase == "loading":
        return {"phase": "loading", "message": worker_message or "正在预加载模型到 GPU", "jobId": job.id}
    if worker_phase == "running":
        return {"phase": "generating", "message": worker_message or "常驻模型正在生成", "jobId": job.id}
    if worker_phase == "error":
        return {"phase": "error", "message": worker_message or "常驻模型服务异常", "jobId": job.id}
    return {"phase": "queued", "message": "模型已常驻 · 正在提交任务", "jobId": job.id}


def _safe_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "reference.png").suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    if suffix not in allowed:
        raise HTTPException(400, "Reference image must be PNG, JPG, or WebP")
    return suffix


def _validate_spec(raw: str) -> dict:
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(400, f"Invalid generation spec: {error.msg}") from error
    chunks = int(spec.get("chunks", 0))
    prompts = spec.get("prompts")
    points = spec.get("pathPoints")
    look_points = spec.get("lookPoints")
    if not 1 <= chunks <= 40:
        raise HTTPException(400, "chunks must be between 1 and 40")
    if not isinstance(prompts, list) or len(prompts) != chunks:
        raise HTTPException(400, "Every chunk needs exactly one prompt")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise HTTPException(400, "Prompt text cannot be empty")
    if not isinstance(points, list) or not points:
        raise HTTPException(400, "Draw or select a camera trajectory")
    if look_points is not None and (not isinstance(look_points, list) or len(look_points) < 2):
        raise HTTPException(400, "View control needs at least two points")
    spec["chunks"] = chunks
    spec["prompts"] = [prompt.strip()[:4000] for prompt in prompts]
    return spec


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                output.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "Reference image exceeds 25 MB")
            output.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Reference image is empty")


def _discover_segments(job: Job) -> None:
    canonical_dir = job.root / "output" / "evoke_ui" / "segments"
    if job.branch_from_chunk > 0:
        generated_dir = job.inference_output / "evoke_ui" / "segments"
        if generated_dir.is_dir():
            for path in generated_dir.glob("segment_*_pred.mp4"):
                match = SEGMENT_RE.match(path.name)
                if not match or path.stat().st_size <= 0:
                    continue
                logical_index = job.branch_from_chunk + int(match.group(1))
                if logical_index >= job.chunks:
                    continue
                destination = canonical_dir / f"segment_{logical_index:03d}_pred.mp4"
                if not destination.is_file() or destination.stat().st_size != path.stat().st_size:
                    _copy_file(path, destination)
    segment_dir = canonical_dir
    if not segment_dir.is_dir():
        return
    found: list[tuple[int, str]] = []
    for path in segment_dir.glob("segment_*_pred.mp4"):
        match = SEGMENT_RE.match(path.name)
        if match and path.stat().st_size > 0:
            found.append((int(match.group(1)), str(path.relative_to(job.root))))
    job.segments = [path for _, path in sorted(found)]


def _finalize_revision(job: Job) -> None:
    """Build the canonical full take from accepted prefix plus newly generated suffix."""
    if not job.project_id or not job.revision_id:
        return
    _discover_segments(job)
    if len(job.segments) != job.chunks:
        raise RuntimeError(f"expected {job.chunks} canonical segments, found {len(job.segments)}")
    final_path = job.root / "output" / "evoke_ui" / "geo_pred.mp4"
    concat_file = job.root / "concat_segments.txt"
    lines = []
    for relative in job.segments:
        path = (job.root / relative).resolve()
        lines.append("file '" + str(path).replace("'", "'\\''") + "'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(final_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()}")


async def _ensure_final_revision(job: Job) -> Path:
    """Lazily assemble one immutable revision, coalescing concurrent requests."""
    lock = FINALIZE_LOCKS.setdefault(job.id, asyncio.Lock())
    async with lock:
        _discover_segments(job)
        if len(job.segments) != job.chunks:
            raise HTTPException(409, "Complete video requires every chunk")
        final_path = job.root / "output" / "evoke_ui" / "geo_pred.mp4"
        segment_paths = [job.root / relative for relative in job.segments]
        newest_segment = max(path.stat().st_mtime_ns for path in segment_paths)
        if not final_path.is_file() or final_path.stat().st_size <= 0 \
                or final_path.stat().st_mtime_ns < newest_segment:
            await asyncio.to_thread(_finalize_revision, job)
        return final_path


async def _run_job(job: Job) -> None:
    async with GPU_LOCK:
        if job.status == "cancelled":
            job.finished_at = time.time()
            return
        job.status = "running"
        job.started_at = time.time()
        job.message = "Submitting to persistent model worker"
        _persist_job(job)
        env = os.environ.copy()
        env.update(
            {
                "MODE": job.mode,
                "JSONL": str(job.root / "case.jsonl"),
                "NUM_CHUNKS": str(job.generated_chunks),
                "MAX_CASES": "1",
                "LOCAL_GPUS": "1",
                "OUT_ROOT": str(job.inference_output),
                "JOYSTICK_HUD": "off",
                "SAVE_SEGMENTS": "1",
                "IN_PROCESS_BATCH": "1",
                "EVOKE_ARGV_SERVER_DIR": str(WORKER_ROOT),
                "EVOKE_SERVER_REQUEST_ID": job.id,
            }
        )
        _apply_model_environment(env)
        if job.mode == "v2v":
            env["REF_VIDEO_SEC"] = f"{job.ref_video_sec:.9f}"
            env["START_SECONDS"] = f"{job.start_seconds:.9f}"
        if os.environ.get("EVOKE_UI_PYTHON_BIN"):
            env["EVOKE_PYTHON_BIN"] = os.environ["EVOKE_UI_PYTHON_BIN"]
        command = ["bash", "scripts/inference/infer_post_distill.sh"]
        log_path = job.root / "inference.log"
        try:
            job.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=REPO_ROOT,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            assert job.process.stdout is not None
            with log_path.open("w", encoding="utf-8") as log_file:
                while line_bytes := await job.process.stdout.readline():
                    line = line_bytes.decode("utf-8", errors="replace").rstrip()
                    log_file.write(line + "\n")
                    log_file.flush()
                    job.log_tail.append(line)
                    job.log_tail = job.log_tail[-160:]
                    _discover_segments(job)
                    if job.segments:
                        job.message = f"Streaming chunk {len(job.segments)} of {job.chunks}"
            job.return_code = await job.process.wait()
            _discover_segments(job)
            if job.status == "cancelling":
                job.status, job.message = "cancelled", "Generation cancelled"
            elif job.return_code == 0:
                job.status, job.message = "complete", "All chunks generated"
            else:
                job.status = "failed"
                job.message = f"Inference exited with code {job.return_code}"
        except Exception as error:  # surfaced to the UI, with details retained in its log.
            job.status, job.message = "failed", str(error)
            job.log_tail.append(f"[ui] {type(error).__name__}: {error}")
        finally:
            job.finished_at = time.time()
            job.process = None
            _persist_job(job)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_ROOT / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    checks = _weights_state()
    gpu = _gpu_state()
    runtime = _runtime_state()
    environment_ready = all(checks.values()) and gpu["available"]
    return {
        "ready": environment_ready and runtime["phase"] in {"ready", "generating"},
        "environmentReady": environment_ready,
        "weights": checks,
        "gpu": gpu,
        "runtime": runtime,
        "previewVae": _preview_vae_state(),
        "geometry": _vigeo_preload_state(),
    }


@app.get("/api/default-case")
async def default_case() -> dict:
    schedule = json.loads(DEFAULT_SCHEDULE.read_text(encoding="utf-8"))
    prompts: list[str] = []
    current_prompt = schedule[0]["prompt"]
    switches = {int(item["start_chunk"]): item["prompt"] for item in schedule}
    for chunk in range(DEFAULT_CHUNKS):
        current_prompt = switches.get(chunk, current_prompt)
        prompts.append(current_prompt)
    return {
        "name": DEFAULT_CASE_NAME,
        "chunks": DEFAULT_CHUNKS,
        "prompts": prompts,
        "referenceUrl": "api/default-case/reference",
    }


@app.get("/api/default-case/reference")
async def default_reference() -> FileResponse:
    return FileResponse(DEFAULT_REFERENCE, media_type="image/jpeg", filename=f"{DEFAULT_CASE_NAME}.jpg")


@app.post("/api/jobs")
async def create_job(reference: UploadFile = File(...), spec: str = Form(...)) -> dict:
    if not all(_weights_state().values()):
        raise HTTPException(503, "Post-distill, base, or ViGeo weights are incomplete")
    config = _validate_spec(spec)
    job_id = uuid.uuid4().hex[:12]
    project_id = uuid.uuid4().hex[:12]
    project_root = PROJECTS_ROOT / project_id
    revision_id = "r0001"
    job_root = project_root / "revisions" / revision_id
    input_root = job_root / "input"
    input_root.mkdir(parents=True, exist_ok=False)
    assets_root = project_root / "assets"
    reference_path = assets_root / f"reference{_safe_suffix(reference)}"
    source_pose_path = assets_root / "trajectory_30fps.npz"
    v2v_pose_path = assets_root / "trajectory_24fps.npz"
    try:
        await _save_upload(reference, reference_path)
        pose_info = build_pose_npz(
            source_pose_path,
            config["pathPoints"],
            config["chunks"],
            config.get("motionScale", 24),
            config.get("verticalLift", 0),
            config.get("fov", 70),
            bool(config.get("followPath", True)),
            config.get("lookPoints"),
            config.get("lookYawDegrees", 90),
            config.get("lookPitchDegrees", 45),
        )
        v2v_pose_info = resample_pose_npz(source_pose_path, v2v_pose_path)
        schedule = [
            {"start_chunk": chunk, "prompt": prompt}
            for chunk, prompt in enumerate(config["prompts"])
        ]
        (input_root / "prompt_schedule.json").write_text(
            json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        case = {
            "name": "evoke_ui",
            "image_path": str(reference_path),
            "pose_path": str(source_pose_path),
            "prompt": config["prompts"][0],
            "segment_prompts_path": str(input_root / "prompt_schedule.json"),
            "pose_fps": POSE_FPS,
            "pose_source_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "pose_type": "vipe",
            "seed": int(config.get("seed", 42)),
        }
        (job_root / "case.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
        (job_root / "spec.json").write_text(
            json.dumps({**config, "pose": pose_info, "v2vPose": v2v_pose_info}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_json(project_root / "project.json", {
            "schemaVersion": REVISION_SCHEMA_VERSION,
            "id": project_id,
            "title": config["prompts"][0][:80],
            "createdAt": time.time(),
            "updatedAt": time.time(),
            "activeRevisionId": revision_id,
            "assets": {
                "reference": str(reference_path.relative_to(project_root)),
                "pose30Fps": str(source_pose_path.relative_to(project_root)),
                "pose24Fps": str(v2v_pose_path.relative_to(project_root)),
            },
        })
    except Exception:
        shutil.rmtree(project_root, ignore_errors=True)
        raise
    job = Job(
        id=job_id,
        root=job_root,
        chunks=config["chunks"],
        project_id=project_id,
        revision_id=revision_id,
    )
    JOBS[job_id] = job
    _persist_job(job)
    asyncio.create_task(_run_job(job))
    return job.snapshot()


def _adopt_legacy_job(job: Job) -> None:
    if job.project_id:
        return
    if job.status != "complete":
        raise HTTPException(409, "Only a completed result can become a v2v parent")
    try:
        case = json.loads((job.root / "case.jsonl").read_text(encoding="utf-8").splitlines()[0])
        spec = json.loads((job.root / "spec.json").read_text(encoding="utf-8"))
        source_reference = Path(case["image_path"])
        source_pose = Path(case["pose_path"])
    except (KeyError, OSError, json.JSONDecodeError, IndexError) as error:
        raise HTTPException(409, f"Legacy result cannot be imported: {error}") from error

    project_id = uuid.uuid4().hex[:12]
    project_root = PROJECTS_ROOT / project_id
    revision_id = "r0001"
    revision_root = project_root / "revisions" / revision_id
    assets_root = project_root / "assets"
    reference_path = assets_root / f"reference{source_reference.suffix.lower()}"
    pose30_path = assets_root / "trajectory_30fps.npz"
    pose24_path = assets_root / "trajectory_24fps.npz"
    try:
        _copy_file(source_reference, reference_path)
        _copy_file(source_pose, pose30_path)
        pose24_info = resample_pose_npz(pose30_path, pose24_path, int(case.get("pose_fps", POSE_FPS)), OUTPUT_FPS)
        revision_root.mkdir(parents=True, exist_ok=False)
        for relative in ("output", "input"):
            source = job.root / relative
            if source.is_dir():
                shutil.copytree(source, revision_root / relative)
        imported_case = {
            **case,
            "image_path": str(reference_path),
            "pose_path": str(pose30_path),
        }
        (revision_root / "case.jsonl").write_text(json.dumps(imported_case, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_json(revision_root / "spec.json", {**spec, "v2vPose": pose24_info})
        if (job.root / "inference.log").is_file():
            _copy_file(job.root / "inference.log", revision_root / "inference.log")
        _write_json(project_root / "project.json", {
            "schemaVersion": REVISION_SCHEMA_VERSION,
            "id": project_id,
            "title": str(spec.get("prompts", ["EVOKE project"])[0])[:80],
            "createdAt": job.created_at,
            "updatedAt": time.time(),
            "activeRevisionId": revision_id,
            "importedFrom": str(job.root),
            "assets": {
                "reference": str(reference_path.relative_to(project_root)),
                "pose30Fps": str(pose30_path.relative_to(project_root)),
                "pose24Fps": str(pose24_path.relative_to(project_root)),
            },
        })
    except Exception:
        shutil.rmtree(project_root, ignore_errors=True)
        raise

    job.root = revision_root
    job.project_id = project_id
    job.revision_id = revision_id
    job.engine_output = revision_root / "output"
    _discover_segments(job)
    _persist_job(job)


def _job_pose_paths(job: Job) -> tuple[Path, Path]:
    """Resolve authored poses, following retry ancestry when files are inherited."""
    spec = json.loads((job.root / "spec.json").read_text(encoding="utf-8"))
    continuation = spec.get("continuation") or {}
    project_root = PROJECTS_ROOT / str(job.project_id)
    if continuation.get("pose30Fps") and continuation.get("pose24Fps"):
        relative30 = Path(str(continuation["pose30Fps"]))
        relative24 = Path(str(continuation["pose24Fps"]))
        revision_root = job.root
        visited: set[str] = set()
        while revision_root.parent == project_root / "revisions":
            candidate = revision_root / relative30, revision_root / relative24
            if all(path.is_file() for path in candidate):
                return candidate
            try:
                manifest = json.loads((revision_root / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                break
            parent_id = str(manifest.get("parentRevisionId") or "")
            if not parent_id or parent_id in visited:
                break
            visited.add(parent_id)
            revision_root = project_root / "revisions" / parent_id
    project = json.loads((project_root / "project.json").read_text(encoding="utf-8"))
    return project_root / project["assets"]["pose30Fps"], project_root / project["assets"]["pose24Fps"]


@app.post("/api/jobs/{job_id}/extend")
async def extend_job(job_id: str, payload: dict = Body(...)) -> dict:
    """Append new chunks as a child revision, retaining the complete parent take."""
    parent = _get_job(job_id)
    _discover_segments(parent)
    if parent.status != "complete" or len(parent.segments) != parent.chunks:
        raise HTTPException(409, "Continuation requires a completed parent")
    _adopt_legacy_job(parent)
    additional = int(payload.get("additionalChunks", 0))
    if additional < 1 or parent.chunks + additional > 40:
        raise HTTPException(400, f"additionalChunks must be between 1 and {40 - parent.chunks}")
    prompts = payload.get("prompts")
    points = payload.get("pathPoints")
    look_points = payload.get("lookPoints")
    if not isinstance(prompts, list) or len(prompts) != additional or any(not str(item).strip() for item in prompts):
        raise HTTPException(400, "Continuation needs one non-empty prompt per appended chunk")
    if not isinstance(points, list) or len(points) < 2:
        raise HTTPException(400, "Continuation needs a camera trajectory")
    if look_points is not None and (not isinstance(look_points, list) or len(look_points) < 2):
        raise HTTPException(400, "Continuation view control needs at least two points")
    prompts = [str(item).strip()[:4000] for item in prompts]
    parent_spec = json.loads((parent.root / "spec.json").read_text(encoding="utf-8"))
    parent_prompts = list(parent_spec.get("prompts") or [])
    if len(parent_prompts) != parent.chunks:
        raise HTTPException(409, "Parent prompt timeline is incomplete")

    project_root = PROJECTS_ROOT / str(parent.project_id)
    revision_id = _next_revision_id(project_root)
    revision_root = project_root / "revisions" / revision_id
    input_root = revision_root / "input"
    input_root.mkdir(parents=True, exist_ok=False)
    pose30_path = input_root / "trajectory_30fps.npz"
    pose24_path = input_root / "trajectory_24fps.npz"
    parent_pose30, _ = _job_pose_paths(parent)
    try:
        pose_info = append_pose_npz(
            parent_pose30,
            pose30_path,
            points,
            additional,
            payload.get("motionScale", 24),
            payload.get("verticalLift", 0),
            payload.get("fov", 70),
            bool(payload.get("followPath", True)),
            look_points,
            payload.get("lookYawDegrees", 90),
            payload.get("lookPitchDegrees", 45),
        )
        pose24_info = resample_pose_npz(pose30_path, pose24_path)
        schedule_path = input_root / "prompt_schedule.json"
        _write_json(schedule_path, [{"start_chunk": index, "prompt": prompt} for index, prompt in enumerate(prompts)])
        prompt_path = input_root / "prompt.json"
        _write_json(prompt_path, {"overall": {"full_prompt": prompts[0]}})
        context_chunks = min(parent.chunks, max(1, min(3, int(payload.get("contextChunks", 3)))))
        start_seconds = _chunk_start_frame(parent.chunks - context_chunks) / OUTPUT_FPS
        ref_video_sec = context_chunks * 1.5
        parent_video = await _ensure_final_revision(parent)
        case = {
            "name": "evoke_ui",
            "video_path": str(parent_video),
            "pose_path": str(pose24_path),
            "prompt_path": str(prompt_path),
            "segment_prompts_path": str(schedule_path),
            "video_fps": OUTPUT_FPS,
            "pose_source_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "pose_type": "vipe",
            "seed": int(payload.get("seed", parent_spec.get("seed", 42))),
        }
        (revision_root / "case.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_json(revision_root / "spec.json", {
            **parent_spec,
            "chunks": parent.chunks + additional,
            "prompts": parent_prompts + prompts,
            "seed": case["seed"],
            "continuation": {
                "parentRevisionId": parent.revision_id,
                "fromChunk": parent.chunks,
                "additionalChunks": additional,
                "contextChunks": context_chunks,
                "pathPoints": points,
                "lookPoints": look_points,
                "lookYawDegrees": payload.get("lookYawDegrees", 90),
                "lookPitchDegrees": payload.get("lookPitchDegrees", 45),
                "motionScale": payload.get("motionScale", 24),
                "verticalLift": payload.get("verticalLift", 0),
                "fov": payload.get("fov", 70),
                "followPath": bool(payload.get("followPath", True)),
                "startSeconds": start_seconds,
                "referenceSeconds": ref_video_sec,
                "pose30Fps": "input/trajectory_30fps.npz",
                "pose24Fps": "input/trajectory_24fps.npz",
                "pose": pose_info,
                "v2vPose": pose24_info,
            },
        })
        for index in range(parent.chunks):
            _copy_file(
                parent.root / parent.segments[index],
                revision_root / "output" / "evoke_ui" / "segments" / f"segment_{index:03d}_pred.mp4",
            )
    except Exception:
        shutil.rmtree(revision_root, ignore_errors=True)
        raise

    job = Job(
        id=uuid.uuid4().hex[:12], root=revision_root, chunks=parent.chunks + additional,
        mode="v2v", run_chunks=additional, branch_from_chunk=parent.chunks,
        project_id=parent.project_id, revision_id=revision_id, parent_revision_id=parent.revision_id,
        engine_output=revision_root / "suffix", ref_video_sec=ref_video_sec, start_seconds=start_seconds,
    )
    JOBS[job.id] = job
    _discover_segments(job)
    _persist_job(job)
    asyncio.create_task(_run_job(job))
    return job.snapshot()


@app.post("/api/jobs/{job_id}/retry")
async def retry_from_chunk(job_id: str, payload: dict = Body(...)) -> dict:
    """Create a child revision, preserving chunks before ``fromChunk``.

    Chunk 0 branches as a fresh i2v take. Later chunks use the retained parent
    video plus the synchronized 24 FPS pose as a bounded v2v context window.
    """
    parent = _get_job(job_id)
    _discover_segments(parent)
    if parent.status != "complete" or len(parent.segments) != parent.chunks:
        raise HTTPException(409, "Retry requires a completed parent with every chunk available")
    _adopt_legacy_job(parent)

    try:
        from_chunk = int(payload.get("fromChunk"))
    except (TypeError, ValueError) as error:
        raise HTTPException(400, "fromChunk must be an integer") from error
    if not 0 <= from_chunk < parent.chunks:
        raise HTTPException(400, f"fromChunk must be between 0 and {parent.chunks - 1}")

    parent_spec = json.loads((parent.root / "spec.json").read_text(encoding="utf-8"))
    prompts = payload.get("prompts", parent_spec.get("prompts"))
    if not isinstance(prompts, list) or len(prompts) != parent.chunks:
        raise HTTPException(400, "Retry needs one prompt for every logical chunk")
    prompts = [str(prompt).strip()[:4000] for prompt in prompts]
    if any(not prompt for prompt in prompts):
        raise HTTPException(400, "Prompt text cannot be empty")
    seed = int(payload.get("seed", parent_spec.get("seed", 42)))
    context_chunks = min(from_chunk, max(1, min(3, int(payload.get("contextChunks", 3)))))

    project_root = PROJECTS_ROOT / str(parent.project_id)
    project = json.loads((project_root / "project.json").read_text(encoding="utf-8"))
    assets = project["assets"]
    reference_path = project_root / assets["reference"]
    pose30_path, pose24_path = _job_pose_paths(parent)
    revision_id = _next_revision_id(project_root)
    revision_root = project_root / "revisions" / revision_id
    input_root = revision_root / "input"
    input_root.mkdir(parents=True, exist_ok=False)

    suffix_prompts = prompts[from_chunk:]
    schedule = [{"start_chunk": index, "prompt": prompt} for index, prompt in enumerate(suffix_prompts)]
    schedule_path = input_root / "prompt_schedule.json"
    _write_json(schedule_path, schedule)
    # infer_batch's v2v caption reader expects the dataset JSON envelope.
    prompt_path = input_root / "prompt.json"
    _write_json(prompt_path, {"overall": {"full_prompt": suffix_prompts[0]}})

    mode = "i2v" if from_chunk == 0 else "v2v"
    start_seconds = 0.0
    ref_video_sec = 0.0
    if mode == "i2v":
        case = {
            "name": "evoke_ui",
            "image_path": str(reference_path),
            "pose_path": str(pose30_path),
            "prompt": suffix_prompts[0],
            "segment_prompts_path": str(schedule_path),
            "pose_fps": POSE_FPS,
            "pose_source_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "pose_type": "vipe",
            "seed": seed,
        }
    else:
        context_start_chunk = from_chunk - context_chunks
        start_frame = _chunk_start_frame(context_start_chunk)
        end_frame = _chunk_start_frame(from_chunk)
        start_seconds = start_frame / OUTPUT_FPS
        ref_video_sec = (end_frame - start_frame) / OUTPUT_FPS
        parent_video = await _ensure_final_revision(parent)
        case = {
            "name": "evoke_ui",
            "video_path": str(parent_video),
            "pose_path": str(pose24_path),
            "prompt_path": str(prompt_path),
            "segment_prompts_path": str(schedule_path),
            "video_fps": OUTPUT_FPS,
            "pose_source_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "pose_type": "vipe",
            "seed": seed,
        }
    (revision_root / "case.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_json(revision_root / "spec.json", {
        **parent_spec,
        "prompts": prompts,
        "seed": seed,
        "branch": {
            "parentRevisionId": parent.revision_id,
            "fromChunk": from_chunk,
            "contextChunks": context_chunks,
            "mode": mode,
            "startSeconds": start_seconds,
            "referenceSeconds": ref_video_sec,
        },
    })

    # Materialize the accepted prefix in the child revision. New suffix segments
    # stream into the same canonical numbering without altering the parent.
    for index in range(from_chunk):
        source = parent.root / parent.segments[index]
        destination = revision_root / "output" / "evoke_ui" / "segments" / f"segment_{index:03d}_pred.mp4"
        _copy_file(source, destination)

    new_job = Job(
        id=uuid.uuid4().hex[:12],
        root=revision_root,
        chunks=parent.chunks,
        mode=mode,
        run_chunks=parent.chunks - from_chunk,
        branch_from_chunk=from_chunk,
        project_id=parent.project_id,
        revision_id=revision_id,
        parent_revision_id=parent.revision_id,
        engine_output=revision_root / ("suffix" if mode == "v2v" else "output"),
        ref_video_sec=ref_video_sec,
        start_seconds=start_seconds,
    )
    JOBS[new_job.id] = new_job
    _discover_segments(new_job)
    _persist_job(new_job)
    asyncio.create_task(_run_job(new_job))
    return new_job.snapshot()


def _get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = _get_job(job_id)
    _discover_segments(job)
    return job.snapshot()


@app.get("/api/session/latest")
async def latest_session() -> dict:
    if not JOBS:
        raise HTTPException(404, "No local session")
    job = max(JOBS.values(), key=lambda item: item.created_at)
    _discover_segments(job)
    spec = json.loads((job.root / "spec.json").read_text(encoding="utf-8"))
    snapshot = job.snapshot()
    snapshot["editor"] = {
        "chunks": job.chunks,
        "prompts": spec.get("prompts", []),
        "pathPoints": (spec.get("continuation") or {}).get("pathPoints", spec.get("pathPoints", [])),
        "lookPoints": (spec.get("continuation") or {}).get("lookPoints", spec.get("lookPoints", [])),
        "lookYawDegrees": (spec.get("continuation") or {}).get("lookYawDegrees", spec.get("lookYawDegrees", 90)),
        "lookPitchDegrees": (spec.get("continuation") or {}).get("lookPitchDegrees", spec.get("lookPitchDegrees", 45)),
        "motionScale": (spec.get("continuation") or {}).get("motionScale", spec.get("motionScale", 24)),
        "verticalLift": (spec.get("continuation") or {}).get("verticalLift", spec.get("verticalLift", 0)),
        "fov": (spec.get("continuation") or {}).get("fov", spec.get("fov", 70)),
        "followPath": (spec.get("continuation") or {}).get("followPath", spec.get("followPath", True)),
        "seed": spec.get("seed", 42),
        "referenceUrl": f"api/jobs/{job.id}/reference",
    }
    return snapshot


@app.get("/api/jobs/{job_id}/reference")
async def job_reference(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.project_id:
        project_root = PROJECTS_ROOT / str(job.project_id)
        project = json.loads((project_root / "project.json").read_text(encoding="utf-8"))
        target = project_root / project["assets"]["reference"]
    else:
        case = json.loads((job.root / "case.jsonl").read_text(encoding="utf-8").splitlines()[0])
        target = Path(case["image_path"])
    if not target.is_file():
        raise HTTPException(404, "Reference image not found")
    media_type, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=media_type)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = _get_job(job_id)

    async def stream() -> AsyncIterator[str]:
        previous = ""
        while True:
            _discover_segments(job)
            payload = json.dumps(job.snapshot(), ensure_ascii=False)
            if payload != previous:
                yield f"data: {payload}\n\n"
                previous = payload
            if job.status in {"complete", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.75)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if job.process and job.process.returncode is None:
        job.status, job.message = "cancelling", "Stopping inference"
        os.killpg(job.process.pid, signal.SIGTERM)
        # The actual CUDA call runs inside the persistent worker, not the lightweight per-job
        # launcher. Restarting the worker is the only safe interruption point today; its pipeline is
        # then preloaded again before another queued job starts.
        asyncio.create_task(_restart_worker())
    elif job.status == "queued":
        job.status, job.message = "cancelled", "Generation cancelled"
    return job.snapshot()


@app.get("/api/jobs/{job_id}/files/{relative_path:path}")
async def job_file(job_id: str, relative_path: str) -> FileResponse:
    job = _get_job(job_id)
    target = (job.root / relative_path).resolve()
    if job.root not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found")
    media_type, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=media_type)


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.status != "complete":
        raise HTTPException(409, "Complete video is available after generation finishes")
    target = await _ensure_final_revision(job)
    project = job.project_id or "project"
    revision = job.revision_id or job.id
    return FileResponse(
        target,
        media_type="video/mp4",
        filename=f"evoke-{project}-{revision}.mp4",
    )


if __name__ == "__main__":
    import uvicorn

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=os.environ.get("EVOKE_UI_HOST", "127.0.0.1"), port=int(os.environ.get("EVOKE_UI_PORT", "7860")))
