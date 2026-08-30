<h1 align="center">
  <sub><picture>
    <source media="(prefers-color-scheme: dark)" srcset="examples/logo/logo_dark.png">
    <img src="examples/logo/logo_light.png" width="38" alt="">
  </picture></sub>&nbsp;&nbsp;&nbsp;
  Alaya-EVOKE: From Linear-Scaling Supervision to Endless World
</h1>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="examples/logo/alaya-lab-horizontal-light.svg">
    <img src="examples/logo/alaya-lab-horizontal-dark.svg" width="152" alt="Alaya Lab">
  </picture>
</p>

<p align="center">
  <a href="https://evoke-world.github.io/Evoke/"><img src="https://img.shields.io/badge/🌐_Project_Page-evoke--world.github.io-1a73e8.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2608.13546"><img src="https://img.shields.io/badge/arXiv-2608.13546-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/AlayaLab/Evoke"><img src="https://img.shields.io/badge/🤗_Weights-AlayaLab/Evoke-ffce1c.svg" alt="Weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <b>A three-step, CFG-free world model that remembers where it has been, takes direction mid-flight,<br>
  and keeps going — 1.5 s of video every 2.11 s on a single H200.</b>
</p>

## ✨ Highlights

- 🏆 **State of the art on WBench** — as a *three-step* world model, while staying competitive in
  visual quality on **VBench-Long** and **VBench-2.0**.
- ⚡ **3 steps, zero CFG.** 1.5 s of 384×640 video every **2.11 s** on one H200 — one forward per
  step, not two. Few-step speed without few-step ceilings.
- 🌍 **Endless, not windowed.** Scene geometry lives in an external, camera-indexed **world state
  bank** instead of the denoiser's context. Only what the current view needs is retrieved, so the
  context stays *bounded* however long the session runs — no trading session length for memory.
- 🎛️ **Re-promptable mid-flight.** Per-chunk conditioning lets you change the prompt *while the
  rollout is running*: the sky ignites, the storm rolls in, no cut and no restart.
- 🧑‍🏫 **A teacher rebuilt for the long horizon.** Chunk-wise grouping, distant-frame retrieval and a
  linear-attention global state make its memory and compute grow **linearly** — which is what makes
  30 s self-forced supervision affordable.

## 📰 Latest News

- **2026-08-30 — Director Web UI.** EVOKE now ships with a local Web UI for per-chunk prompting,
  dual-stick camera control, streaming previews, continuation, and revision branching. See the
  [UI guide and interface preview](ui/README.md).
- **2026-08-14 — Day 0.** Alaya-EVOKE entered the official
  [WBench](https://github.com/meituan-longcat/WBench) table and achieved state of the art in its
  release snapshot, one day after the [first arXiv submission](https://arxiv.org/abs/2608.13546).

## 🎬 Demos

<p align="center">
  <a href="https://www.youtube.com/watch?v=QX7PBBaBGdc">
    <img src="assets/promo.jpg" width="100%" alt="Watch the EVOKE overview on YouTube">
  </a>
  <br>
  <sub><a href="https://www.youtube.com/watch?v=QX7PBBaBGdc"><b>▶&nbsp; Watch the overview</b></a></sub>
</p>

Every clip was produced by the launchers in this repo, on the data bundled in `examples/` — no external
dataset, no cherry-picking across seeds. The `Move` / `Rot` joystick is the camera-control HUD burned
into `geo_pred.mp4`.

### Re-prompting mid-rollout

Per-chunk conditioning lets the prompt change **while the rollout is running** — no cut, no restart.
Each schedule below switches at chunk 3 of 6 (213 frames, 8.9 s).

| | scene | the prompt switches to |
|---|---|---|
| <img src="assets/demo/segment_aurora.gif" width="240"> | frozen tundra, polar daylight | an aurora ignites across the whole sky |
| <img src="assets/demo/segment_meteor.gif" width="240"> | still mountain lake at night | a meteor shower opens overhead |
| <img src="assets/demo/segment_crystalstorm.gif" width="240"> | alien valley of crystal spires | an electrical storm arcs between the spires |
| <img src="assets/demo/segment_gateway.gif" width="240"> | dormant ring megastructure | the gateway ignites, energy column rising |

```bash
MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh
```

### Conditioning modes

4 chunks each (141 frames, 5.9 s; t2v is 140).

| | mode | input | camera |
|---|---|---|---|
| <img src="assets/demo/v2v.gif" width="240"> | `v2v` | reference video + pose track | yes — continues past the reference window |
| <img src="assets/demo/i2v.gif" width="240"> | `i2v` | single first frame + pose track | yes |
| <img src="assets/demo/t2v.gif" width="240"> | `t2v` | prompt only | no — the engine forbids warp + t2v |

```bash
MODE=v2v NUM_CHUNKS=4 bash scripts/inference/infer_post_distill.sh   # or MODE=i2v / MODE=t2v
```

### The other two models

Both run on the same bundled inputs as the demos above, so they are directly comparable.

| | model | steps | note |
|---|---|---|---|
| <img src="assets/demo/stage1.gif" width="240"> | `stage1_camera_control` | 50, CFG 5.0 | the multi-step model that precedes distillation — not distilled, so it is sampled like an ordinary diffusion model. Same case as `v2v` |
| <img src="assets/demo/teacher.gif" width="240"> | `evoke_teacher` | 50, CFG 5.0 | the dual-expert DMD teacher sampled directly, same input image as `i2v`; no camera conditioning |

```bash
NUM_CHUNKS=4 bash scripts/inference/infer_stage1.sh
bash scripts/inference/infer_evoke_teacher.sh
```

## 🧠 How it works

<p align="center"><img src="examples/logo/fig_n_inference.png" width="720" alt="EVOKE inference pipeline"></p>

The student is autoregressive over latent chunks (`latent_window_size = 9`). Each chunk is laid out
along the RoPE frame index as:

```
prefix | long(16) | mid(2) | warp(W) | prev_short(1) | noise(W)
```

| tier | what it is |
|---|---|
| `prefix` | the frame-0 global anchor (i2v: the input image; v2v: the first latent of the reference video) |
| `long` / `mid` | multi-term parametric memory (`history_sizes = [16, 2, 1]`) with coarser patch kernels — long `(4,8,8)`, mid `(2,4,4)`, everything else `(1,2,2)` |
| `warp` | the world state bank rendered into this view; its RoPE overlaps the noise window |
| `prev_short` | the last latent of the previous chunk, the continuity anchor closest to the noise |

Every tier lives at the same latent resolution (`res/8`); compression comes only from the patchify
convolution kernel, and no low-resolution latents are stored. In the short tier
`[prefix | warp | prev_short]` the residual MLP and the per-stage compression act only on the warp
frames in the middle.

The world state bank itself has three operations: **write** — a monocular depth model estimates depth
for the emitted chunk under its known poses, unprojected into a persistent point cloud; **read** — the
current camera pose addresses the bank directly, with sources ranked by co-visibility, up to eight
fused, and a batched z-buffered scatter returning a warped image plus a per-pixel visibility mask; and
**evict** — an optional retention window, which hour-scale runs enable explicitly.

## ⚙️ Environment

Python 3.10 + CUDA 12.4. The pins in `requirements.txt` are the environment actually in use —
torch 2.4 / deepspeed 0.14.5 / flash-attn are load-bearing, not aspirational.

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Two things `pip` will not do: `diffusers` is pinned to a **development fork that is not on PyPI**
(install it first or nothing imports), and `postprocess_viz.py` needs the **`ffmpeg` binary** on
`PATH`. Both depth backends (ViGeo, Depth-Anything-3) are **vendored** under
`evoke/third_party/`, so only their weights are downloaded — see [Weights](#-weights).

## 📦 Weights

Everything goes under `models/` (gitignored). Every released EVOKE directory is the **parent** of a
`transformer/`, because it loads as `from_pretrained(path, subfolder="transformer")`.

```
models/
├── model_index.json                              a map of this layout -- not a loadable pipeline
├── evoke-base/                                   vae / text_encoder / tokenizer / scheduler only
├── ViGeo1.1/vigeo.pt                             depth backend -- REQUIRED
├── DA3/{config.json,model.safetensors}           depth backend -- OPTIONAL
└── evoke/
    ├── stage1_camera_control/transformer/        multi-step camera-controllable model
    ├── stage2_few_step_training/transformer/     few-step distillation (3-step pyramid)
    ├── stage3_long_distillation/transformer/     30s long-video distillation (post-distill init)
    ├── stage3_post_distillation/transformer/     the shipped model
    └── evoke_teacher/{high,low}_noise/           the two DMD teacher experts -- training only
```

```bash
# EVOKE -- the four released models, the teacher, and the base components
hf download AlayaLab/Evoke --local-dir models

# ViGeo -- REQUIRED. The depth backend behind the world state bank; every shipped
# recipe uses it (DEPTH_BACKEND=vigeo, cloud_warp.backend: vigeo).
hf download pkqbajng/ViGeo --local-dir models/ViGeo1.1
```

**Depth-Anything-3 is optional** — nothing in the default path touches it, and you only need it if you
set `DEPTH_BACKEND=da3`. Get the `da3-giant` weights from
[depth-anything-3](https://huggingface.co/spaces/depth-anything/depth-anything-3) and drop
`config.json` + `model.safetensors` into `models/DA3/`. Switching backend is a **recipe change**, not a
speed knob — training and inference must agree on it.

> Both depth backends ship under **CC-BY-NC-4.0**, which is more restrictive than this repo's
> Apache-2.0. Check their licences before any commercial use.

## 🖥️ Director Web UI

The repository includes a local, single-GPU Director UI for the released post-distill model. It
supports a reference image, per-chunk prompts, independent movement/look camera controls, streaming
chunk previews, lazy full-video download, continuation, and non-destructive revision branching. The
bundled default is the six-chunk Meteor case.

Complete the Environment and Weights sections above first, make sure `ffmpeg` is on `PATH`, then run
from the repository root:

```bash
python -m pip install -r ui/requirements.txt
python ui/app.py
```

Open <http://127.0.0.1:7860> and wait for the status to become `READY`. Generated projects, jobs,
videos, logs, and caches are excluded from source releases by `ui/.gitignore`.

The server binds to localhost by default and does not provide authentication or rate limiting. Do
not expose it directly to the public internet. Configuration, model-path overrides, data layout,
and the deliberately exposed reverse-proxy form are documented in **[`ui/README.md`](ui/README.md)**.

## 🚀 Inference

**384 × 640 @ 24 fps.** One chunk = 36 frames = 1.5 s, so `NUM_CHUNKS=20` is a 30 s clip. All four
commands run on the data bundled in `examples/` — one case each, four for `segment` — no external
dataset:

```bash
MODE=t2v     NUM_CHUNKS=20              bash scripts/inference/infer_post_distill.sh   # prompt only
MODE=i2v     NUM_CHUNKS=20              bash scripts/inference/infer_post_distill.sh   # first frame + pose
MODE=v2v     NUM_CHUNKS=20              bash scripts/inference/infer_post_distill.sh   # ref video + pose
MODE=segment NUM_CHUNKS=6  MAX_CASES=0  bash scripts/inference/infer_post_distill.sh   # prompt switches mid-rollout
```

| Launcher | Weights | Steps |
|---|---|---|
| `infer_post_distill.sh` | `models/evoke/stage3_post_distillation` | 3, CFG-free |
| `infer_stage1.sh` | `models/evoke/stage1_camera_control` | 50, CFG 5.0 |
| `infer_evoke_teacher.sh` | `models/evoke/evoke_teacher` | 50, CFG 5.0 (example only) |

Results land in `<OUT_ROOT>/<case>/geo_pred.mp4`. Every distilled model was trained on v2v
conditioning alone (`geo_condition_{i2v,t2v}_ratio: 0.0`), so `MODE=i2v|t2v` on them is **zero-shot**
and the launchers say so at startup. Only `stage1_camera_control` has all three modes in distribution
(ratios 0.1 / 0.2).

Everything else — the mode × model matrix, hour-scale rollouts, per-chunk log format, and how to point
the launchers at your own data — is in **[`scripts/inference/README.md`](scripts/inference/README.md)**.
Every launcher also has `-h`.

## 🗝️ Training

One launcher per released model. Each initialises from **its own released checkpoint**, so you
continue from where we left off — nobody reproduces a stage from scratch, and the pretraining data is
not part of this release:

```bash
bash scripts/training/train_stage1_camera_control.sh      # no teacher (not a distillation)   1x8
bash scripts/training/train_stage2_few_step_training.sh   # teacher: stage1_camera_control    1x8
bash scripts/training/train_stage3_long_distillation.sh   # teacher: evoke_teacher, 2 experts 6x8
bash scripts/training/train_stage3_post_distillation.sh   # teacher: stage1_camera_control    6x8
```

Post-distillation goes back to the stage-1 teacher on purpose: it is a short run that firms up camera
control, not another long-horizon distillation.

Each writes to `models/train/<same-name>/`; move or symlink it into `models/evoke/` to serve it.

All four start with **no external dataset** — they point at the single 60 s clip in `examples/data/`,
so they run as a pipeline check, not a real training run (one clip overfits immediately). For a real
run swap `data_yaml_path` to the production mix named beside it in the config.

Scale is set by `ACCELERATE_CONFIG` — the topology is baked into the accelerate yaml, so do not
override it with `--num_machines`. To merge a LoRA checkpoint into a full transformer, see
`tools/merge_lora_ckpt.py` (use `--dtype fp32`: the delta is ~5e-4 of the weight magnitude and bf16
swallows it).

## 📝 Notes

- **The warp / attention recipe must match between training and inference.** A mismatch silently
  degrades quality rather than failing — every knob in the launchers is annotated with the config
  field it mirrors.
- Resolution is data driven, but keep the width a **multiple of 64** so the long tier and the
  quarter-resolution pyramid stage both divide evenly.

## 👍 Acknowledgement

The EVOKE teacher is built on **[LingBot-World](https://github.com/robbyant/lingbot-world)**. The
vae / text encoder / tokenizer / scheduler in `models/evoke-base` come from the released
**[Helios](https://github.com/PKU-YuanGroup/Helios)** base, which traces them to **Wan**. The optional
Director UI LightTAE decoder adapter is adapted from
**[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)** under Apache-2.0.

## 🔒 License

Apache-2.0, see `LICENSE`. Vendored third-party code keeps its own license and provenance under
`evoke/third_party/*`.

## ✏️ Citation

```bibtex
@misc{evoke2026,
  title         = {Alaya-EVOKE: From Linear-Scaling Supervision to Endless World},
  author        = {Yuanyang Yin and Gongxuan Wang and Yifan Zhan and Chuanhao Li and Kaipeng Zhang and Feng Zhao},
  year          = {2026},
  eprint        = {2608.13546},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.13546},
}
```
