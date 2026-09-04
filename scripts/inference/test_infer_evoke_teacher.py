"""Lightweight tests for the standalone EvokeTeacher inference configuration path.

The heavyweight runtime modules and 14B model are stubbed so this can run with only the Python
standard library:

    python -m unittest scripts.inference.test_infer_evoke_teacher
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


def _identity_no_grad():
    return lambda function: function


def _load_inference_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = _identity_no_grad
    fake_torch.bfloat16 = "bfloat16"

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKLWan = object
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.UMT5EncoderModel = object

    fake_wrapper_module = types.ModuleType("evoke.modules.evoke_teacher.wrapper")
    fake_wrapper_module.EvokeTeacherScoreWrapper = object
    fake_wrapper_module.build_i2v_y = object
    fake_utils_module = types.ModuleType("evoke.utils.utils_base")
    fake_utils_module.encode_prompt = object

    stubs = {
        "numpy": types.ModuleType("numpy"),
        "torch": fake_torch,
        "diffusers": fake_diffusers,
        "transformers": fake_transformers,
        "evoke.modules.evoke_teacher.wrapper": fake_wrapper_module,
        "evoke.utils.utils_base": fake_utils_module,
    }
    script = Path(__file__).with_name("infer_evoke_teacher.py")
    spec = importlib.util.spec_from_file_location("_infer_evoke_teacher_under_test", script)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class _FakeBlock:
    num_nearby_frames = 3
    overlap_size = 1
    per_frame_tokens = 960
    select_scales = ["1x", "2x", "4x", "8x"]

    def __init__(self, chunk_size, num_select_frames):
        self.chunk_size = chunk_size
        self.num_select_frames = num_select_frames


class _FakeExpert:
    def __init__(self, overrides):
        self.blocks = [_FakeBlock(overrides["chunk_size"], overrides["num_select_frames"])]


class _FakeWrapper:
    def __init__(self, overrides):
        self.dit_low = _FakeExpert(overrides)
        self.dit_high = _FakeExpert(overrides)

    def to(self, _device):
        return self

    def eval(self):
        return self


class InferEvokeTeacherSparseConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_inference_module()

    def _build(self, extra_args=()):
        args = self.module.parse_args(["--prompt", "test", *extra_args])
        captured = {}

        def fake_factory(**kwargs):
            captured.update(kwargs)
            return _FakeWrapper(kwargs["model_cfg_overrides"])

        with mock.patch.object(self.module, "EvokeTeacherScoreWrapper", side_effect=fake_factory):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                wrapper = self.module.build_teacher_wrapper(args, "cpu", "bfloat16")
        return wrapper, captured, output.getvalue()

    def test_shipped_defaults_reach_constructed_model(self):
        wrapper, captured, output = self._build()
        self.assertEqual(captured["model_cfg_overrides"], {
            "chunk_size": 9,
            "num_select_frames": 1,
        })
        block = wrapper.dit_low.blocks[0]
        self.assertEqual((block.chunk_size, block.num_select_frames), (9, 1))
        for name in ("chunk_size", "num_select_frames", "num_nearby_frames", "overlap_size",
                     "per_frame_tokens", "select_scales"):
            self.assertIn(f"{name}=", output)

    def test_cli_override_reaches_constructed_model(self):
        wrapper, captured, _ = self._build(("--chunk_size", "12", "--num_select_frames", "2"))
        self.assertEqual(captured["model_cfg_overrides"], {
            "chunk_size": 12,
            "num_select_frames": 2,
        })
        block = wrapper.dit_low.blocks[0]
        self.assertEqual((block.chunk_size, block.num_select_frames), (12, 2))


if __name__ == "__main__":
    unittest.main()
