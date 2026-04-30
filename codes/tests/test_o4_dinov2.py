from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from o4_dinov2 import (
    _extract_patch_tokens,
    _load_dinov2_model,
    _resolve_dinov2_model_spec,
    resolve_dinov2_checkpoint_path,
    resolve_dinov2_repo_path,
    resolve_o4_execution_mode,
)


class ResolveO4ExecutionModeTests(unittest.TestCase):
    def test_baseline_mode_remains_available(self) -> None:
        status = resolve_o4_execution_mode(
            "baseline",
            use_torch=False,
            dinov2_model_name="facebook/dinov2-base",
            dinov2_repo_path=None,
            dinov2_checkpoint_path=None,
        )

        self.assertTrue(status.available)
        self.assertEqual(status.selected_mode, "baseline")
        self.assertEqual(status.descriptor_source, "handcrafted_patch_tokens")

    def test_dinov2_mode_requires_torch(self) -> None:
        status = resolve_o4_execution_mode(
            "dinov2_cost_volume",
            use_torch=False,
            dinov2_model_name="facebook/dinov2-base",
            dinov2_repo_path=None,
            dinov2_checkpoint_path=None,
        )

        self.assertFalse(status.available)
        self.assertEqual(status.selected_mode, "dinov2_cost_volume")
        self.assertIn("requires the torch backend", status.reason)

    def test_dinov2_mode_requires_existing_local_checkpoint(self) -> None:
        status = resolve_o4_execution_mode(
            "dinov2_cost_volume",
            use_torch=True,
            dinov2_model_name="facebook/dinov2-base",
            dinov2_repo_path=None,
            dinov2_checkpoint_path="/tmp/does-not-exist/dinov2_vitb14_reg4_pretrain.pth",
        )

        self.assertFalse(status.available)
        self.assertIn("dinov2_checkpoint_path", status.reason)
        self.assertIn("/tmp/does-not-exist/dinov2_vitb14_reg4_pretrain.pth", status.reason)

    def test_dinov2_mode_is_available_with_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dinov2_vitb14_reg4_pretrain.pth"
            checkpoint.write_bytes(b"test")
            status = resolve_o4_execution_mode(
                "dinov2_cost_volume",
                use_torch=True,
                dinov2_model_name="facebook/dinov2-base",
                dinov2_repo_path=None,
                dinov2_checkpoint_path=checkpoint,
            )

        self.assertTrue(status.available)
        self.assertEqual(status.descriptor_source, "dinov2_direct_model_patch_tokens")
        self.assertIn("pretrained DINOv2 model forward pass", status.reason)
        self.assertIn("checkpoint=", status.reason)

    def test_dinov2_mode_rejects_missing_repo_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dinov2_vitb14_reg4_pretrain.pth"
            checkpoint.write_bytes(b"test")
            status = resolve_o4_execution_mode(
                "dinov2_cost_volume",
                use_torch=True,
                dinov2_model_name="facebook/dinov2-base",
                dinov2_repo_path=Path(tmpdir) / "missing-dinov2",
                dinov2_checkpoint_path=checkpoint,
            )

        self.assertFalse(status.available)
        self.assertIn("repo path was not found", status.reason)

    def test_default_base_checkpoint_mapping_uses_explicit_models_root(self) -> None:
        checkpoint = resolve_dinov2_checkpoint_path("facebook/dinov2-base", None)

        self.assertEqual(
            checkpoint,
            Path("/limx_embop/tos/users/Nemo/self-work/models/dinov2_vitb14_reg4_pretrain.pth"),
        )

    def test_all_supported_variants_resolve_expected_builder_and_checkpoint(self) -> None:
        cases = (
            ("facebook/dinov2-small", "dinov2_vits14_reg", "dinov2_vits14_reg4_pretrain.pth"),
            ("facebook/dinov2-base", "dinov2_vitb14_reg", "dinov2_vitb14_reg4_pretrain.pth"),
            ("dinov2_vits14_reg", "dinov2_vits14_reg", "dinov2_vits14_reg4_pretrain.pth"),
            ("dinov2_vitb14_reg", "dinov2_vitb14_reg", "dinov2_vitb14_reg4_pretrain.pth"),
        )

        for selector, builder_name, checkpoint_name in cases:
            with self.subTest(selector=selector):
                spec = _resolve_dinov2_model_spec(selector)

                self.assertEqual(spec.builder_name, builder_name)
                self.assertEqual(spec.checkpoint_filename, checkpoint_name)
                self.assertEqual(
                    resolve_dinov2_checkpoint_path(selector, None),
                    Path("/limx_embop/tos/users/Nemo/self-work/models") / checkpoint_name,
                )

    def test_large_variant_is_no_longer_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported O4 DINOv2 model selector"):
            _resolve_dinov2_model_spec("facebook/dinov2-large")


class ExtractPatchTokensTests(unittest.TestCase):
    def test_drops_prefix_tokens_from_last_hidden_state(self) -> None:
        tokens = np.arange(1 * 6 * 3, dtype=np.float32).reshape(1, 6, 3)

        model = lambda pixel_values: SimpleNamespace(last_hidden_state=tokens)
        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
        with patch("o4_dinov2._require_torch", return_value=fake_torch):
            resolved = _extract_patch_tokens(model, np.zeros((1, 3, 2, 2), dtype=np.float32), expected_tokens=4)

        self.assertTrue(np.array_equal(resolved, tokens[:, 2:, :]))

    def test_prefers_forward_features_patch_tokens(self) -> None:
        tokens = np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)

        class FakeModel:
            def forward_features(self, pixel_values):
                return {"x_norm_patchtokens": tokens}

        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
        with patch("o4_dinov2._require_torch", return_value=fake_torch):
            resolved = _extract_patch_tokens(FakeModel(), np.zeros((1, 3, 2, 2), dtype=np.float32), expected_tokens=4)

        self.assertTrue(np.array_equal(resolved, tokens))


class LoadDinov2ModelTests(unittest.TestCase):
    def test_resolve_dinov2_repo_path_requires_backbones_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "dinov2" / "hub").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "does not expose dinov2.hub.backbones"):
                resolve_dinov2_repo_path(repo_path)

    def test_requires_local_dinov2_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dinov2_vitb14_reg4_pretrain.pth"
            checkpoint.write_bytes(b"test")
            repo_path = Path(tmpdir) / "repo"
            (repo_path / "dinov2" / "hub").mkdir(parents=True)
            (repo_path / "dinov2" / "hub" / "backbones.py").write_text("# test\n", encoding="utf-8")

            fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: {"model": {}})
            with patch("o4_dinov2._require_torch", return_value=fake_torch):
                with patch("importlib.import_module", side_effect=ImportError("missing local dinov2")):
                    with self.assertRaisesRegex(ImportError, "requires a local DINOv2 Python implementation"):
                        _load_dinov2_model("facebook/dinov2-base", checkpoint, "cpu", repo_path=repo_path)

    def test_rejects_known_checkpoint_variant_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dinov2_vits14_reg4_pretrain.pth"
            checkpoint.write_bytes(b"test")

            with self.assertRaisesRegex(ValueError, "expected_checkpoint='dinov2_vitb14_reg4_pretrain.pth'"):
                _load_dinov2_model("facebook/dinov2-base", checkpoint, "cpu")


if __name__ == "__main__":
    unittest.main()
