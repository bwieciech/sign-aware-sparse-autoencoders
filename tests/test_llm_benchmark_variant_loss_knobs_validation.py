import os
import tempfile
import unittest
from typing import Any, Dict

from pydantic import ValidationError

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig


def _base_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {
            "backend": "local_jsonl",
            "data_files": ["dummy.jsonl"],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "variants": [
            {
                "name": "variant",
                "sae_class": "GatedSAE",
                "sign_aware": True,
                "d_hidden": 8,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


class TestLLMBenchmarkVariantLossKnobsValidation(unittest.TestCase):
    def test_variant_loss_knobs_defaults_are_applied(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))

            # when
            config = LLMBenchmarkConfig.model_validate(payload)

            # then
            variant = config.variants[0]
            self.assertEqual(float(variant.aux_loss_coeff), 1.0)
            self.assertEqual(float(variant.dead_zone_reg_coeff), 0.0)
            self.assertEqual(str(variant.dead_zone_reg_type), "l2")

    def test_variant_loss_knobs_accept_valid_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["aux_loss_coeff"] = 0.75
            payload["variants"][0]["dead_zone_reg_coeff"] = 0.125
            payload["variants"][0]["dead_zone_reg_type"] = "l2"

            # when
            config = LLMBenchmarkConfig.model_validate(payload)

            # then
            variant = config.variants[0]
            self.assertAlmostEqual(float(variant.aux_loss_coeff), 0.75, places=12)
            self.assertAlmostEqual(float(variant.dead_zone_reg_coeff), 0.125, places=12)
            self.assertEqual(str(variant.dead_zone_reg_type), "l2")

    def test_variant_rejects_negative_aux_loss_coeff(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["aux_loss_coeff"] = -0.1

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(payload)

    def test_variant_rejects_non_finite_dead_zone_reg_coeff(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["dead_zone_reg_coeff"] = float("inf")

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(payload)

    def test_variant_rejects_unsupported_dead_zone_reg_type(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["dead_zone_reg_type"] = "l1"

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(payload)

    def test_variant_rejects_non_string_dead_zone_reg_type(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["dead_zone_reg_type"] = 123

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(payload)

    def test_variant_rejects_non_default_aux_loss_for_non_gated_sae(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["sae_class"] = "SAE"
            payload["variants"][0]["aux_loss_coeff"] = 0.5

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(payload)

    def test_variant_rejects_dead_zone_regularization_without_sign_aware_gated_sae(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["sign_aware"] = False
            payload["variants"][0]["dead_zone_reg_coeff"] = 0.1

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(payload)

    def test_variant_accepts_dead_zone_regularization_for_sign_aware_gated_sae(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            payload["variants"][0]["dead_zone_reg_coeff"] = 0.1

            # when
            config = LLMBenchmarkConfig.model_validate(payload)

            # then
            variant = config.variants[0]
            self.assertAlmostEqual(float(variant.dead_zone_reg_coeff), 0.1, places=12)


if __name__ == "__main__":
    unittest.main()
