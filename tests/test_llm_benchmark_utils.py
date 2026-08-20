import json
import os
import tempfile
import unittest
from subprocess import CompletedProcess
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.experiments.llm_benchmark.environment import (
    _collect_pip_freeze,
    _package_version,
    clear_pip_freeze_cache,
    collect_environment_metadata,
)
from sgnsae.experiments.llm_benchmark.paths import (
    _strip_float_trailing_zeros,
    build_subrun_relative_dir,
    format_sweep_value_for_dir,
    slugify_variant_name,
)
from sgnsae.experiments.llm_benchmark.runner import run_llm_sae_benchmark
from sgnsae.experiments.llm_benchmark.runner import _output_subdir_path


def _minimal_valid_config(output_dir: str) -> Dict[str, Any]:
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
                "name": "baseline_relu",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


class TestLLMBenchmarkUtils(unittest.TestCase):
    def test_slugify_variant_name_normalizes(self) -> None:
        # given
        raw = " My Variant! "

        # when
        slug = slugify_variant_name(raw)

        # then
        self.assertEqual(slug, "my_variant")

    def test_slugify_variant_name_rejects_empty_and_all_symbols(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = slugify_variant_name("")
        with self.assertRaises(ValueError):
            _ = slugify_variant_name("!!!")

    def test_format_sweep_value_for_dir_formats_and_validates(self) -> None:
        # given / when / then
        self.assertEqual(format_sweep_value_for_dir(32), "32")
        self.assertEqual(format_sweep_value_for_dir(0.0), "0")
        self.assertEqual(format_sweep_value_for_dir(1e-3), "1e-03")
        self.assertEqual(format_sweep_value_for_dir(1e3), "1e03")
        with self.assertRaises(ValueError):
            _ = format_sweep_value_for_dir(True)
        with self.assertRaises(ValueError):
            _ = format_sweep_value_for_dir("not-a-number")  # type: ignore[arg-type]

    def test_strip_float_trailing_zeros_returns_input_without_exponent(self) -> None:
        # given / when / then
        self.assertEqual(_strip_float_trailing_zeros("1.2300"), "1.2300")

    def test_build_subrun_relative_dir_rejects_unknown_sweep_param(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = build_subrun_relative_dir(
                variant_slug="v",
                seed=0,
                sweep_param="bad",
                sweep_value=1,
            )

    def test_package_version_returns_none_for_missing_package(self) -> None:
        # given / when / then
        self.assertIsNone(_package_version("this_package_should_not_exist_12345"))

    def test_collect_pip_freeze_handles_exceptions_and_nonzero_exit(self) -> None:
        # given / when / then
        with patch(
            "sgnsae.experiments.llm_benchmark.environment.subprocess.run",
            side_effect=RuntimeError("boom"),
        ):
            lines, error = _collect_pip_freeze()
            self.assertIsNone(lines)
            self.assertIsInstance(error, str)

        with patch(
            "sgnsae.experiments.llm_benchmark.environment.subprocess.run",
            return_value=CompletedProcess(
                args=["pip", "freeze"],
                returncode=1,
                stdout="",
                stderr="nope",
            ),
        ):
            lines, error = _collect_pip_freeze()
            self.assertIsNone(lines)
            self.assertIsInstance(error, str)

    def test_collect_environment_metadata_handles_cuda_device_name_error(self) -> None:
        # given
        with patch(
            "sgnsae.experiments.llm_benchmark.environment.torch.cuda.is_available",
            return_value=True,
        ), patch(
            "sgnsae.experiments.llm_benchmark.environment.torch.cuda.get_device_name",
            side_effect=RuntimeError("boom"),
        ):
            # when
            metadata = collect_environment_metadata()

            # then
            self.assertTrue(metadata["torch"]["cuda_available"])
            self.assertIsNone(metadata["torch"]["gpu_name"])

    def test_clear_pip_freeze_cache_forces_fresh_collection(self) -> None:
        # given
        clear_pip_freeze_cache()
        with patch(
            "sgnsae.experiments.llm_benchmark.environment._collect_pip_freeze",
            return_value=(["alpha==1.0"], None),
        ) as collect_mock:
            first = collect_environment_metadata()
            second = collect_environment_metadata()

        self.assertEqual(collect_mock.call_count, 1)
        self.assertEqual(first["pip_freeze"], ["alpha==1.0"])
        self.assertEqual(second["pip_freeze"], ["alpha==1.0"])

        # when
        clear_pip_freeze_cache()
        with patch(
            "sgnsae.experiments.llm_benchmark.environment._collect_pip_freeze",
            return_value=(["beta==2.0"], None),
        ) as collect_mock:
            refreshed = collect_environment_metadata()

        # then
        self.assertEqual(collect_mock.call_count, 1)
        self.assertEqual(refreshed["pip_freeze"], ["beta==2.0"])
        clear_pip_freeze_cache()

    def test_runner_rejects_validate_only_with_stage_override(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with self.assertRaises(ValueError):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["cache"],
                    validate_only=True,
                )
            self.assertFalse(os.path.exists(output_dir))

    def test_runner_rejects_unknown_stage_override(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with self.assertRaises(ValueError):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["not_a_stage"],
                )
            self.assertFalse(os.path.exists(output_dir))

    def test_runner_resume_match_sets_resume_check_passed(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["resume_check"]["status"], "passed")

    def test_runner_existing_config_invalid_json_is_treated_as_no_fingerprint(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                f.write("{not json")

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            with open(os.path.join(output_dir, "config.json"), "r") as f:
                parsed = json.load(f)
            self.assertIn("config_fingerprint", parsed)

    def test_runner_existing_config_missing_fingerprint_is_treated_as_no_fingerprint(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump({"config_fingerprint": None}, f)

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            with open(os.path.join(output_dir, "config.json"), "r") as f:
                parsed = json.load(f)
            self.assertIsInstance(parsed["config_fingerprint"], str)

    def test_runner_existing_run_status_invalid_json_does_not_crash(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "run_status.json"), "w") as f:
                f.write("{not json")

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            with open(os.path.join(output_dir, "run_status.json"), "r") as f:
                parsed = json.load(f)
            self.assertIn("schema_version", parsed)

    def test_runner_invalid_relative_subrun_dir_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            resolved = {
                "output_dir": output_dir,
                "config_fingerprint": "abc",
                "subruns": [{"relative_subrun_dir": None}],
            }

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_benchmark_config",
                return_value=resolved,
            ):
                with self.assertRaises(ValueError):
                    _ = run_llm_sae_benchmark(config_path=config_path)

    def test_runner_relative_subrun_dir_with_parent_segment_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            resolved = {
                "output_dir": output_dir,
                "config_fingerprint": "abc",
                "subruns": [{"relative_subrun_dir": "../escaped"}],
            }

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_benchmark_config",
                return_value=resolved,
            ):
                with self.assertRaises(ValueError):
                    _ = run_llm_sae_benchmark(config_path=config_path)

    def test_output_subdir_path_rejects_empty_relative_path(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = _output_subdir_path("/tmp/run_root", "/")

    def test_output_subdir_path_rejects_commonpath_mismatch(self) -> None:
        # given
        with patch(
            "sgnsae.experiments.llm_benchmark.runner.os.path.commonpath",
            return_value="/tmp/other",
        ):
            # when / then
            with self.assertRaises(ValueError):
                _ = _output_subdir_path("/tmp/run_root", "models/variant/seed_0/k_32")

    def test_output_subdir_path_rejects_commonpath_errors(self) -> None:
        # given
        with patch(
            "sgnsae.experiments.llm_benchmark.runner.os.path.commonpath",
            side_effect=ValueError("mismatch"),
        ):
            # when / then
            with self.assertRaises(ValueError):
                _ = _output_subdir_path("/tmp/run_root", "models/variant/seed_0/k_32")
