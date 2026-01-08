import io
import json
import os
import tempfile
import types
import unittest
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.scripts import run_toy_model_geometry_validation_experiment as script_module


class TestToyModelGeometryValidationCli(unittest.TestCase):
    def _build_raw_config(self, output_dir: str) -> Dict[str, Any]:
        return {
            "output_dir": output_dir,
            "device": "auto",
            "run_id": "unittest_run",
            "data": {
                "seed": 0,
                "num_train": 2,
                "num_val": 2,
                "num_test": 2,
                "num_pairs": 2,
                "direction_mode": "uniform",
                "pair_axis_active_prob": 0.2,
                "pair_feature_prob": 0.5,
                "pair_correlation": -1.0,
                "magnitude_dist": "constant",
                "constant_magnitude": 1.0,
                "noise_std": 0.0,
            },
            "models": [
                {
                    "name": "relu_sae",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 4,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
        }

    def test_cli_main_runs_and_writes_config_and_environment(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            stdout = io.StringIO()
            argv = ["prog", "--config-path", config_path]

            # when
            with (
                patch.object(script_module.sys, "argv", argv),
                patch.object(script_module.sys, "stdout", stdout),
                patch.object(
                    script_module,
                    "validate_toy_model_geometry_validation_dependencies",
                    return_value=None,
                ),
            ):
                script_module.main()

            # then
            output = stdout.getvalue()
            self.assertIn("Loaded Toy Model Geometry Validation config:", output)
            self.assertIn("Run complete. Artifacts written to:", output)

            expected_run_dir = os.path.join(
                tmpdir, "toy_model_geometry_validation", raw_config["run_id"]
            )
            self.assertTrue(os.path.isdir(expected_run_dir))
            self.assertTrue(
                os.path.isfile(os.path.join(expected_run_dir, "config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(expected_run_dir, "environment.json"))
            )

    def test_import_or_raise_fails_with_install_hint_on_missing_module(self) -> None:
        # given
        def raise_module_not_found(_: str) -> types.ModuleType:
            raise ModuleNotFoundError("missing")

        with patch.object(
            script_module.importlib,
            "import_module",
            side_effect=raise_module_not_found,
        ):
            # when & then
            with self.assertRaises(RuntimeError) as context:
                _ = script_module._import_or_raise("not_a_module")

        self.assertIn("not_a_module", str(context.exception))
        self.assertIn(script_module.INSTALL_HINT, str(context.exception))

    def test_validate_dependencies_sets_agg_backend_before_pyplot_import(self) -> None:
        # given
        call_order: list[str] = []
        fake_torch = types.ModuleType("torch")
        fake_numpy = types.ModuleType("numpy")
        fake_pyplot = types.ModuleType("matplotlib.pyplot")

        fake_matplotlib = types.ModuleType("matplotlib")

        def use(backend: str) -> None:
            call_order.append(f"use:{backend}")

        setattr(fake_matplotlib, "use", use)

        fake_scipy = types.ModuleType("scipy")
        fake_einops = types.ModuleType("einops")
        setattr(fake_einops, "einsum", object())

        def patched_import_or_raise(module_name: str) -> types.ModuleType:
            call_order.append(module_name)
            modules_by_name = {
                "torch": fake_torch,
                "numpy": fake_numpy,
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
                "scipy": fake_scipy,
                "einops": fake_einops,
            }
            return modules_by_name[module_name]

        # when
        with patch.object(
            script_module,
            "_import_or_raise",
            side_effect=patched_import_or_raise,
        ):
            script_module.validate_toy_model_geometry_validation_dependencies()

        # then
        self.assertIn("matplotlib", call_order)
        self.assertIn("use:Agg", call_order)
        self.assertIn("matplotlib.pyplot", call_order)
        self.assertLess(call_order.index("matplotlib"), call_order.index("use:Agg"))
        self.assertLess(
            call_order.index("use:Agg"), call_order.index("matplotlib.pyplot")
        )

    def test_validate_dependencies_ignores_matplotlib_backend_errors(self) -> None:
        # given
        fake_torch = types.ModuleType("torch")
        fake_numpy = types.ModuleType("numpy")
        fake_pyplot = types.ModuleType("matplotlib.pyplot")

        fake_matplotlib = types.ModuleType("matplotlib")

        def raise_backend_error(_: str) -> None:
            raise RuntimeError("backend error")

        setattr(fake_matplotlib, "use", raise_backend_error)

        fake_scipy = types.ModuleType("scipy")
        fake_einops = types.ModuleType("einops")
        setattr(fake_einops, "einsum", object())

        def patched_import_or_raise(module_name: str) -> types.ModuleType:
            modules_by_name = {
                "torch": fake_torch,
                "numpy": fake_numpy,
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
                "scipy": fake_scipy,
                "einops": fake_einops,
            }
            return modules_by_name[module_name]

        # when / then
        with patch.object(
            script_module,
            "_import_or_raise",
            side_effect=patched_import_or_raise,
        ):
            script_module.validate_toy_model_geometry_validation_dependencies()

    def test_validate_dependencies_fails_when_einops_einsum_missing(self) -> None:
        # given
        fake_torch = types.ModuleType("torch")
        fake_numpy = types.ModuleType("numpy")
        fake_pyplot = types.ModuleType("matplotlib.pyplot")

        fake_matplotlib = types.ModuleType("matplotlib")

        def use(_: str) -> None:
            return None

        setattr(fake_matplotlib, "use", use)

        fake_scipy = types.ModuleType("scipy")
        fake_einops = types.ModuleType("einops")

        def patched_import_or_raise(module_name: str) -> types.ModuleType:
            modules_by_name = {
                "torch": fake_torch,
                "numpy": fake_numpy,
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
                "scipy": fake_scipy,
                "einops": fake_einops,
            }
            return modules_by_name[module_name]

        # when / then
        with patch.object(
            script_module,
            "_import_or_raise",
            side_effect=patched_import_or_raise,
        ):
            with self.assertRaises(RuntimeError) as context:
                script_module.validate_toy_model_geometry_validation_dependencies()

        self.assertIn("einops.einsum", str(context.exception))
        self.assertIn(script_module.INSTALL_HINT, str(context.exception))

    def test_main_exits_with_clear_error_when_validation_fails(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            argv = ["prog", "--config-path", config_path]
            stderr = io.StringIO()

            # when / then
            with (
                patch.object(script_module.sys, "argv", argv),
                patch.object(script_module.sys, "stderr", stderr),
                patch.object(
                    script_module,
                    "validate_toy_model_geometry_validation_dependencies",
                    side_effect=RuntimeError("boom"),
                ),
            ):
                with self.assertRaises(SystemExit) as context:
                    script_module.main()

            self.assertEqual(context.exception.code, 1)
            self.assertIn("boom", stderr.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
