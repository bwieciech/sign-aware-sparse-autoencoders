import io
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from sgnsae.scripts import run_polarity_dial_experiment


class TestPolarityDialDependencyValidation(unittest.TestCase):
    def test_import_or_raise_fails_with_install_hint_on_missing_module(self) -> None:
        # given
        def raise_module_not_found(_: str) -> types.ModuleType:
            raise ModuleNotFoundError("missing")

        with patch.object(
            run_polarity_dial_experiment.importlib,
            "import_module",
            side_effect=raise_module_not_found,
        ):
            # when & then
            with self.assertRaises(RuntimeError) as context:
                _ = run_polarity_dial_experiment._import_or_raise("not_a_module")

        self.assertIn("not_a_module", str(context.exception))
        self.assertIn(run_polarity_dial_experiment.INSTALL_HINT, str(context.exception))

    def test_validate_polarity_dial_dependencies_succeeds_when_all_present(
        self,
    ) -> None:
        # given / when / then
        run_polarity_dial_experiment.validate_polarity_dial_dependencies()

    def test_validate_polarity_dial_dependencies_sets_agg_backend_before_pyplot_import(
        self,
    ) -> None:
        # given
        call_order = []
        fake_torch = types.ModuleType("torch")
        fake_numpy = types.ModuleType("numpy")
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_einops = types.ModuleType("einops")
        setattr(fake_einops, "einsum", object())

        fake_matplotlib = types.ModuleType("matplotlib")

        def use(backend: str) -> None:
            call_order.append(f"use:{backend}")

        setattr(fake_matplotlib, "use", use)

        fake_scipy = types.ModuleType("scipy")
        fake_optimize = types.ModuleType("scipy.optimize")

        def linear_sum_assignment(*_args: object, **_kwargs: object) -> None:
            return None

        setattr(fake_optimize, "linear_sum_assignment", linear_sum_assignment)
        setattr(fake_scipy, "optimize", fake_optimize)

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
        with (
            patch.dict(
                sys.modules,
                {"scipy": fake_scipy, "scipy.optimize": fake_optimize},
                clear=False,
            ),
            patch.object(
                run_polarity_dial_experiment,
                "_import_or_raise",
                side_effect=patched_import_or_raise,
            ),
        ):
            run_polarity_dial_experiment.validate_polarity_dial_dependencies()

        # then
        self.assertIn("matplotlib", call_order)
        self.assertIn("use:Agg", call_order)
        self.assertIn("matplotlib.pyplot", call_order)
        self.assertLess(call_order.index("matplotlib"), call_order.index("use:Agg"))
        self.assertLess(
            call_order.index("use:Agg"), call_order.index("matplotlib.pyplot")
        )

    def test_validate_polarity_dial_dependencies_ignores_matplotlib_backend_errors(
        self,
    ) -> None:
        # given
        call_order = []
        fake_torch = types.ModuleType("torch")
        fake_numpy = types.ModuleType("numpy")
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_einops = types.ModuleType("einops")
        setattr(fake_einops, "einsum", object())

        fake_matplotlib = types.ModuleType("matplotlib")

        def raise_backend_error(_: str) -> None:
            raise RuntimeError("backend error")

        setattr(fake_matplotlib, "use", raise_backend_error)

        fake_scipy = types.ModuleType("scipy")
        fake_optimize = types.ModuleType("scipy.optimize")

        def linear_sum_assignment(*_args: object, **_kwargs: object) -> None:
            return None

        setattr(fake_optimize, "linear_sum_assignment", linear_sum_assignment)
        setattr(fake_scipy, "optimize", fake_optimize)

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
        with (
            patch.dict(
                sys.modules,
                {"scipy": fake_scipy, "scipy.optimize": fake_optimize},
                clear=False,
            ),
            patch.object(
                run_polarity_dial_experiment,
                "_import_or_raise",
                side_effect=patched_import_or_raise,
            ),
        ):
            run_polarity_dial_experiment.validate_polarity_dial_dependencies()

        # then
        self.assertIn("matplotlib.pyplot", call_order)

    def test_validate_polarity_dial_dependencies_fails_when_scipy_api_missing(
        self,
    ) -> None:
        # given
        fake_scipy = types.ModuleType("scipy")
        fake_optimize = types.ModuleType("scipy.optimize")
        fake_scipy.optimize = fake_optimize
        fake_modules = {"scipy": fake_scipy, "scipy.optimize": fake_optimize}

        # when / then
        with patch.dict(sys.modules, fake_modules, clear=False):
            with self.assertRaises(RuntimeError) as context:
                run_polarity_dial_experiment.validate_polarity_dial_dependencies()

        self.assertIn("linear_sum_assignment", str(context.exception))
        self.assertIn(run_polarity_dial_experiment.INSTALL_HINT, str(context.exception))

    def test_validate_polarity_dial_dependencies_fails_when_einops_einsum_missing(
        self,
    ) -> None:
        # given
        original_import_or_raise = run_polarity_dial_experiment._import_or_raise

        def patched_import_or_raise(module_name: str) -> types.ModuleType:
            if module_name == "einops":
                return types.ModuleType("einops")
            return original_import_or_raise(module_name)

        # when / then
        with patch.object(
            run_polarity_dial_experiment,
            "_import_or_raise",
            side_effect=patched_import_or_raise,
        ):
            with self.assertRaises(RuntimeError) as context:
                run_polarity_dial_experiment.validate_polarity_dial_dependencies()

        self.assertIn("einops.einsum", str(context.exception))
        self.assertIn(run_polarity_dial_experiment.INSTALL_HINT, str(context.exception))

    def test_main_exits_with_clear_error_when_validation_fails(self) -> None:
        # given
        with patch.object(
            run_polarity_dial_experiment,
            "validate_polarity_dial_dependencies",
            side_effect=RuntimeError("boom"),
        ):
            stderr = io.StringIO()
            argv = ["prog", "--config-path", "config.json"]

            # when
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stderr", stderr),
                self.assertRaises(SystemExit) as context,
            ):
                run_polarity_dial_experiment.main()

        # then
        self.assertEqual(context.exception.code, 1)
        self.assertIn("boom", stderr.getvalue())

    def test_main_runs_using_config_loader_and_runner(self) -> None:
        # given
        @dataclass
        class DummyConfig:
            seed: int

        stdout = io.StringIO()
        argv = ["prog", "--config-path", "config.json"]
        with (
            patch.object(
                run_polarity_dial_experiment,
                "validate_polarity_dial_dependencies",
                return_value=None,
            ),
            patch(
                "sgnsae.experiments.polarity_dial.load_experiment_config",
                return_value=DummyConfig(seed=1),
            ),
            patch(
                "sgnsae.experiments.polarity_dial.run_polarity_dial_experiment",
                return_value="run_dir",
            ),
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdout", stdout),
        ):
            # when
            run_polarity_dial_experiment.main()

        # then
        self.assertIn("Loaded Polarity Dial config", stdout.getvalue())
        self.assertIn("Run complete. Artifacts written to", stdout.getvalue())

    def test_main_prints_all_run_dirs_when_runner_returns_list(self) -> None:
        # given
        @dataclass
        class DummyConfig:
            seed: int

        stdout = io.StringIO()
        argv = ["prog", "--config-path", "config.json"]
        with (
            patch.object(
                run_polarity_dial_experiment,
                "validate_polarity_dial_dependencies",
                return_value=None,
            ),
            patch(
                "sgnsae.experiments.polarity_dial.load_experiment_config",
                return_value=DummyConfig(seed=1),
            ),
            patch(
                "sgnsae.experiments.polarity_dial.run_polarity_dial_experiment",
                return_value=["run_dir_a", "run_dir_b"],
            ),
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdout", stdout),
        ):
            # when
            run_polarity_dial_experiment.main()

        # then
        output = stdout.getvalue()
        self.assertIn("Run complete. Artifacts written to: run_dir_a", output)
        self.assertIn("Run complete. Artifacts written to: run_dir_b", output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
