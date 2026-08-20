import argparse
import io
import types
import unittest
from unittest.mock import patch

import sgnsae.scripts.run_llm_sae_benchmark as script_module


class TestLLMBenchmarkCli(unittest.TestCase):
    def test_parse_arguments_reads_positional_config_path(self) -> None:
        # given
        with patch.object(
            script_module.sys,
            "argv",
            ["run_llm_sae_benchmark.py", "config.json"],
        ):
            # when
            args = script_module.parse_arguments()

        # then
        self.assertEqual(args.config_path, "config.json")
        self.assertIsNone(args.config_path_flag)
        self.assertIsNone(args.stages)
        self.assertFalse(args.validate_only)

    def test_parse_stage_list_rejects_empty_entries(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = script_module._parse_stage_list(" , , ")

    def test_parse_stage_list_returns_trimmed_entries(self) -> None:
        # given
        raw = " cache, train ,eval "

        # when
        parsed = script_module._parse_stage_list(raw)

        # then
        self.assertEqual(parsed, ["cache", "train", "eval"])

    def test_parse_stage_list_accepts_examples_stage(self) -> None:
        # given
        raw = "cache,examples,plot"

        # when
        parsed = script_module._parse_stage_list(raw)

        # then
        self.assertEqual(parsed, ["cache", "examples", "plot"])

    def test_resolve_config_path_rejects_positional_and_flag_combination(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = script_module._resolve_config_path("a.json", "b.json")

    def test_resolve_config_path_rejects_missing_path(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = script_module._resolve_config_path(None, None)

    def test_main_exits_code_2_when_config_path_is_invalid(self) -> None:
        # given
        args = argparse.Namespace(
            config_path="a.json",
            config_path_flag="b.json",
            stages=None,
            validate_only=False,
        )

        # when / then
        with patch.object(script_module, "parse_arguments", return_value=args):
            with self.assertRaises(SystemExit) as context:
                script_module.main()
        self.assertEqual(context.exception.code, 2)

    def test_main_exits_code_1_when_dependency_check_fails(self) -> None:
        # given
        args = argparse.Namespace(
            config_path="config.json",
            config_path_flag=None,
            stages=None,
            validate_only=False,
        )

        # when / then
        with patch.object(
            script_module, "parse_arguments", return_value=args
        ), patch.object(
            script_module,
            "_import_or_raise",
            side_effect=RuntimeError("missing dependency"),
        ):
            with self.assertRaises(SystemExit) as context:
                script_module.main()
        self.assertEqual(context.exception.code, 1)

    def test_main_exits_code_2_for_validate_only_with_stages(self) -> None:
        # given
        args = argparse.Namespace(
            config_path="config.json",
            config_path_flag=None,
            stages="cache",
            validate_only=True,
        )

        # when / then
        with patch.object(
            script_module, "parse_arguments", return_value=args
        ), patch.object(
            script_module,
            "_import_or_raise",
            return_value=types.ModuleType("fake"),
        ):
            with self.assertRaises(SystemExit) as context:
                script_module.main()
        self.assertEqual(context.exception.code, 2)

    def test_main_exits_code_2_for_invalid_stage_list(self) -> None:
        # given
        args = argparse.Namespace(
            config_path="config.json",
            config_path_flag=None,
            stages=" , , ",
            validate_only=False,
        )

        # when / then
        with patch.object(
            script_module, "parse_arguments", return_value=args
        ), patch.object(
            script_module,
            "_import_or_raise",
            return_value=types.ModuleType("fake"),
        ):
            with self.assertRaises(SystemExit) as context:
                script_module.main()
        self.assertEqual(context.exception.code, 2)

    def test_main_exits_code_1_when_runner_is_not_implemented(self) -> None:
        # given
        args = argparse.Namespace(
            config_path="config.json",
            config_path_flag=None,
            stages=None,
            validate_only=False,
        )

        # when / then
        with patch.object(
            script_module, "parse_arguments", return_value=args
        ), patch.object(
            script_module,
            "_import_or_raise",
            return_value=types.ModuleType("fake"),
        ), patch(
            "sgnsae.experiments.llm_benchmark.load_benchmark_config",
            return_value=object(),
        ), patch(
            "sgnsae.experiments.llm_benchmark.resolve_benchmark_config",
            return_value={"resolved": True},
        ), patch(
            "sgnsae.experiments.llm_benchmark.run_llm_sae_benchmark",
            side_effect=NotImplementedError("pending"),
        ):
            with self.assertRaises(SystemExit) as context:
                script_module.main()
        self.assertEqual(context.exception.code, 1)

    def test_main_runs_to_completion(self) -> None:
        # given
        args = argparse.Namespace(
            config_path="config.json",
            config_path_flag=None,
            stages=None,
            validate_only=False,
        )
        stdout_stream = io.StringIO()

        # when
        with patch.object(
            script_module, "parse_arguments", return_value=args
        ), patch.object(
            script_module,
            "_import_or_raise",
            return_value=types.ModuleType("fake"),
        ), patch(
            "sgnsae.experiments.llm_benchmark.load_benchmark_config",
            return_value=object(),
        ), patch(
            "sgnsae.experiments.llm_benchmark.resolve_benchmark_config",
            return_value={"resolved": True},
        ), patch(
            "sgnsae.experiments.llm_benchmark.run_llm_sae_benchmark",
            return_value="/tmp/out",
        ), patch.object(
            script_module.sys, "stdout", stdout_stream
        ):
            script_module.main()

        # then
        self.assertIn("Loaded LLM SAE benchmark config:", stdout_stream.getvalue())
        self.assertIn(
            "Run complete. Artifacts written to: /tmp/out", stdout_stream.getvalue()
        )

    def test_import_or_raise_raises_runtime_error_for_missing_module(self) -> None:
        # given
        def raise_module_not_found(_: str) -> types.ModuleType:
            raise ModuleNotFoundError("missing")

        with patch.object(
            script_module.importlib,
            "import_module",
            side_effect=raise_module_not_found,
        ):
            # when / then
            with self.assertRaises(RuntimeError) as context:
                _ = script_module._import_or_raise("not_a_module")

        self.assertIn("not_a_module", str(context.exception))
        self.assertIn(script_module.INSTALL_HINT, str(context.exception))

    def test_import_or_raise_raises_runtime_error_for_import_error(self) -> None:
        # given
        def raise_import_error(_: str) -> types.ModuleType:
            raise ImportError("import failed")

        with patch.object(
            script_module.importlib,
            "import_module",
            side_effect=raise_import_error,
        ):
            # when / then
            with self.assertRaises(RuntimeError) as context:
                _ = script_module._import_or_raise("not_a_module")

        self.assertIn("ImportError", str(context.exception))
        self.assertIn(script_module.INSTALL_HINT, str(context.exception))

    def test_import_or_raise_allows_non_import_errors_to_propagate(self) -> None:
        # given
        def raise_os_error(_: str) -> types.ModuleType:
            raise OSError("library load failure")

        with patch.object(
            script_module.importlib,
            "import_module",
            side_effect=raise_os_error,
        ):
            # when / then
            with self.assertRaises(OSError):
                _ = script_module._import_or_raise("torch")
