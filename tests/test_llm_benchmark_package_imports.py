import subprocess
import sys
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLLMBenchmarkPackageImports(unittest.TestCase):
    def _run_python(self, statements: List[str]) -> str:
        command = "\n".join(statements)
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        return str(completed.stdout).strip()

    def test_importing_config_module_does_not_import_transformers(self) -> None:
        # given
        statements = [
            "import importlib",
            "import sys",
            "sys.modules.pop('transformers', None)",
            "importlib.import_module('sgnsae.experiments.llm_benchmark.config')",
            "print('transformers' in sys.modules)",
        ]

        # when
        output = self._run_python(statements=statements)

        # then
        self.assertEqual(output, "False")

    def test_run_llm_sae_benchmark_wrapper_imports_runner_lazily(self) -> None:
        # given
        import sgnsae.experiments.llm_benchmark as benchmark_package

        # when
        with patch(
            "sgnsae.experiments.llm_benchmark.runner.run_llm_sae_benchmark",
            return_value="/tmp/out",
        ) as runner_call:
            result = benchmark_package.run_llm_sae_benchmark(
                config_path="config.json",
                validate_only=True,
            )

        # then
        self.assertEqual(result, "/tmp/out")
        runner_call.assert_called_once_with(
            config_path="config.json",
            validate_only=True,
        )

    def test_importing_llm_benchmark_package_does_not_import_transformers(
        self,
    ) -> None:
        # given
        statements = [
            "import importlib",
            "import sys",
            "sys.modules.pop('transformers', None)",
            "importlib.import_module('sgnsae.experiments.llm_benchmark')",
            "print('transformers' in sys.modules)",
        ]

        # when
        output = self._run_python(statements=statements)

        # then
        self.assertEqual(output, "False")
