import importlib.util
import json
import os
import tempfile
import unittest
from typing import Any, Dict

from tests.llm_benchmark_test_helpers import legacy_success_training_metrics_payload


def _load_migration_module() -> Any:
    repository_root = os.path.dirname(os.path.dirname(__file__))
    module_path = os.path.join(
        repository_root,
        "scripts",
        "migrate_llm_training_status.py",
    )
    spec = importlib.util.spec_from_file_location(
        "migrate_llm_training_status",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load migration script from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


class TestMigrateLLMTrainingStatus(unittest.TestCase):
    def test_migrate_training_metrics_file_backfills_real_legacy_success_payload(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            _write_json(metrics_path, legacy_success_training_metrics_payload())

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertTrue(migrated)
            with open(metrics_path, "r") as file:
                payload = json.load(file)
            self.assertEqual(payload["status"], "completed")

    def test_migrate_training_metrics_file_rejects_sparse_stub_payload(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            _write_json(
                metrics_path,
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "history": [],
                },
            )

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)

    def test_migrate_training_metrics_file_rejects_payload_missing_variant_name(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            payload = legacy_success_training_metrics_payload()
            del payload["variant_name"]
            _write_json(metrics_path, payload)

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)

    def test_migrate_training_metrics_file_rejects_payload_missing_adam_betas(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            payload = legacy_success_training_metrics_payload()
            del payload["adam_betas"]
            _write_json(metrics_path, payload)

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)

    def test_migrate_training_metrics_file_rejects_terminated_early_payload(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            payload = legacy_success_training_metrics_payload()
            payload["terminated_early"] = True
            _write_json(metrics_path, payload)

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)

    def test_migrate_training_metrics_file_rejects_failure_payload(self) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            payload = legacy_success_training_metrics_payload()
            payload["failure"] = {"type": "exception"}
            _write_json(metrics_path, payload)

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)

    def test_migrate_training_metrics_file_rejects_payload_with_existing_status(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            payload = legacy_success_training_metrics_payload()
            payload["status"] = "completed"
            _write_json(metrics_path, payload)

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)

    def test_migrate_training_metrics_file_rejects_legacy_payload_with_wrong_field_types(
        self,
    ) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            payload = legacy_success_training_metrics_payload()
            payload.update(
                {
                    "schema_version": "1",
                    "seed": "0",
                    "sweep_value": "0.001",
                    "d_in": "768",
                    "train_steps": "1000",
                    "learning_rate": "3e-4",
                    "warmup_steps": "0",
                    "mixed_precision": "false",
                    "mixed_precision_dtype": 123,
                    "resample_dead_latents": "false",
                }
            )
            _write_json(metrics_path, payload)

            # when
            migrated = module.migrate_training_metrics_file(metrics_path)

            # then
            self.assertFalse(migrated)
            with open(metrics_path, "r") as file:
                written_payload = json.load(file)
            self.assertNotIn("status", written_payload)

    def test_walk_counts_only_real_legacy_success_files(self) -> None:
        # given
        module = _load_migration_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            first_subrun_dir = os.path.join(tmpdir, "models", "variant_a", "seed_0")
            second_subrun_dir = os.path.join(tmpdir, "models", "variant_b", "seed_0")
            os.makedirs(first_subrun_dir, exist_ok=True)
            os.makedirs(second_subrun_dir, exist_ok=True)
            _write_json(
                os.path.join(first_subrun_dir, "training_metrics.json"),
                legacy_success_training_metrics_payload(),
            )
            _write_json(
                os.path.join(second_subrun_dir, "training_metrics.json"),
                {
                    "variant_slug": "variant_b",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                    "history": [],
                },
            )

            # when
            updated = module.walk(tmpdir)

            # then
            self.assertEqual(updated, 1)
            with open(
                os.path.join(first_subrun_dir, "training_metrics.json"), "r"
            ) as file:
                first_payload = json.load(file)
            with open(
                os.path.join(second_subrun_dir, "training_metrics.json"), "r"
            ) as file:
                second_payload = json.load(file)
            self.assertEqual(first_payload["status"], "completed")
            self.assertNotIn("status", second_payload)


if __name__ == "__main__":
    unittest.main()
