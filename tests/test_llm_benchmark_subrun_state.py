import json
import os
import tempfile
import unittest
from typing import Any, Dict, Optional

from sgnsae.experiments.llm_benchmark.subrun_state import (
    is_legacy_success_training_metrics_payload,
    legacy_migration_error,
    load_completed_train_steps,
    load_completed_training_history,
    subrun_training_completed,
    subrun_training_state,
)
from tests.llm_benchmark_test_helpers import legacy_success_training_metrics_payload


def _write_training_metrics(
    subrun_dir: str,
    payload: Optional[Dict[str, Any]] = None,
    raw_payload: Optional[str] = None,
) -> None:
    os.makedirs(subrun_dir, exist_ok=True)
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        if raw_payload is not None:
            file.write(raw_payload)
            return
        json.dump(payload, file, indent=2)


class TestLLMBenchmarkSubrunState(unittest.TestCase):
    def test_legacy_success_training_metrics_payload_requires_real_historical_shape(
        self,
    ) -> None:
        # given
        payload = legacy_success_training_metrics_payload()

        # when
        result = is_legacy_success_training_metrics_payload(payload)

        # then
        self.assertTrue(result)

    def test_subrun_training_state_classifies_completed_legacy_and_incomplete(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            completed_dir = os.path.join(tmpdir, "completed")
            legacy_dir = os.path.join(tmpdir, "legacy")
            incomplete_dir = os.path.join(tmpdir, "incomplete")
            malformed_dir = os.path.join(tmpdir, "malformed")
            _write_training_metrics(completed_dir, {"status": "completed"})
            _write_training_metrics(
                legacy_dir,
                legacy_success_training_metrics_payload(),
            )
            _write_training_metrics(
                incomplete_dir,
                {"status": "completed", "terminated_early": True},
            )
            _write_training_metrics(malformed_dir, raw_payload="[")

            # when
            completed_state = subrun_training_state(completed_dir)
            legacy_state = subrun_training_state(legacy_dir)
            incomplete_state = subrun_training_state(incomplete_dir)
            malformed_state = subrun_training_state(malformed_dir)

            # then
            self.assertEqual(completed_state, "completed")
            self.assertEqual(legacy_state, "legacy_unmigrated")
            self.assertEqual(incomplete_state, "incomplete")
            self.assertEqual(malformed_state, "incomplete")
            self.assertTrue(subrun_training_completed(completed_dir))
            self.assertFalse(subrun_training_completed(legacy_dir))

    def test_legacy_success_training_metrics_payload_rejects_invalid_required_fields(
        self,
    ) -> None:
        # given
        invalid_payloads = []

        payload_terminated_early = legacy_success_training_metrics_payload()
        payload_terminated_early["terminated_early"] = True
        invalid_payloads.append(payload_terminated_early)

        payload_with_failure = legacy_success_training_metrics_payload()
        payload_with_failure["failure"] = {"type": "exception"}
        invalid_payloads.append(payload_with_failure)

        payload_missing_keys = legacy_success_training_metrics_payload()
        del payload_missing_keys["train_steps"]
        del payload_missing_keys["batch_size"]
        invalid_payloads.append(payload_missing_keys)

        payload_missing_variant_name = legacy_success_training_metrics_payload()
        payload_missing_variant_name["variant_name"] = ""
        invalid_payloads.append(payload_missing_variant_name)

        payload_missing_variant_slug = legacy_success_training_metrics_payload()
        payload_missing_variant_slug["variant_slug"] = ""
        invalid_payloads.append(payload_missing_variant_slug)

        payload_bad_history = legacy_success_training_metrics_payload()
        payload_bad_history["history"] = {}
        invalid_payloads.append(payload_bad_history)

        payload_bad_loss_knobs = legacy_success_training_metrics_payload()
        payload_bad_loss_knobs["loss_knobs"] = []
        invalid_payloads.append(payload_bad_loss_knobs)

        payload_bad_adam_betas = legacy_success_training_metrics_payload()
        payload_bad_adam_betas["adam_betas"] = [0.9]
        invalid_payloads.append(payload_bad_adam_betas)

        # when / then
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(is_legacy_success_training_metrics_payload(payload))

    def test_legacy_success_training_metrics_payload_rejects_wrong_numeric_and_bool_types(
        self,
    ) -> None:
        # given
        invalid_updates = {
            "schema_version": "1",
            "seed": "0",
            "sweep_param": "",
            "sweep_value": "0.001",
            "d_in": "768",
            "train_steps": "1000",
            "learning_rate": "3e-4",
            "warmup_steps": "0",
            "mixed_precision": "false",
            "mixed_precision_dtype": 123,
            "resample_dead_latents": "false",
        }

        # when / then
        for key, value in invalid_updates.items():
            with self.subTest(key=key, value=value):
                payload = legacy_success_training_metrics_payload()
                payload[key] = value
                self.assertFalse(is_legacy_success_training_metrics_payload(payload))

    def test_legacy_success_training_metrics_payload_rejects_invalid_ranges(
        self,
    ) -> None:
        # given
        invalid_payloads = []

        payload_zero_learning_rate = legacy_success_training_metrics_payload()
        payload_zero_learning_rate["learning_rate"] = 0.0
        invalid_payloads.append(payload_zero_learning_rate)

        payload_negative_learning_rate = legacy_success_training_metrics_payload()
        payload_negative_learning_rate["learning_rate"] = -1e-3
        invalid_payloads.append(payload_negative_learning_rate)

        payload_bad_adam_beta_range = legacy_success_training_metrics_payload()
        payload_bad_adam_beta_range["adam_betas"] = [0.9, 1.0]
        invalid_payloads.append(payload_bad_adam_beta_range)

        # when / then
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(is_legacy_success_training_metrics_payload(payload))

    def test_legacy_migration_error_mentions_subrun_and_command(self) -> None:
        # given
        output_dir = os.path.join("tmp", "run")
        subrun_dir = os.path.join("tmp", "run", "models", "variant", "seed_0")

        # when
        error = legacy_migration_error(output_dir=output_dir, subrun_dir=subrun_dir)

        # then
        self.assertIn("legacy pre-status training_metrics.json", str(error))
        self.assertIn(subrun_dir, str(error))
        self.assertIn("python scripts/migrate_llm_training_status.py", str(error))
        self.assertIn(output_dir, str(error))

    def test_load_completed_train_steps_returns_none_when_file_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "nonexistent")

            # when
            result = load_completed_train_steps(subrun_dir)

            # then
            self.assertIsNone(result)

    def test_load_completed_train_steps_returns_none_when_status_running(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "running")
            _write_training_metrics(subrun_dir, {"status": "running"})

            # when
            result = load_completed_train_steps(subrun_dir)

            # then
            self.assertIsNone(result)

    def test_load_completed_train_steps_returns_none_when_status_failed(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "failed")
            _write_training_metrics(subrun_dir, {"status": "failed"})

            # when
            result = load_completed_train_steps(subrun_dir)

            # then
            self.assertIsNone(result)

    def test_load_completed_train_steps_returns_none_when_no_train_steps_key(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "no_key")
            _write_training_metrics(subrun_dir, {"status": "completed"})

            # when
            result = load_completed_train_steps(subrun_dir)

            # then
            self.assertIsNone(result)

    def test_load_completed_train_steps_returns_none_for_invalid_train_steps(
        self,
    ) -> None:
        # given
        invalid_values = ["abc", 0, -1, True]

        with tempfile.TemporaryDirectory() as tmpdir:
            for value in invalid_values:
                with self.subTest(value=value):
                    subrun_dir = os.path.join(tmpdir, f"invalid_{value}")
                    _write_training_metrics(
                        subrun_dir,
                        {"status": "completed", "train_steps": value},
                    )

                    # when
                    result = load_completed_train_steps(subrun_dir)

                    # then
                    self.assertIsNone(result)

    def test_load_completed_train_steps_returns_value_when_valid(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "valid")
            _write_training_metrics(
                subrun_dir,
                {"status": "completed", "train_steps": 225000},
            )

            # when
            result = load_completed_train_steps(subrun_dir)

            # then
            self.assertEqual(result, 225000)

    def test_load_completed_training_history_returns_none_when_file_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "nonexistent")

            # when
            result = load_completed_training_history(subrun_dir)

            # then
            self.assertIsNone(result)

    def test_load_completed_training_history_returns_none_when_status_running(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "running")
            _write_training_metrics(
                subrun_dir,
                {"status": "running", "history": [{"step": 1}]},
            )

            # when
            result = load_completed_training_history(subrun_dir)

            # then
            self.assertIsNone(result)

    def test_load_completed_training_history_returns_empty_list_when_history_empty(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "empty_history")
            _write_training_metrics(
                subrun_dir,
                {"status": "completed", "history": []},
            )

            # when
            result = load_completed_training_history(subrun_dir)

            # then
            self.assertEqual(result, [])

    def test_load_completed_training_history_returns_history_list_when_valid(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "valid")
            expected_history = [{"step": 50, "loss_total": 0.5}]
            _write_training_metrics(
                subrun_dir,
                {"status": "completed", "history": expected_history},
            )

            # when
            result = load_completed_training_history(subrun_dir)

            # then
            self.assertEqual(result, expected_history)

    def test_load_completed_training_history_returns_none_when_history_not_list(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "invalid_history")
            _write_training_metrics(
                subrun_dir,
                {"status": "completed", "history": "invalid"},
            )

            # when
            result = load_completed_training_history(subrun_dir)

            # then
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
