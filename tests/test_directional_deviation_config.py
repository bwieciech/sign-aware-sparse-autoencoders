import json
import os
import tempfile
import unittest
from typing import Any, Dict
from types import SimpleNamespace
from unittest.mock import patch

from sgnsae.experiments.directional_deviation import config as config_module


class TestDirectionalDeviationConfig(unittest.TestCase):
    def _build_base_raw_config(self, output_dir: str) -> Dict[str, Any]:
        return {
            "output_dir": output_dir,
            "device": "cpu",
            "data": {
                "seed": 1,
                "num_train": 2,
                "num_val": 3,
                "num_test_normal": 1,
                "num_test_high_per_channel": 1,
                "num_test_low_per_channel": 1,
                "num_channels": 4,
                "noise_std": 0.0,
                "normal_mode": "standard_normal",
                "anomaly_eval_mode": "stream",
            },
            "anomaly": {
                "delta": 2.0,
                "shift_mode": "single_channel",
            },
            "thresholds": {
                "latent_quantile": 0.99,
                "score_fpr": 0.01,
                "min_validation_samples": 3,
                "deterministic_algorithms": False,
            },
            "models": [
                {
                    "name": "relu_sae",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
        }

    def test_load_config_from_yaml_and_save_round_trip(self) -> None:
        # given
        yaml_text = "\n".join(
            [
                "output_dir: /tmp/output",
                "device: cpu",
                "data:",
                "  seed: 1",
                "  num_train: 2",
                "  num_val: 3",
                "  num_test_normal: 1",
                "  num_test_high_per_channel: 1",
                "  num_test_low_per_channel: 1",
                "  num_channels: 4",
                "  noise_std: 0.0",
                "  normal_mode: standard_normal",
                "  anomaly_eval_mode: stream",
                "anomaly:",
                "  delta: 2.0",
                "  shift_mode: single_channel",
                "thresholds:",
                "  latent_quantile: 0.99",
                "  score_fpr: 0.01",
                "  min_validation_samples: 3",
                "  deterministic_algorithms: false",
                "models:",
                "  - name: relu_sae",
                "    sae_class: SAE",
                "    sign_aware: false",
                "    single_activation_offset: false",
                "    reset_activation_offsets_on_resample: false",
                "    d_hidden: 2",
                "    sparsity_coeff: 0.01",
                "    learning_rate: 0.1",
                "    num_epochs: 1",
                "    batch_size: 2",
                "    adam_betas: [0.9, 0.999]",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            output_path = os.path.join(tmpdir, "saved.json")
            with open(config_path, "w") as f:
                f.write(yaml_text)

            # when
            loaded = config_module.load_experiment_config(config_path)
            config_module.save_experiment_config(loaded, output_path)

            # then
            self.assertEqual(loaded.device, "cpu")
            self.assertEqual(loaded.data.num_channels, 4)
            self.assertEqual(loaded.models[0].name, "relu_sae")
            with open(output_path, "r") as f:
                serialized = json.load(f)
            self.assertEqual(serialized["output_dir"], "/tmp/output")
            self.assertEqual(serialized["thresholds"]["score_fpr"], 0.01)

    def test_load_config_rejects_missing_sections(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"output_dir": "x", "models": []}, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

    def test_load_config_requires_yaml_dependency(self) -> None:
        # given
        yaml_text = "\n".join(
            [
                "output_dir: /tmp/output",
                "data: {}",
                "anomaly: {}",
                "thresholds: {}",
                "models: []",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(yaml_text)

            # when & then
            with patch.object(config_module, "yaml", None):
                with self.assertRaises(ValueError):
                    config_module.load_experiment_config(config_path)

    def test_data_config_rejects_zero_test_counts(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "num_test_normal": 0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "num_test_high_per_channel": 0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "num_test_low_per_channel": 0}
            )

    def test_data_config_rejects_negative_noise_std(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
            "noise_std": -0.1,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(**base_kwargs)

    def test_data_config_rejects_invalid_train_anomaly_fraction(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "train_anomaly_fraction": -0.1}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "train_anomaly_fraction": 1.1}
            )

    def test_threshold_config_rejects_invalid_quantile_and_fpr(self) -> None:
        # given
        base_kwargs = {
            "latent_quantile": 0.99,
            "score_fpr": 0.01,
            "min_validation_samples": 1,
            "deterministic_algorithms": False,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationThresholdConfig(
                **{**base_kwargs, "latent_quantile": 0.0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationThresholdConfig(
                **{**base_kwargs, "latent_quantile": 1.0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationThresholdConfig(
                **{**base_kwargs, "score_fpr": 0.0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationThresholdConfig(
                **{**base_kwargs, "score_fpr": 1.0}
            )

    def test_experiment_config_rejects_d_hidden_not_less_than_num_channels(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["models"][0]["d_hidden"] = raw_config["data"]["num_channels"]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

    def test_experiment_config_allows_oversized_model_width_when_width_sweep_present(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["models"][0]["d_hidden"] = 999
            raw_config["width_sweep"] = [1, 2, 3]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            loaded = config_module.load_experiment_config(config_path)

            # then
            self.assertEqual(loaded.width_sweep, [1, 2, 3])
            self.assertEqual(loaded.models[0].d_hidden, 999)

    def test_data_config_heteroskedastic_requires_channel_stds(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 2,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "normal_mode": "heteroskedastic_normal"}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "heteroskedastic_normal",
                    "channel_stds": [1.0],
                }
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "heteroskedastic_normal",
                    "channel_stds": [1.0, 0.0],
                }
            )

        config_ok = config_module.DirectionalDeviationDataConfig(
            **{
                **base_kwargs,
                "normal_mode": "heteroskedastic_normal",
                "channel_stds": [1.0, 2.0],
            }
        )
        self.assertEqual(config_ok.normal_mode, "heteroskedastic_normal")

        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "channel_stds": [1.0, 2.0]}
            )

    def test_data_config_validates_channel_stds_uniqueness_when_enabled(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 2,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "heteroskedastic_normal",
                    "channel_stds": [1.0, 1.0],
                }
            )

        config_ok = config_module.DirectionalDeviationDataConfig(
            **{
                **base_kwargs,
                "normal_mode": "heteroskedastic_normal",
                "channel_stds": [1.0, 1.0],
                "ensure_unique_channel_stds": False,
            }
        )
        self.assertFalse(config_ok.ensure_unique_channel_stds)

    def test_data_config_non_gaussian_modes_require_distribution_and_validate_df(
        self,
    ) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "normal_mode": "iid_non_gaussian"}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "iid_non_gaussian",
                    "non_gaussian_dist": "invalid",  # type: ignore[arg-type]
                }
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "iid_non_gaussian",
                    "non_gaussian_dist": "student_t",
                }
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "iid_non_gaussian",
                    "non_gaussian_dist": "student_t",
                    "student_t_df": 2.0,
                }
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{
                    **base_kwargs,
                    "normal_mode": "iid_non_gaussian",
                    "non_gaussian_dist": "laplace",
                    "student_t_df": 3.0,
                }
            )

        config_laplace_ok = config_module.DirectionalDeviationDataConfig(
            **{
                **base_kwargs,
                "normal_mode": "iid_non_gaussian",
                "non_gaussian_dist": "laplace",
            }
        )
        self.assertEqual(config_laplace_ok.non_gaussian_dist, "laplace")

        config_student_t_ok = config_module.DirectionalDeviationDataConfig(
            **{
                **base_kwargs,
                "normal_mode": "iid_non_gaussian",
                "non_gaussian_dist": "student_t",
                "student_t_df": 3.0,
            }
        )
        self.assertEqual(config_student_t_ok.student_t_df, 3.0)

    def test_data_config_non_gaussian_heteroskedastic_requires_channel_stds(
        self,
    ) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 2,
            "normal_mode": "non_gaussian_heteroskedastic",
            "non_gaussian_dist": "logistic",
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(**base_kwargs)
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "channel_stds": [1.0, 1.0]}
            )

        config_ok = config_module.DirectionalDeviationDataConfig(
            **{
                **base_kwargs,
                "channel_stds": [1.0, 1.0],
                "ensure_unique_channel_stds": False,
            }
        )
        self.assertEqual(config_ok.normal_mode, "non_gaussian_heteroskedastic")

    def test_data_config_rejects_non_gaussian_fields_for_gaussian_modes(
        self,
    ) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
            "normal_mode": "standard_normal",
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "non_gaussian_dist": "laplace"}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "student_t_df": 7.0}
            )

    def test_data_config_rejects_invalid_modes(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "normal_mode": "invalid"}  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "anomaly_eval_mode": "invalid"}  # type: ignore[arg-type]
            )

    def test_data_config_validates_seeds(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "seeds": []}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationDataConfig(
                **{**base_kwargs, "seeds": [1, 1]}
            )
        config_ok = config_module.DirectionalDeviationDataConfig(
            **{**base_kwargs, "seeds": [1, 2]}
        )
        self.assertEqual(config_ok.seeds, [1, 2])

    def test_anomaly_config_validation(self) -> None:
        # given
        base_kwargs = {"delta": 2.0, "shift_mode": "single_channel"}

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationAnomalyConfig(
                **{**base_kwargs, "delta": 0.0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationAnomalyConfig(
                delta=2.0, shift_mode="subset"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationAnomalyConfig(
                delta=2.0, shift_mode="single_channel", subset_size=1
            )

        config_ok = config_module.DirectionalDeviationAnomalyConfig(
            delta=2.0, shift_mode="subset", subset_size=1
        )
        self.assertEqual(config_ok.subset_size, 1)

        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationAnomalyConfig(
                delta=2.0, shift_mode="invalid"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationAnomalyConfig(
                delta=2.0, shift_mode="single_channel", channels_to_evaluate=[]
            )

    def test_experiment_config_rejects_subset_size_greater_than_num_channels(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["anomaly"]["shift_mode"] = "subset"
            raw_config["anomaly"]["subset_size"] = (
                raw_config["data"]["num_channels"] + 1
            )
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

    def test_experiment_config_rejects_invalid_channels_to_evaluate(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["anomaly"]["channels_to_evaluate"] = [
                0,
                raw_config["data"]["num_channels"],
            ]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["anomaly"]["channels_to_evaluate"] = [0, 0]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

    def test_experiment_config_rejects_num_val_below_min_validation_samples(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["data"]["num_val"] = 2
            raw_config["thresholds"]["min_validation_samples"] = 3
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

    def test_experiment_config_rejects_invalid_coverage_recall_target(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["coverage_recall_target"] = -0.1
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["coverage_recall_target"] = 1.1
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

    def test_model_variant_config_validation_errors(self) -> None:
        # given
        base_kwargs = {
            "sae_class": "SAE",
            "sign_aware": False,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": False,
            "d_hidden": 2,
            "sparsity_coeff": 0.0,
            "learning_rate": 0.01,
            "num_epochs": 1,
            "batch_size": 2,
            "adam_betas": (0.9, 0.999),
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="", **base_kwargs
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model", **{**base_kwargs, "sae_class": "Invalid"}  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model", **{**base_kwargs, "symmetric_magnitude": True}
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model",
                **{**base_kwargs, "adam_betas": (0.9,)},  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model",
                **{**base_kwargs, "adam_betas": (0.0, 0.9)},
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model",
                **{**base_kwargs, "adam_betas": (0.9, 1.0)},
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model",
                **{**base_kwargs, "sparsity_coeff": -0.1},
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model",
                **{**base_kwargs, "learning_rate": 0.0},
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationModelVariantConfig(
                name="model",
                **{**base_kwargs, "resample_latents_epochs": 0.0},
            )

    def test_experiment_config_validation_errors(self) -> None:
        # given
        data_config = config_module.DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = config_module.DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        threshold_config = config_module.DirectionalDeviationThresholdConfig(
            latent_quantile=0.99,
            score_fpr=0.01,
            min_validation_samples=1,
            deterministic_algorithms=False,
        )
        model_config = config_module.DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="",
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="invalid",  # type: ignore[arg-type]
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[],
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
                log_every_n_steps=0,
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
                log_every_n_epochs=0,
            )

    def test_experiment_config_validates_width_sweep(self) -> None:
        # given
        data_config = config_module.DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = config_module.DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        threshold_config = config_module.DirectionalDeviationThresholdConfig(
            latent_quantile=0.99,
            score_fpr=0.01,
            min_validation_samples=1,
            deterministic_algorithms=False,
        )
        model_config = config_module.DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
                width_sweep=[],
            )
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
                width_sweep=[4],
            )

    def test_experiment_config_rejects_subset_mode_without_subset_size(self) -> None:
        # given
        data_config = config_module.DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        threshold_config = config_module.DirectionalDeviationThresholdConfig(
            latent_quantile=0.99,
            score_fpr=0.01,
            min_validation_samples=1,
            deterministic_algorithms=False,
        )
        model_config = config_module.DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        malformed_anomaly_config = SimpleNamespace(
            delta=1.0,
            shift_mode="subset",
            subset_size=None,
            channels_to_evaluate=None,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.DirectionalDeviationExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                anomaly=malformed_anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
            )

    def test_load_config_rejects_missing_output_dir_and_models_type(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            path_missing_output = os.path.join(tmpdir, "missing_output.json")
            with open(path_missing_output, "w") as f:
                json.dump(
                    {"data": {}, "anomaly": {}, "thresholds": {}, "models": []},
                    f,
                )
            path_models_not_list = os.path.join(tmpdir, "models_not_list.json")
            with open(path_models_not_list, "w") as f:
                json.dump(
                    {
                        "output_dir": "x",
                        "data": {},
                        "anomaly": {},
                        "thresholds": {},
                        "models": {},
                    },
                    f,
                )

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(path_missing_output)
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(path_models_not_list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
