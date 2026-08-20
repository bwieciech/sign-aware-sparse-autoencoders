import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from sgnsae.experiments.toy_model_geometry_validation import config as config_module


class TestToyModelGeometryValidationConfig(unittest.TestCase):
    def _build_base_model_config(self) -> Dict[str, Any]:
        return {
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

    def _build_base_data_config(self) -> Dict[str, Any]:
        return {
            "seed": 1,
            "num_train": 2,
            "num_val": 3,
            "num_test": 4,
            "num_pairs": 2,
            "direction_mode": "uniform",
            "pair_axis_active_prob": 0.2,
            "pair_feature_prob": 0.5,
            "pair_correlation": -1.0,
            "magnitude_dist": "constant",
            "constant_magnitude": 1.0,
            "noise_std": 0.0,
        }

    def _build_base_raw_config(self, output_dir: str) -> Dict[str, Any]:
        return {
            "output_dir": output_dir,
            "device": "cpu",
            "data": self._build_base_data_config(),
            "models": [self._build_base_model_config()],
            "evaluation": {
                "min_axis_abs_cos": 0.9,
                "min_activation_corr": 0.8,
                "min_regime_samples": 5,
                "hist_num_bins": 10,
            },
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
                "  num_test: 4",
                "  num_pairs: 2",
                "  direction_mode: uniform",
                "  pair_axis_active_prob: 0.2",
                "  pair_feature_prob: 0.5",
                "  pair_correlation: -1.0",
                "  magnitude_dist: constant",
                "  constant_magnitude: 1.0",
                "  noise_std: 0.0",
                "evaluation:",
                "  min_axis_abs_cos: 0.9",
                "  min_activation_corr: 0.8",
                "  min_regime_samples: 5",
                "  hist_num_bins: 10",
                "models:",
                "  - name: relu_sae",
                "    sae_class: SAE",
                "    sign_aware: false",
                "    single_activation_offset: false",
                "    reset_activation_offsets_on_resample: false",
                "    d_hidden: 4",
                "    sparsity_coeff: 0.01",
                "    learning_rate: 0.1",
                "    num_epochs: 1",
                "    batch_size: 2",
                "    adam_betas: [0.9, 0.999]",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            output_path = os.path.join(tmpdir, "saved", "config.json")
            with open(config_path, "w") as f:
                f.write(yaml_text)

            # when
            loaded = config_module.load_experiment_config(config_path)
            config_module.save_experiment_config(loaded, output_path)

            # then
            self.assertEqual(loaded.device, "cpu")
            self.assertEqual(loaded.data.num_pairs, 2)
            self.assertEqual(loaded.models[0].name, "relu_sae")
            self.assertEqual(loaded.evaluation.hist_num_bins, 10)
            with open(output_path, "r") as f:
                serialized = json.load(f)
            self.assertEqual(serialized["output_dir"], "/tmp/output")
            self.assertEqual(serialized["models"][0]["name"], "relu_sae")

    def test_load_config_requires_yaml_dependency(self) -> None:
        # given
        yaml_text = "\n".join(
            [
                "output_dir: /tmp/output",
                "data: {}",
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
                    _ = config_module.load_experiment_config(config_path)

    def test_load_config_rejects_missing_sections(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"output_dir": "x", "models": []}, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

    def test_load_config_rejects_models_when_not_list(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["models"] = {"not": "a list"}
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

    def test_load_config_rejects_missing_output_dir(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            del raw_config["output_dir"]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

    def test_load_config_rejects_evaluation_when_not_object(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_base_raw_config(tmpdir)
            raw_config["evaluation"] = ["not", "an", "object"]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

    def test_data_config_rejects_invalid_direction_mode(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "direction_mode": "not_a_mode"}
            )

    def test_data_config_accepts_active_pairs_per_sample_as_list(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when
        config = config_module.Toy2DDataConfig(
            **{**base_kwargs, "active_pairs_per_sample": [0, 1, 2]}
        )

        # then
        self.assertEqual(config.active_pairs_per_sample, [0, 1, 2])

    def test_data_config_rejects_empty_active_pairs_per_sample_list(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "active_pairs_per_sample": []}
            )

    def test_data_config_rejects_non_int_active_pairs_per_sample_list(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "active_pairs_per_sample": [0, "1"]}
            )

    def test_data_config_rejects_active_pairs_per_sample_list_out_of_range(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "active_pairs_per_sample": [3]}
            )

    def test_data_config_rejects_active_pairs_per_sample_when_not_int_or_null(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "active_pairs_per_sample": "1"}
            )

    def test_data_config_rejects_active_pairs_per_sample_scalar_out_of_range(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "active_pairs_per_sample": 3}
            )

    def test_data_config_accepts_seed_list(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when
        config = config_module.Toy2DDataConfig(**{**base_kwargs, "seed": [1, 2, 3]})

        # then
        self.assertEqual(config.seed, [1, 2, 3])

    def test_data_config_rejects_empty_seed_list(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(**{**base_kwargs, "seed": []})

    def test_data_config_rejects_seed_when_not_int(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(**{**base_kwargs, "seed": "nope"})

    def test_data_config_rejects_seed_list_with_non_integer_value(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(**{**base_kwargs, "seed": [1, "bad"]})

    def test_data_config_rejects_missing_pair_correlation_when_grid_not_set(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{
                    **base_kwargs,
                    "pair_correlation": None,
                }
            )

    def test_data_config_rejects_pair_axis_active_prob_out_of_range(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "pair_axis_active_prob": 1.1}
            )

    def test_data_config_rejects_pair_feature_prob_outside_open_unit_interval(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "pair_feature_prob": 0.0}
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "pair_feature_prob": 1.0}
            )

    def test_data_config_rejects_pair_correlation_outside_range(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "pair_correlation": 2.0}
            )

    def test_data_config_rejects_infeasible_pair_correlation_includes_feasible_interval(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError) as context:
            _ = config_module.Toy2DDataConfig(
                **{
                    **base_kwargs,
                    "pair_feature_prob": 0.1,
                    "pair_correlation": -1.0,
                }
            )
        self.assertIn("feasible interval is [", str(context.exception))
        self.assertIn("pair_feature_prob=0.1", str(context.exception))

    def test_calculate_pair_correlation_feasible_interval_rejects_degenerate_probabilities(
        self,
    ) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = config_module._calculate_pair_correlation_feasible_interval(0.0)
        with self.assertRaises(ValueError):
            _ = config_module._calculate_pair_correlation_feasible_interval(1.0)

    def test_data_config_accepts_pair_correlation_grid(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when
        config = config_module.Toy2DDataConfig(
            **{
                **base_kwargs,
                "pair_correlation_grid": [-1.0, -0.5, 0.0],
            }
        )

        # then
        self.assertEqual(config.pair_correlation_grid, [-1.0, -0.5, 0.0])

    def test_data_config_rejects_empty_pair_correlation_grid(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{
                    **base_kwargs,
                    "pair_correlation_grid": [],
                }
            )

    def test_data_config_rejects_invalid_magnitude_dist(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{**base_kwargs, "magnitude_dist": "not_a_dist"}
            )

    def test_data_config_rejects_lognormal_sigma_non_positive(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{
                    **base_kwargs,
                    "magnitude_dist": "lognormal",
                    "lognormal_sigma": 0.0,
                }
            )

    def test_data_config_rejects_constant_magnitude_non_positive(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(
                **{
                    **base_kwargs,
                    "constant_magnitude": 0.0,
                }
            )

    def test_data_config_rejects_negative_noise_std(self) -> None:
        # given
        base_kwargs = self._build_base_data_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DDataConfig(**{**base_kwargs, "noise_std": -0.1})

    def test_model_variant_config_rejects_empty_name(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(**{**base_kwargs, "name": ""})

    def test_model_variant_config_rejects_invalid_sae_class(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "sae_class": "BadSAE"}
            )

    def test_model_variant_config_rejects_symmetric_magnitude_without_sign_aware(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{
                    **base_kwargs,
                    "sign_aware": False,
                    "symmetric_magnitude": True,
                }
            )

    def test_model_variant_config_rejects_non_positive_d_hidden(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(**{**base_kwargs, "d_hidden": 0})

    def test_model_variant_config_rejects_negative_sparsity_coeff(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "sparsity_coeff": -0.1}
            )

    def test_model_variant_config_rejects_non_positive_learning_rate(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "learning_rate": 0.0}
            )

    def test_model_variant_config_rejects_non_positive_resample_latents_epochs(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "resample_latents_epochs": 0.0}
            )

    def test_model_variant_config_rejects_negative_sparsity_warmup_epochs(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when / then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "sparsity_coeff_warmup_epochs": -1}
            )

    def test_model_variant_config_rejects_sparsity_warmup_longer_than_training(
        self,
    ) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when / then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{
                    **base_kwargs,
                    "num_epochs": 2,
                    "sparsity_coeff_warmup_epochs": 3,
                }
            )

    def test_model_variant_config_normalizes_null_warmup_steps_to_zero(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when
        config = config_module.Toy2DModelVariantConfig(
            **{**base_kwargs, "warmup_steps": None}
        )

        # then
        self.assertEqual(config.warmup_steps, 0)

    def test_model_variant_config_rejects_invalid_warmup_steps(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when / then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "warmup_steps": -1}
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "warmup_steps": 1.5}
            )

    def test_model_variant_config_rejects_invalid_adam_betas(self) -> None:
        # given
        base_kwargs = self._build_base_model_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "adam_betas": [0.9]}
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DModelVariantConfig(
                **{**base_kwargs, "adam_betas": [0.0, 0.999]}
            )

    def test_evaluation_config_rejects_out_of_range_thresholds(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DEvaluationConfig(min_axis_abs_cos=1.1)
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DEvaluationConfig(min_activation_corr=-0.1)
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DEvaluationConfig(min_sign_agreement=1.1)

    def test_evaluation_config_rejects_non_positive_counts(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DEvaluationConfig(min_regime_samples=0)
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DEvaluationConfig(hist_num_bins=0)

    def test_experiment_config_rejects_invalid_top_level_values(self) -> None:
        # given
        data_config = config_module.Toy2DDataConfig(**self._build_base_data_config())
        model_config = config_module.Toy2DModelVariantConfig(
            **self._build_base_model_config()
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="",
                device="cpu",
                data=data_config,
                models=[model_config],
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="x",
                device="tpu",
                data=data_config,
                models=[model_config],
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                models=[],
            )

    def test_experiment_config_rejects_duplicate_model_names(self) -> None:
        # given
        data_config = config_module.Toy2DDataConfig(**self._build_base_data_config())
        model_config = config_module.Toy2DModelVariantConfig(
            **self._build_base_model_config()
        )
        model_config_2 = config_module.Toy2DModelVariantConfig(
            **{**self._build_base_model_config(), "d_hidden": 5}
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                models=[model_config, model_config_2],
            )

    def test_experiment_config_accepts_cuda_index_device(self) -> None:
        # given
        data_config = config_module.Toy2DDataConfig(**self._build_base_data_config())
        model_config = config_module.Toy2DModelVariantConfig(
            **self._build_base_model_config()
        )

        # when
        config = config_module.Toy2DExperimentConfig(
            output_dir="x",
            device="cuda:0",
            data=data_config,
            models=[model_config],
        )

        # then
        self.assertEqual(config.device, "cuda:0")

    def test_experiment_config_rejects_non_positive_logging_and_preview_values(
        self,
    ) -> None:
        # given
        data_config = config_module.Toy2DDataConfig(**self._build_base_data_config())
        model_config = config_module.Toy2DModelVariantConfig(
            **self._build_base_model_config()
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                models=[model_config],
                log_every_n_steps=0,
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                models=[model_config],
                log_every_n_epochs=0,
            )
        with self.assertRaises(ValueError):
            _ = config_module.Toy2DExperimentConfig(
                output_dir="x",
                device="cpu",
                data=data_config,
                models=[model_config],
                sample_preview_count=0,
            )

    def test_package_exports_config_symbols(self) -> None:
        # given
        from sgnsae.experiments import toy_model_geometry_validation

        exported_names: List[str] = list(
            getattr(toy_model_geometry_validation, "__all__")
        )

        # when / then
        for required in (
            "Toy2DDataConfig",
            "Toy2DEvaluationConfig",
            "Toy2DModelVariantConfig",
            "Toy2DExperimentConfig",
            "load_experiment_config",
            "save_experiment_config",
            "run_toy_model_geometry_validation_experiment",
        ):
            self.assertIn(required, exported_names)
