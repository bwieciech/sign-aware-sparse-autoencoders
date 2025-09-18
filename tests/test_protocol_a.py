import unittest
from unittest import mock

import torch

from sgnsae.data import (
    BUNDLE_TYPE_ANTICORRELATED,
    Toy2DBundle,
    Toy2DData,
    Toy2DGenerationConfig,
    generate_toy2d_dataset,
)
from sgnsae.eval import DecoderSignUsage
from sgnsae.eval.toy2d_eval import evaluate_toy2d_model
from sgnsae.eval.visualize_toy2d import HAS_MATPLOTLIB, plot_decoder_arrangement
from sgnsae.models import SAE
from sgnsae.train import OptimizerConfig, TrainingConfig, detect_device, train_model


class DummyModel(torch.nn.Module):
    def __init__(self, latents: torch.Tensor, decoder: torch.Tensor):
        super().__init__()
        self.register_parameter("dummy_param", torch.nn.Parameter(torch.zeros(1)))
        self.n_instances = 1
        self.d_hidden = latents.shape[1]
        self.W_dec = torch.nn.Parameter(decoder.unsqueeze(0))
        self.latents = latents

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        preds = x
        lats = self.latents[:batch]
        return {
            "x_hat": preds.unsqueeze(0),
            "latents": lats.unsqueeze(0),
        }

    def to(self, device):  # pragma: no cover - keep API compatibility
        super().to(device)
        self.latents = self.latents.to(device)
        return self


class TestProtocolAInfrastructure(unittest.TestCase):
    def test_dataset_generation_respects_anticorrelation(self):
        config = Toy2DGenerationConfig(
            num_samples=100,
            num_correlated_bundles=1,
            correlated_bundle_size=2,
            num_anticorrelated_bundles=1,
            anticorrelated_bundle_size=3,
            correlated_on_probability=0.5,
            anticorrelated_on_probability=0.9,
            angular_std=0.0,
            magnitude_range=(0.0, 1.0),
            noise_std=0.0,
            seed=42,
        )
        data = generate_toy2d_dataset(config)

        anti_indices = data.anticorrelated_pairs()
        for i, j in anti_indices:
            co_activations = (data.activations[:, i] > 0) & (data.activations[:, j] > 0)
            self.assertEqual(co_activations.any().item(), 0)

    def test_train_model_runs_for_small_problem(self):
        config = Toy2DGenerationConfig(
            num_samples=256,
            num_correlated_bundles=1,
            correlated_bundle_size=2,
            num_anticorrelated_bundles=0,
            anticorrelated_bundle_size=0,
            correlated_on_probability=0.5,
            anticorrelated_on_probability=0.0,
            angular_std=0.0,
            magnitude_range=(0.0, 1.0),
            noise_std=0.0,
            seed=0,
        )
        data = generate_toy2d_dataset(config)
        dataset = data.to_dataset()

        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=4,
            resample_latents_threshold=10,
            sign_aware=False,
            device="cpu",
        )
        train_cfg = TrainingConfig(
            steps=5,
            batch_size=32,
            sparsity_coeff=1e-3,
            optimizer=OptimizerConfig(learning_rate=1e-3, beta1=0.0, beta2=0.99),
            log_every=1,
            resample_check_interval=1,
            latent_activity_epsilon=1e-8,
            grad_clip=None,
            shuffle=True,
            seed=0,
        )
        result = train_model(model, dataset, train_cfg)
        self.assertEqual(result.inactivity_counters.shape, (1, 4))
        self.assertGreaterEqual(len(result.history), 1)
        decoder_norms = model.W_dec.detach().norm(dim=1)
        self.assertTrue(
            torch.allclose(
                decoder_norms,
                torch.ones_like(decoder_norms),
                atol=1e-5,
            )
        )

    def test_detect_device_prefers_cuda_when_available(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(detect_device(None), "cuda")

    def test_detect_device_respects_override(self):
        self.assertEqual(detect_device("cpu"), "cpu")

    def test_collapse_detection_identifies_signed_latent(self):
        features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        activations = torch.tensor(
            [
                [0.8, 0.0],
                [0.0, 0.9],
            ],
            dtype=torch.float32,
        )
        observations = activations @ features
        bundle = Toy2DBundle(
            name="anti_0",
            indices=(0, 1),
            bundle_type=BUNDLE_TYPE_ANTICORRELATED,
            base_angle=0.0,
        )
        config = Toy2DGenerationConfig(
            num_samples=2,
            num_correlated_bundles=0,
            correlated_bundle_size=0,
            num_anticorrelated_bundles=1,
            anticorrelated_bundle_size=2,
            correlated_on_probability=0.0,
            anticorrelated_on_probability=1.0,
            angular_std=0.0,
            magnitude_range=(0.0, 1.0),
            noise_std=0.0,
            seed=0,
        )
        data = Toy2DData(
            observations=observations,
            activations=activations,
            features=features,
            bundles=[bundle],
            config=config,
            feature_to_bundle=[0, 0],
        )
        dataset = data.to_dataset()
        latents = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
        decoder = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        model = DummyModel(latents, decoder)
        evaluation = evaluate_toy2d_model(model, "dummy", dataset, data)
        self.assertAlmostEqual(evaluation.collapse_rate, 1.0)
        self.assertEqual(evaluation.collapsed_pairs, [(0, 1)])

    def test_collapse_detection_handles_flipped_orientation(self):
        features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        activations = torch.tensor(
            [
                [0.8, 0.0],
                [0.0, 0.9],
            ],
            dtype=torch.float32,
        )
        observations = activations @ features
        bundle = Toy2DBundle(
            name="anti_0",
            indices=(0, 1),
            bundle_type=BUNDLE_TYPE_ANTICORRELATED,
            base_angle=0.0,
        )
        config = Toy2DGenerationConfig(
            num_samples=2,
            num_correlated_bundles=0,
            correlated_bundle_size=0,
            num_anticorrelated_bundles=1,
            anticorrelated_bundle_size=2,
            correlated_on_probability=0.0,
            anticorrelated_on_probability=1.0,
            angular_std=0.0,
            magnitude_range=(0.0, 1.0),
            noise_std=0.0,
            seed=0,
        )
        data = Toy2DData(
            observations=observations,
            activations=activations,
            features=features,
            bundles=[bundle],
            config=config,
            feature_to_bundle=[0, 0],
        )
        dataset = data.to_dataset()
        latents = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
        decoder = torch.tensor([[-1.0], [0.0]], dtype=torch.float32)
        model = DummyModel(latents, decoder)
        evaluation = evaluate_toy2d_model(model, "dummy", dataset, data)
        self.assertAlmostEqual(evaluation.collapse_rate, 1.0)
        self.assertEqual(evaluation.collapsed_pairs, [(0, 1)])

    def test_alignment_summary_reports_zero_for_perfect_alignment(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        activations = torch.zeros(4, 2, dtype=torch.float32)
        observations = activations @ features
        data = Toy2DData(
            observations=observations,
            activations=activations,
            features=features,
            bundles=[],
            config=Toy2DGenerationConfig(
                num_samples=4,
                num_correlated_bundles=0,
                correlated_bundle_size=0,
                num_anticorrelated_bundles=0,
                anticorrelated_bundle_size=0,
                correlated_on_probability=0.0,
                anticorrelated_on_probability=0.0,
                angular_std=0.0,
                magnitude_range=(0.0, 1.0),
                noise_std=0.0,
                seed=0,
            ),
            feature_to_bundle=[-1, -1],
        )
        dataset = data.to_dataset()
        latents = torch.zeros(4, 2)
        decoder = features.t()
        model = DummyModel(latents, decoder)

        evaluation = evaluate_toy2d_model(model, "aligned", dataset, data)
        self.assertTrue(torch.allclose(evaluation.alignment_angles_deg, torch.zeros(2)))
        self.assertAlmostEqual(evaluation.alignment_mean_deg, 0.0)
        self.assertAlmostEqual(evaluation.alignment_median_deg, 0.0)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for this test")
    def test_visualization_returns_figure(self):
        config = Toy2DGenerationConfig(
            num_samples=32,
            num_correlated_bundles=1,
            correlated_bundle_size=2,
            num_anticorrelated_bundles=1,
            anticorrelated_bundle_size=2,
            correlated_on_probability=0.5,
            anticorrelated_on_probability=0.5,
            angular_std=0.0,
            magnitude_range=(0.0, 1.0),
            noise_std=0.0,
            seed=1,
        )
        data = generate_toy2d_dataset(config)
        decoder_vectors = torch.eye(2)
        sign_usage = [
            DecoderSignUsage(positive_rate=0.5, negative_rate=0.5)
            for _ in range(decoder_vectors.shape[0])
        ]
        fig = plot_decoder_arrangement(decoder_vectors, data, sign_usage, title="test")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.axes[0].get_title(), "test")


if __name__ == "__main__":
    unittest.main()
