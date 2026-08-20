import unittest

import torch

from sgnsae.models.topk_utils import (
    deterministic_topk_mask,
    deterministic_topk_mask_per_instance,
)


class TestDeterministicTopkMaskPerInstance(unittest.TestCase):
    def test_uniform_k_matches_scalar_function_bitwise(self) -> None:
        # given
        torch.manual_seed(42)
        batch, n_instances, d_hidden = 8, 3, 10
        scores = torch.randn(batch, n_instances, d_hidden)
        k = 4
        topk_k_per_instance = torch.tensor([k, k, k], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)
        expected = deterministic_topk_mask(scores, k)

        # then
        self.assertTrue(torch.equal(result, expected))

    def test_different_k_values_produce_correct_per_instance_active_counts(
        self,
    ) -> None:
        # given
        torch.manual_seed(0)
        batch, d_hidden = 5, 6
        n_instances = 3
        scores = torch.randn(batch, n_instances, d_hidden)
        topk_k_per_instance = torch.tensor([1, 3, 5], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)

        # then
        for i, k in enumerate([1, 3, 5]):
            active_counts = result[:, i, :].sum(dim=-1)
            self.assertTrue(
                torch.all(active_counts == k),
                f"Instance {i}: expected {k} active per row, got {active_counts}",
            )

    def test_tie_breaking_matches_scalar_function_per_instance(self) -> None:
        # given
        batch, n_instances, d_hidden = 1, 2, 6
        scores = torch.zeros(batch, n_instances, d_hidden)
        scores[0, 0, :] = torch.tensor([5.0, 3.0, 3.0, 3.0, 1.0, 0.0])
        scores[0, 1, :] = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 0.0])
        topk_k_per_instance = torch.tensor([3, 4], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)

        # then
        expected_0 = deterministic_topk_mask(scores[:, 0:1, :], 3)
        expected_1 = deterministic_topk_mask(scores[:, 1:2, :], 4)
        self.assertTrue(torch.equal(result[:, 0:1, :], expected_0))
        self.assertTrue(torch.equal(result[:, 1:2, :], expected_1))

        # Leftmost-first tie-breaking: instance 0 picks index 0 (5.0) + indices 1,2 (first two 3.0s)
        self.assertEqual(result[0, 0].tolist(), [True, True, True, False, False, False])
        # Instance 1 picks first four 2.0s (indices 0-3)
        self.assertEqual(result[0, 1].tolist(), [True, True, True, True, False, False])

    def test_k_equals_one(self) -> None:
        # given
        torch.manual_seed(1)
        batch, n_instances, d_hidden = 4, 2, 8
        scores = torch.randn(batch, n_instances, d_hidden)
        topk_k_per_instance = torch.tensor([1, 1], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)

        # then
        for i in range(n_instances):
            active_counts = result[:, i, :].sum(dim=-1)
            self.assertTrue(torch.all(active_counts == 1))

    def test_k_equals_d_hidden(self) -> None:
        # given
        batch, n_instances, d_hidden = 3, 2, 5
        scores = torch.randn(batch, n_instances, d_hidden)
        topk_k_per_instance = torch.tensor([d_hidden, d_hidden], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)

        # then
        self.assertTrue(torch.all(result))

    def test_single_instance(self) -> None:
        # given
        torch.manual_seed(2)
        batch, d_hidden = 6, 10
        scores = torch.randn(batch, 1, d_hidden)
        topk_k_per_instance = torch.tensor([3], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)
        expected = deterministic_topk_mask(scores, 3)

        # then
        self.assertTrue(torch.equal(result, expected))

    def test_single_batch_element(self) -> None:
        # given
        torch.manual_seed(3)
        n_instances, d_hidden = 3, 8
        scores = torch.randn(1, n_instances, d_hidden)
        topk_k_per_instance = torch.tensor([2, 4, 6], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)

        # then
        for i, k in enumerate([2, 4, 6]):
            self.assertEqual(result[0, i].sum().item(), k)

    def test_nan_in_scores_produces_valid_mask_nan_not_selected(self) -> None:
        # given
        batch, n_instances, d_hidden = 2, 2, 6
        scores = torch.tensor(
            [
                [
                    [float("nan"), 5.0, 3.0, 2.0, 1.0, 0.0],
                    [4.0, float("nan"), float("nan"), 1.0, 0.5, 0.0],
                ],
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
                    [float("nan"), float("nan"), 3.0, 2.0, 1.0, 0.0],
                ],
            ]
        )
        topk_k_per_instance = torch.tensor([2, 3], dtype=torch.int64)

        # when
        result = deterministic_topk_mask_per_instance(scores, topk_k_per_instance)

        # then
        self.assertEqual(result.shape, scores.shape)
        for b in range(batch):
            for i, k in enumerate([2, 3]):
                instance_mask = result[b, i]
                self.assertEqual(instance_mask.sum().item(), k)
                nan_positions = torch.isnan(scores[b, i])
                self.assertFalse(
                    torch.any(instance_mask & nan_positions).item(),
                    f"NaN position selected at batch={b}, instance={i}",
                )

    def test_mismatched_k_length_and_n_instances_raises_value_error(self) -> None:
        # given
        scores = torch.randn(2, 2, 6)
        topk_k_per_instance = torch.tensor([1, 2, 3], dtype=torch.int64)

        # when / then
        with self.assertRaises(ValueError):
            deterministic_topk_mask_per_instance(scores, topk_k_per_instance)


if __name__ == "__main__":
    unittest.main()
