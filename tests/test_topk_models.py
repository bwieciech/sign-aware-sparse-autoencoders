import unittest
from typing import List, Type, TypeVar, Union

import torch

from sgnsae.experiments.llm_benchmark.training import (
    _copy_instance_0_parameters_to_all_instances,
)
from sgnsae.models import AbsTopKGatedMagnitudeSAE, AbsTopKSAE, TopKSAE
from sgnsae.models.sae_base import SAEBase

TOPK_CLASSES: List[Type[SAEBase]] = [TopKSAE, AbsTopKSAE, AbsTopKGatedMagnitudeSAE]

_T = TypeVar("_T", bound=SAEBase)


def _make_model(
    cls: Type[_T],
    n_instances: int,
    d_in: int,
    d_hidden: int,
    topk_k: Union[int, List[int]],
) -> _T:
    return cls(
        n_instances=n_instances,
        d_in=d_in,
        d_hidden=d_hidden,
        topk_k=topk_k,
        resample_latents_threshold=1_000_000,
        device="cpu",
    )


class TestTopKConstructorValidation(unittest.TestCase):
    def test_scalar_topk_k_with_single_instance_backward_compatible(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when
                model = _make_model(cls, n_instances=1, d_in=4, d_hidden=6, topk_k=3)

                # then
                self.assertEqual(model.topk_k, 3)
                self.assertIsNone(model.topk_k_per_instance)
                self.assertIsNone(model._topk_k_tensor)

    def test_scalar_topk_k_with_multiple_instances(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when
                model = _make_model(cls, n_instances=3, d_in=4, d_hidden=6, topk_k=2)

                # then
                self.assertEqual(model.topk_k, 2)
                self.assertIsNone(model.topk_k_per_instance)
                self.assertIsNone(model._topk_k_tensor)

    def test_list_topk_k_with_matching_n_instances(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when
                model = _make_model(
                    cls, n_instances=3, d_in=4, d_hidden=6, topk_k=[1, 3, 5]
                )

                # then
                self.assertEqual(model.topk_k, 1)
                self.assertEqual(model.topk_k_per_instance, [1, 3, 5])
                self.assertTrue(
                    torch.equal(
                        model._topk_k_tensor,
                        torch.tensor([1, 3, 5], dtype=torch.int64),
                    )
                )

    def test_list_topk_k_with_wrong_length_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(
                        cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[1, 2, 3]
                    )

    def test_list_topk_k_with_k_exceeding_d_hidden_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[3, 7])

    def test_list_topk_k_with_k_zero_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[0, 3])

    def test_list_topk_k_with_k_negative_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[-1, 3])

    def test_empty_list_topk_k_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[])

    def test_copy_instance_0_does_not_overwrite_topk_k_per_instance(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given
                model = _make_model(
                    cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[2, 4]
                )

                # when
                _copy_instance_0_parameters_to_all_instances(model)

                # then
                self.assertEqual(model.topk_k_per_instance, [2, 4])
                self.assertEqual(model.topk_k, 2)
                self.assertTrue(
                    torch.equal(
                        model._topk_k_tensor,
                        torch.tensor([2, 4], dtype=torch.int64),
                    )
                )

    def test_list_topk_k_with_float_values_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(
                        cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[2.5, 4.0]
                    )

    def test_scalar_topk_k_with_float_value_raises_value_error(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given / when / then
                with self.assertRaises(ValueError):
                    _make_model(cls, n_instances=1, d_in=4, d_hidden=6, topk_k=2.5)

    def test_topk_k_tensor_not_in_state_dict(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given
                model = _make_model(
                    cls, n_instances=2, d_in=4, d_hidden=6, topk_k=[2, 4]
                )

                # when
                state_dict_keys = set(model.state_dict().keys())

                # then
                self.assertNotIn("_topk_k_tensor", state_dict_keys)
                self.assertNotIn("topk_k", state_dict_keys)
                self.assertNotIn("topk_k_per_instance", state_dict_keys)


class TestTopKForwardWithPerInstanceK(unittest.TestCase):
    def test_forward_shape_and_sparsity_with_per_instance_k(self) -> None:
        for cls in TOPK_CLASSES:
            with self.subTest(cls=cls.__name__):
                # given
                torch.manual_seed(42)
                n_instances, d_in, d_hidden = 2, 8, 6
                batch_size = 5
                model = _make_model(
                    cls,
                    n_instances=n_instances,
                    d_in=d_in,
                    d_hidden=d_hidden,
                    topk_k=[2, 4],
                )
                # TopKSAE applies ReLU before masking; bias pre-activations
                # positive so that all top-k selected latents are nonzero.
                if cls is TopKSAE:
                    with torch.no_grad():
                        model.b_enc.fill_(10.0)
                x = torch.randn(batch_size, d_in)

                # when
                output = model(x)

                # then
                latents = output["latents"]
                self.assertEqual(latents.shape, (n_instances, batch_size, d_hidden))
                self.assertEqual(output["x_hat"].shape, (n_instances, batch_size, d_in))

                for b in range(batch_size):
                    nonzero_0 = (latents[0, b] != 0).sum().item()
                    nonzero_1 = (latents[1, b] != 0).sum().item()
                    self.assertEqual(
                        nonzero_0,
                        2,
                        f"{cls.__name__}: instance 0 expected 2 active, got {nonzero_0}",
                    )
                    self.assertEqual(
                        nonzero_1,
                        4,
                        f"{cls.__name__}: instance 1 expected 4 active, got {nonzero_1}",
                    )


if __name__ == "__main__":
    unittest.main()
