import unittest
from typing import Any, Dict

import torch

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.latent_examples import (
    _select_latent_indices_from_statistics,
    render_highlighted_text,
)


def _base_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {
            "backend": "local_jsonl",
            "data_files": ["dummy.jsonl"],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "variants": [
            {
                "name": "baseline_relu",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


class TestLatentExamplesTextHighlighting(unittest.TestCase):
    def test_render_highlighted_text_wraps_contiguous_active_token_spans(self) -> None:
        # given
        token_pieces = ["The", " quick", " brown", " fox"]
        active_positions = [1, 2]

        # when
        highlighted = render_highlighted_text(
            token_pieces=token_pieces,
            active_token_indices=active_positions,
            left_marker="|>",
            right_marker="<|",
        )

        # then
        self.assertEqual(highlighted, "The|> quick brown<| fox")

    def test_render_highlighted_text_ignores_out_of_range_positions(self) -> None:
        # given
        token_pieces = ["A", " B", " C", " D"]
        active_positions = [-1, 1, 3, 9]

        # when
        highlighted = render_highlighted_text(
            token_pieces=token_pieces,
            active_token_indices=active_positions,
            left_marker="<<",
            right_marker=">>",
        )

        # then
        self.assertEqual(highlighted, "A<< B>> C<< D>>")

    def test_select_latent_indices_uses_configured_selection_mode(self) -> None:
        # given
        config = LLMBenchmarkConfig.model_validate(_base_config(output_dir="/tmp/out"))
        max_abs = torch.tensor([1.0, 4.0, 4.0, 2.0], dtype=torch.float64)
        frequency = torch.tensor([5.0, 2.0, 9.0, 9.0], dtype=torch.float64)

        # when
        by_abs = _select_latent_indices_from_statistics(
            max_abs_by_latent=max_abs,
            activation_count_by_latent=frequency,
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"latent_selection": "top_by_max_abs", "max_latents": 3}
                    )
                }
            ),
            d_hidden=4,
        )
        by_freq = _select_latent_indices_from_statistics(
            max_abs_by_latent=max_abs,
            activation_count_by_latent=frequency,
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"latent_selection": "top_by_freq", "max_latents": 2}
                    )
                }
            ),
            d_hidden=4,
        )
        explicit = _select_latent_indices_from_statistics(
            max_abs_by_latent=max_abs,
            activation_count_by_latent=frequency,
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={
                            "latent_selection": "explicit",
                            "explicit_latents": [3, 3, 5, 1],
                            "max_latents": 3,
                        }
                    )
                }
            ),
            d_hidden=4,
        )

        # then
        self.assertEqual(by_abs, [1, 2, 3])
        self.assertEqual(by_freq, [2, 3])
        self.assertEqual(explicit, [3, 1])
