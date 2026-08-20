import importlib
import unittest


class TestLlmBenchmarkPaperV2Configs(unittest.TestCase):
    def test_revision_v2_stage_and_model_modules_import(self) -> None:
        # given / when
        imported_examples_module = importlib.import_module(
            "sgnsae.experiments.llm_benchmark.examples"
        )
        imported_hybrid_module = importlib.import_module(
            "sgnsae.models.abstopk_gated_magnitude_sae"
        )

        # then
        self.assertTrue(hasattr(imported_examples_module, "generate_latent_examples"))
        self.assertTrue(hasattr(imported_hybrid_module, "AbsTopKGatedMagnitudeSAE"))


if __name__ == "__main__":
    unittest.main()
