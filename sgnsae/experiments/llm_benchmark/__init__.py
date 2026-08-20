from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    load_benchmark_config,
    resolve_benchmark_config,
)


def run_llm_sae_benchmark(*args, **kwargs):
    from sgnsae.experiments.llm_benchmark.runner import (
        run_llm_sae_benchmark as run_llm_sae_benchmark_impl,
    )

    return run_llm_sae_benchmark_impl(*args, **kwargs)


__all__ = [
    "LLMBenchmarkConfig",
    "load_benchmark_config",
    "resolve_benchmark_config",
    "run_llm_sae_benchmark",
]
