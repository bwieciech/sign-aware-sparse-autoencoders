import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

from sgnsae.experiments.llm_benchmark.overhead import (
    build_variant_model,
    summarize_variant_parameters,
    summarize_run_parameters,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--d-in", type=int, default=None)
    parser.add_argument("--widths", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="SAE,GatedSAE,SignedSoftThresholdSAE,JumpReLUSAE,TopKSAE,AbsTopKSAE",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def _resolve_runtime_device(requested_device: str) -> str:
    normalized = str(requested_device).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return normalized


def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        if device == "cpu":
            return torch.float32
        return torch.float16
    if normalized == "bfloat16":
        if device == "cpu":
            return torch.float32
        return torch.bfloat16
    raise ValueError(
        "--dtype must be one of {'float32','float16','bfloat16'}; "
        f"received '{dtype_name}'."
    )


def _parse_widths(raw_widths: Optional[str]) -> List[int]:
    if raw_widths is None:
        return [512, 1024, 2048, 4096]
    values = [entry.strip() for entry in raw_widths.split(",") if entry.strip()]
    if not values:
        raise ValueError("--widths must contain at least one integer width.")
    parsed: List[int] = []
    for value in values:
        try:
            width = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid width value: {value}.") from exc
        if width <= 0:
            raise ValueError(f"Width values must be positive, received {width}.")
        parsed.append(width)
    return parsed


def _load_run_variants(run_dir: str) -> Tuple[int, List[Dict[str, Any]]]:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"Missing run config at {config_path}.")
    with open(config_path, "r") as file:
        config_payload = json.load(file)
    if not isinstance(config_payload, dict):
        raise ValueError("Run config must be a JSON object.")
    variants = config_payload.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("Run config variants must be a non-empty list.")

    manifest_path = os.path.join(run_dir, "activations", "activations_manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Missing activation manifest at {manifest_path}.")
    with open(manifest_path, "r") as file:
        manifest_payload = json.load(file)
    activation = manifest_payload.get("activation", {})
    if not isinstance(activation, dict):
        raise ValueError("Activation manifest missing activation dictionary.")
    d_in = activation.get("d_in")
    if isinstance(d_in, bool) or not isinstance(d_in, int) or d_in <= 0:
        raise ValueError(f"Invalid activation.d_in in manifest: {d_in}.")
    return int(d_in), [
        dict(variant) for variant in variants if isinstance(variant, dict)
    ]


def _manual_variant_configs(
    variants_csv: str, widths: List[int]
) -> List[Dict[str, Any]]:
    classes = [entry.strip() for entry in variants_csv.split(",") if entry.strip()]
    if not classes:
        raise ValueError("--variants must contain at least one class name.")
    configs: List[Dict[str, Any]] = []
    for sae_class in classes:
        for width in widths:
            config = {
                "name": f"{sae_class}_{width}",
                "variant_slug": f"{sae_class.lower()}_{width}",
                "sae_class": sae_class,
                "sign_aware": False,
                "d_hidden": int(width),
            }
            if sae_class in {"TopKSAE", "AbsTopKSAE"}:
                config["topk_k"] = int(min(32, width))
            else:
                config["sparsity_coeff"] = 1e-3
            configs.append(config)
    return configs


def _time_forward_and_optional_backward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    iters: int,
    include_backward: bool,
    device: str,
) -> List[float]:
    timings: List[float] = []
    for _ in range(int(iters)):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(inputs)
        reconstruction = output.get("x_hat")
        if reconstruction is None:
            reconstruction = output.get("y_pred")
        if reconstruction is None:
            raise ValueError("Model output must include x_hat or y_pred.")
        if include_backward:
            loss = reconstruction.float().pow(2).mean()
            loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(float(elapsed_ms))
        model.zero_grad(set_to_none=True)
    return timings


def _benchmark_variant(
    variant_config: Dict[str, Any],
    d_in: int,
    batch_size: int,
    runtime_device: str,
    dtype: torch.dtype,
    warmup_iters: int,
    iters: int,
    include_backward: bool,
) -> Dict[str, Any]:
    model = build_variant_model(
        variant_config=variant_config,
        d_in=int(d_in),
        device=runtime_device,
    )
    model = model.to(device=runtime_device, dtype=dtype)
    model.train(mode=bool(include_backward))
    inputs = torch.randn(
        int(batch_size),
        int(d_in),
        device=runtime_device,
        dtype=dtype,
    )

    if runtime_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _ = _time_forward_and_optional_backward(
        model=model,
        inputs=inputs,
        iters=int(warmup_iters),
        include_backward=bool(include_backward),
        device=runtime_device,
    )
    timings = _time_forward_and_optional_backward(
        model=model,
        inputs=inputs,
        iters=int(iters),
        include_backward=bool(include_backward),
        device=runtime_device,
    )
    if not timings:
        raise ValueError("Benchmark produced no timing samples.")
    peak_memory_bytes: Optional[int]
    if runtime_device == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
    else:
        peak_memory_bytes = None
    parameter_summary = summarize_variant_parameters(
        variant_config=variant_config,
        d_in=int(d_in),
    )
    return {
        "variant_slug": variant_config.get("variant_slug"),
        "name": variant_config.get("name"),
        "sae_class": variant_config.get("sae_class"),
        "d_hidden": int(variant_config["d_hidden"]),
        "timings_ms": timings,
        "forward_ms_mean": float(statistics.fmean(timings)),
        "forward_ms_std": (
            float(statistics.pstdev(timings)) if len(timings) > 1 else 0.0
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "parameter_summary": parameter_summary,
    }


def _plot_report(report: Dict[str, Any], out_dir: str) -> None:
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        return

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        sae_class = entry.get("sae_class")
        if not isinstance(sae_class, str):
            continue
        grouped.setdefault(sae_class, []).append(entry)

    if not grouped:
        return

    os.makedirs(out_dir, exist_ok=True)
    figure_forward, axis_forward = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    figure_memory, axis_memory = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    for sae_class in sorted(grouped.keys()):
        points = sorted(grouped[sae_class], key=lambda item: int(item["d_hidden"]))
        widths = [int(point["d_hidden"]) for point in points]
        forward = [float(point["forward_ms_mean"]) for point in points]
        axis_forward.plot(widths, forward, marker="o", label=sae_class)
        peak_memory = [
            (
                float(point["peak_memory_bytes"]) / (1024.0**2)
                if point.get("peak_memory_bytes") is not None
                else None
            )
            for point in points
        ]
        if any(value is not None for value in peak_memory):
            filtered_pairs = [
                (width, value)
                for width, value in zip(widths, peak_memory)
                if value is not None
            ]
            axis_memory.plot(
                [pair[0] for pair in filtered_pairs],
                [pair[1] for pair in filtered_pairs],
                marker="o",
                label=sae_class,
            )

    axis_forward.set_xlabel("d_hidden")
    axis_forward.set_ylabel("forward ms")
    axis_forward.set_title("Forward Time vs Width")
    axis_forward.grid(True, linestyle="--", alpha=0.35)
    axis_forward.legend(fontsize=8, loc="best")
    figure_forward.tight_layout()
    figure_forward.savefig(os.path.join(out_dir, "forward_ms_vs_width.png"))
    plt.close(figure_forward)

    axis_memory.set_xlabel("d_hidden")
    axis_memory.set_ylabel("peak memory (MiB)")
    axis_memory.set_title("Peak Memory vs Width")
    axis_memory.grid(True, linestyle="--", alpha=0.35)
    axis_memory.legend(fontsize=8, loc="best")
    figure_memory.tight_layout()
    figure_memory.savefig(os.path.join(out_dir, "peak_mem_vs_width.png"))
    plt.close(figure_memory)


def main() -> None:
    args = parse_arguments()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.iters <= 0:
        raise ValueError("--iters must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative.")

    runtime_device = _resolve_runtime_device(args.device)
    dtype = _resolve_dtype(dtype_name=args.dtype, device=runtime_device)

    if args.run_dir is not None:
        d_in, variant_configs = _load_run_variants(run_dir=args.run_dir)
        parameter_summary = summarize_run_parameters(output_dir=args.run_dir)
    else:
        widths = _parse_widths(raw_widths=args.widths)
        if args.d_in is None or args.d_in <= 0:
            raise ValueError("--d-in must be provided and positive in standalone mode.")
        d_in = int(args.d_in)
        variant_configs = _manual_variant_configs(
            variants_csv=args.variants,
            widths=widths,
        )
        parameter_summary = {
            "schema_version": 1,
            "output_dir": None,
            "d_in": d_in,
            "variants": [
                summarize_variant_parameters(variant_config=variant, d_in=d_in)
                for variant in variant_configs
            ],
        }

    out_dir = args.out_dir or args.run_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for variant_config in variant_configs:
        results.append(
            _benchmark_variant(
                variant_config=variant_config,
                d_in=int(d_in),
                batch_size=int(args.batch_size),
                runtime_device=runtime_device,
                dtype=dtype,
                warmup_iters=int(args.warmup_iters),
                iters=int(args.iters),
                include_backward=bool(args.include_backward),
            )
        )

    report = {
        "schema_version": 1,
        "settings": {
            "run_dir": args.run_dir,
            "d_in": int(d_in),
            "batch_size": int(args.batch_size),
            "device": runtime_device,
            "dtype": str(dtype),
            "iters": int(args.iters),
            "warmup_iters": int(args.warmup_iters),
            "include_backward": bool(args.include_backward),
        },
        "parameter_summary": parameter_summary,
        "results": results,
    }
    report_path = os.path.join(out_dir, "overhead_report.json")
    with open(report_path, "w") as file:
        json.dump(report, file, indent=2)
    if bool(args.plot):
        _plot_report(report=report, out_dir=out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
