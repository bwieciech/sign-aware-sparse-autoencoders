import posixpath
import re
from typing import Union


_SLUG_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9_]+")
_SLUG_MULTIPLE_UNDERSCORES_PATTERN = re.compile(r"_+")


def slugify_variant_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Variant name must be a non-empty string.")
    slug = name.strip().lower()
    slug = slug.replace(" ", "_")
    slug = _SLUG_NON_ALNUM_PATTERN.sub("_", slug)
    slug = _SLUG_MULTIPLE_UNDERSCORES_PATTERN.sub("_", slug)
    slug = slug.strip("_")
    if not slug:
        raise ValueError(f"Variant name '{name}' produces an empty slug.")
    return slug


def _strip_float_trailing_zeros(value: str) -> str:
    if "e" not in value:
        return value
    mantissa, exponent = value.split("e", maxsplit=1)
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = exponent.replace("+", "")
    return f"{mantissa}e{exponent}"


def format_sweep_value_for_dir(sweep_value: Union[int, float]) -> str:
    if isinstance(sweep_value, bool):
        raise ValueError("Sweep value must be numeric; received a boolean.")
    if isinstance(sweep_value, int):
        return str(sweep_value)
    if not isinstance(sweep_value, (int, float)):
        raise ValueError(f"Sweep value must be numeric; received {type(sweep_value)}.")
    value_float = float(sweep_value)
    if value_float == 0.0:
        return "0"
    formatted = _strip_float_trailing_zeros(f"{value_float:.6e}")
    return formatted


def build_subrun_relative_dir(
    variant_slug: str, seed: int, sweep_param: str, sweep_value: Union[int, float]
) -> str:
    if sweep_param not in {"lambda", "k"}:
        raise ValueError(
            "sweep_param must be one of {'lambda', 'k'}; received " f"{sweep_param}."
        )
    formatted_value = format_sweep_value_for_dir(sweep_value)
    return posixpath.join(
        "models",
        variant_slug,
        f"seed_{seed}",
        f"{sweep_param}_{formatted_value}",
    )
