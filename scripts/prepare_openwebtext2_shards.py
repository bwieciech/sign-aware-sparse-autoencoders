import argparse
import glob
import io
import json
import os
import urllib.request
from typing import Dict, Iterable, List, Optional


def _iter_json_records(path: str) -> Iterable[Dict[str, object]]:
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as exc:
            raise ValueError(
                "Reading .jsonl.zst files requires the 'zstandard' package."
            ) from exc

        with open(path, "rb") as compressed_file:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed_file) as stream:
                text_stream = io.TextIOWrapper(stream, encoding="utf-8")
                for line_number, line in enumerate(text_stream, start=1):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        parsed = json.loads(stripped_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed parsing '{path}' at line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"Expected JSON object in '{path}' at line {line_number}."
                        )
                    yield parsed
        return

    with open(path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                parsed = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed parsing '{path}' at line {line_number}: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object in '{path}' at line {line_number}."
                )
            yield parsed


def convert_to_shards(
    input_glob: str,
    output_dir: str,
    records_per_shard: int,
    text_field: str,
    overwrite: bool,
) -> List[str]:
    if not input_glob.strip():
        raise ValueError("input_glob must be a non-empty string.")
    if not output_dir.strip():
        raise ValueError("output_dir must be a non-empty string.")
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive.")

    input_paths = sorted(glob.glob(input_glob))
    if not input_paths:
        raise ValueError(f"No files matched input_glob='{input_glob}'.")

    os.makedirs(output_dir, exist_ok=True)
    existing_shard_paths = sorted(glob.glob(os.path.join(output_dir, "shard_*.jsonl")))
    if existing_shard_paths and not overwrite:
        raise ValueError(
            f"Output dir '{output_dir}' already contains "
            f"{len(existing_shard_paths)} shard_*.jsonl files. "
            "Pass --overwrite or choose a different output dir."
        )
    if overwrite:
        for existing_path in existing_shard_paths:
            os.remove(existing_path)

    shard_index = 0
    written_records = 0
    created_files: List[str] = []
    current_shard_file: Optional[io.TextIOWrapper] = None
    current_shard_path: Optional[str] = None

    def _open_next_shard() -> io.TextIOWrapper:
        nonlocal shard_index
        nonlocal current_shard_path
        current_shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.jsonl")
        shard_index += 1
        created_files.append(current_shard_path)
        return open(current_shard_path, "w")

    try:
        for input_path in input_paths:
            for record in _iter_json_records(input_path):
                if text_field not in record:
                    available_keys = sorted(record.keys())
                    raise ValueError(
                        f"Missing text_field='{text_field}' in '{input_path}'. "
                        f"Available keys: {available_keys}"
                    )

                if current_shard_file is None:
                    current_shard_file = _open_next_shard()

                current_shard_file.write(json.dumps(record, ensure_ascii=False))
                current_shard_file.write("\n")
                written_records += 1

                if written_records % records_per_shard == 0:
                    current_shard_file.close()
                    current_shard_file = None
    finally:
        if current_shard_file is not None:
            current_shard_file.close()

    return created_files


def download_openwebtext2_files(
    base_url: str,
    filenames: List[str],
    output_dir: str,
    overwrite: bool,
) -> List[str]:
    if not base_url.strip():
        raise ValueError("base_url must be a non-empty string.")
    if not filenames:
        raise ValueError("filenames must be non-empty.")

    os.makedirs(output_dir, exist_ok=True)
    downloaded_paths: List[str] = []
    normalized_base_url = base_url.rstrip("/")

    for filename in filenames:
        normalized_filename = str(filename).strip()
        if not normalized_filename:
            raise ValueError("filenames cannot contain empty entries.")
        target_path = os.path.join(output_dir, normalized_filename)
        if os.path.exists(target_path) and not overwrite:
            downloaded_paths.append(target_path)
            continue

        target_parent = os.path.dirname(target_path)
        if target_parent:
            os.makedirs(target_parent, exist_ok=True)

        source_url = f"{normalized_base_url}/{normalized_filename}"
        urllib.request.urlretrieve(source_url, target_path)
        downloaded_paths.append(target_path)

    return downloaded_paths


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--input-glob", required=True, type=str)
    convert_parser.add_argument("--output-dir", required=True, type=str)
    convert_parser.add_argument("--records-per-shard", default=100000, type=int)
    convert_parser.add_argument("--text-field", default="text", type=str)
    convert_parser.add_argument("--overwrite", action="store_true")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--base-url", required=True, type=str)
    download_parser.add_argument("--filenames", required=True, type=str)
    download_parser.add_argument("--output-dir", required=True, type=str)
    download_parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    if args.command == "convert":
        created_files = convert_to_shards(
            input_glob=str(args.input_glob),
            output_dir=str(args.output_dir),
            records_per_shard=int(args.records_per_shard),
            text_field=str(args.text_field),
            overwrite=bool(args.overwrite),
        )
        print(f"Created {len(created_files)} shard files.")
        return

    filenames = [
        entry.strip() for entry in str(args.filenames).split(",") if entry.strip()
    ]
    downloaded_files = download_openwebtext2_files(
        base_url=str(args.base_url),
        filenames=filenames,
        output_dir=str(args.output_dir),
        overwrite=bool(args.overwrite),
    )
    print(f"Downloaded {len(downloaded_files)} files.")


if __name__ == "__main__":
    main()
