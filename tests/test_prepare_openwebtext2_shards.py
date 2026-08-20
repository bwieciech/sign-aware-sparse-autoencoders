import io
import json
import os
import runpy
import sys
import tempfile
import unittest
import types
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from scripts.prepare_openwebtext2_shards import (
    _iter_json_records,
    _parse_arguments,
    convert_to_shards,
    download_openwebtext2_files,
    main,
)


def _write_jsonl(path: str, records: List[Dict[str, str]]) -> None:
    with open(path, "w") as file:
        for record in records:
            file.write(json.dumps(record))
            file.write("\n")


class TestPrepareOpenwebtext2Shards(unittest.TestCase):
    def test_iter_json_records_plain_jsonl_validation(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.jsonl")
            with open(path, "w") as file:
                file.write("\n")
                file.write(json.dumps({"text": "ok"}))
                file.write("\n")

            # when
            records = list(_iter_json_records(path))

            # then
            self.assertEqual(records, [{"text": "ok"}])

            # when / then
            with open(path, "w") as file:
                file.write("{bad\n")
            with self.assertRaises(ValueError):
                _ = list(_iter_json_records(path))

            with open(path, "w") as file:
                file.write("[1,2,3]\n")
            with self.assertRaises(ValueError):
                _ = list(_iter_json_records(path))

    def test_iter_json_records_zst_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            zst_path = os.path.join(tmpdir, "input.jsonl.zst")
            with open(zst_path, "wb") as file:
                file.write(b"placeholder")

            original_import = __import__

            def _import_without_zstandard(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "zstandard":
                    raise ImportError("missing")
                return original_import(name, *args, **kwargs)

            # when / then
            with patch("builtins.__import__", side_effect=_import_without_zstandard):
                with self.assertRaises(ValueError):
                    _ = list(_iter_json_records(zst_path))

            class _FakeReader:
                def __init__(self, payload: bytes) -> None:
                    self.payload = payload
                    self.stream = io.BytesIO(payload)

                def __enter__(self) -> io.BytesIO:
                    return self.stream

                def __exit__(
                    self,
                    exc_type: Any,
                    exc: Any,
                    traceback: Any,
                ) -> bool:
                    del exc_type
                    del exc
                    del traceback
                    return False

            class _FakeDecompressor:
                def __init__(self, payload: bytes) -> None:
                    self.payload = payload

                def stream_reader(self, compressed_file: Any) -> _FakeReader:
                    del compressed_file
                    return _FakeReader(self.payload)

            def _fake_zstandard(payload: bytes) -> Any:
                return types.SimpleNamespace(
                    ZstdDecompressor=lambda: _FakeDecompressor(payload)
                )

            with patch.dict(
                sys.modules,
                {"zstandard": _fake_zstandard(b'\n{"text":"ok"}\n')},
            ):
                records = list(_iter_json_records(zst_path))
            self.assertEqual(records, [{"text": "ok"}])

            with patch.dict(
                sys.modules,
                {"zstandard": _fake_zstandard(b"{bad\n")},
            ):
                with self.assertRaises(ValueError):
                    _ = list(_iter_json_records(zst_path))

            with patch.dict(
                sys.modules,
                {"zstandard": _fake_zstandard(b"[1,2,3]\n")},
            ):
                with self.assertRaises(ValueError):
                    _ = list(_iter_json_records(zst_path))

    def test_convert_to_shards_writes_deterministic_shards(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir, exist_ok=True)

            _write_jsonl(
                os.path.join(input_dir, "b.jsonl"),
                [{"text": "b1"}, {"text": "b2"}],
            )
            _write_jsonl(
                os.path.join(input_dir, "a.jsonl"),
                [{"text": "a1"}, {"text": "a2"}],
            )

            # when
            created = convert_to_shards(
                input_glob=os.path.join(input_dir, "*.jsonl"),
                output_dir=output_dir,
                records_per_shard=2,
                text_field="text",
                overwrite=False,
            )

            # then
            self.assertEqual(
                [os.path.basename(path) for path in created],
                ["shard_00000.jsonl", "shard_00001.jsonl"],
            )

            first_shard_path = os.path.join(output_dir, "shard_00000.jsonl")
            with open(first_shard_path, "r") as file:
                first_shard_lines = [json.loads(line) for line in file if line.strip()]
            self.assertEqual(first_shard_lines, [{"text": "a1"}, {"text": "a2"}])

    def test_convert_to_shards_raises_on_missing_text_field(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            _write_jsonl(input_path, [{"body": "no-text"}])

            # when / then
            with self.assertRaises(ValueError) as context:
                _ = convert_to_shards(
                    input_glob=input_path,
                    output_dir=output_dir,
                    records_per_shard=1,
                    text_field="text",
                    overwrite=False,
                )
            self.assertIn("Missing text_field='text'", str(context.exception))

    def test_convert_to_shards_validation_and_overwrite_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError):
                _ = convert_to_shards(
                    input_glob="",
                    output_dir=output_dir,
                    records_per_shard=1,
                    text_field="text",
                    overwrite=False,
                )
            with self.assertRaises(ValueError):
                _ = convert_to_shards(
                    input_glob=os.path.join(input_dir, "*.jsonl"),
                    output_dir="",
                    records_per_shard=1,
                    text_field="text",
                    overwrite=False,
                )
            with self.assertRaises(ValueError):
                _ = convert_to_shards(
                    input_glob=os.path.join(input_dir, "*.jsonl"),
                    output_dir=output_dir,
                    records_per_shard=0,
                    text_field="text",
                    overwrite=False,
                )
            with self.assertRaises(ValueError):
                _ = convert_to_shards(
                    input_glob=os.path.join(input_dir, "*.jsonl"),
                    output_dir=output_dir,
                    records_per_shard=1,
                    text_field="text",
                    overwrite=False,
                )

            input_path = os.path.join(input_dir, "a.jsonl")
            _write_jsonl(input_path, [{"text": "a"}])
            os.makedirs(output_dir, exist_ok=True)
            stale_shard_path = os.path.join(output_dir, "shard_00000.jsonl")
            with open(stale_shard_path, "w") as file:
                file.write("stale\n")

            with self.assertRaises(ValueError):
                _ = convert_to_shards(
                    input_glob=input_path,
                    output_dir=output_dir,
                    records_per_shard=1,
                    text_field="text",
                    overwrite=False,
                )

            created = convert_to_shards(
                input_glob=input_path,
                output_dir=output_dir,
                records_per_shard=1,
                text_field="text",
                overwrite=True,
            )
            self.assertEqual(len(created), 1)
            with open(created[0], "r") as file:
                lines = [line.strip() for line in file if line.strip()]
            self.assertEqual(lines, ['{"text": "a"}'])

    def test_convert_to_shards_closes_open_file_on_exception(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            with open(input_path, "w") as file:
                file.write(json.dumps({"text": "ok"}))
                file.write("\n")
                file.write(json.dumps({"bad": "missing"}))
                file.write("\n")

            # when / then
            with self.assertRaises(ValueError):
                _ = convert_to_shards(
                    input_glob=input_path,
                    output_dir=output_dir,
                    records_per_shard=10,
                    text_field="text",
                    overwrite=False,
                )

    def test_download_openwebtext2_files_calls_urlretrieve(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "downloads")

            def _fake_urlretrieve(url: str, target_path: str) -> Tuple[str, object]:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w") as file:
                    file.write(url)
                return target_path, None

            # when
            with patch(
                "scripts.prepare_openwebtext2_shards.urllib.request.urlretrieve",
                side_effect=_fake_urlretrieve,
            ) as mocked_download:
                downloaded = download_openwebtext2_files(
                    base_url="https://example.com/openwebtext2",
                    filenames=["part-000.jsonl.zst", "part-001.jsonl.zst"],
                    output_dir=output_dir,
                    overwrite=False,
                )

            # then
            self.assertEqual(mocked_download.call_count, 2)
            self.assertEqual(len(downloaded), 2)
            self.assertTrue(os.path.isfile(downloaded[0]))
            with open(downloaded[0], "r") as file:
                payload = file.read()
            self.assertIn(
                "https://example.com/openwebtext2/part-000.jsonl.zst", payload
            )

    def test_download_openwebtext2_files_validation_and_skip_existing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "downloads")
            os.makedirs(output_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError):
                _ = download_openwebtext2_files(
                    base_url="",
                    filenames=["a.jsonl.zst"],
                    output_dir=output_dir,
                    overwrite=False,
                )
            with self.assertRaises(ValueError):
                _ = download_openwebtext2_files(
                    base_url="https://example.com",
                    filenames=[],
                    output_dir=output_dir,
                    overwrite=False,
                )
            with self.assertRaises(ValueError):
                _ = download_openwebtext2_files(
                    base_url="https://example.com",
                    filenames=[" "],
                    output_dir=output_dir,
                    overwrite=False,
                )

            existing_path = os.path.join(output_dir, "part-000.jsonl.zst")
            with open(existing_path, "w") as file:
                file.write("existing")

            with patch(
                "scripts.prepare_openwebtext2_shards.urllib.request.urlretrieve"
            ) as mocked_download:
                result = download_openwebtext2_files(
                    base_url="https://example.com/",
                    filenames=["part-000.jsonl.zst"],
                    output_dir=output_dir,
                    overwrite=False,
                )

            self.assertEqual(result, [existing_path])
            self.assertEqual(mocked_download.call_count, 0)

    def test_parse_arguments_and_main_convert_download_paths(self) -> None:
        # given / when / then
        with patch(
            "sys.argv",
            [
                "prepare_openwebtext2_shards.py",
                "convert",
                "--input-glob",
                "in/*.jsonl",
                "--output-dir",
                "out",
            ],
        ):
            args = _parse_arguments()
        self.assertEqual(args.command, "convert")
        self.assertEqual(args.records_per_shard, 100000)

        with patch(
            "sys.argv",
            [
                "prepare_openwebtext2_shards.py",
                "download",
                "--base-url",
                "https://example.com",
                "--filenames",
                "a,b",
                "--output-dir",
                "out",
            ],
        ):
            args = _parse_arguments()
        self.assertEqual(args.command, "download")

        with patch(
            "scripts.prepare_openwebtext2_shards._parse_arguments",
            return_value=types.SimpleNamespace(
                command="convert",
                input_glob="in/*.jsonl",
                output_dir="out",
                records_per_shard=2,
                text_field="text",
                overwrite=False,
            ),
        ), patch(
            "scripts.prepare_openwebtext2_shards.convert_to_shards",
            return_value=["a", "b"],
        ) as convert_mock, patch(
            "builtins.print"
        ) as print_mock:
            main()
        self.assertEqual(convert_mock.call_count, 1)
        self.assertEqual(print_mock.call_count, 1)

        with patch(
            "scripts.prepare_openwebtext2_shards._parse_arguments",
            return_value=types.SimpleNamespace(
                command="download",
                base_url="https://example.com",
                filenames="a,b",
                output_dir="out",
                overwrite=False,
            ),
        ), patch(
            "scripts.prepare_openwebtext2_shards.download_openwebtext2_files",
            return_value=["a"],
        ) as download_mock, patch(
            "builtins.print"
        ) as print_mock:
            main()
        self.assertEqual(download_mock.call_count, 1)
        self.assertEqual(print_mock.call_count, 1)

    def test_main_guard_executes_when_run_as_script(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            _write_jsonl(input_path, [{"text": "script mode"}])
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "scripts",
                "prepare_openwebtext2_shards.py",
            )

            # when
            with patch(
                "sys.argv",
                [
                    script_path,
                    "convert",
                    "--input-glob",
                    input_path,
                    "--output-dir",
                    output_dir,
                    "--records-per-shard",
                    "1",
                ],
            ), patch("builtins.print") as print_mock:
                runpy.run_path(script_path, run_name="__main__")

            # then
            self.assertEqual(print_mock.call_count, 1)
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "shard_00000.jsonl"))
            )
