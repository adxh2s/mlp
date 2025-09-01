from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.instrumentation.file_manager import FileManager


@pytest.fixture()
def fm() -> FileManager:
    return FileManager()


def test_check_path_exists_and_types(tmp_path: Path, fm: FileManager) -> None:
    d = tmp_path / "dir"
    f = tmp_path / "file.txt"
    d.mkdir()
    f.write_text("hello", encoding="utf-8")

    assert fm.check_path_exists(d) is True
    assert fm.check_path_exists(f) is True
    assert fm.is_dir(d) is True
    assert fm.is_dir(f) is False
    assert fm.is_file(f) is True
    assert fm.is_file(d) is False
    assert fm.check_path_exists(tmp_path / "missing") is False


def test_permissions(tmp_path: Path, fm: FileManager) -> None:
    f = tmp_path / "file.txt"
    f.write_text("data", encoding="utf-8")
    assert fm.has_perm_read(f) is True
    assert fm.has_perm_write(f) is True
    assert isinstance(fm.has_perm_exec(f), bool)


def test_ensure_dir_idempotent(tmp_path: Path, fm: FileManager) -> None:
    d = tmp_path / "a" / "b" / "c"
    p1 = fm.ensure_dir(d)
    p2 = fm.ensure_dir(d)
    assert p1 == d == p2
    assert d.exists() and d.is_dir()


def test_list_files_by_ext(tmp_path: Path, fm: FileManager) -> None:
    (tmp_path / "a.py").write_text("# py", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("k: v", encoding="utf-8")
    (tmp_path / "c.yml").write_text("k: v", encoding="utf-8")
    (tmp_path / "d.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "e.py").write_text("# py in sub", encoding="utf-8")

    found = fm.list_files_by_ext(tmp_path, [".py", ".yaml", ".yml"])
    names = [p.name for p in found]
    assert "a.py" in names
    assert "b.yaml" in names
    assert "c.yml" in names
    assert "d.txt" not in names
    assert "e.py" not in names  # non récursif


def test_make_timestamp_name_format(fm: FileManager, tmp_path: Path) -> None:
    src = tmp_path / "data.csv"
    src.write_text("x,y\n1,2\n", encoding="utf-8")
    stamped = fm.make_timestamp_name(src)
    assert stamped.endswith("_data.csv")
    # YYYYMMDD_HHMMSS = 15, présence d'un underscore
    assert len(stamped.split("_", 1)) >= 8


def test_copy_file_with_and_without_rename(tmp_path: Path, fm: FileManager) -> None:
    src = tmp_path / "file.bin"
    src.write_bytes(b"\x00\x01\x02")

    dest_dir = tmp_path / "out"
    copied = fm.copy_file(src, dest_dir)
    assert copied.exists() and copied.name == "file.bin"
    assert copied.read_bytes() == b"\x00\x01\x02"

    copied2 = fm.copy_file(src, dest_dir, rename="renamed.bin")
    assert copied2.exists() and copied2.name == "renamed.bin"
    assert copied2.read_bytes() == b"\x00\x01\x02"


def test_compress_file_gz(tmp_path: Path, fm: FileManager) -> None:
    src = tmp_path / "raw.dat"
    src.write_bytes(b"abcdef" * 100)
    gz_path = fm.compress_file_gz(src)
    assert gz_path.exists() and gz_path.suffix == ".gz"
    assert src.exists()  # original conservé

    src2 = tmp_path / "raw2.dat"
    src2.write_bytes(b"012345" * 100)
    gz_path2 = fm.compress_file_gz(src2, delete_original=True)
    assert gz_path2.exists()
    assert not src2.exists()


def test_read_write_csv(tmp_path: Path, fm: FileManager) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "data.csv"
    fm.write_file(df, path)
    assert path.exists()
    df2 = fm.read_file(path)
    pd.testing.assert_frame_equal(df, df2)


@pytest.mark.parametrize("ext", [".xlsx", ".xls"])
def test_read_write_excel(tmp_path: Path, fm: FileManager, ext: str) -> None:
    df = pd.DataFrame({"a": [10, 20]})
    path = tmp_path / f"book{ext}"
    fm.write_file(df, path)
    assert path.exists()
    df2 = fm.read_file(path)
    pd.testing.assert_frame_equal(df, df2)


def test_read_write_json(tmp_path: Path, fm: FileManager) -> None:
    df = pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    path = tmp_path / "data.json"
    fm.write_file(df, path)
    assert path.exists()
    df2 = fm.read_file(path)
    df2 = df2[df.columns.tolist()]
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_read_file_unsupported_extension_raises(tmp_path: Path, fm: FileManager) -> None:
    path = tmp_path / "data.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        fm.read_file(path)


def test_write_file_non_dataframe_text_and_bytes(tmp_path: Path, fm: FileManager) -> None:
    p_txt = tmp_path / "msg.txt"
    fm.write_file("hello", p_txt)
    assert p_txt.read_text(encoding="utf-8") == "hello"

    p_bin = tmp_path / "blob.bin"
    fm.write_file(b"\x10\x20", p_bin)
    assert p_bin.read_bytes() == b"\x10\x20"
