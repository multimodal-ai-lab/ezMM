import base64
from pathlib import Path

import pytest

from ezmm import File, FileTooLargeError, MultimodalSequence, get_max_file_size, set_max_file_size
from ezmm.common import item_registry


@pytest.fixture
def max_file_size():
    """Restores the maximum file size after the test."""
    original = get_max_file_size()
    yield
    set_max_file_size(original)


def test_file():
    file = File("in/sample.pdf")
    assert file.kind == "file"
    assert file.reference == f"<file:{file.id}>"
    assert file.mime_type == "application/pdf"
    assert file.size == Path("in/sample.pdf").stat().st_size


def test_binary():
    data = Path("in/sample.pdf").read_bytes()
    file = File(binary_data=data, mime_type="application/pdf")
    assert file.file_path.suffix == ".pdf"
    assert file.bytes == data
    assert base64.b64decode(file.get_base64_encoded()) == data


def test_binary_with_suffix():
    file = File(binary_data=Path("in/table.csv").read_bytes(), suffix=".csv")
    assert file.file_path.suffix == ".csv"


def test_html_is_download_link():
    file = File("in/table.csv")
    html = file.as_html()
    assert html.startswith("<a ")
    assert "download=" in html
    assert file.file_url in html


def test_file_in_sequence():
    pdf = File("in/sample.pdf")
    csv = File("in/table.csv")
    seq = MultimodalSequence("See", pdf, "and", csv)
    assert seq.has_files()
    assert seq.files == [pdf, csv]
    assert MultimodalSequence(str(seq)) == seq


def test_default_max_file_size():
    assert get_max_file_size() == 100 * 1024 ** 2


def test_max_file_size_path(max_file_size):
    set_max_file_size(100)
    with pytest.raises(FileTooLargeError):
        File("in/sample.pdf")
    assert item_registry.count_items("file") == 0


def test_max_file_size_binary(max_file_size):
    set_max_file_size(100)
    with pytest.raises(FileTooLargeError):
        File(binary_data=Path("in/sample.pdf").read_bytes())
    temp_dir = item_registry.path / "items"
    assert not temp_dir.exists() or not any(temp_dir.iterdir())  # Nothing written


def test_max_file_size_disabled(max_file_size):
    set_max_file_size(None)
    assert File("in/sample.pdf")


def test_max_file_size_only_applies_to_files(max_file_size):
    from ezmm import Image
    set_max_file_size(100)
    assert Image("in/roses.jpg")  # Much larger than 100 bytes
