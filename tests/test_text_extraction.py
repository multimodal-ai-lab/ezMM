import zipfile

import pytest

from ezmm import embedding
from ezmm.embedding import read_text, read_document_text, read_pdf

requires_embed = pytest.mark.skipif(not embedding.is_available(), reason="Requires ezmm[embed]")


@pytest.mark.parametrize("data, expected", [
    ("# Roses\n\nGrüße aus dem Garten.".encode("utf-8"), "# Roses\n\nGrüße aus dem Garten."),
    ("name,größe\nrose,5".encode("utf-8-sig"), "name,größe\nrose,5"),  # BOM gets removed
    ("name;Größe\nRose;5".encode("latin-1"), "name;Größe\nRose;5"),
    ("Preis: 5 € – günstig".encode("cp1252"), "Preis: 5 € – günstig"),
    ("Hello UTF-16 wörld".encode("utf-16"), "Hello UTF-16 wörld"),
    ("Hello UTF-32 wörld".encode("utf-32"), "Hello UTF-32 wörld"),
])
def test_read_text_encodings(tmp_path, data, expected):
    path = tmp_path / "file.txt"
    path.write_bytes(data)
    assert read_text(path) == expected


def test_read_text_binary(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(bytes(range(256)) * 4)
    assert read_text(path) is None
    assert read_text("in/roses.jpg") is None
    assert read_text("in/tone.wav") is None


def test_read_text_truncated_character(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes("ä".encode("utf-8") * 10)
    assert read_text(path, max_bytes=5) == "ää"  # The cut-off third "ä" gets dropped


def _zip(path, members: dict[str, str]):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_read_office_text(tmp_path):
    docx = _zip(tmp_path / "doc.docx", {
        "word/document.xml": '<w:document><w:body><w:p><w:r><w:t>Hel</w:t></w:r><w:r><w:t>lo</w:t></w:r>'
                             '<w:r><w:tab/><w:t>roses &amp; tulips</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t>'
                             '</w:r></w:p></w:body></w:document>',
        "word/styles.xml": "<w:styles>Ignored</w:styles>"})
    assert read_document_text(docx) == "Hello roses & tulips\nSecond"

    pptx = _zip(tmp_path / "slides.pptx", {
        f"ppt/slides/slide{i}.xml": f"<p:sld><a:p><a:r><a:t>Slide {i}</a:t></a:r></a:p></p:sld>" for i in (1, 2, 10)})
    assert read_document_text(pptx).split("\n") == ["Slide 1", "Slide 2", "Slide 10"]

    xlsx = _zip(tmp_path / "sheet.xlsx", {
        "xl/sharedStrings.xml": "<sst><si><t>name</t></si><si><t>flowers</t></si></sst>"})
    assert read_document_text(xlsx) == "name\nflowers"

    odt = _zip(tmp_path / "doc.odt", {
        "content.xml": "<office:text><text:h>Title</text:h><text:p>Some<text:s/>text</text:p></office:text>"})
    assert read_document_text(odt) == "Title\nSome text"

    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not a zip file")
    assert read_document_text(broken) is None


@requires_embed
def test_read_pdf():
    pages, text = read_pdf("in/sample.pdf")
    assert len(pages) == 1
    assert text.strip() == "Hello from ezMM!"


@requires_embed
def test_text_file_content_gets_embedded(tmp_path):
    """Text files are embedded by their content, not only by their name."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    roses, taxes = tmp_path / "a" / "notes.md", tmp_path / "b" / "notes.md"
    roses.write_text("# Garden\nThe red roses are blooming beautifully this summer.", encoding="cp1252")
    taxes.write_text("# Finance\nThe quarterly tax report is due next week.", encoding="utf-16")
    query = embedding.embed_query("flowers in bloom")
    assert embedding.cos_sim(query, embedding.embed_file(roses)) > embedding.cos_sim(query, embedding.embed_file(taxes))

    prepared = embedding.prepare_input(roses, "file")
    assert prepared.startswith("title: notes.md | text: # Garden")


@requires_embed
def test_text_truncated_to_context(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("Roses and tulips grow in the garden. " * 5000, encoding="utf-8")
    prepared = embedding.prepare_input(path, "file")
    n_tokens = embedding.count_tokens(prepared)
    assert embedding.MAX_CONTEXT_TOKENS - 2 * embedding.CONTEXT_RESERVE < n_tokens <= embedding.MAX_CONTEXT_TOKENS

    # PDFs: the text fills the context left by the page images
    pdf_input = embedding.prepare_input("in/sample.pdf", "file")
    assert pdf_input["text"] == "title: sample.pdf | text: <|image|>Hello from ezMM!"
    assert len(pdf_input["image"]) == 1
