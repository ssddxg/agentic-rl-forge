from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import agentic_rl_forge.data.corpus as corpus_module
from agentic_rl_forge.data import (
    CorpusDocument,
    build_text_corpus,
    corpus_digest,
    inspect_corpus,
    load_corpus_jsonl,
)


def _jsonl(*records: object) -> bytes:
    return b"".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for record in records
    )


def test_load_corpus_accepts_bom_scalar_ids_and_preserves_metadata(tmp_path: Path) -> None:
    path = tmp_path / "资料.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf"
        + _jsonl(
            {"id": 7, "contents": "巴黎是法国首都。", "source": "城市.md", "rank": 1},
            {"id": "berlin", "contents": "Berlin is in Germany.", "source": "城市.md"},
        )
    )

    documents = load_corpus_jsonl(path)
    report = inspect_corpus(path)

    assert documents[0].document_id == "7"
    assert documents[0].metadata == {"rank": 1, "source": "城市.md"}
    assert report.document_count == 2
    assert report.source_count == 1
    assert report.total_characters == sum(len(item.contents) for item in documents)
    assert report.min_characters == min(len(item.contents) for item in documents)
    assert report.max_characters == max(len(item.contents) for item in documents)
    assert report.average_characters == report.total_characters / 2
    assert len(report.file_sha256) == 64
    assert report.corpus_sha256 == corpus_digest(documents)


@pytest.mark.parametrize(
    ("payload", "line_number", "message"),
    (
        (b"not-json\n", 1, "not valid JSON"),
        (_jsonl(["not", "object"]), 1, "not a JSON object"),
        (_jsonl({"contents": "missing id"}), 1, "does not contain 'id'"),
        (_jsonl({"id": "  ", "contents": "text"}), 1, "empty document id"),
        (_jsonl({"id": "a"}), 1, "does not contain 'contents'"),
        (_jsonl({"id": "a", "contents": "  "}), 1, "non-empty string"),
        (
            _jsonl(
                {"id": "same", "contents": "first"},
                {"id": "same", "contents": "second"},
            ),
            2,
            "duplicates document id",
        ),
    ),
)
def test_load_corpus_rejects_invalid_records_with_path_and_line(
    tmp_path: Path,
    payload: bytes,
    line_number: int,
    message: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(path))}:{line_number}.*{message}",
    ):
        load_corpus_jsonl(path)


def test_load_corpus_rejects_invalid_utf8_and_empty_files(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-utf8.jsonl"
    invalid.write_bytes(b'{"id":"a","contents":"ok"}\n\xff')
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"\xef\xbb\xbf\n\n")

    with pytest.raises(ValueError, match=rf"{re.escape(str(invalid))}:2.*UTF-8"):
        load_corpus_jsonl(invalid)
    with pytest.raises(ValueError, match=rf"{re.escape(str(empty))}:1.*empty"):
        load_corpus_jsonl(empty)


def test_corpus_digest_is_order_and_metadata_key_order_independent() -> None:
    left = CorpusDocument("a", "alpha", {"source": "one", "nested": {"x": 1, "y": 2}})
    right = CorpusDocument("b", "beta", {"labels": ["x", "y"]})
    reordered_left = CorpusDocument(
        "a",
        "alpha",
        {"nested": {"y": 2, "x": 1}, "source": "one"},
    )

    assert corpus_digest((left, right)) == corpus_digest((right, reordered_left))
    assert corpus_digest((left, right)) != corpus_digest(
        (left, CorpusDocument("b", "changed", {"labels": ["x", "y"]}))
    )
    with pytest.raises(ValueError, match="duplicate document id"):
        corpus_digest((left, left))


def test_build_text_corpus_recurses_chunks_deduplicates_and_skips_hidden(
    tmp_path: Path,
) -> None:
    source = tmp_path / "中文资料"
    nested = source / "章节"
    hidden = source / ".private"
    nested.mkdir(parents=True)
    hidden.mkdir()
    (nested / "介绍.md").write_bytes(b"\xef\xbb\xbf" + "甲乙丙丁戊己庚辛壬癸".encode())
    (source / "duplicate.txt").write_text("甲乙丙丁戊己庚辛壬癸", encoding="utf-8")
    (source / "empty.rst").write_text(" \n", encoding="utf-8")
    (source / ".ignored.md").write_text("secret", encoding="utf-8")
    (hidden / "ignored.txt").write_text("secret", encoding="utf-8")
    (source / "ignored.bin").write_bytes(b"binary")
    output = tmp_path / "corpus.jsonl"

    report = build_text_corpus(source, output, chunk_size=5, chunk_overlap=1)
    documents = load_corpus_jsonl(output)

    assert [item.contents for item in documents] == ["甲乙丙丁戊", "戊己庚辛壬", "壬癸"]
    assert report.scanned_file_count == 3
    assert report.included_file_count == 1
    assert report.skipped_empty_file_count == 1
    assert report.duplicate_chunk_count == 3
    assert report.document_count == 3
    assert report.source_count == 1
    assert {item.metadata["source"] for item in documents} == {"duplicate.txt"}
    assert [item.metadata["chunk_index"] for item in documents] == [0, 1, 2]
    assert {item.metadata["chunk_count"] for item in documents} == {3}
    assert report.file_sha256 == inspect_corpus(output).file_sha256


def test_build_text_corpus_is_deterministic_and_protects_existing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    (source / "b.md").write_text("second document", encoding="utf-8")
    (source / "a.txt").write_text("first document", encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_report = build_text_corpus(source, first)
    second_report = build_text_corpus(source, second)
    original = first.read_bytes()

    assert first.read_bytes() == second.read_bytes()
    assert first_report.corpus_sha256 == second_report.corpus_sha256
    with pytest.raises(FileExistsError, match="output already exists"):
        build_text_corpus(source, first)
    assert first.read_bytes() == original

    (source / "a.txt").write_text("updated document", encoding="utf-8")
    replaced = build_text_corpus(source, first, force=True)
    assert first.read_bytes() != original
    assert replaced.corpus_sha256 != first_report.corpus_sha256


def test_build_text_corpus_falls_back_when_hard_links_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("portable corpus", encoding="utf-8")
    output = tmp_path / "corpus.jsonl"

    def unsupported_link(source_path: Path, destination_path: Path) -> None:
        del source_path, destination_path
        raise OSError("hard links are unavailable")

    monkeypatch.setattr(corpus_module.os, "link", unsupported_link)

    report = build_text_corpus(source, output)

    assert report.document_count == 1
    assert [document.contents for document in load_corpus_jsonl(output)] == ["portable corpus"]
    assert not tuple(tmp_path.glob(".corpus-*.tmp"))


def test_hard_link_fallback_does_not_overwrite_a_racing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("candidate corpus", encoding="utf-8")
    output = tmp_path / "corpus.jsonl"
    competing_payload = b"written by another process"

    def racing_unsupported_link(source_path: Path, destination_path: Path) -> None:
        del source_path
        Path(destination_path).write_bytes(competing_payload)
        raise OSError("hard links are unavailable")

    monkeypatch.setattr(corpus_module.os, "link", racing_unsupported_link)

    with pytest.raises(FileExistsError, match="output already exists"):
        build_text_corpus(source, output)

    assert output.read_bytes() == competing_payload
    assert not tuple(tmp_path.glob(".corpus-*.tmp"))


def test_failed_fallback_does_not_remove_a_replaced_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("candidate corpus", encoding="utf-8")
    output = tmp_path / "corpus.jsonl"
    competing_payload = b"replacement from another process"

    def unsupported_link(source_path: Path, destination_path: Path) -> None:
        del source_path, destination_path
        raise OSError("hard links are unavailable")

    def failed_replace(source_path: Path, destination_path: Path) -> None:
        del source_path
        destination = Path(destination_path)
        destination.unlink()
        destination.write_bytes(competing_payload)
        raise OSError("publish failed")

    monkeypatch.setattr(corpus_module.os, "link", unsupported_link)
    monkeypatch.setattr(corpus_module.os, "replace", failed_replace)

    with pytest.raises(OSError, match="publish failed"):
        build_text_corpus(source, output)

    assert output.read_bytes() == competing_payload
    assert not tuple(tmp_path.glob(".corpus-*.tmp"))


def test_failed_fallback_cleans_up_its_unchanged_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("candidate corpus", encoding="utf-8")
    output = tmp_path / "corpus.jsonl"

    def unsupported_link(source_path: Path, destination_path: Path) -> None:
        del source_path, destination_path
        raise OSError("hard links are unavailable")

    def failed_replace(source_path: Path, destination_path: Path) -> None:
        del source_path, destination_path
        raise OSError("publish failed")

    monkeypatch.setattr(corpus_module.os, "link", unsupported_link)
    monkeypatch.setattr(corpus_module.os, "replace", failed_replace)

    with pytest.raises(OSError, match="publish failed"):
        build_text_corpus(source, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".corpus-*.tmp"))


def test_build_text_corpus_ignores_file_and_directory_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "documents"
    outside = tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    (source / "real.txt").write_text("included", encoding="utf-8")
    (outside / "linked.txt").write_text("excluded file", encoding="utf-8")
    (outside / "nested.txt").write_text("excluded directory", encoding="utf-8")
    try:
        (source / "file-link.txt").symlink_to(outside / "linked.txt")
        (source / "directory-link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available on this host")

    output = tmp_path / "corpus.jsonl"
    report = build_text_corpus(source, output)
    documents = load_corpus_jsonl(output)

    assert report.scanned_file_count == 1
    assert [item.contents for item in documents] == ["included"]


def test_build_text_corpus_leaves_no_partial_output_after_failure(tmp_path: Path) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    (source / "bad.txt").write_bytes(b"valid line\n\xff")
    output = tmp_path / "nested" / "corpus.jsonl"

    with pytest.raises(ValueError, match="not valid UTF-8"):
        build_text_corpus(source, output)

    assert not output.exists()
    assert not output.parent.exists()
    assert not tuple(tmp_path.rglob(".corpus-*.tmp"))


def test_build_text_corpus_validates_limits_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("content", encoding="utf-8")
    output = tmp_path / "corpus.jsonl"

    with pytest.raises(ValueError, match="chunk_overlap"):
        build_text_corpus(source, output, chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="max_file_bytes"):
        build_text_corpus(source, output, max_file_bytes=1)

    assert not output.exists()
