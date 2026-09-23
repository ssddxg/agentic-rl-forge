from __future__ import annotations

import unicodedata
from itertools import pairwise

TOKENIZER_VERSION = "unicode-nfkc-cjk-unigram-bigram-v1"


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _append_cjk_tokens(tokens: list[str], characters: list[str]) -> None:
    if characters:
        tokens.extend(characters)
    if len(characters) > 1:
        tokens.extend(left + right for left, right in pairwise(characters))
    characters.clear()


def _append_word(tokens: list[str], characters: list[str]) -> None:
    if characters:
        tokens.append("".join(characters))
    characters.clear()


def tokenize_text(text: str) -> list[str]:
    """Normalize text and split it into words and searchable CJK bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    word: list[str] = []
    cjk_run: list[str] = []

    for character in normalized:
        if _is_cjk(character):
            _append_word(tokens, word)
            cjk_run.append(character)
            continue
        if character.isalnum() or character == "_":
            _append_cjk_tokens(tokens, cjk_run)
            word.append(character)
            continue
        _append_word(tokens, word)
        _append_cjk_tokens(tokens, cjk_run)

    _append_word(tokens, word)
    _append_cjk_tokens(tokens, cjk_run)
    return tokens
