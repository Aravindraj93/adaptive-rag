"""Configurable token normalization for the CPU-first lexical baseline."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def _light_english_stem(token: str) -> str:
    """Conservative suffix normalization, not a full linguistic stemmer."""

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        return root
    if len(token) > 4 and token.endswith("ed"):
        root = token[:-2]
        if root.endswith("at") or root.endswith("it"):
            return root + "e"
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        return root
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


@dataclass(frozen=True, slots=True)
class NormalizedTokenizer:
    """Unicode-aware word tokenizer with optional lightweight suffix folding."""

    fold_accents: bool = False
    light_stemming: bool = True

    def __call__(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        if self.fold_accents:
            normalized = "".join(
                char
                for char in unicodedata.normalize("NFKD", normalized)
                if not unicodedata.combining(char)
            )
        tokens = _TOKEN.findall(normalized)
        if self.light_stemming:
            return [_light_english_stem(token) for token in tokens]
        return tokens

    def to_dict(self) -> dict[str, bool | str]:
        return {
            "name": "normalized-v1",
            "fold_accents": self.fold_accents,
            "light_stemming": self.light_stemming,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "NormalizedTokenizer":
        if value.get("name") != "normalized-v1":
            raise ValueError("unsupported tokenizer in index manifest")
        return cls(
            fold_accents=bool(value.get("fold_accents", False)),
            light_stemming=bool(value.get("light_stemming", True)),
        )

