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


_CAMEL_CASE_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_CASE_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_ALPHA_DIGIT = re.compile(r"[a-zA-Z]+|[0-9]+")


@dataclass(frozen=True, slots=True)
class NormalizedTokenizer:
    """Unicode-aware word tokenizer with optional lightweight suffix folding."""

    fold_accents: bool = False
    light_stemming: bool = True
    split_identifiers: bool = False

    def __call__(self, text: str) -> list[str]:
        if self.split_identifiers:
            return self._tokenize_with_identifiers(text)
        return self._tokenize_standard(text)

    def _tokenize_standard(self, text: str) -> list[str]:
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

    def _tokenize_with_identifiers(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text)
        if self.fold_accents:
            normalized = "".join(
                char
                for char in unicodedata.normalize("NFKD", normalized)
                if not unicodedata.combining(char)
            )
        # Preserve original tokens as well as camelCase splits
        camel = _CAMEL_CASE_1.sub(r"\1 \2", normalized)
        camel = _CAMEL_CASE_2.sub(r"\1 \2", camel).casefold()
        base_tokens = _TOKEN.findall(normalized.casefold())
        camel_tokens = _TOKEN.findall(camel)

        raw_parts: list[str] = []
        seen: set[str] = set()
        for tok in base_tokens + camel_tokens:
            if tok not in seen:
                seen.add(tok)
                raw_parts.append(tok)
            if "_" in tok:
                for part in tok.split("_"):
                    if len(part) >= 2 and part not in seen:
                        seen.add(part)
                        raw_parts.append(part)
                    for sub in _ALPHA_DIGIT.findall(part):
                        if len(sub) >= 2 and sub not in seen:
                            seen.add(sub)
                            raw_parts.append(sub)
            else:
                for sub in _ALPHA_DIGIT.findall(tok):
                    if len(sub) >= 2 and sub not in seen:
                        seen.add(sub)
                        raw_parts.append(sub)

        if self.light_stemming:
            result: list[str] = []
            seen_out: set[str] = set()
            for part in raw_parts:
                stemmed = _light_english_stem(part) if not part.isdigit() else part
                if stemmed not in seen_out:
                    seen_out.add(stemmed)
                    result.append(stemmed)
                if stemmed != part and part not in seen_out:
                    seen_out.add(part)
                    result.append(part)
            return result
        return raw_parts

    def to_dict(self) -> dict[str, bool | str]:
        return {
            "name": "normalized-v1",
            "fold_accents": self.fold_accents,
            "light_stemming": self.light_stemming,
            "split_identifiers": self.split_identifiers,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "NormalizedTokenizer":
        if value.get("name") != "normalized-v1":
            raise ValueError("unsupported tokenizer in index manifest")
        return cls(
            fold_accents=bool(value.get("fold_accents", False)),
            light_stemming=bool(value.get("light_stemming", True)),
            split_identifiers=bool(value.get("split_identifiers", False)),
        )


@dataclass(frozen=True, slots=True)
class CharNGramTokenizer:
    """Character n-gram tokenizer for sub-word and technical symbol matching."""

    min_n: int = 3
    max_n: int = 4

    def __call__(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens: list[str] = []
        seen: set[str] = set()
        words = _TOKEN.findall(normalized)
        sub_words = re.split(r"[_\.\s]+", normalized)
        for word in words + [w for w in sub_words if len(w) >= 2]:
            if word not in seen:
                seen.add(word)
                tokens.append(word)
            word_len = len(word)
            for n in range(self.min_n, min(self.max_n + 1, word_len + 1)):
                for i in range(word_len - n + 1):
                    gram = word[i : i + n]
                    if gram not in seen:
                        seen.add(gram)
                        tokens.append(gram)
        return tokens

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": "char-ngram-v1",
            "min_n": self.min_n,
            "max_n": self.max_n,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CharNGramTokenizer":
        if value.get("name") != "char-ngram-v1":
            raise ValueError("unsupported tokenizer in index manifest")
        return cls(
            min_n=int(value.get("min_n", 3)),
            max_n=int(value.get("max_n", 4)),
        )


