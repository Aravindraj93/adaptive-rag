"""Small maintained embedding adapters with no model or network dependency."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

from .tokenization import NormalizedTokenizer


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    """Map tokens to a fixed-size signed hashing vector.

    This is a deterministic CPU baseline for plumbing and fusion tests. It captures
    token overlap, not semantic similarity, and does not replace a trained model.
    """

    dimensions: int = 256
    tokenizer: NormalizedTokenizer = field(default_factory=NormalizedTokenizer)

    def __post_init__(self) -> None:
        if self.dimensions < 8:
            raise ValueError("dimensions must be at least 8")

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self.tokenizer(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += -1.0 if (value >> 32) & 1 else 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            return [value / magnitude for value in vector]
        return vector

