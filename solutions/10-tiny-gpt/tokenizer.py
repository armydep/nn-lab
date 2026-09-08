"""Character-level tokenizer."""

from __future__ import annotations


class CharTokenizer:
    """Maps characters to integer ids and back.

    Real LLMs use subword tokenizers (BPE) with ~50k tokens, which trade a
    larger vocabulary for shorter sequences. The interface is the same:
    encode(str) -> list[int], decode(list[int]) -> str.
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    def encode(self, s: str) -> list[int]:
        return [self._stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)
