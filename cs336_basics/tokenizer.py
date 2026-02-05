from __future__ import annotations

import regex as re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN


BytesPair = Tuple[bytes, bytes]


def _get_pairs(tokens: Tuple[bytes, ...]) -> set[BytesPair]:
    return set(zip(tokens, tokens[1:]))


def _merge_pair(tokens: Tuple[bytes, ...], pair: BytesPair) -> Tuple[bytes, ...]:
    """
    Merge all occurrences of `pair` in `tokens` in a single pass.
    """

    if len(tokens) < 2:
        return tokens

    a, b = pair
    out: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return tuple(out)


class Tokenizer:
    """
    GPT-2 style BPE tokenizer operating directly on bytes.

    The unit tests provide vocab/merges already converted back to raw bytes, so we
    can implement BPE directly over byte tokens.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: Iterable[BytesPair],
        special_tokens: Optional[Iterable[str]] = None,
    ):
        self.vocab: Dict[int, bytes] = dict(vocab)
        self.byte_to_int: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Ensure all single-byte tokens exist (0..255).
        for i in range(256):
            b = bytes([i])
            if b not in self.byte_to_int:
                new_id = len(self.vocab)
                self.vocab[new_id] = b
                self.byte_to_int[b] = new_id

        # BPE merge ranks: lower rank == higher priority.
        self.merge_ranks: Dict[BytesPair, int] = {pair: rank for rank, pair in enumerate(merges)}

        # Special tokens (string) -> token id
        self.special_tokens: Dict[str, int] = {}
        if special_tokens:
            # Longest-first to correctly handle overlaps.
            for tok in sorted(list(special_tokens), key=len, reverse=True):
                tok_b = tok.encode("utf-8")
                if tok_b in self.byte_to_int:
                    tok_id = self.byte_to_int[tok_b]
                else:
                    tok_id = len(self.vocab)
                    self.vocab[tok_id] = tok_b
                    self.byte_to_int[tok_b] = tok_id
                self.special_tokens[tok] = tok_id

        # For splitting on specials.
        self._special_split_re = None
        if self.special_tokens:
            pat = "(" + "|".join(re.escape(k) for k in sorted(self.special_tokens.keys(), key=len, reverse=True)) + ")"
            self._special_split_re = re.compile(pat)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bpe(self, token_bytes: bytes) -> List[int]:
        """
        Run BPE over a single pretoken (bytes), returning token ids.
        """

        # Start from single-byte symbols.
        tokens: Tuple[bytes, ...] = tuple(bytes([b]) for b in token_bytes)
        if len(tokens) == 0:
            return []

        while True:
            pairs = _get_pairs(tokens)
            if not pairs:
                break

            # Select best pair by rank
            best_pair = None
            best_rank = None
            for p in pairs:
                r = self.merge_ranks.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break

            tokens = _merge_pair(tokens, best_pair)

        # Map final byte tokens to ids (must exist in vocab).
        return [self.byte_to_int[t] for t in tokens]

    def encode(self, text: str, progress_bar: bool = False) -> List[int]:
        """
        Encode full text to token ids, preserving special tokens as indivisible units.
        """

        # Split on special tokens (if any).
        chunks: List[str]
        if self._special_split_re is not None:
            chunks = [c for c in self._special_split_re.split(text) if c != ""]
        else:
            chunks = [text]

        ids: List[int] = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
                continue
            # GPT-2 pretokenization
            for pretoken in re.findall(GPT2_PRETOKENIZER_PATTERN, chunk):
                ids.extend(self._bpe(pretoken.encode("utf-8")))
        return ids

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """
        Streaming encoder over an iterable of strings (e.g., a file object).
        Yields token ids one-by-one.
        """

        for text in texts:
            for _id in self.encode(text):
                yield _id

    def decode(self, ids: List[int]) -> str:
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")
