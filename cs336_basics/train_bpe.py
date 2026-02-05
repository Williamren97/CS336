"""
Fast BPE trainer used by the unit tests.

Implementation closely follows the reference solution from
`ZitongYang/cs336-assignment1-basics`:
`https://raw.githubusercontent.com/ZitongYang/cs336-assignment1-basics/master/cs336_basics/train_bpe.py`
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from typing import Iterable

import regex as re
from tqdm import tqdm

from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN


def _find_pretokens(text: str) -> Counter[str]:
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))


def _read_text_file(input_path: str, num_workers: int, special_tokens: Iterable[str]):
    """
    Read a text file and return a frequency table of pretokens, represented as tuples of bytes.

    Special tokens are removed from the text before pretokenization so that they do not participate
    in merges (and so we never create tokens containing b"<|", per the unit tests).
    """

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Treat special tokens as boundaries by splitting on them (do NOT delete and concatenate).
    # We drop the special tokens from pretokenization so they can't be merged into other tokens.
    special_tokens = list(special_tokens)
    if special_tokens:
        pat = "(" + "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + ")"
        parts = [p for p in re.split(pat, text) if p and p not in special_tokens]
        if num_workers == 1:
            pretokens = sum((_find_pretokens(p) for p in parts), Counter())
        else:
            # Parallelize across parts (coarse-grained).
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                pretokens_iter = executor.map(_find_pretokens, parts)
            pretokens = sum(pretokens_iter, Counter())
    else:
        if num_workers == 1:
            pretokens = _find_pretokens(text)
        else:
            chunk_size = max(1, len(text) // num_workers)
            text_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                pretokens_iter = executor.map(_find_pretokens, text_chunks)
            pretokens = sum(pretokens_iter, Counter())

    def to_tuple_of_bytes(pretoken: str) -> tuple[bytes, ...]:
        return tuple(bytes([b]) for b in pretoken.encode("utf-8"))

    return {to_tuple_of_bytes(pretoken): freq for pretoken, freq in pretokens.items()}


def _update_byte_tuple(byte_tuple: tuple[bytes, ...], merge_loc: int):
    """
    Merge the byte_tuple at merge_loc and return (new_tuple, prefix, suffix).
    """

    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc : merge_loc + 2]
    suffix = byte_tuple[merge_loc + 2 :]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix


def _merge_all(byte_tuple: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Merge all occurrences of `pair` in `byte_tuple` in one pass.
    """
    if len(byte_tuple) < 2:
        return byte_tuple
    a, b = pair
    out: list[bytes] = []
    i = 0
    while i < len(byte_tuple):
        if i < len(byte_tuple) - 1 and byte_tuple[i] == a and byte_tuple[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(byte_tuple[i])
            i += 1
    return tuple(out)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: Iterable[str],
    progress_bar: bool = False,
    num_workers: int = 1,
):
    """
    Train a byte pair encoding tokenizer on the input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.

    Returns:
        Tuple of the learned vocab and merges.
    """
    special_tokens = list(special_tokens)

    # Initialize vocab with 256 bytes + special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    pretoken_freq: dict[tuple[bytes, ...], int] = _read_text_file(input_path, num_workers, special_tokens)

    # Initial pair frequency table + inverted index: pair -> set(pretoken_tuple)
    pair_freq: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for pretoken_tuple, freq in pretoken_freq.items():
        for a, b in zip(pretoken_tuple, pretoken_tuple[1:]):
            pair = (a, b)
            pair_freq[pair] += freq
            pair_to_pretokens.setdefault(pair, set()).add(pretoken_tuple)

    merges: list[tuple[bytes, bytes]] = []
    pbar = tqdm(total=vocab_size - len(vocab), disable=not progress_bar)

    next_id = max(vocab.keys()) + 1
    while len(vocab) < vocab_size:
        # Drop dead pairs to keep `max()` scans smaller.
        for k in [k for k, v in pair_freq.items() if v <= 0]:
            del pair_freq[k]
            pair_to_pretokens.pop(k, None)

        most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        merges.append(most_freq_pair)
        new_token = most_freq_pair[0] + most_freq_pair[1]

        vocab[next_id] = new_token
        next_id += 1

        affected = list(pair_to_pretokens.get(most_freq_pair, set()))
        pair_to_pretokens.pop(most_freq_pair, None)

        for old_tuple in affected:
            freq = pretoken_freq.get(old_tuple)
            if freq is None:
                continue  # stale

            # Remove old tuple's pair contributions
            if len(old_tuple) >= 2:
                for a, b in zip(old_tuple, old_tuple[1:]):
                    pair_freq[(a, b)] -= freq

            # Merge and update frequency table
            new_tuple = _merge_all(old_tuple, most_freq_pair)
            del pretoken_freq[old_tuple]
            pretoken_freq[new_tuple] = pretoken_freq.get(new_tuple, 0) + freq

            # Add new tuple's pair contributions and update inverted index
            if len(new_tuple) >= 2:
                for a, b in zip(new_tuple, new_tuple[1:]):
                    pair = (a, b)
                    pair_freq[pair] += freq
                    pair_to_pretokens.setdefault(pair, set()).add(new_tuple)

        pbar.update(1)

    pbar.close()
    return vocab, merges