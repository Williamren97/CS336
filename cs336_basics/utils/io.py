from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import torch

# Regex pattern used by the GPT-2 pre-tokenizer.
GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache()
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Mapping between every possible byte (0..255) and a printable unicode character.
    Taken from the GPT-2 code; used to make merges/vocab human-inspectable.
    """

    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def get_tokenizer_from_vocab_merges_path(vocab_path: str | os.PathLike, merges_path: str | os.PathLike):
    """
    Load a GPT-2 vocab.json + merges.txt and return them as:
    - vocab: dict[int, bytes]
    - merges: list[tuple[bytes, bytes]]

    Note: we return raw bytes (not the GPT-2 byte-unicode remapping) since the tests
    also operate on bytes.
    """

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)

    gpt2_bpe_merges: list[tuple[str, str]] = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return vocab, merges


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    # Load onto CPU so tests are deterministic and don't require CUDA/MPS.
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def save_vocab_and_merges(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
):
    """
    Helper for debugging / serialization (not used by unit tests).
    """

    byte_to_unicode = gpt2_bytes_to_unicode()
    reversed_vocab = {"".join(byte_to_unicode[b] for b in bytes_token): k for k, bytes_token in vocab.items()}
    reversed_merges = [
        " ".join(
            [
                "".join(byte_to_unicode[b] for b in merge[0]),
                "".join(byte_to_unicode[b] for b in merge[1]),
            ]
        )
        for merge in merges
    ]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(reversed_vocab, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in reversed_merges:
            f.write(merge + "\n")

