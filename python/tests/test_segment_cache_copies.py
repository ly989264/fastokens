"""A `copy.deepcopy` of the compat shim shares the original's segment cache
(`FASTOKENS_SEGMENT_CACHE`): results through either stay identical to
HuggingFace `tokenizers`, with the copy's own truncation, and after the copy
extends its vocabulary. Synthetic byte-level BPE tokenizer, no network.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from fastokens._compat import _TokenizerShim
from fastokens._native import Tokenizer

hf_tokenizers = pytest.importorskip("tokenizers")


def tokenizer_json() -> str:
    alphabet = sorted(hf_tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alphabet)}
    merges = ["Ġ t", "h e", "Ġt he", "i n", "in g", "Ċ Ċ", "u s", "e r", "us er"]
    for m in merges:
        vocab[m.replace(" ", "")] = len(vocab)
    n = len(vocab)

    def added(id, content, special):
        return {"id": id, "content": content, "single_word": False, "lstrip": False,
                "rstrip": False, "normalized": False, "special": special}

    return json.dumps({
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [added(n, "<|im_start|>", True), added(n + 1, "<|im_end|>", True)],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    })


PIECES = ["the", " user", " said", "Hello", "WORLD", "中文", " 123", "\n", "  \n", " /", "{", "}\n", "();", "é", "😀"]


def chat_requests(seed: int, turns: int = 12) -> list[str]:
    rng = random.Random(seed)
    history, out = [], []
    for _ in range(turns):
        body = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 200)))
        history.append(f"<|im_start|>{body}<|im_end|>")
        if len(history) > 3 and rng.random() < 0.2:
            del history[rng.randrange(1, len(history))]
        out.append("".join(history))
    return out


@pytest.fixture
def hf():
    return hf_tokenizers.Tokenizer.from_str(tokenizer_json())


@pytest.fixture
def shim(monkeypatch):
    monkeypatch.setenv("FASTOKENS_SEGMENT_CACHE", "64")
    return _TokenizerShim(tokenizer_json())


def test_a_copy_matches_hf_on_what_the_original_cached(hf, shim):
    requests = chat_requests(1)
    for request in requests[:-1]:
        shim.encode(request, add_special_tokens=False)
    clone = copy.deepcopy(shim)
    for request in requests:
        want = hf.encode(request, add_special_tokens=False).ids
        assert clone.encode(request, add_special_tokens=False).ids == want
        assert shim.encode(request, add_special_tokens=False).ids == want


def test_a_copy_applies_its_own_truncation(hf, shim):
    request = chat_requests(2)[-1]
    want = hf.encode(request, add_special_tokens=False).ids
    assert shim.encode(request, add_special_tokens=False).ids == want
    clone = copy.deepcopy(shim)
    clone.enable_truncation(max_length=7)
    assert clone.encode(request, add_special_tokens=False).ids == want[:7]
    assert shim.encode(request, add_special_tokens=False).ids == want
    assert shim.truncation is None


def test_a_copy_that_adds_tokens_stays_correct_and_so_does_the_original(hf, shim):
    body = "say hi to <|new|> now and <|new|> again"
    request = f"<|im_start|>{body}<|im_end|>{body}<|im_end|>"
    assert shim.encode(request, add_special_tokens=False).ids == hf.encode(request, add_special_tokens=False).ids
    clone = copy.deepcopy(shim)
    hf_clone = hf_tokenizers.Tokenizer.from_str(tokenizer_json())
    token = hf_tokenizers.AddedToken("<|new|>", special=True)
    hf_clone.add_tokens([token])
    clone.add_special_tokens([token])
    for _ in range(2):
        assert clone.encode(request, add_special_tokens=False).ids == hf_clone.encode(request, add_special_tokens=False).ids
        assert shim.encode(request, add_special_tokens=False).ids == hf.encode(request, add_special_tokens=False).ids


def test_without_the_env_var_nothing_changes(hf, monkeypatch):
    monkeypatch.delenv("FASTOKENS_SEGMENT_CACHE", raising=False)
    shim = _TokenizerShim(tokenizer_json())
    clone = copy.deepcopy(shim)
    request = chat_requests(3)[-1]
    want = hf.encode(request, add_special_tokens=False).ids
    assert clone.encode(request, add_special_tokens=False).ids == want
    # Through the binding: sharing with itself is harmless.
    tok = Tokenizer.from_json_str(tokenizer_json())
    tok.share_segment_cache_from(tok)
    assert tok.encode(request).ids == want
