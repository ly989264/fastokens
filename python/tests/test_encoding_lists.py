"""`Encoding` list properties build plain `list[int]` equal to their contents.

The getters take fast paths (a repeated value becomes one `[v] * n`; token ids
reuse shared `int` objects), so check every shape of input against the values
it was built from. These build encodings directly and need no network.
"""

from fastokens._native import Encoding


def _check(values, got):
    assert type(got) is list
    assert got == values
    assert all(type(v) is int for v in got)


def test_ids_round_trip_mixed_values():
    ids = [0, 1, 255, 256, 151_643, 7, 7, 3, 200_000]
    enc = Encoding(ids)
    _check(ids, enc.ids)
    _check([1] * len(ids), enc.attention_mask)
    _check([0] * len(ids), enc.type_ids)
    _check([0] * len(ids), enc.special_tokens_mask)


def test_repeated_and_empty_lists():
    _check([42] * 1000, Encoding([42] * 1000).ids)
    enc = Encoding([])
    _check([], enc.ids)
    _check([], enc.attention_mask)


def test_ids_beyond_the_interned_range():
    ids = [1, 2**21, 2**32 - 1, 5]
    _check(ids, Encoding(ids).ids)


def test_lists_are_independent_copies():
    enc = Encoding([10, 20, 30])
    first = enc.ids
    first.append(99)
    first[0] = -1
    _check([10, 20, 30], enc.ids)


def test_setters_replace_contents():
    enc = Encoding([1, 2, 3])
    enc.ids = [4, 5]
    enc.attention_mask = [1, 0]
    enc.type_ids = [0, 1]
    enc.special_tokens_mask = [1, 1]
    _check([4, 5], enc.ids)
    _check([1, 0], enc.attention_mask)
    _check([0, 1], enc.type_ids)
    _check([1, 1], enc.special_tokens_mask)


def test_padding_makes_masks_non_uniform():
    enc = Encoding([5, 6, 7])
    enc.pad(5, direction="left", pad_id=9, pad_type_id=1)
    _check([9, 9, 5, 6, 7], enc.ids)
    _check([0, 0, 1, 1, 1], enc.attention_mask)
    _check([1, 1, 0, 0, 0], enc.type_ids)
    enc.truncate(4, direction="right")
    _check([9, 9, 5, 6], enc.ids)
    _check([0, 0, 1, 1], enc.attention_mask)


def test_untracked_fields_accept_and_ignore_values():
    enc = Encoding([1, 2])
    enc.sequence_ids = [0, 0]
    enc.word_ids = [None, 1]
    enc.words = [None, 1]
    enc.set_sequence_id(1)
    merged = Encoding.merge([enc, Encoding([3])])
    _check([1, 2, 3], merged.ids)
