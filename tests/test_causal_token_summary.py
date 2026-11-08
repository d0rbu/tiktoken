"""Eligibility summaries preserve the original scalar encoder's complete IDs."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given, settings, strategies as st
from tiktoken import _tiktoken

from .test_prefix_character_offsets import model


def reference(core, text, allowed, eos):
    ids = core.encode(text, allowed)
    return len(ids), ids.count(eos), bool(ids) and ids[-1] == eos


@pytest.mark.parametrize("guarded", [True, False])
@pytest.mark.parametrize("policy", ["all", "partial", "none"])
@settings(max_examples=100, deadline=None)
@given(text=st.text(st.characters(exclude_categories=("Cs",)), max_size=150))
def test_summary_matches_original_scalar_ids(text, guarded, policy):
    core, _, allowed = model(guarded)
    allowed = (
        allowed if policy == "all" else frozenset({"<S>"}) if policy == "partial" else frozenset()
    )
    for value in (text, text + "<S>", "<S>" + text + "<S>long<T>éS"):
        for eos in (0, 97, 300, 301, 2**32 - 1):
            assert core._causal_token_summary(value, allowed, eos) == reference(
                core, value, allowed, eos
            )


@pytest.mark.parametrize("eos", [97, 300])
def test_aliases_count_actual_ids_including_ordinary_special_overlap(eos):
    ranks = {bytes([i]): i for i in range(256)}
    specials = {"<A>": eos, "<B>": eos}
    core = _tiktoken.CoreBPE(ranks, specials, r"[\s\S]+")
    for text in ("", "a", "a<A><B>", "<A>a<B>a", "é🙂<A>"):
        assert core._causal_token_summary(text, set(specials), eos) == reference(
            core, text, set(specials), eos
        )


@pytest.mark.parametrize(
    "text,allowed", [(None, set()), (b"abc", set()), ("\ud800", set()), ("abc", {1}), ("abc", None)]
)
def test_encoder_argument_failures_are_preserved(text, allowed):
    core, _, _ = model()
    with pytest.raises(Exception) as original:
        core.encode(text, allowed)
    with pytest.raises(type(original.value)):
        core._causal_token_summary(text, allowed, 300)


@pytest.mark.parametrize("eos", [-1, 2**32, "300", None])
def test_invalid_native_rank_rejects_without_wrapping(eos):
    core, _, allowed = model()
    with pytest.raises((TypeError, OverflowError)):
        core._causal_token_summary("abc", allowed, eos)


def test_shared_core_concurrent_summaries_preserve_ids():
    core, _, allowed = model()
    texts = ["aéS😀 abc<S>long  \n" * i for i in range(16)]
    expected = [reference(core, text, allowed, 300) for text in texts]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda text: core._causal_token_summary(text, allowed, 300), texts))
    assert actual == expected
