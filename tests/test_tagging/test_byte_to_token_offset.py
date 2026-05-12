"""byte-range → token-range conversion via tokenizer offset_mapping.

Critical for joining span tags (byte-based) to Controller.Lookup
inputs (token-based)."""
from skillcacher.proxy.assemble import byte_range_to_token_range


TEST_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def test_full_text_round_trips():
    text = "hello world this is a test"
    tok_start, tok_end = byte_range_to_token_range(text, 0, len(text.encode()), TEST_TOKENIZER)
    assert tok_start == 0
    # tok_end equals total token count for this slice
    from transformers import AutoTokenizer
    tot = len(AutoTokenizer.from_pretrained(TEST_TOKENIZER).encode(text, add_special_tokens=False))
    assert tok_end == tot


def test_substring_range_maps_to_subset():
    text = "alpha beta gamma delta"
    # bytes covering 'beta gamma' = offsets 6..16 (exclusive)
    tok_start, tok_end = byte_range_to_token_range(text, 6, 16, TEST_TOKENIZER)
    # Should be at least one token, fewer than the whole encoding.
    from transformers import AutoTokenizer
    full = len(AutoTokenizer.from_pretrained(TEST_TOKENIZER).encode(text, add_special_tokens=False))
    assert 0 < (tok_end - tok_start) < full


def test_empty_range_returns_empty():
    text = "hello"
    tok_start, tok_end = byte_range_to_token_range(text, 2, 2, TEST_TOKENIZER)
    assert tok_start == tok_end


def test_range_at_token_boundary_is_exact():
    """A range that starts/ends exactly on token boundaries should map cleanly."""
    text = "the quick brown fox"
    # Tokenize once to find a real token boundary
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEST_TOKENIZER)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    # Take the byte range covering tokens 1..3
    if len(enc.offset_mapping) >= 3:
        b_start = enc.offset_mapping[1][0]
        b_end = enc.offset_mapping[2][1]
        ts, te = byte_range_to_token_range(text, b_start, b_end, TEST_TOKENIZER)
        assert ts == 1
        assert te == 3
