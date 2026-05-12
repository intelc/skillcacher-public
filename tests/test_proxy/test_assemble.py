"""Tests for assemble_and_tokenize: rendering an Anthropic Messages
request into a prompt-token sequence + per-token span tags."""
from skillcacher.proxy.assemble import assemble_and_tokenize


# Use a small ungated tokenizer for tests.
TEST_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def test_simple_user_message_tokenizes():
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "hello world"}], "max_tokens": 1}
    tokens, tags = assemble_and_tokenize(body, TEST_TOKENIZER)
    assert len(tokens) > 0
    assert len(tokens) == len(tags)
    assert all(isinstance(t, int) and t >= 0 for t in tokens)


def test_system_block_tagged_as_system_static():
    body = {
        "model": "claude-x",
        "system": "you are helpful",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    tokens, tags = assemble_and_tokenize(body, TEST_TOKENIZER)
    assert "system_static" in tags


def test_skill_frontmatter_block_tagged_as_skill_body():
    skill_text = "---\nname: test\n---\n\n# Body\n\nThis is a skill body."
    body = {
        "model": "claude-x",
        "system": [{"type": "text", "text": skill_text}],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    tokens, tags = assemble_and_tokenize(body, TEST_TOKENIZER)
    assert "skill_body" in tags


def test_tool_def_tagged():
    body = {
        "model": "claude-x",
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    tokens, tags = assemble_and_tokenize(body, TEST_TOKENIZER)
    assert "tool_def" in tags
