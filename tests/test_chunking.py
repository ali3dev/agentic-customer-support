from src.rag.chunking import (
    chunk_markdown,
    extract_markdown_sections,
    split_text,
)


def test_extract_markdown_heading_hierarchy():
    markdown = """# Returns
## Refunds
Refunds take 5 days.
"""

    sections = extract_markdown_sections(markdown)

    assert len(sections) == 1
    assert sections[0]["section_title"] == "Returns > Refunds"
    assert "Refunds take 5 days." in sections[0]["text"]


def test_extract_content_before_first_heading():
    sections = extract_markdown_sections(
        "Introduction text.\n\n# Policy\nPolicy text."
    )

    assert sections[0]["section_title"] == "Introduction"
    assert sections[1]["section_title"] == "Policy"


def test_empty_markdown_returns_no_chunks():
    assert chunk_markdown("", "doc-1", "test.md") == []


def test_split_text_respects_max_chars():
    text = "A" * 1000

    chunks = split_text(text, max_chars=200, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_id_stays_stable_when_earlier_section_is_added():
    original = """# Refunds
Refunds take 5 business days.
"""

    updated = """# Introduction
Welcome to our support policy.

# Refunds
Refunds take 5 business days.
"""

    first = chunk_markdown(original, "policy-1", "policy.md")
    second = chunk_markdown(updated, "policy-1", "policy.md")

    original_id = first[0]["chunk_id"]
    updated_refund = next(
        chunk for chunk in second
        if chunk["section_title"] == "Refunds"
    )

    assert original_id == updated_refund["chunk_id"]


def test_headings_inside_code_blocks_are_not_sections():
    markdown = """# Installation

```python
# This is a code comment
print("hello")
```

## Configuration
Set your API key.
"""

    sections = extract_markdown_sections(markdown)

    assert len(sections) == 2
    assert sections[0]["section_title"] == "Installation"
    assert "# This is a code comment" in sections[0]["text"]
    assert sections[1]["section_title"] == "Installation > Configuration"


def test_final_chunk_includes_heading_within_max_chars():
    markdown = """# Returns and Refund Policy

Customers can request a return within 30 days.
"""

    chunks = chunk_markdown(
        markdown,
        "policy-1",
        "returns.md",
        max_chars=100,
        overlap=10,
    )

    assert all(len(chunk["text"]) <= 100 for chunk in chunks)
