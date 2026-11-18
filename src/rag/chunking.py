import hashlib
import re


def split_text(
    text: str,
    max_chars: int = 1500,
    overlap: int = 150,
) -> list[str]:
    """Split text into overlapping chunks using preferred boundaries."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between 0 and max_chars.")

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            for separator in separators:
                boundary = text.rfind(separator, start, end)

                if boundary > start + max_chars // 2:
                    end = boundary + len(separator)
                    break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap
        start = max(next_start, start + 1)

    return chunks


def extract_markdown_sections(markdown: str) -> list[dict]:
    """Extract Markdown sections while preserving heading hierarchy."""

    sections = []
    heading_stack = []
    current_lines = []
    current_title = "Introduction"

    fence_char = None
    fence_length = 0

    def save_section():
        content = "\n".join(current_lines).strip()

        if content:
            sections.append({
                "section_title": current_title,
                "text": content,
            })

    for line in markdown.splitlines():
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)

        if fence_char is not None:
            current_lines.append(line)

            if (
                fence_match
                and fence_match.group(1)[0] == fence_char
                and len(fence_match.group(1)) >= fence_length
            ):
                fence_char = None
                fence_length = 0

            continue

        if fence_match:
            fence_char = fence_match.group(1)[0]
            fence_length = len(fence_match.group(1))
            current_lines.append(line)
            continue

        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

        if match:
            save_section()

            level = len(match.group(1))
            heading = match.group(2).strip()

            heading_stack[:] = heading_stack[:level - 1]
            heading_stack.append(heading)

            current_title = " > ".join(heading_stack)
            current_lines = []
        else:
            current_lines.append(line)

    save_section()

    return sections


def chunk_markdown(
    markdown: str,
    document_id: str,
    source_path: str,
    max_chars: int = 1500,
    overlap: int = 150,
) -> list[dict]:
    """Convert a Markdown document into metadata-rich chunks."""

    sections = extract_markdown_sections(markdown)

    chunks = []
    ordinal = 0

    for section in sections:
        section_title = section["section_title"]
        prefix = f"{section_title}\n\n"

        body_max_chars = max_chars - len(prefix)

        if body_max_chars <= 0:
            raise ValueError(
                "max_chars must be larger than the heading prefix."
            )

        text_parts = split_text(
            section["text"],
            max_chars=body_max_chars,
            overlap=overlap,
        )

        for part in text_parts:
            chunk_text = f"{prefix}{part}"

            content_hash = hashlib.sha256(
                chunk_text.encode("utf-8")
            ).hexdigest()

            identity = (
                f"{document_id}:{section_title}:{content_hash}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:24]

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": chunk_text,
                "source_path": source_path,
                "section_title": section_title,
                "ordinal": ordinal,
            })

            ordinal += 1

    return chunks
