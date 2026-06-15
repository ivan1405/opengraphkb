import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    return re.sub(r"```[a-zA-Z]*\n?", "", text).strip()


def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length, adding a suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form."""
    return unicodedata.normalize("NFC", text)
