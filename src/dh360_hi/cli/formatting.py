"""Terminal output and table formatting utilities for the CLI."""

from __future__ import annotations

import json
import sys
from typing import Any


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format data into a clean, aligned ASCII table.

    Args:
        headers: List of column header names.
        rows: List of row records (each a list matching headers length).

    Returns:
        Formatted ASCII table string.
    """
    if not headers:
        return ""

    str_rows = [[str(cell) if cell is not None else "-" for cell in row] for row in rows]
    col_widths = [len(h) for h in headers]

    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(cell))

    # Build header line
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    # Build data lines
    row_lines = [
        " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) for row in str_rows
    ]

    return "\n".join([header_line, separator_line] + row_lines)


def format_key_value(pairs: list[tuple[str, Any]], title: str | None = None) -> str:
    """Format key-value pairs with clean padding.

    Args:
        pairs: List of (Key, Value) tuples.
        title: Optional section header.

    Returns:
        Formatted string.
    """
    if not pairs:
        return ""

    max_key_len = max(len(k) for k, _ in pairs)
    lines: list[str] = []
    if title:
        lines.append(f"=== {title} ===")

    for k, v in pairs:
        val_str = str(v) if v is not None else "-"
        lines.append(f"  {k.ljust(max_key_len)} : {val_str}")

    return "\n".join(lines)


def format_json(data: Any) -> str:
    """Serialize data to formatted JSON string."""
    return json.dumps(data, indent=2, default=str)


def print_error(message: str, code: str | None = None) -> None:
    """Print an error message to stderr without tracebacks."""
    if code:
        sys.stderr.write(f"Error [{code}]: {message}\n")
    else:
        sys.stderr.write(f"Error: {message}\n")
