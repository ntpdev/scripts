#!/usr/bin/env python3
import unittest
from pathlib import Path

from pydantic import BaseModel

import chatutils as cu


class DiffItem(BaseModel):
    start_position: int
    diff: str


def apply_diff(filename: Path, diffs: list[DiffItem]):
    lines = filename.read_text(encoding="utf-8").splitlines()
    for d in diffs:
        lines = apply_diff_impl(lines, d.start_position, [s for s in d.diff.splitlines() if s != "@@"])
    filename.write_text("\n".join(lines), encoding="utf-8")
    return f"success: applied {len(diffs)} diffs to {filename.name}"


def apply_diff_impl(lines: list[str], start_position: int, diff: list[str]) -> list[str]:
    """
    Apply a unified diff to a list of lines.

    Args:
        lines: Original content as a list of strings
        start_position: Line number (1-indexed) where the diff should start
        diff: List of diff lines with prefixes ' ' (unchanged), '-' (delete), '+' (insert)

    Returns:
        Updated list of lines after applying the diff
    """
    # cu.print_block(diff, True)
    n = len(lines)
    start_position = max(1, min(start_position, n + 1))
    start_idx = start_position - 1

    # Validate diff format - all lines should begin with ' ', '+', or '-'
    for i, line in enumerate(diff):
        if line.strip() and not line.startswith((" ", "+", "-")):
            raise ValueError(f"Invalid diff format on line {i + 1}:{line} line must begin with ' ', '+', or '-'")

    # Find the first non-blank diff line with ' ' or '-'
    first_match_line = None
    first_match_content = None
    for d in diff:
        if d[0] in (" ", "-") and d[1:].strip() != "":
            first_match_line = d
            first_match_content = d[1:]
            break

    if first_match_line is None:
        raise ValueError("Diff does not contain any non-blank ' ' or '-' lines for synchronization")

    # Search for a matching line in the source within start_idx +/- 10
    window_start = max(0, start_idx - 10)
    window_end = min(n, start_idx + 10 + 1)
    found = False
    for idx in range(window_start, window_end):
        if lines[idx] == first_match_content:
            actual_start = idx
            found = True
            break

    if not found:
        raise ValueError(f"Could not find a matching line for diff line '{first_match_content.strip()}' in source lines {window_start + 1} to {window_end}")

    # Apply the diff
    result = lines[:actual_start]  # Lines before the diff
    i = actual_start
    j = 0

    while j < len(diff) and i < n:
        action = diff[j][0]
        diff_line = diff[j][1:]
        source = lines[i]

        if action == " ":  # Unchanged line
            if source != diff_line:
                raise ValueError(f"Mismatch at unchanged line {i}: expected '{diff_line}', got '{source}'")
            result.append(source)
            i += 1
            j += 1

        elif action == "-":  # Delete line
            if source != diff_line:
                raise ValueError(f"Mismatch at delete line {i}: expected '{diff_line}', got '{source}'")
            i += 1
            j += 1

        elif action == "+":  # Insert line
            result.append(diff_line)
            j += 1

        else:
            raise ValueError(f"Invalid diff action: {action}")

    # Check if we've processed all diff lines
    if j < len(diff):
        raise ValueError(f"Not all diff lines were applied. Remaining: {len(diff) - j}")

    # Add remaining lines after the diff
    result.extend(lines[i:])

    return result


if __name__ == "__main__":
    unittest.main()