#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import chatutils as cu


@dataclass
class EditItem:
    search: list[str]
    replace: list[str]
    start_position: int = -1
    end_position: int = -1


def find_block(lines: list[str], search: list[str]) -> list[tuple[int, int]]:
    """
    Find all occurrences of a block of lines that match the search pattern ignoring whitespace.
    Returns list of (start, end) index tuples.
    """
    if not search:
        return []

    stripped = [ln.strip() for ln in lines]
    pattern = [s.strip() for s in search]
    if not pattern:
        return []

    n, m = len(stripped), len(pattern)
    return [(i, i + m) for i in range(n - m + 1) if stripped[i : i + m] == pattern]


def update_edit_position(source: list[str], edit: EditItem) -> None:
    """
    Locate the unique occurrence of edit.search in source, and updates the EditItem
    with start_position and end_position.
    """
    matches = find_block(source, edit.search)
    if (count := len(matches)) == 0:
        raise ValueError(f"No occurrences of search block found. Search pattern: {edit.search!r}")

    if count > 1:
        xs = [start for start, _ in matches]
        raise ValueError(f"Multiple ({count}) matches found at line numbers {xs}. Search pattern: {edit.search!r}")

    edit.start_position, edit.end_position = matches[0]


def edit_file_impl(source: list[str], edit: EditItem) -> list[str]:
    """Replace the search block by the replace block.
    Preserve the indentation of the original and the relative indentation of the replacement."""

    def count_indentation(s: str) -> int:
        return len(s) - len(s.lstrip())

    cu.print_block(edit.search, True)
    cu.print_block(edit.replace, True)
    update_edit_position(source, edit)

    original_indent = count_indentation(source[edit.start_position])
    replacement = []

    if edit.replace:
        # Use the first line's indent of the replacement block as the reference
        first_line_indent = count_indentation(edit.replace[0])
        for repl in edit.replace:
            current_indent = count_indentation(repl)
            relative_indent = current_indent - first_line_indent
            adjusted_indent = max(original_indent + relative_indent, 0)
            stripped_line = repl.lstrip()
            s = " " * adjusted_indent + stripped_line
            replacement.append(s)

    # Replace the matched block with the new content
    return source[: edit.start_position] + replacement + source[edit.end_position :]


def edit_file(filename: Path, edits: list[EditItem]) -> str:
    lines = filename.read_text(encoding="utf-8").splitlines()
    for e in edits:
        lines = edit_file_impl(lines, e)
    filename.write_text("\n".join(lines), encoding="utf-8")
    xs = [e.start_position + 1 for e in edits]
    return f"success: applied {len(edits)} edits at line numbers {xs} to {filename.name}"


if __name__ == "__main__":
    # Example Usage

    # Define the initial source content using dedent
    source_content = dedent("""\
        def my_function():
            # This is a function
            print("Hello, world!")
            # End of the function
        pass
        """)

    # Convert the source content string into a list of lines
    source_lines = source_content.splitlines(keepends=True)

    # Define the search pattern using dedent
    search_pattern = dedent("""\
        # This is a function
        print("Hello, world!")
        # End of the function
        """).splitlines()  # Split search lines without keeping ends for find_block

    # Define the replacement content using dedent
    replace_content = dedent("""\
        # This block has been replaced
        print("New message!")
        """).splitlines()  # Split replace lines without keeping ends

    # Create an EditItem
    edit_item = EditItem(search=search_pattern, replace=replace_content)

    # Perform the edit
    try:
        edited_lines = edit_file_impl(source_lines, edit_item)

        # Print the original and edited content
        cu.print_block(source_lines, True)
        cu.print_block(search_pattern, True)
        cu.print_block(replace_content, True)
        cu.print_block(edited_lines, True)
    except ValueError as e:
        print(f"Error during editing: {e}")

    # Example of deleting a block (empty replace list)
    print("\n--- Example of Deletion ---")
    source_for_deletion = dedent("""\
        Line 1
        Block to delete - line 1
        Block to delete - line 2
        Line 4
        """).splitlines(keepends=True)

    search_to_delete = dedent("""\
        Block to delete - line 1
        Block to delete - line 2
        """).splitlines()

    delete_item = EditItem(search=search_to_delete, replace=[])  # Empty replace list

    try:
        deleted_lines = edit_file_impl(source_for_deletion, delete_item)
        cu.print_block(source_for_deletion, True)
        cu.print_block(search_to_delete, True)
        cu.print_block(deleted_lines, True)
    except ValueError as e:
        print(f"Error during deletion: {e}")