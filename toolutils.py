#!/usr/bin/env python3
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

import chatutils as cu

console = Console()


@dataclass
class ErrorLocation:
    file_path: Path
    line_number: int
    function_name: str
    source: list[str] | None = None

    def get_source(self) -> list[str]:
        if self.source is None:
            self.source = get_source_code(self.file_path, self.function_name)
        return self.source


@dataclass
class EditItem:
    search: list[str]
    replace: list[str]
    filename: str = ""
    start_position: int = -1
    end_position: int = -1

    def as_diff_block(self) -> str:
        """
        Convert the EditItem back to a diff block string.
        """
        lines = []
        lines.append(f"```\n--- search {self.filename}")
        lines.extend(self.search)
        lines.append("--- replace")
        lines.extend(self.replace)
        lines.append("--- end\n```\n")
        return "\n".join(lines)


class VirtualFile:
    def __init__(self, path: Path | None = None, name: str | None = None, content: str | None = None, read_only: bool = False) -> None:
        if name and content:
            self.name = name
            self.path = None
            self.lines = content.splitlines()
            self.read_only = True
        elif path:
            self.name = path.name
            self.path = path
            self.lines = []
            self.read_only = read_only
        else:
            raise ValueError("Either 'path' or both 'name' and 'content' must be provided")
        self.modified = False

    def _ensure_loaded(self) -> None:
        if self.path and not self.lines:
            self.lines = self.path.read_text(encoding="utf-8").splitlines()
            console.print(f"VirtualFile.read_text {self.path} lines {len(self.lines)}", style="yellow")

    def read_text(self, line_number: int = 1, window_size: int = 99, show_line_numbers: bool = False) -> list[str]:
        """
        Returns a subset of lines centered around a given line number. Line numbers are a 1-based index.
        """
        self._ensure_loaded()
        n = len(self.lines)
        if window_size < 1:
            window_size = n

        # If input has window_size or fewer lines, return all lines
        if n <= window_size:
            start_idx = 0
            result = self.lines[:]
        else:
            # Convert 1-based line_number to 0-based index
            center_idx = line_number - 1

            # Try to center around the target line
            start_idx = center_idx - window_size // 2
            end_idx = start_idx + window_size

            if start_idx < 0:
                start_idx = 0
                end_idx = window_size
            elif end_idx > n:
                end_idx = n
                start_idx = n - window_size

            result = self.lines[start_idx:end_idx]

        if show_line_numbers:
            start_line = start_idx + 1
            width = len(str(n))
            result = [f"{start_line + i:>{width}}: {line}" for i, line in enumerate(result)]

        return result

    def write_text(self) -> str:
        """Writes the current content back to the underlying file if not read-only."""
        if self.read_only or not self.modified:
            return f"SUCCESS: file was not modified '{self.name}'"

        try:
            self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")  # Add newline back at the end of the file
            self.modified = False
            return f"SUCCESS: saved '{self.name}' to {self.path}."
        except Exception as e:
            return f"ERROR: {e} writing '{self.name}' to {self.path}"

    def edit(self, edits: list[EditItem]) -> str:
        """
        Applies patches to the file. returns SUCCESS or ERROR.
        """
        try:
            self._ensure_loaded()

            xs = self.lines.copy()
            for e in edits:
                xs = edit_file_impl(xs, e)

            self.lines = xs
            self.modified = True
            return f"SUCCESS: applied {len(edits)} patches to '{self.name}'."
        except Exception as e:
            return f"ERROR: failed to apply patch to '{self.name}'. No changes have been made to the file.\n{e}"

    def as_markdown(self) -> str:
        x = "\n".join(self.read_text(window_size=0))
        return f"## {self.name}\n\n```\n{x}\n```\n\n"

    def __repr__(self):
        return f"VirtualFile(name='{self.name}', path='{self.path}', modified={self.modified}, read_only={self.read_only}, loaded={self.lines is not None})"


class VirtualFileSystem:
    def __init__(self):
        self.file_mapping: dict[str, VirtualFile] = {}

    def create_mapping(self, fn: Path) -> bool:
        """Checks if fn exists and creates a VirtualFile. Returns True if successful, False otherwise."""
        if fn.is_file():
            self.file_mapping[fn.name] = VirtualFile(path=fn)
            return True
        return False

    def create_unmapped(self, fname: str, contents: str) -> bool:
        self.file_mapping[fname] = VirtualFile(name=fname, content=contents)
        return True

    def get_file(self, fname: str | Path) -> VirtualFile:
        """
        Retrieves a VirtualFile by name or path.
        Raises ValueError if no mapping is found.
        """
        p = fname if isinstance(fname, Path) else Path(fname)

        vf = self.file_mapping.get(p.name)

        if vf is None:
            err = f"Error: File not found {fname}"
            console.print(err, style="red")
            raise ValueError(err)

        return vf

    def read_text(self, fn: str | Path, line_number: int = 1, window_size: int = 50, show_line_numbers: bool = False) -> list[str]:
        """
        Reads and returns a section of a file's content around a given line number.
        """
        return self.get_file(fn).read_text(line_number=line_number, window_size=window_size, show_line_numbers=show_line_numbers)

    def edit(self, fn: str | Path, edits: list[EditItem]) -> str:
        """Finds a file by name and applies edits to it."""
        return self.get_file(fn).edit(edits)

    def _apply_edits(self, fn: str | Path, edits: list[EditItem]) -> str:
        return self.get_file(fn).edit(edits)

    def apply_edits(self, blocks: list[cu.CodeBlock]) -> str:
        """extract markdown blocks, extract edit instructions and apply to files"""
        if edits := parse_edits([s for b in blocks for s in b.lines]):
            if any(not self.file_mapping.get(e.filename) for e in edits):
                return "ERROR: patch does not contain a filename. No changes applied."
            grouped = defaultdict(list)
            for e in edits:
                grouped[e.filename].append(e)
            msgs = []
            for filename, file_patches in grouped.items():
                msgs.append(self._apply_edits(filename, file_patches))
            return "\n".join(msgs)
        return "ERROR: no edit blocks found. Check blocks are marked as diff"

    def save_all(self) -> dict[str, str]:
        """Saves all modified, non-read-only files."""
        save_results: dict[str, str] = {}
        console.print("\nSaving all modified files:", style="yellow")
        for file_name, virtual_file in self.file_mapping.items():
            if virtual_file.modified and not virtual_file.read_only:
                save_results[file_name] = virtual_file.write_text()
            elif virtual_file.modified and virtual_file.read_only:
                save_results[file_name] = f"File '{file_name}' not saved as it is read-only."
            else:
                save_results[file_name] = f"File '{file_name}' not saved as it has not been modified."
        return save_results

    def __repr__(self):
        return f"VirtualFileSystem(files={list(self.file_mapping.keys())})"


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
        raise ValueError(f"No occurrences of search block found. Failed patch\n\n{edit.as_diff_block()}")

    if count > 1:
        xs = [start for start, _ in matches]
        raise ValueError(f"Multiple ({count}) matches found at line numbers {xs}. Add more context lines to search block. Failed patch\n\n{edit.as_diff_block()}")

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
    """apply edits ignoring filename in edits"""
    lines = filename.read_text(encoding="utf-8").splitlines()
    for e in edits:
        lines = edit_file_impl(lines, e)
    filename.write_text("\n".join(lines), encoding="utf-8")
    xs = [e.start_position + 1 for e in edits]
    return f"success: applied {len(edits)} edits at line numbers {xs} to {filename.name}"


def parse_edits(input: str | list[str]) -> list[EditItem]:
    """
    Parse a multi-line string into a list of EditItems
    """

    def collect_lines_until(start_idx: int, stop_prefix: str) -> tuple[list[str], int]:
        collected = []
        i = start_idx

        while i < n_lines and not lines[i].strip().startswith(stop_prefix):
            collected.append(lines[i])
            i += 1

        # Skip the stop line if we found it
        if i < n_lines and lines[i].strip().startswith(stop_prefix):
            i += 1

        return collected, i

    lines = input.splitlines() if isinstance(input, str) else input
    n_lines = len(lines)
    edit_items = []
    i = 0
    filename = ""

    while i < n_lines:
        # Look for start of edit block
        s = lines[i].strip()
        if s.startswith("--- search"):
            parts = s.split()
            if len(parts) > 2:
                filename = filename or parts[2]
            i += 1
            search_lines, i = collect_lines_until(i, "--- replace")
            replace_lines, i = collect_lines_until(i, "--- end")

            # Create EditItem with the collected lines
            edit_items.append(EditItem(search=search_lines, replace=replace_lines, filename=filename))
        else:
            i += 1

    return edit_items


def expand_to_blank_lines(lines: list[str], start: int, end: int) -> list[str]:
    """
    Expands the slice [start:end] to include contiguous lines up to the
    nearest blank line before and after.
    """
    # Expand upwards
    while start > 0 and lines[start - 1].strip() != "":
        start -= 1
    # Expand downwards
    while end < len(lines) and lines[end].strip() != "":
        end += 1
    return lines[start:end]


def get_source_code(
    filename: Path,
    name: str,
    kind: str = "function",  # "function" or "class"
) -> list[str]:
    """
    Extracts the source code of a function or class from a Python file using AST,
    including contiguous comments and decorators before and after, up to blank lines.

    Args:
        filename: Path to the Python source file.
        name: Name of the function or class to extract.
        kind: "function" or "class".

    Returns:
        The source code as a list of strings (lines), or an empty list if not found.
    """
    source = filename.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source, str(filename))

    node_type = ast.FunctionDef if kind == "function" else ast.ClassDef

    for node in ast.walk(tree):
        if isinstance(node, node_type) and node.name == name:
            start = node.lineno - 1  # line numbers are 1-based
            end = node.end_lineno
            return expand_to_blank_lines(source_lines, start, end)
    return []


ERROR_LOCATION_REGEX = re.compile(r'^\s*File "(.*?)", line (\d+), in (.*)$')


def extract_error_locations(message: str) -> list[ErrorLocation]:
    """
    Extracts error location information (file, line, function) from a traceback message.

    Args:
        message: The error traceback message as a string.

    Returns:
        A list of ErrorLocation objects.
    """
    error_locations: list[ErrorLocation] = []

    for line in message.splitlines():
        match = re.match(ERROR_LOCATION_REGEX, line)
        if match:
            file_path = Path(match.group(1))
            line_number = int(match.group(2))
            function_name = match.group(3)
            error_locations.append(ErrorLocation(file_path, line_number, function_name))

    return error_locations


if __name__ == "__main__":
    print("hello")
