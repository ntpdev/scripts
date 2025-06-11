#!/usr/bin/env python3
import ast
import re
from dataclasses import dataclass
from pathlib import Path

import chatutils as cu


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
    start_position: int = -1
    end_position: int = -1


class VirtualFile:
    def __init__(path: Path, read_only: bool = False):
        self.name: str = path.name
        self.path: Path = path
        self.lines: list[str] | None = None # Using Python 3.10 type hint
        self.modified: bool = False
        self.read_only: bool = read_only

    def _ensure_loaded(self):
        if self.lines is None:
            self.lines = self.path.read_text(encoding="utf-8").splitlines()
            console.print(f"VirtualFile.read_text {self.path}", style="yellow")

    def read_text(self) -> list[str]:
        """Reads underlying file if not already loaded."""
        self._ensure_loaded()
        return self.lines if self.lines is not None else []

    def write_text(self) -> str:
        """Writes the current content back to the underlying file if not read-only."""
        if self.read_only:
            return f"Cannot write: '{self.name}' is read-only."

        if self.lines is None:
            return f"Cannot write: Content of '{self.name}' is not loaded."

        try:
            self.path.write_text('\n'.join(self.lines) + '\n', encoding="utf-8") # Add newline back at the end of the file
            self.modified = False # Mark as not modified after successful write
            return f"Successfully wrote '{self.name}' to {self.path}."
        except Exception as e:
            return f"Error writing '{self.name}' to {self.path}: {e}"

    def edit(self, edits: list[EditItem]) -> str:
        """
        Applies a list of EditItem objects to the file lines.
        (Unimplemented)
        """
        self._ensure_loaded()
        if self.read_only:
             return f"Cannot edit: '{self.name}' is read-only."

        # Implementation for applying edits goes here
        # This would involve iterating through edits,
        # finding search patterns within self.lines (respecting start/end_position),
        # and replacing them with replace patterns.
        # After successful edits, set self.modified = True

        console.print(f"Edit function for '{self.name}' is not yet implemented.", style="info")
        return "Edit function not implemented." # Placeholder return value

    def read_file_content(self, line_number: int, show_line_numbers: bool = False, window_size: int = 50) -> list[str]:
        """
        Reads and returns a section of the file content around a given line number.
        Optionally shows line numbers.
        """
        self._ensure_loaded()

        total_lines = len(self.lines)
        if total_lines == 0:
            return [f"File '{self.name}' is empty."]

        # Adjust line_number to be 0-based index
        target_index = line_number - 1

        if not (0 <= target_index < total_lines):
            return [f"Error: Line number {line_number} is out of range (1-{total_lines}) for '{self.name}'."]

        # Calculate the start and end indices for the window
        start_index = max(0, target_index - window_size // 2)
        end_index = min(total_lines - 1, target_index + window_size // 2)

        output_lines: list[str] = []
        for i in range(start_index, end_index + 1):
            line_content = self.lines[i]
            if show_line_numbers:
                # Format line number with padding
                output_lines.append(f"{i + 1:>{len(str(total_lines))}}: {line_content}")
            else:
                output_lines.append(line_content)

        return output_lines


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

    def read_file(self, filename: str | Path, line_number: int, show_line_numbers: bool = False, window_size: int = 50) -> list[str]:
        """
        Reads and returns a section of a file's content around a given line number.
        """
        return self.get_file(filename).read_file_content(line_number, show_line_numbers, window_size)


    def edit_file(self, fname: str | Path, edits: list[EditItem]) -> str:
        """Finds a file by name and applies edits to it."""
        return self.get_file(fname).edit(edits)


    def save_all(self) -> dict[str, str]:
        """Saves all modified, non-read-only files."""
        save_results: dict[str, str] = {}
        console.print("\n[bold]Saving all modified files:[/bold]", style="info")
        for file_name, virtual_file in self.file_mapping.items():
            if virtual_file.modified and not virtual_file.read_only:
                save_results[file_name] = virtual_file.write_text()
            elif virtual_file.modified and virtual_file.read_only:
                save_results[file_name] = f"Skipped saving '{file_name}': Read-only file."
            else:
                save_results[file_name] = f"Skipped saving '{file_name}': Not modified."
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
            raise ValueError(f"No occurrences of search block found. Search pattern: {edit.search!r}")

    if count > 1:
        xs = [start for start, _ in matches]
        raise ValueError(f"Multiple ({count}) matches found at line numbers {xs}. Search pattern: {edit.search!r}")
        raise ValueError(f"No occurrences of search block found. Search pattern: {edit.search!r}")


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