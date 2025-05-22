#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent, shorten

from rich.console import Console

console = Console()
THINK_PATTERN = re.compile("(.*?)<think>(.*?)</think>(.*)", re.DOTALL)
latex_to_unicode = {
    "approx": "≈",
    "dashv": "⊣",
    "div": "÷",
    "equiv": "≡",
    "exists": "∃",
    "forall": "∀",
    "geq": "≥",
    "iff": "⇔",
    "implies": "⇒",
    "in": "∈",
    "land": "∧",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "leq": "≤",
    "lor": "∨",
    "neg": "¬",
    "neq": "≠",
    "pm": "±",
    "rightarrow": "→",
    "sim": "∼",
    "sqrt": "√",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "therefore": "∴",
    "times": "×",
    "vdash": "⊢",
    "wedge": "∧",
}


@dataclass
class CodeBlock:
    language: str
    lines: list[str]


def print_block(lines: str | list[str], line_numbers: bool = False, style: str = "") -> None:
    xs = lines if isinstance(lines, list) else lines.splitlines()
    text = ""
    for i, s in enumerate(xs):
        if line_numbers:
            text += f"{i + 1:>3} {s if s.endswith('\n') else s + '\n'}"
        else:
            text += s if s.endswith("\n") else s + "\n"
    console.print(text, markup=False, style=style)


def make_fullpath(fn: str) -> Path:
    return Path.home() / "Documents" / "chats" / fn


def get_python() -> Path:
    """Returns the path to the Python executable in the virtual environment."""
    if "VIRTUAL_ENV" in os.environ:
        # Virtual environment is activated
        venv_path = Path(os.environ["VIRTUAL_ENV"])
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"
    # Fallback to the current Python interpreter
    console.print("no virtual env found. running default python environment", style="red")
    return Path(sys.executable)


def find_last_file(dir: Path) -> tuple[Path | None, int]:
    matching_files = [(x, int(x.stem[1:])) for x in dir.glob("z*.md") if x.stem[1:].isdigit()]

    if not matching_files:
        return None, 0

    matching_files.sort(key=lambda e: e[1], reverse=True)
    return matching_files[0]


def save_content(msg: str):
    last_path, last = find_last_file(Path.home() / "Documents" / "chats")
    next = last + 1
    name = f"z{next:02}.md"
    fn = make_fullpath(name)
    fn.write_text(msg, encoding="utf-8")
    console.print(f"saved {name} {shorten(msg, width=70, placeholder='...')}", style="yellow")


def save_code(fn: str, code: CodeBlock) -> Path:
    full_path = make_fullpath(fn)
    console.print(f"saving code {full_path}", style="yellow")
    full_path.write_text("\n".join(code.lines), encoding="utf-8")
    return full_path


def save_and_execute_python(code: CodeBlock, timeout: int = 30):
    try:
        script_path = save_code("temp.py", code)

        result = subprocess.run([str(get_python()), script_path], cwd=script_path.parent, capture_output=True, text=True, timeout=timeout)

        if result.stdout:
            print_block(result.stdout, line_numbers=True, style="yellow")
        if result.stderr:
            print_block(result.stderr, line_numbers=True, style="green")
    except Exception as e:
        console.print(f"ERROR: {e.__class__.__name__} {str(e)}", style="red")
        return None, str(e)


def save_and_execute_bash(code: CodeBlock):
    try:
        script_path = save_code("temp", code)

        os.chmod(script_path, 0o755)  # Set executable permissions
        result = subprocess.run(["bash", script_path], capture_output=True, text=True, timeout=5)

        if result.stdout:
            print_block(result.stdout, line_numbers=True, style="yellow")
        if result.stderr:
            print_block(result.stderr, line_numbers=True, style="green")
        return result.stdout, result.stderr
    except Exception as e:
        console.print(f"ERROR: {e.__class__.__name__} {str(e)}", style="red")
        return None, str(e)


def save_and_execute_powershell(code: CodeBlock):
    script_path = save_code("temp.ps1", code)
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.stdout:
            print_block(result.stdout, line_numbers=True, style="yellow")
        if result.stderr:
            print_block(result.stderr, line_numbers=True, style="green")
        return result.stdout, result.stderr
    except Exception as e:
        console.print(f"ERROR: {e.__class__.__name__} {str(e)}", style="red")
        return None, str(e)


def search_for_language(s: str) -> str:
    s_lower = s.lower()
    return next((lang for lang in ["python", "bash", "powershell"] if lang in s_lower), "")


def extract_code_block(contents: str, sep: str) -> CodeBlock | None:
    """extracts the first code block"""
    xs = contents.splitlines()
    inside = False
    code = None
    for x in xs:
        if x.strip().startswith(sep):
            inside = not inside
            if inside:
                code = CodeBlock(x.strip()[len(sep) :].lower(), [])
            else:
                if len(code.language) == 0:
                    code.language = search_for_language(contents)
                return code
        else:
            if inside:
                # infer language based on content
                if (len(code.language) == 0) and ("print(" in x or "import" in x):
                    code.language = "python"
                # check for line with name of language why?
                em = search_for_language(x)
                if len(em) > 0:
                    code.language = em
                else:
                    code.lines.append(x)

    return code


def execute_python_script(code: str) -> str:
    r = execute_script(CodeBlock("python", code.splitlines()))
    return r if r else "SUCCESS: script executed successfully but there was no output. include a print statement"


def execute_script(code: CodeBlock):
    print_block(code.lines, True)
    execution_functions = {
        "python": save_and_execute_python,
        "bash": save_and_execute_bash,
        "powershell": save_and_execute_powershell,
    }

    msg = ""
    if code.language in execution_functions:
        output, err = execution_functions[code.language](code)
        msg = err if err else output
    else:
        console.print("unrecognised code block found")
    return msg.strip()


def translate_latex(s: str) -> str:
    for k, v in latex_to_unicode.items():
        s = s.replace("\\" + k, v)
    return s


def input_multi_line() -> str:
    if (inp := input().strip()) != "{":
        return translate_latex(inp)
    lines = []
    while (line := input()) != "}":
        lines.append(line)
    return translate_latex("\n".join(lines))


def translate_thinking(s: str) -> str:
    match = THINK_PATTERN.search(s)
    if match:
        before, thought, after = match.groups()
        return f"{before}## thinking\n{thought}\n## answer\n{after}"
    return s


def load_textfile(s: str, line_numbers: bool = False) -> str | None:
    """loads a text file. if it is a code or data file wrap in markdown code block."""
    lmap = {
        ".py": "python",
        ".htm": "html",
        ".html": "html",
        ".java": "java",
        ".yaml": "yaml",
        ".json": "json",
        ".xml": "xml",
    }
    fname = make_fullpath(s)
    try:
        content = fname.read_text(encoding="utf-8")
        if line_numbers:
            content = "\n".join(f"[{i + 1:03d}] {s}" for i, s in enumerate(content.splitlines()))
        console.print(f"loaded file {fname} length {len(content)}", style="yellow")
        if fname.suffix in lmap:
            return f"\n## {fname.name}\n\n```{lmap[fname.suffix]}\n{content}\n```\n"
        return content
    except FileNotFoundError as e:
        console.print(f"{e.__class__.__name__}: {e}", style="red")
    return None


def run_linter(fn: Path) -> str:
    code = CodeBlock("powershell", [f"uvx ruff check --fix {fn}"])
    out, err = save_and_execute_powershell(code)
    return (
        dedent(f"""\
            ruff check --fix {fn.name}

            # output""") + "\n\n" + out)


def run_python_unittest(fn: Path, func_name: str | None = None) -> str:
    """given a python file, run the test with filename test_x in folder ./tests"""
    if not fn.is_file():
        raise ValueError(f"file {fn} does not exist")

    test_module = f"tests.test_{fn.stem}.Test_{func_name}" if func_name else f"tests.test_{fn.stem}"

    code = CodeBlock("powershell", [f"Set-Location -Path '{fn.parent}'", f"uv run python -m unittest -v {test_module}"])
    out, err = save_and_execute_powershell(code)
    failed_tests = "FAIL" in err

    # unittest sends the test output to stderr
    # print_block(out, True)
    # print_block(err, True)
    s = (
        dedent(f"""\
            running module test

            > python -m unittest {test_module}
        """)
        + "\n\n"
        + err
    )
    return failed_tests, s


def extract_diff(diff: str) -> list[tuple[list[str], list[str]]]:
    """Extract all diff blocks from a string. Returns list of (search, replace) line tuples.

    diff format for each block:
    <<<
    search_lines
    ===
    replace_lines
    >>>

    Args:
        diff: String containing 0 or more diff blocks

    Returns:
        List of tuples where each tuple contains (search_lines, replace_lines)
        for each diff block found in the input
    """

    def collect_lines(start_idx, end_marker, error_msg):
        collected = []
        idx=start_idx

        while idx < len(lines) and not lines[idx].startswith(end_marker):
            if lines[idx].startswith("+") or lines[idx].startswith("-"):
                raise ValueError("Invalid diff block. Found line starting with + or - in diff block. Do not use unified diff format\n" + lines[idx])
            collected.append(lines[idx])
            idx += 1

        if idx >= len(lines):
            raise ValueError(error_msg)

        return collected, idx

    lines = diff.splitlines()
    diffs = []
    i = 0

    while i < len(lines):
        _, i = collect_lines(i, "<<<", "No diff block found. Block must start with <<<")
        search_lines, i = collect_lines(i + 1, "===", "Invalid diff block. Found <<< without matching === marker")
        replace_lines, i = collect_lines(i + 1, ">>>", "Invalid diff block. Found === without matching >>> marker")
        diffs.append((search_lines, replace_lines))
        i += 1

    return diffs


def find_block(lines: list[str], search: list[str]) -> tuple[int, int] | None:
    """
    Find the start and end indexes of lines that match the search pattern exactly once.

    Args:
        lines: List of strings to search through
        search: List of strings to search for (pattern to match)

    Returns:
        Tuple of (start, end) indexes if the pattern appears exactly once, None otherwise
        The slice lines[start:end] will match the search pattern when ignoring whitespace
    """
    if not search:
        return None

    # Preprocess both lines and search patterns by stripping whitespace
    stripped_lines = [line.strip() for line in lines]
    stripped_search = [s.strip() for s in search]

    # Handle empty search list edge case
    if not stripped_search:
        return None

    search_len = len(stripped_search)
    lines_len = len(stripped_lines)
    match_indices = []

    for i in range(lines_len - search_len + 1):
        match = True
        for j in range(search_len):
            if stripped_lines[i + j] != stripped_search[j]:
                match = False
                break
        if match:
            match_indices.append((i, i + search_len))

    return match_indices[0] if len(match_indices) == 1 else None


def apply_diff_impl(lines: list[str], search: list[str], replace: list[str]) -> list[str]:
    """Replace the search block by the replace block.
    Preserve the indentation of the original and the relative indentation of the replacement."""

    def count_indentation(s: str) -> int:
        count = 0
        for char in s:
            if char in " \t":
                count += 1
            else:
                break
        return count

    match = find_block(lines, search)
    if match is None:
        raise ValueError(f"No matching block found. {search[0]}")
    start_idx, end_idx = match

    original_indent = count_indentation(lines[start_idx])
    replacement = []

    if replace:
        # Use the first line's indent of the replacement block as the reference
        first_line_indent = count_indentation(replace[0])
        for repl in replace:
            current_indent = count_indentation(repl)
            relative_indent = current_indent - first_line_indent
            adjusted_indent = max(original_indent + relative_indent, 0)
            stripped_line = repl.lstrip()
            s = " " * adjusted_indent + stripped_line + "\n"
            replacement.append(s)

    # Replace the matched block with the new content
    return lines[:start_idx] + replacement + lines[end_idx:]


def apply_diff(p: Path, diff: str) -> str:
    diffs = extract_diff(diff)
    console.print(f"found {len(diffs)} diffs", style="yellow")
    with p.open(encoding="utf-8") as f:
        lines = f.readlines()

    for diff in diffs:
        lines = apply_diff_impl(lines, diff[0], diff[1])
    return "".join(lines)


def apply_simple_diff_impl(original_lines: list[str], start_line: int, diff_str: str) -> list[str]:
    """
    Core implementation that applies a simple diff string to a list of lines.

    Args:
        original_lines: The original content as a list of lines (without newlines)
        start_line: The 1-indexed line number where the diff starts
        diff_str: The simple diff string to apply (format: " " match, "-" delete, "+" insert)

    Returns:
        The updated content as a list of lines (without newlines)

    Raises:
        ValueError: If the diff format is invalid or cannot be applied
    """
    updated_content_lines: list[str] = []
    original_ptr = 0  # Current 0-indexed position in original_lines

    # Convert 1-indexed start_line to 0-indexed
    diff_start_0idx = start_line - 1

    # Add lines before the diff start
    updated_content_lines.extend(original_lines[:diff_start_0idx])
    original_ptr = diff_start_0idx

    # Process the diff lines
    diff_lines = diff_str.splitlines()

    for diff_line in diff_lines:
        if not diff_line:  # Skip empty lines
            continue

        op = diff_line[0]  # Operation character: ' ', '+', or '-'
        line_content = diff_line[1:]  # The actual line content without the operation prefix

        if op == " ":  # Context line (unchanged line)
            if original_ptr >= len(original_lines) or original_lines[original_ptr] != line_content:
                raise ValueError(f"Diff application error: Context line mismatch at original line {original_ptr + 1}. Expected '{line_content}', Got '{original_lines[original_ptr] if original_ptr < len(original_lines) else 'EOF'}'. Diff line: '{diff_line}'")
            updated_content_lines.append(line_content)
            original_ptr += 1

        elif op == "-":  # Removed line
            if original_ptr >= len(original_lines) or original_lines[original_ptr] != line_content:
                breakpoint()
                raise ValueError(f"Diff application error: Removed line mismatch at original line {original_ptr + 1}. Expected '{line_content}', Got '{original_lines[original_ptr] if original_ptr < len(original_lines) else 'EOF'}'. Diff line: '{diff_line}'")
            original_ptr += 1  # Advance pointer, but do not add to updated_content_lines (it's removed)

        elif op == "+":  # Added line
            updated_content_lines.append(line_content)
            # original_ptr does not advance for added lines as they don't consume original content

        else:
            raise ValueError(f"Diff application error: Unexpected character '{op}' in diff line: '{diff_line}'")

    # Add any remaining lines from the original file after the diff
    updated_content_lines.extend(original_lines[original_ptr:])

    return updated_content_lines


def apply_simple_diff(fn: Path, start_line: int, diff_str: str) -> str:
    """
    Applies a simple diff string to the content of a file.

    Args:
        fn: Path to the file to modify
        start_line: The 1-indexed line number where the diff starts
        diff_str: The simple diff string (format: " " match, "-" delete, "+" insert)

    Returns:
        The updated file content as a string
    """
    # Read the original content, determine its trailing newline status
    # splitlines() removes newlines, so we must add them back if original had one.
    raw_original_content = fn.read_text(encoding="utf-8")
    original_content_lines = raw_original_content.splitlines()
    original_ends_with_newline = raw_original_content.endswith("\n")

    # Apply the diff to the content lines
    print_block(diff_str, True, style="yellow")
    updated_lines = apply_simple_diff_impl(original_content_lines, start_line, diff_str)

    # Reconstruct the string from the list of lines
    result = "\n".join(updated_lines)

    # Preserve the original file's trailing newline status
    if original_ends_with_newline:
        result += "\n"

    return result
