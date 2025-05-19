#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

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
    s = f"saved {name} {msg if len(msg) < 70 else msg[:70] + ' ...'}"
    console.print(s, style="yellow")


def save_code(fn: str, code: CodeBlock) -> Path:
    full_path = make_fullpath(fn)
    console.print(f"saving code {full_path}", style="red")
    full_path.write_text("\n".join(code.lines), encoding="utf-8")
    return full_path


def save_and_execute_python(code: CodeBlock, timeout: int = 30):
    try:
        script_path = save_code("temp.py", code)

        result = subprocess.run([str(get_python()), script_path], cwd=script_path.parent, capture_output=True, text=True, timeout=timeout)

        if len(result.stdout) > 0:
            console.print(result.stdout, style="yellow")
        else:
            console.print(result.stderr, style="red")
        return result.stdout, result.stderr
    except Exception as e:
        console.print(f"ERROR: {e.__class__.__name__} {str(e)}", style="red")
        return None, str(e)


def save_and_execute_bash(code: CodeBlock):
    try:
        script_path = save_code("temp", code)

        os.chmod(script_path, 0o755)  # Set executable permissions
        result = subprocess.run(["bash", script_path], capture_output=True, text=True, timeout=5)

        if len(result.stdout) > 0:
            console.print(result.stdout, style="yellow")
        else:
            console.print(result.stderr, style="red")
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

        if len(result.stdout) > 0:
            console.print(result.stdout, style="yellow")
        else:
            console.print(result.stderr, style="red")
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
    output = None
    err = None
    msg = None
    xs = (f"{i + 1:02d} {s}" for i, s in enumerate(code.lines))
    block = f"```{code.language}\n{'\n'.join(xs)}\n```"
    console.print(Markdown(block), style="white")
    if code.language == "python":
        output, err = save_and_execute_python(code)
        if err:
            msg = err
        else:
            msg = output
    elif code.language == "bash":
        output, err = save_and_execute_bash(code)
        if err:
            msg = err
        else:
            msg = output
    elif code.language == "powershell":
        output, err = save_and_execute_powershell(code)
        if err:
            msg = err
        else:
            msg = output
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


def load_textfile(s: str) -> str | None:
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
        console.print(f"loaded file {fname} length {len(content)}", style="yellow")
        if fname.suffix in lmap:
            return f"\n## {fname.name}\n\n```{lmap[fname.suffix]}\n{content}\n```\n"
        return content
    except FileNotFoundError as e:
        console.print(f"{e.__class__.__name__}: {e}", style="red")
    return None


def print_block(lines: str | list[str]) -> None:
    xs = lines if isinstance(lines, list) else lines.splitlines()
    text = ""
    for i, s in enumerate(xs):
        text += f"{i + 1:02d} {s if s.endswith('\n') else s + '\n'}"
    console.print(text)


def extract_diff(diff: str) -> list[tuple[list[str], list[str]]]:
    """Extract all diffs from a string. Returns list of (search, replace) line tuples.

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
    lines = diff.splitlines()
    diffs = []
    current_diff = None

    for line in lines:
        if line.startswith("<<<"):
            # Start new diff block
            if current_diff is not None:
                raise ValueError("Unclosed diff block before new <<< marker")
            current_diff = {"search": [], "state": "search"}
        elif line.startswith("==="):
            # Transition to replace section
            if current_diff is None or current_diff["state"] != "search":
                raise ValueError("=== marker without preceding <<< or in wrong state")
            current_diff["state"] = "replace"
        elif line.startswith(">>>"):
            # Finalize current diff block
            if current_diff is None or current_diff["state"] != "replace":
                raise ValueError(">>> marker without preceding === or in wrong state")
            diffs.append((current_diff["search"], current_diff["replace"]))
            current_diff = None
        else:
            # Add line to current section
            if current_diff is not None:
                if current_diff["state"] == "search":
                    current_diff["search"].append(line)
                else:
                    if "replace" not in current_diff:
                        current_diff["replace"] = []
                    current_diff["replace"].append(line)

    if current_diff is not None:
        raise ValueError("Unterminated diff block at end of input")

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
    """Replace the search block by the replace block. preserve the indentation of the original"""

    def count_indentation(s: str) -> int:
        count = 0
        for char in s:
            if char in " \t":
                count += 1
            else:
                break
        return count

    # Find the matching block using find_match_lines
    match = find_block(lines, search)
    if match is None:
        raise ValueError(f"No matching block found. {search[0]}")
    start_idx, end_idx = match

    console.print(f"found matching block at {start_idx} to {end_idx} replacing with {len(replace)} lines", style="yellow")
    console.print(f"{search[0]} → {replace[0]}", style="yellow")
    # print_block(search)
    # print_block(replace)

    # Determine indentation from the first line of the matched block
    indentation = " " * count_indentation(lines[start_idx])

    # Handle the replacement
    if not replace:
        replacement = []
    else:
        # Calculate minimum indentation in replacement block
        min_indent = min((count_indentation(s) for s in replace if s.strip() != ""), default=0)
        replacement = []
        for repl in replace:
            repl_stripped = repl[min_indent:] if len(repl) >= min_indent else repl.lstrip()
            replacement.append(indentation + repl_stripped + "\n")

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
