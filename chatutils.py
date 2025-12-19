#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

# on Linux clipbaord support with desktop requires xclip
# sudo apt install xclip

IS_LINUX = sys.platform.startswith("linux")
THINK_PATTERN = re.compile("(.*?)<think>(.*?)</think>(.*)", re.DOTALL)
LATEX_TO_UNICODE = {
    "alpha": "α",
    "approx": "≈",
    "beta": "β",
    "dashv": "⊣",
    "deg": "°",
    "div": "÷",
    "equiv": "≡",
    "exists": "∃",
    "forall": "∀",
    "gamma": "γ",
    "geq": "≥",
    "iff": "⇔",
    "implies": "⇒",
    "in": "∈",
    "infty": "∞",
    "land": "∧",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "leq": "≤",
    "lor": "∨",
    "neg": "¬",
    "neq": "≠",
    "pi": "π",
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

console = Console()

class ChatInput:
    def __init__(self):
        self.history = InMemoryHistory()
        self.completer = self._create_completer()
        self.key_bindings = self._create_key_bindings()
        self.session = PromptSession(
            completer=self.completer,
            complete_while_typing=Condition(lambda: self._should_complete()),
            history=self.history,
            clipboard=PyperclipClipboard(),
            multiline=True,
            key_bindings=self.key_bindings,
        )

    def _should_complete(self):
        """Only show completion when current word starts with %"""
        buffer = get_app().current_buffer
        word = buffer.document.get_word_before_cursor(WORD=True)
        return word is not None and word.startswith("%")
        cursor_pos = app.current_buffer.cursor_position
        text = app.current_buffer.text

        # Find the start of the current word
        word_start = cursor_pos
        while word_start > 0 and text[word_start - 1] not in " \n\t":
            word_start -= 1

        # Check if current word starts with %
        current_word = text[word_start:cursor_pos]
        return current_word.startswith("%")

    def _create_key_bindings(self):
        """Create key bindings for LaTeX translation"""
        kb = KeyBindings()

        @kb.add("c-j")  # Ctrl+Enter to submit
        def handle_submit(event):
            event.current_buffer.validate_and_handle()

        @kb.add("$")
        def handle_dollar(event):
            buffer = event.current_buffer
            text = buffer.text
            cursor_pos = buffer.cursor_position

            # Insert the $ character first
            buffer.insert_text("$")

            # Look for a previous $ to form a LaTeX expression
            dollar_start = -1
            for i in range(cursor_pos - 1, -1, -1):
                if text[i] == "$":
                    dollar_start = i
                    break

            if dollar_start != -1:
                # Extract the content between the dollars
                latex_content = text[dollar_start + 1 : cursor_pos]

                # Check if it matches any LaTeX symbol
                if latex_content in LATEX_TO_UNICODE:
                    # Replace the entire $content$ with the unicode symbol
                    buffer.cursor_position = dollar_start
                    buffer.delete(len(latex_content) + 2)  # Delete $content$
                    buffer.insert_text(LATEX_TO_UNICODE[latex_content])

        return kb

    def _create_completer(self):
        """Create completer for common commands"""
        commands = ["%attach", "%drop", "%exec", "%load", "%log", "%resp", "%reset", "%save", "%tmpl", "%tool", "%web"]
        return WordCompleter(commands, ignore_case=True)

    def get_input(self) -> str:
        """Get multi-line input from user. Returns 'x' on exit."""
        try:
            return self.session.prompt()
        except (EOFError, KeyboardInterrupt):
            return "x"


def input_multi_line() -> str:
    """Enhanced input with history, multi-line editing, and command completion"""
    global chat_input
    if "chat_input" not in globals():
        chat_input = ChatInput()

    return chat_input.get_input()


def translate_latex(s: str) -> str:
    for k, v in LATEX_TO_UNICODE.items():
        s = s.replace("\\" + k, v)
    return s


@dataclass
class CodeBlock:
    language: str
    lines: list[str]


def print_block(lines: str | list[str], line_numbers: bool = False, style: str = "") -> None:
    xs = lines if isinstance(lines, list) else lines.splitlines()
    text = ""
    for i, s in enumerate(xs):
        if line_numbers:
            s2 = s if s.endswith("\n") else s + "\n"
            text += f"{i + 1:>3} {s2}"
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
        return result.stdout, result.stderr
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


def extract_markdown_blocks(contents: str | list[str], sep: str = "```") -> list[CodeBlock]:
    lines = contents.splitlines() if isinstance(contents, str) else contents
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith(sep):
            lang = line[len(sep) :].strip().lower()
            i += 1
            body_start = i
            while i < n and not lines[i].strip().startswith(sep):
                i += 1
            blocks.append(CodeBlock(language=lang, lines=lines[body_start:i]))
            i += 1  # skip closing sep
        else:
            i += 1
    return blocks


def extract_first_code_block(contents: str | list[str], sep: str = "```") -> CodeBlock | None:
    """extracts the first code block"""
    xs = extract_markdown_blocks(contents, sep)
    return xs[0] if xs else None


def execute_python_script(code: str) -> str:
    r = execute_script(CodeBlock("python", code.splitlines()))
    return r or "SUCCESS: script executed successfully but there was no output. include a print statement"


def execute_script(code: CodeBlock):
    print_block(code.lines, True)
    if code.language == "shell":
        code.language = "bash" if IS_LINUX else "powershell"

    msg = ""
    if code.language == "python":
        output, err = save_and_execute_python(code)
        msg = err if err else output
    elif code.language == "bash":
        output, err = save_and_execute_bash(code)
        msg = err if err else output
    elif code.language == "powershell":
        output, err = save_and_execute_powershell(code)
        msg = err if err else output
    else:
        console.print("unrecognised code block found")

    return msg.strip()


def translate_thinking(s: str) -> str:
    match = THINK_PATTERN.search(s)
    if match:
        before, thought, after = match.groups()
        return f"{before}## thinking\n{thought}\n## answer\n{after}"
    return s


def load_textfile(fname: Path, line_numbers: bool = False) -> str | None:
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


def run_linter(fname: Path) -> tuple[bool, str]:
    code = CodeBlock("shell", [f"cd '{fname.parent}' && uvx ruff check --fix {fname.name}"])
    out = execute_script(code)
    return ("error" not in out, out)


def run_python_unittest(fname: Path, func_name: str | None = None) -> tuple[bool, str]:
    """run the test associated with python file fname in folder ./tests. return True if all tests pass."""

    def make_test_cls_name(s: str) -> str:
        return "Test" + "".join(word.capitalize() for word in s.split("_"))

    if not fname.is_file():
        raise ValueError(f"file {fname} does not exist")

    test_module = f"tests.test_{fname.stem}.{make_test_cls_name(func_name)}" if func_name else f"tests.test_{fname.stem}"

    # bash && is like ; on powershell but in this case powershell && works fine
    code = CodeBlock("shell", [f"cd '{fname.parent}' && uv run python -m unittest -v {test_module}"])
    err = execute_script(code)
    failed_tests = "FAIL" in err

    # unittest sends the test output to stderr
    # print_block(out, True)
    # print_block(err, True)
    s = f"python -m unittest {test_module}\n\n" + err
    return not failed_tests, s


def run_mypy(fn: Path) -> tuple[bool, str]:
    code = CodeBlock("shell", [f"cd '{fn.parent}' && uvx --with pydantic mypy --pretty --ignore-missing-imports --follow-imports=skip --strict-optional {fn.name}"])
    out = execute_script(code)
    return ("error" not in out, out)
