#!/usr/bin/env python3
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

console = Console()
THINK_PATTERN = re.compile('(.*?)<think>(.*?)</think>(.*)', re.DOTALL)
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
    xs = (f"{i+1:02d} {s}" for i, s in enumerate(code.lines))
    block = f"```{code.language}\n{"\n".join(xs)}\n```"
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
