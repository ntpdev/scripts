#!/usr/bin/env python3
from dataclasses import dataclass
import os
import platform
from pathlib import Path
import sys
from typing import Optional
from rich.console import Console
from rich.markup import escape
import subprocess
import unittest


console = Console()
latex_to_unicode = {
    "neg": "¬",
    "forall": "∀",
    "exists": "∃",
    "in": "∈",
    "vdash": "⊢",
    "dashv": "⊣",
    "sim": "∼",
    "approx": "≈",
    "land": "∧",
    "wedge": "∧",
    "lor": "∨",
    "rightarrow": "→",
    "implies": "⇒",
    "iff": "⇔",
    "subset": "⊂",
    "supset": "⊃",
    "leftrightarrow": "↔",
    "therefore": "∴",
    "neq": "≠",
    "equiv": "≡",
    "times": "×",
    "div": "÷",
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
        else:
            return venv_path / "bin" / "python"
    else:
        # Fallback to the current Python interpreter
        console.print("no virtual env found. running default python environment", style="red")
        return Path(sys.executable)


def find_last_file(dir: Path) -> tuple[Optional[Path], int]:
    matching_files = [(x, int(x.stem[1:])) for x in dir.glob("z*.md") if x.stem[1:].isdigit()]

    if not matching_files:
        return None, 0

    matching_files.sort(key=lambda e: e[1], reverse=True)
    return matching_files[0]


def save_content(msg: str):
    fn, next = find_last_file(Path.home() / "Documents" / "chats")
    next += 1
    fout = f"z{next:02}.md"
    filename = make_fullpath(fout)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(msg)
    s = f"saved {fout} {msg if len(msg) < 70 else msg[:70] + ' ...'}"
    console.print(s, style="red")


def save_code(fn: str, code: CodeBlock) -> Path:
    full_path = make_fullpath(fn)
    console.print(f"saving code {full_path}", style="red")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\n".join(code.lines))
    return full_path


def save_and_execute_python(code: CodeBlock):
    try:
        script_path = save_code("temp.py", code)

        result = subprocess.run([str(get_python()), script_path], cwd=script_path.parent, capture_output=True, text=True, timeout=60)

        if len(result.stdout) > 0:
            console.print(result.stdout, style="yellow")
        else:
            console.print(result.stderr, style="red")
        return result.stdout, result.stderr
    except Exception as e:
        console.print(f"ERROR: {e}", style="red")
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
        console.print(f"ERROR: {e}", style="red")
        return None, str(e)


def save_and_execute_powershell(code: CodeBlock):
    script_path = save_code("temp.ps1", code)
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if len(result.stdout) > 0:
            console.print(result.stdout, style="yellow")
        else:
            console.print(result.stderr, style="red")
        return result.stdout, result.stderr
    except Exception as e:
        console.print(f"ERROR: {e}", style="red")
        return None, str(e)


def search_for_language(s: str) -> str:
    xs = [e for e in ["python", "bash", "powershell"] if e in s]
    return xs[0] if len(xs) > 0 else ""


def search_for_exact_language(s: str) -> str:
    for e in ["python", "bash", "powershell"]:
        if s == e:
            return s
    return ""


def extract_code_block(contents: str, sep: str) -> CodeBlock:
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
                    code.language = search_for_language(contents.lower())
                return code
        else:
            if inside:
                # infer language based on content
                if (len(code.language) == 0) and ("print(" in x or "import" in x):
                    code.language = "python"
                # check for line with name of language
                em = search_for_exact_language(x)
                if len(em) > 0:
                    code.language = em
                else:
                    code.lines.append(x)

    return code


def execute_script(code: CodeBlock):
    output = None
    err = None
    msg = None
    for i, s in enumerate(code.lines):
        k = i + 1
        console.print(f"{k:02d} {escape(s)}", style="red")
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
    return msg


def translate_latex(s: str) -> str:
    for k, v in latex_to_unicode.items():
        s = s.replace("\\" + k, v)
    return s


def input_multi_line() -> str:
    if (inp := input().strip()) != "{":
        return inp
    lines = []
    while (line := input().strip()) != "}":
        lines.append(line)

    return "\n".join(lines)


class TestChat(unittest.TestCase):
    def test_save_and_execute_python(self):
        s = """
import math
print(f'{math.pi * 7 ** 2:.2f}')
"""

        c = CodeBlock("python", s.split("\n"))
        out, err = save_and_execute_python(c)
        self.assertEqual(out, "153.94\n")
        self.assertEqual(len(err), 0)

    def test_save_and_execute_python_err(self):
        s = """
x,y = 7, 0
print(f'{x / y}')
"""

        c = CodeBlock("python", s.split("\n"))
        out, err = save_and_execute_python(c)
        self.assertEqual(len(out), 0)
        self.assertTrue("ZeroDivisionError" in err)

    def test_save_and_execute_powershell(self):
        s = """
Get-ChildItem -Path ~\\Documents\\*.txt |
  Sort-Object -Property Length -Descending |
  Select-Object -First 5
"""
        if platform.system() != "Linux":
            c = CodeBlock("powershell", s.split("\n"))
            out, err = save_and_execute_powershell(c)
            self.assertGreater(len(out.split("\n")), 5)
            self.assertEqual(len(err), 0)

    def test_save_and_execute_bash(self):
        s = """
ls ~/Documents/*.txt
"""
        if platform.system() == "Linux":
            c = CodeBlock("bash", s.split("\n"))
            out, err = save_and_execute_bash(c)
            self.assertGreater(len(out.split("\n")), 5)
            self.assertEqual(len(err), 0)

    def test_execute_script(self):
        c = CodeBlock("python", ['print("hello from python")'])
        out = execute_script(c)
        self.assertEqual(out, "hello from python\n")

    def test_execute_script2(self):
        if platform.system() == "Windows":
            c = CodeBlock("powershell", ['Write-Output "hello from powershell"'])
            out = execute_script(c)
            self.assertTrue(out.startswith("hello from powershell"))

    def test_execute_script3(self):
        if platform.system() == "Linux":
            c = CodeBlock("bash", ['echo hello from bash $(date +"%Y-%m-%d")'])
            out = execute_script(c)
            self.assertTrue(out.startswith("hello from bash"))

    def test_extract_code_block(self):
        s = """
here is code
```python
args = parser.parse_args()
chat(args.llm)
```
text after block
"""

        c = extract_code_block(s, "```")
        self.assertEqual(c.language, "python")
        self.assertEqual(c.lines, ["args = parser.parse_args()", "chat(args.llm)"])

    def test_extract_code_block_infer(self):
        s = """
here is code
```
args = parser.parse_args()
chat(args.llm)
print(args.llm)
```
text after block
"""

        c = extract_code_block(s, "```")
        self.assertEqual(c.language, "python")
        self.assertEqual(c.lines, ["args = parser.parse_args()", "chat(args.llm)", "print(args.llm)"])

    def test_extract_code_block_infer2(self):
        s = """
```
Get-ChildItem -Path "~\\Documents" -Filter *.txt |
  Sort-Object -Property Length -Descending |
  Select-Object -First 5 -ExpandProperty Name, Length
```
This PowerShell command will list the 5 largest `.txt` files in the `~\\Documents` directory.'
            """
        c = extract_code_block(s, "```")
        console.print(c)
        self.assertEqual(c.language, "powershell")
        self.assertEqual(len(c.lines), 3)

    def test_translate_latex(self):
        s = "if A \\rightarrow B \\lor \\negC \\neq D"
        r = translate_latex(s)
        self.assertEqual(r, "if A → B ∨ ¬C ≠ D")


if __name__ == "__main__":
    unittest.main()
