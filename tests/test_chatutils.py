#!/usr/bin/env python3
import platform
import unittest
from textwrap import dedent

# from rich import inspect
from chatutils import CodeBlock, execute_script, extract_code_block, save_and_execute_bash, save_and_execute_powershell, save_and_execute_python, translate_latex
from ftutils import evaluate_expression

# uv run python -m unittest tests.test_chatutils


class TestSaveAndExecutePython(unittest.TestCase):
    def test_save_and_execute_python(self):
        s = dedent("""\
            import math
            print(f'{math.pi * 7 ** 2:.2f}')
            """)

        c = CodeBlock("python", s.split("\n"))
        out, err = save_and_execute_python(c)
        self.assertEqual(out.strip(), "153.94")
        self.assertEqual(len(err), 0)

    def test_save_and_execute_python_err(self):
        s = dedent("""\
            x,y = 7, 0
            print(f'{x / y}')
            """)

        c = CodeBlock("python", s.split("\n"))
        out, err = save_and_execute_python(c)
        self.assertEqual(len(out), 0)
        self.assertTrue("ZeroDivisionError" in err)


class TestSaveAndExecutePowershell(unittest.TestCase):
    def test_save_and_execute_powershell(self):
        s = dedent("""\
            Get-ChildItem -Path ~\\Documents\\*.txt |
            Sort-Object -Property Length -Descending |
            Select-Object -First 5
            """)
        if platform.system() != "Linux":
            c = CodeBlock("powershell", s.split("\n"))
            out, err = save_and_execute_powershell(c)
            self.assertGreater(len(out.split("\n")), 5)
            self.assertEqual(len(err), 0)


class TestSaveAndExecuteBash(unittest.TestCase):
    def test_save_and_execute_bash(self):
        s = "ls ~/Documents/*.txt"
        if platform.system() == "Linux":
            c = CodeBlock("bash", s.split("\n"))
            out, err = save_and_execute_bash(c)
            self.assertGreater(len(out.split("\n")), 1)
            self.assertEqual(len(err), 0)


class TestExecuteScript(unittest.TestCase):
    def test_execute_script(self):
        c = CodeBlock("python", ['print("hello from python")'])
        out = execute_script(c)
        self.assertEqual(out.strip(), "hello from python")

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


class TestExtractCodeBlock(unittest.TestCase):
    def test_extract_code_block(self):
        s = dedent("""\
                here is code
                ```python
                args = parser.parse_args()
                chat(args.llm)
                ```
                text after block
                """)

        c = extract_code_block(s, "```")
        self.assertEqual(c.language, "python")
        self.assertEqual(c.lines, ["args = parser.parse_args()", "chat(args.llm)"])

    def test_extract_code_block_infer(self):
        # preserve indentation
        s = dedent("""\
            here is code
            ```
            args = parser.parse_args()
            if x := chat(args.llm):
              print(args.llm)
            ```
            text after block
            """)

        c = extract_code_block(s, "```")
        self.assertEqual(c.language, "python")
        self.assertEqual(c.lines, ["args = parser.parse_args()", "if x := chat(args.llm):", "  print(args.llm)"])

    def test_extract_code_block_infer2(self):
        s = dedent("""\
            ```
            Get-ChildItem -Path "~\\Documents" -Filter *.txt |
            Sort-Object -Property Length -Descending |
            Select-Object -First 5 -ExpandProperty Name, Length
            ```)
            This PowerShell command will list the 5 largest `.txt` files in the `~\\Documents` directory.'
            """)
        c = extract_code_block(s, "```")
        self.assertEqual(c.language, "powershell")
        self.assertEqual(len(c.lines), 3)


class TestEvaluateExpression(unittest.TestCase):
    def test_evaluate_expression(self):
        r = evaluate_expression("31 * 997")
        self.assertEqual(r, "30907")

        # test preserve indents in multi-line
        x = dedent("""\
            from math import factorial
            n = 9
            def multinomial(x, y, z):
                return factorial(n) // (factorial(x) * factorial(y) * factorial(z))
            total_ways = sum(multinomial(n - v - s, v, s) for s in range(1, n) for v in range(s+1, n) if (n - v - s) > v)
            total_ways, total_ways % 1000
            """)
        r = evaluate_expression(x)
        self.assertEqual(r, "(2016, 16)")

        # test split single with ; between statements
        x = "a,b,c = 1,-1,-1; d = b**2 - 4*a*c; round((-b + math.sqrt(d)) / (2*a), 3),  round((-b - math.sqrt(d)) / (2*a),3)"
        r = evaluate_expression(x)
        self.assertEqual(r, "(1.618, -0.618)")

        # test imports
        x = "import sympy; x = sympy.symbols('x'); expr= x**2 - 2*x - 3; sympy.solve(expr, x)"
        r = evaluate_expression(x)
        self.assertEqual(r, "[-1, 3]")

        # test error handling
        x = "x = 3; y = 2; y / (x - 3)"
        r = evaluate_expression(x)
        self.assertEqual(r, "ERROR: ZeroDivisionError: division by zero")


class TestTranslateLatex(unittest.TestCase):
    def test_translate_latex(self):
        r = translate_latex("if A \\rightarrow B \\lor \\negC \\neq D")
        self.assertEqual(r, "if A → B ∨ ¬C ≠ D")


if __name__ == "__main__":
    unittest.main()
