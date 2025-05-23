#!/usr/bin/env python3
import platform
import unittest
from textwrap import dedent

# from rich import inspect
from chatutils import CodeBlock, apply_diff_impl, apply_unified_diff_impl , execute_script, extract_code_block, extract_diff, save_and_execute_bash, save_and_execute_powershell, save_and_execute_python, translate_latex, print_block
from ftutils import evaluate_expression

# uv run python -m unittest tests.test_chatutils


class Test_save_and_execute_python(unittest.TestCase):
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


class Test_save_and_execute_powershell(unittest.TestCase):
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


class Test_save_and_execute_bash(unittest.TestCase):
    def test_save_and_execute_bash(self):
        s = "ls ~/Documents/*.txt"
        if platform.system() == "Linux":
            c = CodeBlock("bash", s.split("\n"))
            out, err = save_and_execute_bash(c)
            self.assertGreater(len(out.split("\n")), 5)
            self.assertEqual(len(err), 0)


class Test_execute_script(unittest.TestCase):
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


class Test_extract_code_block(unittest.TestCase):
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


class Test_evaluate_expression(unittest.TestCase):
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


class Test_translate_latex(unittest.TestCase):
    def test_translate_latex(self):
        r = translate_latex("if A \\rightarrow B \\lor \\negC \\neq D")
        self.assertEqual(r, "if A → B ∨ ¬C ≠ D")


class Test_apply_diff_impl(unittest.TestCase):
    def test_single_line_replace(self):
        lines = dedent("""\
            foo bar
                foo
            baz
        """).splitlines(keepends=True)
        expected = dedent("""\
            foo bar
                qux
            baz
            """).splitlines(keepends=True)
        result = apply_diff_impl(lines, ["foo"], ["qux"])
        self.assertEqual(result, expected)

        result = apply_diff_impl(lines, ["  foo"], ["  qux"])
        self.assertEqual(result, expected)

    def test_multi_line_replace(self):
        lines = dedent("""\
            before
              foo
            after
        """).splitlines(keepends=True)
        expected = dedent("""\
            before
              bar
              quz
            after
        """).splitlines(keepends=True)

        result = apply_diff_impl(lines, ["foo"], ["bar", "quz"])
        self.assertEqual(result, expected)

        result = apply_diff_impl(lines, ["foo"], ["  bar", "  quz"])
        self.assertEqual(result, expected)

    def test_preserve_relative_whitespace_increasing(self):
        # Test single line replacement
        lines = dedent("""\
            before
              foo
            after
        """).splitlines(keepends=True)

        expected = dedent("""\
            before
              bar
                quz
            after
        """).splitlines(keepends=True)

        result = apply_diff_impl(lines, ["foo"], ["bar", "  quz"])
        self.assertEqual(result, expected)

        result = apply_diff_impl(lines, ["foo"], ["  bar", "    quz"])
        self.assertEqual(result, expected)

    def test_preserve_relative_whitespace_decreasing(self):
        # Test single line replacement
        lines = dedent("""\
            before
              foo
            after
        """).splitlines(keepends=True)

        expected = dedent("""\
            before
              bar
            quz
            after
        """).splitlines(keepends=True)

        result = apply_diff_impl(lines, ["foo"], ["  bar", "quz"])
        self.assertEqual(result, expected)

        result = apply_diff_impl(lines, ["foo"], ["    bar", "  quz"])
        self.assertEqual(result, expected)

    def test_two_line_match_delete_line(self):
        lines = dedent("""\
            before
            foo bar
            foo bar
            baz
            after
        """).splitlines(keepends=True)

        expected = dedent("""\
            before
            foo bar
            inserted
            after
        """).splitlines(keepends=True)

        result = apply_diff_impl(lines, ["foo bar", "baz"], ["inserted"])
        self.assertEqual(result, expected)

    def test_error_no_match(self):
        lines = dedent("""\
            foo bar
            baz
        """).splitlines()
        with self.assertRaises(ValueError):
            apply_diff_impl(lines, "foo", "qux")

    def test_error_multiple_matches(self):
        lines = dedent("""\
                foo
            ignore
            foo
        """).splitlines()
        with self.assertRaises(ValueError):
            apply_diff_impl(lines, "foo", "qux")

    def test_apply_diff(self):
        code = dedent("""\
            # example
            def fib(n):
                return n if n < 2 else fib(n - 1) + fib(n - 2)

            def fib_iter(n):
                # return a generator iterator
                a, b = 0, 1
                for _ in range(n):
                    yield a
                    a, b = b, a + b

            class dummy:
                def collatz(n):
                    # return an iterator for the collatz sequence for n

            print(fib(10))
        """).splitlines(keepends=True)
        raw_diff = "<<<\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n===\nfrom collections.abc import Iterator\n\ndef fib(n: int) -> int:\n    # Compute the nth Fibonacci number recursively.\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n>>>\n<<<\ndef fib_iter(n):\n    # return a generator iterator\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n===\ndef fib_iter(n: int) -> Iterator:\n    # Return a generator iterator for Fibonacci numbers up to the nth term.\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n>>>\n<<<\ndef collatz(n):\n    # return an iterator for the collatz sequence for n\n===\ndef collatz_iter(n: int) -> Iterator:\n    # Return an iterator for the Collatz sequence starting at n.\n    while n > 1:\n        yield n\n        if n % 2 == 0:\n            n //= 2\n        else:\n            n = 3 * n + 1\n    yield 1\n>>>\n<<<\nprint(fib(10))\n===\nprint(fib(10))\n>>>\n"
        expected = dedent("""\
            # example
            from collections.abc import Iterator

            def fib(n: int) -> int:
                # Compute the nth Fibonacci number recursively.
                return n if n < 2 else fib(n - 1) + fib(n - 2)

            def fib_iter(n: int) -> Iterator:
                # Return a generator iterator for Fibonacci numbers up to the nth term.
                a, b = 0, 1
                for _ in range(n):
                    yield a
                    a, b = b, a + b

            class dummy:
                def collatz_iter(n: int) -> Iterator:
                    # Return an iterator for the Collatz sequence starting at n.
                    while n > 1:
                        yield n
                        if n % 2 == 0:
                            n //= 2
                        else:
                            n = 3 * n + 1
                    yield 1

            print(fib(10))
        """).splitlines(keepends=True)
        # print_block(raw_diff, True)
        # print_block(code, True)

        diffs = extract_diff(raw_diff)
        self.assertEqual(len(diffs), 4)
        modified = code.copy()
        for d in diffs:
            modified = apply_diff_impl(modified, d[0], d[1])
        # print_block(modified, True)
        self.assertEqual(len(modified), len(expected))
        for i in range(len(modified)):
            self.assertEqual(modified[i], expected[i])


class Test_extract_diff(unittest.TestCase):
    def test_extract_diff(self):
        diff = dedent("""\
            <<<
            foo
            ===
            bar
            >>>
        """)
        xs = extract_diff(diff)
        self.assertEqual(len(xs), 1)
        search, replace = xs[0]
        self.assertEqual(search, ["foo"])
        self.assertEqual(replace, ["bar"])

    def test_extract_diff2(self):
        diff = dedent("""\
            ignored
            <<<
            foo
              qux
            ===
            bar
              baz
            >>>
        """)
        xs = extract_diff(diff)
        self.assertEqual(len(xs), 1)
        search, replace = xs[0]
        self.assertEqual(search, ["foo", "  qux"])
        self.assertEqual(replace, ["bar", "  baz"])

    def test_extract_diff_no_marker(self):
        diff = dedent("""\
            <<<
            foo
            bar
              baz
            >>>
        """)
        with self.assertRaises(ValueError):
            extract_diff(diff)


class Test_apply_unified_diff_impl(unittest.TestCase):
    def test_basic_addition(self):
        original = dedent("""\
            line1
            line2
            line3""").splitlines()
        
        diff = dedent("""\
            --- a/file
            +++ b/file
            @@ -1,3 +1,4 @@
             line1
            +new line
             line2
             line3""")
        
        expected = dedent("""\
            line1
            new line
            line2
            line3""").splitlines()
        
        result = apply_unified_diff_impl(original, diff)
        self.assertEqual(result, expected)

    def test_basic_addition2(self):
        original = dedent("""\
            line1
            line2
            line3""").splitlines()
        
        diff = dedent("""\
            --- a/file
            +++ b/file
            @@ -1,3 +1,4 @@
             line1
            -line2
            +new line
            +  indented line
             line3""")
        
        expected = dedent("""\
            line1
            new line
              indented line
            line3""").splitlines()
        
        result = apply_unified_diff_impl(original, diff)
        self.assertEqual(result, expected)

    def test_basic_deletion(self):
        original = dedent("""\
            line1
            line2
            line3""").splitlines()
        
        diff = dedent("""\
            --- a/file
            +++ b/file
            @@ -1,3 +1,2 @@
             line1
            -line2
             line3""")
        
        expected = dedent("""\
            line1
            line3""").splitlines()
        
        result = apply_unified_diff_impl(original, diff)
        self.assertEqual(result, expected)

    def test_multiple_non_overlapping_hunks(self):
        original = dedent("""\
            line1
            line2
            line3
            line4
            line5
            line6""").splitlines()
        
        diff = dedent("""\
            --- a/file
            +++ b/file
            @@ -1,3 +1,4 @@
             line1
            +inserted1
             line2
             line3
            @@ -4,3 +5,4 @@
             line4
            +inserted2
             line5
             line6""")
        
        expected = dedent("""\
            line1
            inserted1
            line2
            line3
            line4
            inserted2
            line5
            line6""").splitlines()
        
        result = apply_unified_diff_impl(original, diff)
        self.assertEqual(result, expected)

    def test_no_changes(self):
        original = dedent("""\
            line1
            line2
            line3""").splitlines()
        
        diff = dedent("""\
            --- a/file
            +++ b/file
            @@ -1,3 +1,3 @@
             line1
             line2
             line3""")
        
        expected = original.copy()  # Should be unchanged
        result = apply_unified_diff_impl(original, diff)
        self.assertEqual(result, expected)

    def test_invalid_diff_raises_value_error(self):
        original = ["line1", "line2"]
        diff = "invalid diff format"
        
        with self.assertRaises(ValueError):
            apply_unified_diff_impl(original, diff)

    def test_line_mismatch_raises_value_error(self):
        original = dedent("""\
            line1
            wrong_line  # This will cause mismatch
            line3""").splitlines()
        
        diff = dedent("""\
            --- a/file
            +++ b/file
            @@ -1,3 +1,3 @@
             line1
            -line2  # Expected this line
            +new_line
             line3""")
        
        with self.assertRaises(ValueError):
            apply_unified_diff_impl(original, diff)

    # def test_empty_file_addition(self):
    #     original = []  # Empty file
        
    #     diff = dedent("""\
    #         --- a/file
    #         +++ b/file
    #         @@ -0,0 +1,2 @@
    #         +line1
    #         +line2""")
        
    #     expected = dedent("""\
    #         line1
    #         line2""").splitlines()
        
    #     result = apply_unified_diff_impl(original, diff)
    #     self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
