#!/usr/bin/env python3
import unittest
from textwrap import dedent

from toolutils import apply_diff_impl


class TestApplyDiffImpl(unittest.TestCase):
    def test_simple_replacement(self):
        original = (
            dedent("""
            line 1
            line 2
            line 3
            line 4
            line 5
        """)
            .strip()
            .splitlines()
        )

        diff = (
            dedent("""
            -line 3
            +modified line 3
        """)
            .strip()
            .splitlines()
        )

        expected = (
            dedent("""
            line 1
            line 2
            modified line 3
            line 4
            line 5
        """)
            .strip()
            .splitlines()
        )

        result = apply_diff_impl(original, 2, diff)
        self.assertEqual(result, expected)

    def test_insertion(self):
        original = dedent("""\
            line 1
            line 2
            line 3
        """).splitlines()

        diff = dedent("""\
             line 2
            +new line
        """).splitlines()

        expected = dedent("""\
            line 1
            line 2
            new line
            line 3
        """).splitlines()

        result = apply_diff_impl(original, 1, diff)
        self.assertEqual(result, expected)

    def test_deletion(self):
        original = dedent("""\
            line 1
            line 2
            line 3
            line 4
        """).splitlines()

        diff = dedent("""\
             line 2
            -line 3
        """).splitlines()

        expected = dedent("""\
            line 1
            line 2
            line 4
        """).splitlines()

        result = apply_diff_impl(original, 2, diff)
        self.assertEqual(result, expected)

    def test_complex_diff(self):
        original = dedent("""\
            def hello():
                print("Hello")
                print("World")
                return True
        """).splitlines()

        diff = dedent("""\
             def hello():
            -    print("Hello")
            -    print("World")
            +    print("Hello, World!")
                 return True
        """).splitlines()

        expected = dedent("""\
            def hello():
                print("Hello, World!")
                return True
        """).splitlines()

        result = apply_diff_impl(original, 0, diff)
        self.assertEqual(result, expected)

    def test_reorder(self):
        original = dedent("""\
            apple
            melon
            banana
            pear
        """).splitlines()

        diff = dedent("""\
             apple
            -melon
             banana
            +melon
        """).splitlines()

        expected = dedent("""\
            apple
            banana
            melon
            pear
        """).splitlines()

        result = apply_diff_impl(original, 0, diff)
        self.assertEqual(result, expected)

    def test_insert_start(self):
        original = dedent("""\
            apple
            melon
            banana
            pear
        """).splitlines()

        diff = dedent("""\
            +strawberry
             apple
        """).splitlines()

        expected = dedent("""\
            strawberry
            apple
            melon
            banana
            pear
        """).splitlines()

        result = apply_diff_impl(original, 0, diff)
        self.assertEqual(result, expected)

    def test_approximate_start_position(self):
        original = ["line " + str(i) for i in range(1, 21)]

        diff = dedent("""\
             line 10
            +inserted line
        """).splitlines()

        # Test with start_position slightly off
        result = apply_diff_impl(original, 12, diff)

        expected = original.copy()
        expected.insert(10, "inserted line")

        self.assertEqual(result, expected)

    def test_invalid_diff_format(self):
        original = ["line 1", "line 2", "line 3"]

        # Diff with invalid prefix
        invalid_diff = ["line 1", "? invalid prefix"]

        with self.assertRaises(ValueError):
            apply_diff_impl(original, 0, invalid_diff)

    def test_no_matching_line(self):
        original = ["line 1", "line 2", "line 3"]

        diff = ["-non existent line", "+new line"]

        with self.assertRaises(ValueError):
            apply_diff_impl(original, 0, diff)


if __name__ == "__main__":
    unittest.main()
