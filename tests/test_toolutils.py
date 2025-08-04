#!/usr/bin/env python3
import unittest
from textwrap import dedent
from pathlib import Path

from toolutils import edit_file_impl, EditItem, extract_error_locations, get_source_code, VirtualFile, VirtualFileSystem


class TestEditFileImpl(unittest.TestCase):
    def test_simple_replacement(self):
        original = dedent("""\
            line 1
            line 2
            line 3
            line 4
            line 5
        """).splitlines()

        edit = EditItem(search=["line 3"], replace=["modified line 3"])

        expected = dedent("""\
            line 1
            line 2
            modified line 3
            line 4
            line 5
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_insertion(self):
        original = dedent("""\
            line 1
            line 2
            line 3
        """).splitlines()

        edit = EditItem(search=["line 2"], replace=["line 2", "new line"])

        expected = dedent("""\
            line 1
            line 2
            new line
            line 3
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_deletion(self):
        original = dedent("""\
            line 1
            line 2
            line 3
            line 4
        """).splitlines()

        edit = EditItem(search=["line 2", "line 3"], replace=["line 2"])

        expected = dedent("""\
            line 1
            line 2
            line 4
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_preserve_indent(self):
        original = dedent("""\
            def hello():
                print("Hello")
                print("World")
                return True
        """).splitlines()

        edit = EditItem(search=['print("Hello")', 'print("World")', "return True"], replace=['print("Hello, World!")', "return True"])
        expected = dedent("""\
            def hello():
                print("Hello, World!")
                return True
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_preserve_relative_indent(self):
        original = dedent("""\
            def hello(n):
                print("Hello")
                if n:
                    print("World")
                return True
        """).splitlines()

        edit = EditItem(
            search=[
                'print("World")',
            ],
            replace=[
                '    print("World")',
                'print("new")',
            ],
        )
        expected = dedent("""\
            def hello(n):
                print("Hello")
                if n:
                    print("World")
                print("new")
                return True
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_reorder(self):
        original = dedent("""\
            apple
            melon
            banana
            pear
        """).splitlines()

        edit = EditItem(search=["apple", "melon", "banana"], replace=["apple", "banana", "melon"])

        expected = dedent("""\
            apple
            banana
            melon
            pear
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_insert_start(self):
        original = dedent("""\
            apple
            melon
            banana
            pear
        """).splitlines()

        edit = EditItem(search=["apple"], replace=["strawberry", "apple"])

        expected = dedent("""\
            strawberry
            apple
            melon
            banana
            pear
        """).splitlines()

        result = edit_file_impl(original, edit)
        self.assertEqual(result, expected)

    def test_error_multiple_matches(self):
        original = ["line 1", "line 2", "line 1", "line 4"]

        edit = EditItem(search=["line 1"], replace=["new line"])

        with self.assertRaises(ValueError):
            edit_file_impl(original, edit)

    def test_error_no_match(self):
        original = ["line 1", "line 2", "line 3"]

        edit = EditItem(search=["non existent line"], replace=["new line"])

        with self.assertRaises(ValueError):
            edit_file_impl(original, edit)


class TestVirtualFile(unittest.TestCase):
    def test_load_real_file(self):
        f = VirtualFile(Path("~/Documents/chats/code.md").expanduser())
        print(f)
        # xs = f.read_text(line_number=14, window_size=9)
        # for s in xs:
        #     print(s)
        s = f.edit([EditItem(["## example code"], ["### xxx yyy", "inserted line"])])
        print(s)
        xs = f.read_text(line_number=14, window_size=9, show_line_numbers=True)
        for s in xs:
            print(s)
        print(f)
        s = f.edit([EditItem(["### xxx yyy", "inserted line"], ["## example code"])])
        print(s)
        xs = f.read_text(line_number=14, window_size=9, show_line_numbers=True)
        for s in xs:
            print(s)

    def test_use_unbacked_file(self):
        f = VirtualFile(name="hello", content="this\nis my\nnew file")
        edit = EditItem(search=["is my"], replace=["was my", "apple"])

        f.edit([edit])

        self.assertEqual(f.read_text(), ["this", "was my", "apple", "new file"])


class TestVirtualFileSystem(unittest.TestCase):
    def test_load_real_file(self):
        print("TestVirtualFileSystem")
        vfs = VirtualFileSystem()
        vfs.create_mapping(Path("~/Documents/chats/code.md").expanduser())
        xs = vfs.read_text("code.md", line_number=14, window_size=9, show_line_numbers=True)
        for s in xs:
            print(s)
    
    def test_parse_edits(self):
        FNAME = "test"
        vfs = VirtualFileSystem()
        vfs.create_unmapped(FNAME, "pear\napple\nbanana")

        input_text = dedent("""\
            <<<<<<<
            apple
            =======
            APPLE
            >>>>>>>
            """)
        s = vfs.edit_diff(FNAME, input_text)
        self.assertEqual(s, "SUCCESS: applied 1 edit")

        xs = vfs.read_text(FNAME)
        self.assertEqual(xs, ["pear", "APPLE", "banana"])

    def test_parse_edits_multiple(self):
        FNAME = "test"
        vfs = VirtualFileSystem()
        vfs.create_unmapped(FNAME, "pear\napple\n  banana\nmelon")

        input_text = dedent("""\
            <<<<<<< search
            apple
            =======
            APPLE
            >>>>>>> replace

            <<<<<<<
              banana
            melon
            =======
            add some more fruits
            >>>>>>>
            """)
        s = vfs.edit_diff(FNAME, input_text)
        self.assertEqual(s, "SUCCESS: applied 2 edit")

        xs = vfs.read_text(FNAME)
        self.assertEqual(xs, ["pear", "APPLE", "  add some more fruits"])


class TestExtractErrorLocations(unittest.TestCase):
    def test_extract_error_locations(self):
        return
        # error_message = r"""
        # ERROR

        # ======================================================================
        # ERROR: test_simple_replacement (tests.test_toolutils.TestEditFileImpl.test_simple_replacement)
        # ----------------------------------------------------------------------
        # Traceback (most recent call last):
        # File "C:\code\scripts\tests\test_toolutils.py", line 32, in test_simple_replacement
        #     result = edit_file_impl(original, edit)
        # File "C:\code\scripts\toolutils.py", line 60, in edit_file_impl
        #     update_edit_position(source, edit)
        #     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
        # File "C:\code\scripts\toolutils.py", line 42, in update_edit_position
        #     raise ValueError(f"No occurrences of search block found. Search pattern: {edit.search!r}")
        # ValueError: No occurrences of search block found. Search pattern: ['line 3x']

        # ----------------------------------------------------------------------
        # Ran 9 tests in 0.020s

        # FAILED (errors=1)
        # """

        # extracted_locations = extract_error_locations(error_message)

        # self.assertEqual(len(extracted_locations), 3)
        # self.assertTrue(extracted_locations[0].file_path.exists())
        # self.assertEqual(extracted_locations[0].line_number, 32)
        # self.assertEqual(extracted_locations[0].function_name, "test_simple_replacement")

        # for location in extracted_locations:
        #     self.assertGreater(len(location.get_source()), 0)


if __name__ == "__main__":
    unittest.main()
