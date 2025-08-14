#!/usr/bin/env python3
import unittest
from pathlib import Path
from textwrap import dedent

from chatutils import extract_markdown_blocks
from toolutils import EditItem, VirtualFile, VirtualFileSystem, edit_file_impl, parse_edits


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
    # def test_load_real_file(self):
    #     f = VirtualFile(Path("~/Documents/chats/code.md").expanduser())
    #     s = f.edit([EditItem(["## example code"], ["### xxx yyy", "inserted line"])])
    #     xs = f.read_text(line_number=14, window_size=9, show_line_numbers=True)
    #     s = f.edit([EditItem(["### xxx yyy", "inserted line"], ["## example code"])])
    #     xs = f.read_text(line_number=14, window_size=9, show_line_numbers=True)

    def test_use_unbacked_file(self):
        f = VirtualFile(name="hello.txt", content="this\nis my\nnew file")
        edit = EditItem(search=["is my"], replace=["was my", "apple"])

        msg = f.edit([edit])

        self.assertTrue(msg.startswith("SUCCESS:"))
        self.assertEqual(f.read_text(), ["this", "was my", "apple", "new file"])

    def test_not_found_error(self):
        f = VirtualFile(name="hello.txt", content="this\nis my\nnew file")
        edit = EditItem(search=["no is my"], replace=["was my", "apple"])

        msg = f.edit([edit])
        print(msg)

        self.assertTrue(msg.startswith("ERROR:"))


class TestVirtualFileSystem(unittest.TestCase):
    def test_load_real_file(self):
        print("TestVirtualFileSystem")
        vfs = VirtualFileSystem()
        vfs.create_mapping(Path("~/Documents/chats/code.md").expanduser())
        xs = vfs.read_text("code.md", line_number=14, window_size=9, show_line_numbers=True)
        for s in xs:
            print(s)

    def test_apply_edits_not_inside_a_block(self):
        fname = "test"
        vfs = VirtualFileSystem()
        vfs.create_unmapped(fname, "pear\napple\nbanana")

        input_text = dedent("""\
            --- search
            apple
            --- replace
            APPLE
            --- end
            """)
        blocks = extract_markdown_blocks(input_text)
        s = vfs.apply_edits(blocks)

        self.assertTrue(s.startswith("ERROR:"))

    def test_apply_edits_no_diff(self):
        fname = "test"
        vfs = VirtualFileSystem()
        vfs.create_unmapped(fname, "pear\napple\nbanana")

        input_text = dedent("""\
            ```
            --- search
            apple
            --- replace
            APPLE
            --- end
            ```
            """)
        blocks = extract_markdown_blocks(input_text)
        s = vfs.apply_edits(blocks)
        self.assertTrue(s.startswith("ERROR:"))

    def test_apply_edits_single(self):
        fname = "test"
        vfs = VirtualFileSystem()
        vfs.create_unmapped(fname, "pear\napple\nbanana")

        input_text = dedent("""\
            ```diff
            --- search test
            apple
            --- replace
            APPLE
            --- end
            ```
            """)
        blocks = extract_markdown_blocks(input_text)
        s = vfs.apply_edits(blocks)
        self.assertTrue(s.startswith("SUCCESS:"))

        xs = vfs.read_text(fname)
        self.assertEqual(xs, ["pear", "APPLE", "banana"])

    def test_apply_edits_multiple(self):
        fname = "foo.py"
        fname2 = "bar.py"
        vfs = VirtualFileSystem()
        vfs.create_unmapped(fname, "pear\napple\nbanana")
        vfs.create_unmapped(fname2, "pear\napple\nbanana")

        input_text = dedent("""\
            ```
            --- search foo.py
            apple
            --- replace
            APPLE
            --- end
            ```
                            
            ```
            --- search bar.py
            apple
            --- replace
            banana
              apple
            --- end
            ```

            ```diff
            --- search foo.py
            APPLE
            --- replace
            APPLES
            --- end
            ```
            """)
        blocks = extract_markdown_blocks(input_text)
        s = vfs.apply_edits(blocks)
        self.assertTrue(s.startswith("SUCCESS:"))

        xs = vfs.read_text(fname)
        self.assertEqual(xs, ["pear", "APPLES", "banana"])

        ys = vfs.read_text(fname2)
        self.assertEqual(ys, ["pear", "banana", "  apple", "banana"])


class TestParseEdits(unittest.TestCase):
    def test_filename_extraction(self):
        input_str = dedent("""
            --- search file1.txt
            old line 1
            old line 2
            --- replace
            new line 1
            new line 2
            --- end
            """).strip()
        edits = parse_edits(input_str)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].filename, "file1.txt")

    def test_search_and_replace_blocks(self):
        input_str = dedent("""
            --- search file2.txt
            foo
            bar
            --- replace
            baz
            qux
            --- end
            """).strip()
        edits = parse_edits(input_str)
        self.assertEqual(edits[0].search, ["foo", "bar"])
        self.assertEqual(edits[0].replace, ["baz", "qux"])

    def test_multiple_edits(self):
        input_str = dedent("""
            --- search file3.txt
            a
            --- replace
            b
            --- end
            ignore
            --- search
            x
            y
            --- replace
            z
            --- end
            """).strip()
        edits = parse_edits(input_str)
        self.assertEqual(len(edits), 2)
        self.assertEqual(edits[0].search, ["a"])
        self.assertEqual(edits[0].replace, ["b"])
        self.assertEqual(edits[1].search, ["x", "y"])
        self.assertEqual(edits[1].replace, ["z"])

    def test_no_filename(self):
        input_str = dedent("""
            --- search
            foo
            --- replace
            bar
            --- end
            """).strip()
        edits = parse_edits(input_str)
        self.assertEqual(edits[0].filename, "")

    def test_list_input(self):
        input_list = ["--- search file4.txt", "old", "--- replace", "new", "--- end"]
        edits = parse_edits(input_list)
        self.assertEqual(len(edits), 1)
        e = edits[0]
        self.assertEqual(e.filename, "file4.txt")
        self.assertEqual(e.search, ["old"])
        self.assertEqual(e.replace, ["new"])


class TestExtractErrorLocations(unittest.TestCase):
    def test_extract_error_locations(self):
        return
        # error_message = r"""
        # ERROR

        # --- replace--- replace--- replace--- replace--- replace--- replace--- replace--- replace--- replace--- replace
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
