#!/usr/bin/python3
import unittest

import fib
from datetime import date

import numpy as np
from rich.pretty import pprint

graph = {
    "A": ["C"],
    "B": ["C", "D"],
    "C": ["E"],
    "D": ["F"],
    "E": ["F", "H"],
    "F": ["G"],
    "G": ["H"],
    "H": [],
}


class TestDataFrame(unittest.TestCase):
    def test_weekly_df(self):
        df = fib.weekly_df(date(2024, 9, 1), 10)
        self.assertEqual(df.index[0].date(), date(2024, 9, 2))
        self.assertEqual(df.index[-1].date(), date(2024, 11, 4))

        xs = df.index.values
        for i in range(xs.size):
            self.assertEqual(fib.binary_search(xs, xs[i]), i)

        # same search
        ys = [fib.binary_search(xs, i) for i in np.arange("2024-10-21", "2024-11-09", dtype="datetime64[D]")]
        self.assertEqual(
            ys,
            [7, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1],
        )

        zs = [fib.np_search(xs, i) for i in np.arange("2024-10-21", "2024-11-09", dtype="datetime64[D]")]
        self.assertEqual(
            zs,
            [7, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1],
        )

    def test_fib(self):
        xs = [fib.fib(x) for x in range(10)]
        self.assertEqual(xs, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])

        # fib.fib2 which is memoized would be equally efficient
        f = fib.fibm(20)
        xs = [f(x) for x in range(11)]
        self.assertEqual(xs, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])

        pprint(list(fib.fibIter(18)))

    def test_colatz(self):
        xs = fib.collatzIter(7)
        self.assertEqual(list(xs), [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1])

    def test_top_sort(self):
        xs = fib.topological_sort(graph)
        # sort is not unique as there are items at same depth
        self.assertEqual(xs, ["A", "B", "C", "D", "E", "F", "G", "H"])

    def test_np_date(self):
        dates = np.array(["2023-01-01", "2023-02-15", "2023-03-31"], dtype="datetime64")
        pprint(dates)
        dt = dates[1].astype(date)
        self.assertEqual(dt, date(2023, 2, 15))

    def test_iter(self):
        input = [5, -7, 3, 5, 2, -2, 4, -1]
        # replace neg values by 0
        self.assertEqual([x if x > 0 else 0 for x in input], [5, 0, 3, 5, 2, 0, 4, 0])
        # filter out
        self.assertEqual([x for x in input if x > 0], [5, 3, 5, 2, 4])
        # generator expresions can be used on the fly. will sum the positive integers
        self.assertEqual(sum(x for x in input if x > 0), 19)
        # count number positive
        self.assertEqual(sum(1 for x in input if x > 0), 5)
        # not as nice using reduce
        # functools.reduce(lambda acc, e : acc + e if e > 0 else acc, xs, 0)
        self.assertEqual(fib.foldr1(lambda x, y: x + y, input), 9)

if __name__ == "__main__":
    unittest.main()
