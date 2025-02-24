#!/usr/bin/python3
import math
from collections import defaultdict, deque
from datetime import date, timedelta
from decimal import Decimal
from functools import cache
from io import StringIO
from typing import Iterator

import numpy as np
import pandas as pd
import plotly.express as px
from rich.console import Console
from rich.pretty import pprint

# import graphviz
from scipy.optimize import brentq

console = Console()


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def fibm(maxn):
    memo = [-1] * maxn
    memo[0] = 0
    memo[1] = 1

    def impl(n):
        if memo[n] < 0:
            # print(f'calculating {n}')
            memo[n] = impl(n - 1) + impl(n - 2)
        return memo[n]

    return impl


def fib_iter(n: int) -> Iterator[int]:
    """return a generator iterator"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# use functools cache decorator
@cache
def fib2(n):
    return n if n < 2 else fib2(n - 1) + fib2(n - 2)


def collatz_iter(n):
    while n > 1:
        yield n
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    yield n


def draw_collatz_seq_length():
    # draw barchart of lengths of various sequences
    d = {x: sum(1 for _ in collatz_iter(x)) for x in range(10, 100)}
    df = pd.DataFrame.from_dict(d, orient="index")

    fig = px.bar(df, x=df.index, y=df.iloc[:, 0], title="Collatz Seq length")
    fig.show()
    breakpoint()


# generator expression for a sequence of squares
def gen_expr():
    # note the brackets create an unevaluated generator
    # which can be evaluated later on demand eg list(ex) or sum(ex)
    ex = (x**2 for x in range(10))
    for x in ex:
        console.print(x, style="cyan")


def foldr1(op, xs: list[int]) -> int:
    """structural pattern matching"""
    match xs:
        case [h]:
            return h
        case [h, *t]:
            return op(h, foldr1(op, t))
        case _:
            raise ValueError("empty list")


def pattern_match_commands(s: str) -> int:
    """pattern match a list with constants"""
    match s.split():
        case ["neg", x]:
            return -int(x)
        case ["add", x, y]:
            return int(x) + int(y)
        case _:
            raise ValueError(f"bad command {s}")


def parse_lines(xs):
    """generator function. parse input lines with space separated values. first line used as key names"""
    colnames = None
    for x in xs:
        ys = x.split()
        if len(ys) > 0:
            if colnames:
                yield {k: v for k, v in zip(colnames, ys)}
            else:
                colnames = ys


def as_typed_values(x):
    """parse string values to types. stateless so can be used with map"""
    x["start"] = date.fromisoformat(x["start"])
    x["finish"] = date.fromisoformat(x["finish"])
    x["price"] = Decimal(x["price"])
    return x


def parsing_text():
    """example of chaining generators together"""
    s = """
    # this file contains comments
    start finish price
    2022-04-01	2023-03-31	288.71
    2022-11-20	2023-03-31	69.70
    2022-06-01	2023-03-31	240.46
    2023-04-01	2024-03-31	328.87
    2023-04-01	2023-06-06	39.00
    # another comment
    2022-06-01	2022-11-20	94.85
    2022-06-01	2023-03-31	-240.00"""

    # start pipeline with a generator that returns lines skipping comments
    # then add mapping stages line -> dict of string -> dict with types
    # p = (e for e in StringIO(s) if not e.strip().startswith('#'))
    # or we can use builtin filter
    p = filter(lambda e: not e.strip().startswith("#"), StringIO(s))
    # note parse lines is stateful so cant use map()
    p1 = parse_lines(p)
    pipeline = map(as_typed_values, p1)
    # can use pipeline in a aggregration function
    # sum(e['price'] for e in pipeline)
    # or just iterating
    for x in pipeline:
        console.print(x, style="green")


def check_seq(num1s: int):
    """generator state machine matches at least num1s followed by 1 or more 2's then a 1"""
    while True:
        y = yield False
        found = 0
        if y == 1:
            while y == 1:
                found += 1
                y = yield False
            if y == 2 and found >= num1s:
                while y == 2:
                    y = yield False
                if y == 1:
                    y = yield True


def draw_gantt_chart():
    s = """start finish task
    2022-04-01	2023-03-31	£288.71
    2022-11-20	2023-03-31	£69.70
    2022-06-01	2023-03-31	£240.46
    2023-04-01	2024-03-31	£328.87
    2023-04-01	2023-06-06	£39.00
    2022-06-01	2022-11-20	£94.85
    2022-06-01	2023-03-31	-£240.00"""
    df = pd.read_csv(StringIO(s), sep="\\s+")
    df.sort_values("start", inplace=True)

    # Create the chart
    fig = px.timeline(df, x_start="start", x_end="finish", y="task", color="task")

    # Customize the chart
    fig.update_layout(title="Gantt Chart", xaxis_title="Date", yaxis_title="Task")

    # Show the chart
    fig.show()


# def draw_diag():
#     g = graphviz.Graph('G', filename='z.gv', format='svg')
#     g.edge('run', 'intr')
#     g.edge('intr', 'runbl')
#     g.edge('runbl', 'run')
#     g.edge('run', 'kernel')
#     g.view()


def init_cashflow(drawdown, n, start_date):
    d = start_date.day
    start_date -= timedelta(days=d)
    ix = pd.date_range(start=start_date, periods=n + 1, freq="MS")
    ix = ix + timedelta(days=d - 1)
    days = ix.to_series().diff().dt.days.fillna(0).astype(int)
    df = pd.DataFrame(
        {
            "days": days,
            "capital": float(drawdown),
            "repayment": 0.0,
            "interest": 0.0,
            "int_paid": 0.0,
            "frac_int": 0.0,
            "outstanding": 0.0,
        },
        index=ix,
    )
    df.iat[0, 6] = df.iat[0, 1]
    return df


def evaluate_cashflow(df, repayment, interest_rate):
    """return the final outstanding amount given a fixed repayment amount"""
    df.iloc[1:, 2] = repayment
    prev = None
    for i in df.index:
        # copy prev outstanding to current capital
        if prev:
            df.at[i, "capital"] = df.at[prev, "outstanding"]
        # calculate row
        carried_interest = df.at[prev, "frac_int"] if prev else 0.0
        df.at[i, "interest"] = df.at[i, "days"] * df.at[i, "capital"] * interest_rate / 365 + carried_interest
        df.at[i, "int_paid"] = math.floor(df.at[i, "interest"] * 100) / 100
        df.at[i, "frac_int"] = df.at[i, "interest"] - df.at[i, "int_paid"]
        df.at[i, "outstanding"] = df.at[i, "capital"] - df.at[i, "repayment"] + df.at[i, "int_paid"]
        prev = i
    return df.iat[-1, 6]


def solve_cashflow(df, interest_rate):
    ubound = df.iat[0, 1]
    return brentq(lambda x: evaluate_cashflow(df, x, interest_rate), 0, ubound)


def example_cashflow():
    drawdown = 1200
    interest_rate = 0.06
    df = init_cashflow(drawdown, 12, date.today())
    r = solve_cashflow(df, interest_rate)
    # re-evaluate with rounded repayment
    evaluate_cashflow(df, round(r, 2), interest_rate)
    console.print("--- example cashflow ---", style="yellow")
    console.print(f"drawdown {drawdown:.2f}, interest rate {interest_rate * 100:.4f}%, monthly payment {round(r, 2)}")
    console.print(
        f"total paid {df['repayment'].sum():.2f} , total interest {df['int_paid'].sum():.2f}, total frac {df['frac_int'].sum()}"
    )
    console.print(df)


def topological_sort(graph):
    """
    Performs a topological sort of a directed graph.

    Args:
    graph: A dictionary representing the graph where keys are vertices and values are lists of their neighbors.

    Returns:
    A list representing the topological sort of the graph, or None if the graph contains a cycle.
    """
    in_degree = defaultdict(int)
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    pprint(in_degree)
    q = deque(u for u in graph if in_degree[u] == 0)
    sorted_list = []

    while q:
        pprint(q)
        u = q.popleft()
        sorted_list.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    if len(sorted_list) < len(graph):
        return None  # Graph contains a cycle

    return sorted_list


def weekly_df(dt: date, sz: int) -> pd.DataFrame:
    """Creates a DataFrame with weekly index starting on first monday after dt."""

    dates = pd.date_range(start=dt, periods=sz, freq="W-MON")
    rnds = np.random.randint(0, 100, size=sz)
    return pd.DataFrame({"random_ints": rnds}, index=dates)


def binary_search(xs, target):
    """return index of entry"""
    lo = 0
    hi = xs.size
    while hi > lo:
        mid = (hi + lo) // 2
        m = xs[mid]
        if target < m:
            hi = mid
        elif target > m:
            lo = mid + 1
        else:
            return mid
    return -1


def np_search(xs, target):
    # ss returns the left index to insert target <= xs[n] ie xs[n] is the ceiling element
    n = np.searchsorted(xs, target)
    return -1 if n >= xs.size or xs[n] != target else n


if __name__ == "__main__":
    # draw_collatz_seq_length()

    s = "neg 7"
    console.print(f"command {s} = {pattern_match_commands(s)}", style="green")
    s = "add 2 3"
    console.print(f"command {s} = {pattern_match_commands(s)}", style="green")
    # draw_gantt_chart()
    # draw_diag()
    example_cashflow()

    parsing_text()

    # create generator and make first call to initialise
    stm = check_seq(3)
    stm.send(None)
    for i in [0, 1, 2, 1, 1, 1, 2, 2, 1, 0, 1]:
        x = stm.send(i)
        console.print(f"iter {i} ret {x}", style="cyan")
