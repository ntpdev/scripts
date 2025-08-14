#!/usr/bin/python3
import sys
from collections import deque, namedtuple
from pathlib import Path
from datetime import datetime, time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rich.console import Console

from tsutils import aggregate, aggregate_to_time_bars, aggregate_min_volume, calc_tlb, day_index, load_overlapping_files, make_filename, calc_strat

console = Console()
Price = namedtuple("Price", ["date", "value"])


def export_daily(df: pd.DataFrame, fname: str) -> None:
    dt = df.index[-1]
    s = make_filename(f"{fname}-{dt.year}{dt.month:02d}{dt.day:02d}.csv")
    print(f"exporting daily to {s}")
    df.to_csv(s)


def export_ninja(df: pd.DataFrame, outfile: str) -> None:
    print(f"exporting in Ninja Trader format {outfile} {len(df)}")
    with open(outfile, "w") as f:
        for i, r in df.iterrows():
            s = f"{i.strftime('%Y%m%d %H%M%S')};{r['Open']:4.2f};{r['High']:4.2f};{r['Low']:4.2f};{r['Close']:4.2f};{r['Volume']}\n"
            f.write(s)


def export_min_vol(df: pd.DataFrame, outfile: str) -> None:
    df2 = aggregate_min_volume(df, 2500)
    print(f"exporting minVol file {outfile} {len(df2)}")
    df2.to_csv(outfile)


def export_3lb(df: pd.DataFrame, outfile: str) -> None:
    tlb, rev = calc_tlb(df["close"], 3)
    print(f"exporting 3lb file {outfile} {len(tlb)}")
    tlb.to_csv(outfile)


# create a new DF which aggregates bars between inclusive indexes
def aggregate_bars(df: pd.DataFrame, idxs_start: Sequence[pd.Timestamp], idxs_end: Sequence[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    dts = []
    for s, e in zip(idxs_start, idxs_end):
        dts.append(e.date())
        r = {}
        r["Open"] = df.Open[s]
        r["High"] = df.High[s:e].max()
        r["Low"] = df.Low[s:e].min()
        r["Close"] = df.Close[e]
        r["Volume"] = df.Volume[s:e].sum()
        vwap = np.average(df.WAP[s:e], weights=df.Volume[s:e])
        r["VWAP"] = round(vwap, 2)
        rows.append(r)
    daily = pd.DataFrame(rows, index=dts)
    daily["Change"] = daily["Close"].sub(daily["Close"].shift())
    daily["DayChg"] = daily["Close"].sub(daily["Open"])
    daily["Range"] = daily["High"].sub(daily["Low"])
    return daily


def aggregrate_bars_between(df: pd.DataFrame, tm_open: time, tm_close: time) -> pd.DataFrame:
    rows = []
    # find indexes of open & close bars
    ops = df.at_time(tm_open).index
    cls = df.at_time(tm_close).index
    for op, cl in zip(ops, cls):
        # slicing a dataframe by index uses an inclusive range
        acc = aggregate(df.loc[op:cl])
        rows.append(acc)
    return pd.DataFrame(rows)


def islastbar(d: datetime) -> bool:
    return (d.hour == 20 and (d.minute == 14 or d.minute == 59)) or (d.hour == 13 and d.minute == 29)


def isfirstbar(d: datetime) -> bool:
    return (d.hour == 22 and d.minute == 00) or (d.hour == 13 and d.minute == 30)


def hilo(df: pd.DataFrame, rev: float) -> None:
    hw = df.High.iloc[0]
    hwp = 0
    lw = df.Low.iloc[0]
    lwp = 0
    hf = False
    lf = False
    c = 0
    for i, r in df.iterrows():
        if r.High > hw:
            hw = r.High
            hwp = i
            hf = True
        elif hw - r.High > rev and hf:
            print(f"High {df.High.loc[hwp]:%.2f}")
            lf = False
            lw = r.Low
            lwp = i

        if r.Low < lw:
            lw = r.Low
            lwp = i
            lf = True
        elif r.Low - lw > rev and lf:
            #           print('Low  %.2f' % df.Low.loc[lwp:i])
            print(df.loc[lwp:i])
            hf = False
            hw = r.High
            hwp = i
            print(df.iloc[50:70])
            c = c + 1
            if c == 2:
                sys.exit(0)


def find_local_minima_with_threshold(data: pd.DataFrame, value_column: str, n: int, alpha: float, start_time: str = "14:30:00", end_time: str = "21:00:00") -> np.ndarray:
    """
    Identify local minima in a time series DataFrame with additional threshold criteria,
    only within a specified daily time window.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data indexed by pandas Timestamps.
    value_column : str
        Column name for the values to analyze.
    n : int
        Half window width; total window width = 2*n + 1.
    alpha : float
        Threshold ratio (e.g., 0.5 for 50%) for comparing right side and left side differences.
    start_time : str, optional
        Start time in HH:MM:SS (24-hour) format for filtering candidate minima. Default is "09:30:00".
    end_time : str, optional
        End time in HH:MM:SS (24-hour) format for filtering candidate minima. Default is "16:00:00".

    Returns:
    --------
    np.ndarray
        Array of timestamps where local minima with threshold criteria are found and within specified time range.
    """
    values = data[value_column]

    # Rolling minimum with center alignment
    rolling_min = values.rolling(window=2 * n + 1, center=True).min()

    # Candidate minima where rolling min equals center value
    candidates = values == rolling_min

    # Exclude edges without full window
    candidates[:n] = False
    candidates[-n:] = False

    candidate_indices = np.where(candidates)[0]

    results = []

    # filter out candidate indices outside time range
    time_mask = (data.index.time >= pd.to_datetime(start_time).time()) & (data.index.time < pd.to_datetime(end_time).time())
    candidate_indices = candidate_indices[time_mask[candidate_indices]]

    for i in candidate_indices:
        p = values.iloc[i]
        left_diff = values.iloc[i - n : i].max() - p
        right_diff = values.iloc[i + 1 : i + n + 1].max() - p

        if right_diff >= alpha * left_diff:
            results.append(data.index[i])

    return np.array(results)


# https://firstratedata.com/i/futures/ES
def fn1() -> None:
    column_names = ["TimeStamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv("d:\\esz19.txt", names=column_names, parse_dates=["TimeStamp"], index_col=["TimeStamp"])
    dfd = df.resample("1H").agg({"open": "first", "close": "last", "high": "max", "low": "min", "volume": "sum"})
    dfd = dfd[dfd.volume > 1000]
    print(dfd.tail(19))


def print_summary(df: pd.DataFrame) -> None:
    di = day_index(df)

    for i, r in di.iterrows():
        console.print(df[r["rth_first"] : r["rth_last"]]["volume"].median())

    console.print("\n--- Daily bars ---", style="yellow")
    df2 = aggregate_to_time_bars(df, di, "first", "last")
    export_daily(df2, "es-daily")
    console.print(df2)

    console.print("\n--- RTH bars ---", style="yellow")
    df_rth = aggregate_to_time_bars(df, di, "rth_first", "rth_last")
    export_daily(df_rth, "es-daily-rth")
    console.print(df_rth)

    console.print("\n--- 3LB ---", style="yellow")
    export_3lb(df_rth, make_filename("es-rth-3lb.csv"))


def previous_max(xs: pd.Series) -> list[pd.Timestamp]:
    ys = deque()
    for i, x in xs.items():
        while ys and x > ys[-1].value:
            ys.pop()
        if len(ys) == 0 or ys[-1].value != x:
            ys.append(Price(i, x))
    return [p.date for p in ys]


def previous_min(xs: pd.Series) -> list[pd.Timestamp]:
    ys = deque()  # deque of tuple[pd.Timestamp, float]
    for i, x in xs.items():
        while ys and x < ys[-1].value:
            ys.pop()
        if len(ys) == 0 or ys[-1].value != x:
            ys.append(Price(i, x))
    return [p.date for p in ys]


def select_with_gap(df: pd.DataFrame, xs: list[pd.Timestamp], n: int) -> pd.DataFrame:
    """return df filtered by list of timestamps"""
    df2 = df.loc[xs]
    ser = df2.index.to_series().diff().dt.total_seconds().div(60).fillna(0).astype(int)
    sel = ser > n
    sel.iat[0] = True
    return df.loc[ser[sel].index]


# standardise but use *100 so +1 std is 100
def normalise_as_perc(v, n=20):
    return (100 * (v - v.rolling(window=n).mean()) / v.rolling(window=n).std()).fillna(0).round(0).astype(int)


def test_find(df: pd.DataFrame, dt: str, n: int):
    """return n rows starting or ending with dt"""
    d = pd.to_datetime(dt, format="ISO8601")
    x = 1 if abs(n) < 2 else n

    # first slice is by datetime index and is inclusive
    return df[d:][:x] if x > 0 else df[:d][x:]


def test() -> None:
    dates = ["2022-09-02", "2022-09-06", "2022-09-07", "2022-09-08", "2022-09-09", "2022-09-12", "2022-09-13", "2022-09-14", "2022-09-15", "2022-09-16"]
    prices = [295.17, 293.05, 298.97, 300.52, 307.09, 310.74, 293.7, 296.03, 291.1, 289.32]

    df = pd.DataFrame({"price": prices}, index=pd.to_datetime(dates))
    console.print(test_find(df, "2022-09-08", 3))
    console.print(test_find(df, "2022-09-12", -3))


if __name__ == "__main__":
    # test()
    # p = Path("c:/temp/ultra")
    p = Path.home() / "Documents" / "data"
    df = load_overlapping_files(p, "nqu5*.csv")
    print_summary(df)
    di = day_index(df)
    row = di.iloc[-1]
    day = df[row["first"] : row["last"]]
    tms = previous_min(day["low"])
    lows = select_with_gap(df, tms, 9)
    console.print("\n--- lows", style="yellow")
    console.print(lows)
    tms = previous_max(day["high"])
    highs = select_with_gap(df, tms, 9)
    console.print("\n--- highs", style="yellow")
    console.print(highs)
    minima = find_local_minima_with_threshold(df, "low", 15, 0.5)
    console.print("\n--- local minima", style="yellow")
    console.print(df.loc[minima])
    df["voln"] = normalise_as_perc(df["volume"])
    df["strat"] = calc_strat(df)
    df["rng"] = ((df["high"] - df["low"]) * 4).astype(int)
    # print minima with rows before and after
    delta = pd.Timedelta(minutes=10)
    for p in range(-9, 0):
        df2 = df.loc[minima[p] - delta : minima[p] + delta]
        console.print(df2)
        input("Press Enter to continue...")

    # for i,r in day.iterrows():

    # df['vwap'] = calc_vwap(df)
    # exportNinja(df, make_filename('ES 09-22.Last.txt'))
    # exportMinVol(df, make_filename('es-minvol.csv'))
