#!/usr/bin/python3
import sys
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from tsutils import aggregate, aggregate_daily_bars, aggregate_min_volume, calc_tlb, day_index, load_overlapping_files, make_filename

console = Console()
Price = namedtuple("Price", ["date", "value"])


def export_daily(df, fname):
    dt = df.index[-1]
    s = make_filename(f"{fname}-{dt.year}{dt.month:02d}{dt.day:02d}.csv")
    print(f"exporting daily to {s}")
    df.to_csv(s)


def export_ninja(df, outfile):
    print(f"exporting in Ninja Trader format {outfile} {len(df)}")
    with open(outfile, "w") as f:
        for i, r in df.iterrows():
            s = f"{i.strftime('%Y%m%d %H%M%S')};{r['Open']:4.2f};{r['High']:4.2f};{r['Low']:4.2f};{r['Close']:4.2f};{r['Volume']}\n"
            f.write(s)


def export_min_vol(df, outfile):
    df2 = aggregate_min_volume(df, 2500)
    print(f"exporting minVol file {outfile} {len(df2)}")
    df2.to_csv(outfile)


def export_3lb(df, outfile):
    tlb, rev = calc_tlb(df["close"], 3)
    print(f"exporting 3lb file {outfile} {len(tlb)}")
    tlb.to_csv(outfile)


# create a new DF which aggregates bars between inclusive indexes
def aggregate_bars(df, idxs_start, idxs_end):
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


def aggregrate_bars_between(df, tm_open, tm_close):
    rows = []
    # find indexes of open & close bars
    ops = df.at_time(tm_open).index
    cls = df.at_time(tm_close).index
    for op, cl in zip(ops, cls):
        # slicing a dataframe by index uses an inclusive range
        acc = aggregate(df.loc[op:cl])
        rows.append(acc)
    return pd.DataFrame(rows)


def islastbar(d):
    return (d.hour == 20 and (d.minute == 14 or d.minute == 59)) or (d.hour == 13 and d.minute == 29)


def isfirstbar(d):
    return (d.hour == 22 and d.minute == 00) or (d.hour == 13 and d.minute == 30)


def hilo(df, rev):
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


# https://firstratedata.com/i/futures/ES
def fn1():
    column_names = ["TimeStamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv("d:\\esz19.txt", names=column_names, parse_dates=["TimeStamp"], index_col=["TimeStamp"])
    dfd = df.resample("1H").agg({"open": "first", "close": "last", "high": "max", "low": "min", "volume": "sum"})
    dfd = dfd[dfd.volume > 1000]
    print(dfd.tail(19))


def print_summary(df):
    di = day_index(df)

    for i, r in di.iterrows():
        console.print(df[r["rth_first"] : r["rth_last"]]["volume"].median())

    console.print("\n--- Daily bars ---", style="yellow")
    df2 = aggregate_daily_bars(df, di, "first", "last")
    export_daily(df2, "es-daily")
    console.print(df2)

    console.print("\n--- RTH bars ---", style="yellow")
    df_rth = aggregate_daily_bars(df, di, "rth_first", "rth_last")
    export_daily(df_rth, "es-daily-rth")
    console.print(df_rth)

    console.print("\n--- 3LB ---", style="yellow")
    export_3lb(df_rth, make_filename("es-rth-3lb.csv"))


def previous_max(xs: pd.Series) -> list[pd.Timestamp]:
    ys = deque()
    for i, x in xs.items():
        while len(ys) > 0 and x > ys[-1].value:
            ys.pop()
        if len(ys) == 0 or ys[-1].value != x:
            ys.append(Price(i, x))
    return [p.date for p in ys]


def previous_min(xs: pd.Series) -> list[pd.Timestamp]:
    ys = deque()  # deque of tuples pd.Timestamp, float
    for i, x in xs.items():
        while len(ys) > 0 and x < ys[-1].value:
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


def test_find(df, dt, n: int):
    """return n rows starting or ending with dt"""
    d = pd.to_datetime(dt, format="ISO8601")
    x = 1 if abs(n) < 2 else n

    # first slice is by datetime index and is inclusive
    return df[d:][:x] if x > 0 else df[:d][x:]


def test():
    dates = ["2022-09-02", "2022-09-06", "2022-09-07", "2022-09-08", "2022-09-09", "2022-09-12", "2022-09-13", "2022-09-14", "2022-09-15", "2022-09-16"]
    prices = [295.17, 293.05, 298.97, 300.52, 307.09, 310.74, 293.7, 296.03, 291.1, 289.32]

    df = pd.DataFrame({"price": prices}, index=pd.to_datetime(dates))
    console.print(test_find(df, "2022-09-08", 3))
    console.print(test_find(df, "2022-09-12", -3))


if __name__ == "__main__":
    # test()
    p = Path("c:/temp/ultra")
    # p = Path.home() / 'Documents' / 'data'
    df = load_overlapping_files(p, "zesh5*.csv")
    # print_summary(df)
    di = day_index(df)
    row = di.iloc[-1]
    day = df[row["rth_first"] : row["rth_last"]]
    tms = previous_min(day["low"])
    lows = select_with_gap(df, tms, 9)
    console.print("\n--- lows", style="yellow")
    console.print(lows)
    tms = previous_max(day["high"])
    highs = select_with_gap(df, tms, 9)
    console.print("\n--- highs", style="yellow")
    console.print(highs)

    # for i,r in day.iterrows():

    # df['vwap'] = calc_vwap(df)
    # exportNinja(df, make_filename('ES 09-22.Last.txt'))
    # exportMinVol(df, make_filename('es-minvol.csv'))
