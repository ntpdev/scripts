#!/usr/bin/python3
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table

from tsutils import aggregate_to_time_bars, calc_vwap, day_index, load_file, load_files, load_files_as_dict, load_overlapping_files, save_m1_timeseries

console = Console()


def test_tick():
    # load the string tick into a pandas dataframe. make the column Date the index
    tick = """
,Date,Open,High,Low,Close,Volume,WAP,BarCount
0,20240724 14:30:00,-71.00,-63.00,-520.00,-402.00,0,0.000,31
1,20240724 14:31:00,-402.00,-340.00,-491.00,-379.00,0,0.000,31
2,20240724 14:32:00,-379.00,-303.00,-432.00,-380.00,0,0.000,31
3,20240724 14:33:00,-380.00,-195.00,-398.00,-215.00,0,0.000,31
4,20240724 14:34:00,-215.00,-48.00,-282.00,-48.00,0,0.000,31
5,20240724 14:35:00,-48.00,41.00,-169.00,-10.00,0,0.000,31
    """

    tick_df = pd.read_csv(StringIO(tick), index_col="Date", parse_dates=["Date"])
    tick_df.drop(columns=[tick_df.columns[0], "Volume", "WAP", "BarCount"], inplace=True)
    tick_df.columns = tick_df.columns.str.lower()

    # load the string futures into a pandas dataframe. make the column Date the index.
    futures = """
,Date,Open,High,Low,Close,Volume,WAP,BarCount
928,20240724 14:28:00,5405.00,5411.50,5404.25,5407.00,5314,5408.550,1515
929,20240724 14:29:00,5407.00,5407.75,5402.75,5404.25,4080,5405.425,1320
930,20240724 14:30:00,5404.50,5405.50,5398.00,5403.25,16809,5401.400,6437
931,20240724 14:31:00,5403.25,5404.25,5394.00,5398.50,10444,5397.850,4031
932,20240724 14:32:00,5398.50,5401.75,5396.50,5398.25,9685,5398.875,3545
933,20240724 14:33:00,5398.25,5400.75,5391.00,5391.25,9019,5394.875,3408
934,20240724 14:34:00,5391.25,5392.00,5386.75,5387.00,10123,5389.325,3358
935,20240724 14:35:00,5387.00,5393.25,5387.00,5389.00,11340,5390.125,4120
936,20240724 14:36:00,5389.25,5398.00,5387.50,5396.00,10767,5393.700,3706
937,20240724 14:37:00,5396.25,5401.00,5394.50,5397.00,11314,5397.475,3958
    """

    futures_df = pd.read_csv(StringIO(futures), index_col="Date", parse_dates=["Date"])
    futures_df.drop(columns=[futures_df.columns[0], "BarCount"], inplace=True)
    futures_df.columns = futures_df.columns.str.lower()

    # add tick cols matching on index
    futures_df["tick_high"] = tick_df["high"]
    futures_df["tick_low"] = tick_df["low"]

    # print the resulting dataframe
    console.print(futures_df)
    console.print(tick_df)


# create a new DF which aggregates bars between inclusive indexes
def aggregate_bars(df, idxs_start, idxs_end):
    rows = []
    dts = []
    for s, e in zip(idxs_start, idxs_end):
        dts.append(e.date())
        r = {}
        r["open"] = df.Open[s]
        r["high"] = df.High[s:e].max()
        r["low"] = df.Low[s:e].min()
        r["close"] = df.Close[e]
        r["volume"] = df.Volume[s:e].sum()
        vwap = np.average(df.WAP[s:e], weights=df.Volume[s:e])
        r["vwap"] = round(vwap, 2)
        rows.append(r)
    daily = pd.DataFrame(rows, index=dts)
    daily["change"] = daily["close"].sub(daily["close"].shift())
    daily["day_chg"] = daily["close"].sub(daily["open"])
    daily["range"] = daily["high"].sub(daily["low"])
    return daily


# return boolean index
def find_price_intersection(df, n, start, end):
    return (df["low"].iloc[start:end] <= n) & (df["high"].iloc[start:end] >= n)


def calc_hilo(df):
    idx_end = 2600
    hi_count = []
    lo_count = []
    hi_count.append(0)
    lo_count.append(0)
    for i in range(1, idx_end):
        current = df.High.iloc[i]
        ch = 0
        for k in range(i - 1, 0, -1):
            prev = df.High.iloc[k]
            if current > prev:
                ch = ch + 1
            elif current < prev:
                break

        current = df.Low.iloc[i]
        cl = 0
        for k in range(i - 1, 0, -1):
            prev = df.Low.iloc[k]
            if current < prev:
                cl = cl - 1
            elif current > prev:
                break

        #        print(f'{i:4d} {ch:4d} {cl:4d} {df.index[i]}  {df.High.iloc[i]:.2f} {df.Low.iloc[i]:.2f}')
        hi_count.append(ch)
        lo_count.append(cl)

    df["HiCount"] = pd.Series(hi_count, index=df.index[0:idx_end], dtype="Int32")
    df["LoCount"] = pd.Series(lo_count, index=df.index[0:idx_end], dtype="Int32")


def make_colour(hi, lo):
    return "green" if hi > abs(lo) else "red"


@dataclass
class RmseCompare:
    n: int
    rmse: float


def compare_emas():
    """compare 19 ema on M5 to emas on M1. Nearest is around 83-90 depends on data"""
    df = load_file("c:\\users\\niroo\\documents\\data\\ESM4 20240304.csv")
    a = df.close.resample("5T").first()
    # adjust=False is needed to match usual ema calc
    b = a.ewm(span=19, adjust=False).mean()
    dfm5 = pd.DataFrame({"close_m5": a, "ema_m5": b})

    xs = []
    for i in range(79, 99):
        df["ema"] = df.close.ewm(span=i, adjust=False).mean()
        df2 = df.merge(dfm5, how="inner", left_index=True, right_index=True)
        rmse = ((df2.ema - df2.ema_m5) ** 2).mean() ** 0.5
        xs.append(RmseCompare(i, rmse))

    df_rmse = pd.DataFrame(xs)
    df_rmse.sort_values(by="rmse", ascending=True, inplace=True)
    print(df_rmse)


def main():
    print(f"Hello world {datetime.now()}")
    # df = load_files('/media/niroo/ULTRA/esh1*')
    df = load_files("esm1*.csv")
    df["VWAP"] = calc_vwap(df)
    calc_hilo(df)
    print("--- RTH bars ---")

    cols = []
    idx_end = 2600
    c = "grey"
    cols.append(c)
    for i in range(1, idx_end):
        hi = df.HiCount.iloc[i]
        lo = df.LoCount.iloc[i]
        if hi > 19:
            c = "green"
        elif lo < -19:
            c = "red"
        cols.append(c)
    # df['Colour'] = pd.Series(cols, index=df.index[0:200], dtype='Int32')
    df["Colour"] = pd.Series(cols, index=df.index[0:idx_end])

    print(df.iloc[80:120])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index[0:idx_end], y=df.High.iloc[0:idx_end] - df.Low.iloc[0:idx_end], base=df.Low.iloc[0:idx_end], marker=dict(color=df.Colour.iloc[0:idx_end])))
    #    color_discrete_map={ '0' : 'blue' }))
    # fig.show()


def plot_hl_times(df, daily_index, start_col, end_col, period=30):
    rows = []

    def bar(x, y):
        return (x - y).seconds // (period * 60)  # find 15 min block

    for i, r in daily_index.dropna(subset=[start_col, end_col]).iterrows():
        day = df[r[start_col] : r[end_col]]
        idx_hi = day["high"].idxmax()
        idx_lo = day["low"].idxmin()
        start = day.index[0]
        rows.append({"high_tm": idx_hi, "high": day.at[idx_hi, "high"], "low_tm": idx_lo, "low": day.at[idx_lo, "low"], "x": bar(idx_hi, start), "y": bar(idx_lo, start)})

    df2 = pd.DataFrame(rows)
    # hist, bin_edges = np.histogram(df2['x'].to_numpy(), bins=13, range=(0,13))
    all_bars = np.concatenate((df2["x"].to_numpy(), df2["y"].to_numpy()))
    counts = np.bincount(all_bars)
    counts_perc = 100 * counts / np.sum(counts)
    # max_bars = 390 // period
    fig = go.Figure()
    fig.add_trace(go.Bar(y=counts_perc))
    # fig.add_trace(go.Histogram(x=df2['y'], nbinsx=max_bars, marker_color='red'))
    fig.show()
    return df2


def calculate_interval_range(df, di):
    """
    Calculate ranges

    Args:
        df: DataFrame with 1 minute datetime index and OHLCV columns
        di: Day index DataFrame with trade dates as index and timestamp columns

    Returns:
        DataFrame with daily ranges indexed by trade date
    """
    data = []

    for trade_date, row in di.iterrows():
        first = row["first"]
        last = row["last"]
        rth_first = row["rth_first"]

        # initialize ranges
        gl_range = None
        h1_range = None

        # Globex range: first -> rth_first - 1min (or first -> last)
        if pd.notna(rth_first):
            globex_end = rth_first - pd.Timedelta(minutes=1)
            globex_slice = df.loc[first:globex_end]
        else:
            globex_slice = df.loc[first:last]

        if not globex_slice.empty:
            gl_range = globex_slice["high"].max() - globex_slice["low"].min()

        # First hour range: rth_first -> rth_first + 59min
        if pd.notna(rth_first):
            first_hour_end = rth_first + pd.Timedelta(minutes=59)
            h1_slice = df.loc[rth_first:first_hour_end]

            h1_range = h1_slice["high"].max() - h1_slice["low"].min()

        data.append({"date": trade_date, "globex": gl_range, "first_hour": h1_range})

    return pd.DataFrame(data).set_index("date")


def print_range_analysis(df, di, rth_df, daily_df, trading_days=10):
    """
    Create a range analysis table for the last N trading days using Rich console

    Args:
        rth_df: DataFrame with 'range' column and datetime index (existing RTH data)
        daily_df: DataFrame with 'range' column and datetime index (existing daily data)
        minute_df: DataFrame with 1-minute OHLCV data and datetime index
        trading_days: Number of trading days to analyze (default 10)
    """
    # Calculate first hour ranges from minute data
    interval_ranges = calculate_interval_range(df, di)

    # Get the last N trading days (rows) from each dataset
    rth_recent = rth_df["range"].tail(trading_days)
    daily_recent = daily_df["range"].tail(trading_days)
    globex_recent = interval_ranges["globex"].tail(trading_days)
    first_hour_recent = interval_ranges["first_hour"].tail(trading_days)

    # Calculate statistics
    def calc_stats(series):
        if len(series) == 0:
            return {"min": None, "max": None, "median": None, "last": None}
        return {"min": series.min(), "max": series.max(), "median": series.median(), "last": series.iloc[-1]}

    # Prepare period names and their stats in a list of tuples
    periods = [
        ("rth", calc_stats(rth_recent)),
        ("daily", calc_stats(daily_recent)),
        ("globex", calc_stats(globex_recent)),
        ("1st hour", calc_stats(first_hour_recent)),
    ]

    # Create Rich table
    table = Table(title=f"{trading_days} trading day range analysis", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("period", style="cyan", no_wrap=True)
    table.add_column("min", justify="right", style="yellow")
    table.add_column("max", justify="right", style="yellow")
    table.add_column("median", justify="right", style="yellow")
    table.add_column("last", justify="right", style="yellow")

    # Format number function
    def format_num(val):
        return f"{val:.2f}" if val is not None else "N/A"

    # Add rows in a loop
    for period_name, stats in periods:
        table.add_row(period_name, format_num(stats["min"]), format_num(stats["max"]), format_num(stats["median"]), format_num(stats["last"]))

    # Print the table
    console.print(table)

    # Print additional info
    console.print(f"\nAnalysis period: Last {trading_days} trading days")
    console.print(f"First hour period: 14:30-15:29 (inclusive)")
    console.print(f"First hour trading days available: {len(first_hour_recent)}")

    return table


def print_summary(df):
    di = day_index(df)
    console.print("\n--- Day index ---", style="yellow")
    console.print(di)

    console.print("\n--- Daily bars ---", style="yellow")
    daily = aggregate_to_time_bars(df, di, "first", "last")
    console.print(daily.iloc[-15:], style="cyan")
    console.print(f"range min,median,max = {daily['range'].min():.2f} {daily['range'].median():.2f} {daily['range'].max():.2f}", style="green")

    console.print("\n--- RTH bars ---", style="yellow")
    rth = aggregate_to_time_bars(df, di, "rth_first", "rth_last")
    console.print(rth.iloc[-15:], style="cyan")
    # range analysis
    print_range_analysis(df, di, rth, daily)
    console.print(f"range min,median,max = {rth['range'].min():.2f} {rth['range'].median():.2f} {rth['range'].max():.2f}", style="green")
    if rth.shape[0] > 10:
        console.print("last 10 days only", style="yellow")
        s = rth.iloc[-10:]
        console.print(f"range min,median,max = {s['range'].min():.2f} {s['range'].median():.2f} {s['range'].max():.2f}", style="green")
        console.print(f"change total, rth-only = {s['change'].sum():.2f} {s['day_chg'].sum():.2f}", style="green")
    # df2 = plot_hl_times(df, di, 'rth_first', 'rth_last', 15)
    # console.print(df2, style='cyan')


def whole_day_concat(path: Path, fspec: str, fnout: str):
    """combines all files in fspec into one file. takes whole days only"""
    dfs = load_files_as_dict(path, fspec)
    hw = pd.Timestamp("2020-01-01")
    comb = None
    for fn, df in dfs.items():
        di = day_index(df)
        for i in range(di.shape[0]):
            start = di.iloc[i, 0]
            end = di.iloc[i, 1]
            if start > hw:
                day = df[(df.index >= start) & (df.index <= end)]
                rows = day.shape[0]
                # 1380 only correct when no daylight savings change
                # for stocks bars=390 and some days might be partial due to half-day holidays
                # if rows == 390:
                if rows == 1380:
                    print(f"{fn} {start} {end} {rows} {hw}")
                    hw = end
                    comb = day if comb is None else pd.concat([comb, day], axis=0, join="outer")
                else:
                    print(f"--skipping incomplete {fn} {start} {end} {rows} ")
            else:
                print(f"--skipping overlap {fn} {start} {end} {hw}")

    if comb is not None:
        save_m1_timeseries(comb, fnout)
        comb["Date"] = pd.to_datetime(comb["Date"])
        comb2 = comb.set_index(comb["Date"])
        print_summary(comb2)


@dataclass
class Fileinfo:
    fname: str
    days: int
    start: pd.Timestamp
    end: pd.Timestamp

    def __repr__(self):
        return f"Fileinfo(fname='{self.fname}', days={self.days:02d}, start='{self.start}', end='{self.end})'"


def check_overlap(p, spec):
    console.print("\n--- checking overlap", style="yellow")
    dfs = load_files_as_dict(p, spec)
    xs = []
    for path, df in dfs.items():
        di = day_index(df)
        xs.append(Fileinfo(path.name, len(di), di.iloc[0]["first"], di.iloc[-1]["last"]))
    console.print(xs)
    
    # Convert to list to maintain order with file paths
    df_list = list(dfs.items())
    
    for i in range(len(xs) - 1):
        if xs[i].end >= xs[i + 1].start:
            console.print(f"overlap found fileinfo {i} {xs[i].fname} and {i + 1} {xs[i + 1].fname}", style="red")
            console.print(f"from {xs[i + 1].start} to {xs[i].end}")
            
            # Extract overlapping data from both files
            _, df_i = df_list[i]
            _, df_i_plus_1 = df_list[i + 1]
            
            overlap_start = xs[i + 1].start
            overlap_end = xs[i].end
            
            # Get overlapping rows from each DataFrame
            overlap_i = df_i.loc[overlap_start:overlap_end]
            overlap_i_plus_1 = df_i_plus_1.loc[overlap_start:overlap_end]
            
            # Find common index values
            common_idx = overlap_i.index.intersection(overlap_i_plus_1.index)
            
            if len(common_idx) > 0:
                # Compare rows with matching timestamps
                df1_common = overlap_i.loc[common_idx].sort_index()
                df2_common = overlap_i_plus_1.loc[common_idx].sort_index()
                
                are_identical = df1_common.equals(df2_common)
                
                if are_identical:
                    console.print(f"  ✓ {len(common_idx)} overlapping rows are IDENTICAL", style="green")
                else:
                    console.print(f"  ✗ {len(common_idx)} overlapping rows contain DIFFERENCES", style="red")
                    
                    # Find which rows differ
                    comparison = df1_common.compare(df2_common)
                    if not comparison.empty:
                        console.print(f"  Differences found in {len(comparison)} rows:")
                        console.print(comparison)
            else:
                console.print(f"  No common timestamps in overlap range", style="yellow")
            
            if xs[i + 1].end <= xs[i].end:
                console.print(f"file {i + 1} {xs[i + 1].fname} not needed as contained", style="red")


def print_combined_summary(p: Path, spec: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    console.print("\n--- combined summary", style="yellow")
    df = load_overlapping_files(p, spec)
    di = day_index(df)
    console.print(di.head(), style="cyan")
    
    # Vectorized approach: assign trade date to each bar based on session boundaries
    df_with_trade_date = df.copy()
    
    # Use searchsorted on 'first' timestamps to find which session each bar belongs to
    # side='right' means bars >= session start belong to that session
    session_indices = di['first'].searchsorted(df.index, side='right') - 1
    
    # Clip indices to valid range
    session_indices = np.clip(session_indices, 0, len(di) - 1)
    
    # Map session indices to trade dates
    df_with_trade_date['trade_date'] = di.index[session_indices]
    
    # Group by trade date and calculate volume and bar count
    daily_stats = df_with_trade_date.groupby('trade_date').agg({
        'volume': 'sum',
        'trade_date': 'size'  # Count of bars
    }).rename(columns={'trade_date': 'bars'})
    
    # Create summary table
    table = Table(title="Combined Data Summary", show_header=True, header_style="bold magenta")
    table.add_column("Trade Date", style="cyan", no_wrap=True)
    table.add_column("Bars", justify="right", style="blue")
    table.add_column("Volume", justify="right", style="green")
    table.add_column("Duration (min)", justify="right", style="yellow")
    
    for trade_date, row in di.iterrows():
        stats = daily_stats.loc[trade_date] if trade_date in daily_stats.index else None
        bars = int(stats['bars']) if stats is not None else 0
        volume = int(stats['volume']) if stats is not None else 0
        duration = int(row['duration'])
        table.add_row(
            trade_date.strftime('%Y-%m-%d'),
            f"{bars:,}",
            f"{volume:,.0f}",
            f"{duration:,}"
        )
    
    # Add totals row
    table.add_section()
    table.add_row(
        "TOTAL",
        f"{daily_stats['bars'].sum():,.0f}",
        f"{daily_stats['volume'].sum():,.0f}",
        f"{di['duration'].sum():,.0f}",
        style="bold"
    )
    
    console.print(table)
    console.print(f"\nTotal trading days: {len(di)}")
    return df, di


if __name__ == "__main__":
    # whole_day_concat(Path("c:/temp/z"), 'esu5*.csv', 'zesu5')
    # test_tick()
    check_overlap(Path.home() / "Documents" / "data", "esz5*")
    df, di = print_combined_summary(Path.home() / "Documents" / "data", "esz5*")
    # s = di.at["2025-09-08", "first"]
    # e = di.at["2025-12-18", "last"]
    # save_m1_timeseries(df[s:e], "zESZ5-combined")
    # check_overlap(Path("c:/temp/z"), "esu5*.csv")
    # df_es = load_overlapping_files(Path("c:/temp/ultra"), "esu5*.csv")
    # df_es = load_overlapping_files(Path.home() / "Documents" / "data", "esh5*.csv")
    # print_summary(df_es)
    # compare_emas()
    # df_tick = simple_concat('ztick-nyse*.csv', 'x')
    # di = day_index(df_tick)
