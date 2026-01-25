#!/usr/bin/python3
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.signal import find_peaks

# import tsutils as ts
# df = ts.load_file(ts.make_filename('esu1 20210705.csv'))

# Time series utility functions

console = Console()


def find_initial_swing(s: pd.Series, perc_rev: float) -> tuple[int, int, int]:
    """return direction and start index and end index of the first incomplete swing."""
    hw = s.iloc[0]
    hwi = 0
    lw = s.iloc[0]
    lwi = 0
    for i in range(s.size):
        x = s.iat[i]
        if x > hw:
            hw = x
            hwi = i
        elif x < lw:
            lw = x
            lwi = i
        if pdiff(lw, hw, perc_rev):
            if lwi < hwi:
                return (1, lwi, hwi)
            return (-1, hwi, lwi)
    return (0, 0, 0)


def find_swings(s: pd.Series, perc_rev: float) -> list[int] | pd.DataFrame:
    """return df of swings. The final row is the current extreme of the in-progress swing"""
    dirn, start_index, end_index = find_initial_swing(s, perc_rev)
    xs = []
    if dirn == 0:
        return xs
    xs.append(start_index)
    extm = s.iloc[end_index]
    extm_index = end_index
    for i in range(end_index + 1, s.size):
        x = s.iat[i]
        if dirn == 1:
            if x > extm:
                extm = x
                extm_index = i
            #                print(f'new hi {extm_index} {extm}')
            elif pdiff(extm, x, perc_rev):
                #                    print(f'reversal {x}')
                xs.append(extm_index)
                dirn = -1
                extm = x
                extm_index = i
        else:
            if x < extm:
                extm = x
                extm_index = i
            #                print(f'new lo {extm_index} {extm}')
            elif pdiff(extm, x, perc_rev):
                #                    print(f'reversal {x}')
                xs.append(extm_index)
                dirn = 1
                extm = x
                extm_index = i
    xs.append(extm_index)  # store the last unconfirmed extreme
    day_index = np.array(xs)
    swing = s.iloc[day_index[:-1]]
    ends = s.iloc[day_index[1:]]
    df = pd.DataFrame({"start": swing, "end": ends.to_numpy(), "dtend": ends.index, "days": np.diff(day_index) + 1})
    df["change"] = ((df["end"] / df["start"] - 1.0) * 100.0).round(2)
    df["mae"] = [round(calculate_mae(s[x : y + 1]), 2) for x, y in zip(xs, xs[1:])]
    return df


def calculate_mae(s: pd.Series) -> float:
    """calculate the mae for the series"""
    if s.iloc[0] < s.iloc[-1]:  # uptrend
        mx = s.expanding().max()
        excursion = (s - mx) / mx
        return excursion.min() * 100.0
    mx = s.expanding().min()
    excursion = (s - mx) / mx
    return excursion.max() * 100.0


# return true if perc diff gt
def pdiff(s: float, e: float, p: float) -> bool:
    return 100 * abs(e / s - 1) >= p


# inclusive end
def aggregate(df: pd.DataFrame) -> dict[str, Any]:
    acc = {}
    for i, r in df.iterrows():
        acc = combine(acc, i, r, 1) if acc else single(i, r, 1)
    return acc


def aggregate_min_volume(df: pd.DataFrame, minvol: float | int) -> pd.DataFrame:
    rows = []
    acc = {}
    #    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    selector = df.index.to_series().diff() != timedelta(minutes=1)
    openbar = (df.index.minute == 0) & selector
    lastbar = selector.shift(-1, fill_value=True)
    eur_open = date(2021, 1, 1)
    rth_open = date(2021, 1, 1)
    for i, r in df.iterrows():
        if openbar.loc[i]:
            eur_open = i + timedelta(hours=8, minutes=59)
            rth_open = i + timedelta(hours=15, minutes=29)
        acc = combine(acc, i, r, 1) if acc else single(i, r, 1)
        if acc["volume"] >= minvol or lastbar.loc[i] or i == eur_open or i == rth_open:
            rows.append(acc)
            acc = None
    if acc:
        rows.append(acc)
    df2 = pd.DataFrame(rows)
    df2.set_index("date", inplace=True)
    return df2


def single(dt_fst: datetime, fst: pd.Series, period: int) -> dict[str, Any]:
    r = {}
    r["date"] = dt_fst
    r["dateCl"] = dt_fst + timedelta(minutes=period)
    r["open"] = fst["open"]
    r["high"] = fst["high"]
    r["low"] = fst["low"]
    r["close"] = fst["close"]
    r["volume"] = fst["volume"]
    r["vwap"] = fst["vwap"]
    if "ema" in fst:
        r["ema"] = fst["ema"]
    return r


def combine(acc: dict[str, Any], dt_snd: datetime, snd: pd.Series, period: int) -> dict[str, Any]:
    r = {}
    r["date"] = acc["date"]
    r["dateCl"] = dt_snd + timedelta(minutes=period)
    r["open"] = acc["open"]
    r["high"] = max(acc["high"], snd["high"])
    r["low"] = min(acc["low"], snd["low"])
    r["close"] = snd["close"]
    r["volume"] = acc["volume"] + snd["volume"]
    r["vwap"] = snd["vwap"]
    if "ema" in snd:
        r["ema"] = snd["ema"]
    return r


# def count_back(xs: pd.Series, i: int) -> int:
#     current = xs.iloc[i]
#     c = 0
#     for k in range(i - 1, -1, -1):
#         prev = xs.iloc[k]
#         if c > 0:
#             if current >= prev:
#                 c += 1
#             else:
#                 break
#         elif c < 0:
#             if current <= prev:
#                 c -= 1
#             else:
#                 break
#         else:
#             c = 1 if current >= prev else -1

#     return c


# def calc_hilo(ser: pd.Series) -> pd.Series:
#     cs = []
#     cs.append(0)
#     for i in range(1, ser.size):
#         cs.append(count_back(ser, i))
#     return pd.Series(cs, ser.index)


def count_prior(s: pd.Series, compare: Callable[[float, float], bool]) -> np.ndarray:
    """Counts elements prior to the current element based on a comparison function.

    Returns:
        A NumPy array where `result[i]` is the count of elements prior to `s[i]`
        that satisfy the `compare` condition.

    Example:
        count_prior(daily["close"], lambda x,y: x > y)
    """
    xs = s.values
    n = len(xs)
    result = np.zeros(n, dtype=int)
    stack = []
    for i, x in enumerate(xs):
        count = 0
        while stack and compare(x, xs[stack[-1]]):
            count += result[stack.pop()] + 1
        result[i] = count
        stack.append(i)
    return result


def calc_hilo(s: pd.Series) -> np.ndarray:
    hs = count_prior(s, lambda x, y: x > y)
    ls = count_prior(s, lambda x, y: x < y)
    return np.where(hs > ls, hs, -ls)


def day_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame with [first, last, rth_first, rth_last, duration] indexed by trade date.
    Trading days are found based on gaps in the m1 data.
    """
    # Parameters for RTH session
    rth_start_offset = pd.Timedelta(hours=15, minutes=30)
    rth_duration = pd.Timedelta(hours=6, minutes=29)

    # Find session boundaries
    first_bar_selector = df.index.to_series().diff() != pd.Timedelta(minutes=1)
    last_bar_selector = first_bar_selector.shift(-1, fill_value=True)

    # Build DataFrame of session boundaries
    idx = pd.DataFrame({"first": df.index[first_bar_selector], "last": df.index[last_bar_selector]})

    # Calculate RTH open and close
    rth_start = idx["first"] + rth_start_offset
    idx["rth_first"] = rth_start.mask(idx["last"] < rth_start)
    idx["rth_last"] = np.minimum(idx["last"], idx["rth_first"] + rth_duration)

    # Calculate session duration in minutes
    idx["duration"] = ((idx["last"] - idx["first"]).dt.total_seconds() / 60) + 1

    # Set trade date as the date of the last bar
    idx.index = idx["last"].dt.normalize()
    idx.index.name = "date"

    return idx


def create_day_summary(df: pd.DataFrame, di: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily summary statistics for GLOBEX and RTH sessions.

    Efficiently processes intraday data to extract key metrics for each trading day
    including GLOBEX highs/lows, RTH OHLC, extremes timing, and first hour statistics.

    Returns
    -------
    Daily summary with trade dates as index and columns:
    - glbx_high, glbx_low: GLOBEX session extremes
    - rth_open, rth_high, rth_low, close: RTH OHLC
    - rth_high_tm, rth_low_tm: timestamps of RTH extremes
    - rth_h1_high, rth_h1_low: first hour RTH extremes
    """

    # Create intervals for GLOBEX sessions (first to EU close)
    eu_close_times = pd.concat([di["last"], di["rth_first"] - pd.Timedelta(minutes=1)], axis=1).min(axis=1)
    glbx_intervals = pd.IntervalIndex.from_arrays(di["first"], eu_close_times, closed="both")

    # Create intervals for RTH sessions - only for days with valid RTH times
    rth_mask = di["rth_first"].notna()

    # Initialize session mapping series
    df_enhanced = df.copy()
    df_enhanced["glbx_session"] = pd.Series(dtype="object", index=df.index)
    df_enhanced["rth_session"] = pd.Series(dtype="object", index=df.index)
    df_enhanced["rth_h1_session"] = pd.Series(dtype="object", index=df.index)

    # Map GLOBEX sessions
    glbx_session_idx = glbx_intervals.get_indexer(df.index)
    df_enhanced["glbx_session"] = pd.Series(di.index[glbx_session_idx], index=df.index).where(glbx_session_idx >= 0)

    # Map RTH sessions only if there are valid RTH days
    if rth_mask.any():
        rth_intervals = pd.IntervalIndex.from_arrays(di.loc[rth_mask, "rth_first"], di.loc[rth_mask, "rth_last"], closed="both")
        rth_session_idx = rth_intervals.get_indexer(df.index)

        # Map valid RTH session indices back to original date index
        valid_rth_mask = rth_session_idx >= 0
        if valid_rth_mask.any():
            rth_dates = di.index[rth_mask].to_numpy()[rth_session_idx[valid_rth_mask]]
            df_enhanced.loc[valid_rth_mask, "rth_session"] = rth_dates

        # Map RTH first hour sessions
        rth_h1_end = di["rth_first"] + pd.Timedelta(minutes=59)
        rth_h1_intervals = pd.IntervalIndex.from_arrays(di.loc[rth_mask, "rth_first"], rth_h1_end[rth_mask], closed="both")
        rth_h1_session_idx = rth_h1_intervals.get_indexer(df.index)

        # Map valid RTH H1 session indices back to original date index
        valid_rth_h1_mask = rth_h1_session_idx >= 0
        if valid_rth_h1_mask.any():
            rth_h1_dates = di.index[rth_mask].to_numpy()[rth_h1_session_idx[valid_rth_h1_mask]]
            df_enhanced.loc[valid_rth_h1_mask, "rth_h1_session"] = rth_h1_dates

    # Vectorized aggregations
    # GLOBEX stats
    glbx_stats = df_enhanced[df_enhanced["glbx_session"].notna()].groupby("glbx_session").agg({"high": "max", "low": "min"}).rename(columns={"high": "glbx_high", "low": "glbx_low"})

    # RTH stats with extremes timing
    rth_data = df_enhanced[df_enhanced["rth_session"].notna()]
    if not rth_data.empty:
        rth_stats = rth_data.groupby("rth_session").agg({"open": "first", "high": ["max", "idxmax"], "low": ["min", "idxmin"], "close": "last"})
        # Flatten RTH column names
        rth_stats.columns = ["rth_open", "rth_high", "rth_high_tm", "rth_low", "rth_low_tm", "close"]
    else:
        # Create empty DataFrame with correct structure
        rth_stats = pd.DataFrame(columns=["rth_open", "rth_high", "rth_high_tm", "rth_low", "rth_low_tm", "close"], index=pd.Index([], name="rth_session"))

    # RTH first hour stats
    rth_h1_data = df_enhanced[df_enhanced["rth_h1_session"].notna()]
    if not rth_h1_data.empty:
        rth_h1_stats = rth_h1_data.groupby("rth_h1_session").agg({"high": "max", "low": "min"}).rename(columns={"high": "rth_h1_high", "low": "rth_h1_low"})
    else:
        # Create empty DataFrame with correct structure
        rth_h1_stats = pd.DataFrame(columns=["rth_h1_high", "rth_h1_low"], index=pd.Index([], name="rth_h1_session"))

    # Combine all statistics
    result = pd.concat([glbx_stats, rth_stats, rth_h1_stats], axis=1)
    result.index.name = "date"

    # Fill NaN values for sessions without RTH data
    result = result.reindex(di.index)

    return result[["glbx_high", "glbx_low", "rth_open", "rth_high", "rth_low", "close", "rth_high_tm", "rth_low_tm", "rth_h1_high", "rth_h1_low"]]


def print_day_summary(df_summary: pd.DataFrame) -> None:
    """Print RTH OHLC summary table using rich.

    Args:
        df_summary: Output from create_day_summary() with trade dates as index
    """
    table = Table(title="RTH Daily Summary", show_header=True, header_style="bold cyan")

    table.add_column("date", justify="left")
    table.add_column("open", justify="right")
    table.add_column("high", justify="right")
    table.add_column("low", justify="right")
    table.add_column("close", justify="right")

    for trade_date, row in df_summary.iterrows():
        date_str = trade_date.strftime("%Y-%m-%d")
        rth_open = f"{row['rth_open']:.2f}" if pd.notna(row['rth_open']) else "-"
        rth_high = f"{row['rth_high']:.2f}" if pd.notna(row['rth_high']) else "-"
        rth_low = f"{row['rth_low']:.2f}" if pd.notna(row['rth_low']) else "-"
        rth_close = f"{row['close']:.2f}" if pd.notna(row['close']) else "-"

        table.add_row(date_str, rth_open, rth_high, rth_low, rth_close)

    console.print(table)


def aggregate_to_time_bars(df: pd.DataFrame, di: pd.DataFrame, start_col: str, end_col: str, target_freq: str | None = None) -> pd.DataFrame:
    """
    Aggregate 1-minute OHLCV data into fixed time bars. If no target_freq is specified, aggregate to daily bars.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and OHLCV columns (open, high, low, close, volume).
        Should contain 1-minute frequency data. May contain optional columns like
        'ema' and 'vwap'.
    di : pd.DataFrame
        DataFrame with trade dates as index and columns 'start_time', 'end_time'
        defining the inclusive RTH periods for each trading session.
    target_freq : str, optional
        Target frequency for aggregation using pandas offset aliases
        (e.g., '5min' for 5-minute, '15T' for 15-minute, '1H' for hourly).
        If None, aggregates to daily bars (one bar per trading session).
    """
    # Create IntervalIndex for efficient RTH period lookup
    intervals = pd.IntervalIndex.from_arrays(di[start_col], di[end_col], closed="both")

    # Map timestamps to trading sessions using IntervalIndex
    session_idx = intervals.get_indexer(df.index)
    valid_mask = session_idx >= 0

    # Filter to RTH data and assign trade dates
    df_rth = df[valid_mask].copy()
    df_rth["date"] = di.index[session_idx[valid_mask]]

    # Build aggregation dictionary using comprehension
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg.update({col: "last" for col in ["ema", "vwap"] if col in df.columns})

    if target_freq:
        # Intraday aggregation - group by time buckets
        df_rth["date"] = df_rth.index.floor(target_freq)

    result = df_rth.groupby("date").agg(agg)

    if not target_freq:
        result["change"] = result["close"].diff()
        result["gap"] = result["open"] - result["close"].shift()
        result["day_chg"] = result["close"] - result["open"]
        result["range"] = result["high"] - result["low"]
        result["strat"] = calc_strat(result)

    return result


# return a row which aggregates bars between inclusive indexes
def aggregate_bars(df: pd.DataFrame, dt: datetime, s: pd.Timestamp, e: pd.Timestamp) -> dict[str, Any]:
    r = {}
    r["date"] = dt
    r["open"] = df.at[s, "open"]
    r["high"] = df.high[s:e].max()
    r["low"] = df.low[s:e].min()
    r["close"] = df.at[e, "close"]
    r["volume"] = df.volume[s:e].sum()
    # contract expiry is opening price of day so day has no volume
    vwap = 0 if r["volume"] <= 0 else np.average(df.wap[s:e], weights=df.volume[s:e])
    r["vwap"] = round(vwap, 2)
    return r


def pivot_cum_vol_avg_by_day(df: pd.DataFrame, di: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Create a pivot table with time of day as rows, trading dates as columns, and cum_vol_avg as values.
    Trading sessions start at 23:00 (bar_num = 0).

    Args:
        df: normal 1 min ohlcv data
        di: daily index

    Returns:
        DataFrame with time (HH:MM format) as index, trading dates as columns, cum_vol_avg as values
    """
    # create dataframe [day, first] trade date and datetime of first bar of day (typically 23:00)
    dr = di[["first"]].copy()
    dr["day"] = dr.index

    # remove the index in order to do the merge
    dc = df.copy()
    dc.reset_index(inplace=True)

    # merge the 2 sorted data frames - this will add the trade date to each row
    m = pd.merge_asof(dc, dr, left_on="Date", right_on="first", direction="backward")
    m.drop(columns=["first"], inplace=True)

    # number each bar and calc cumulative volume trade day
    m["cum_vol"] = m.groupby("day")["volume"].cumsum()
    m["bar_num"] = m.groupby("day").cumcount()

    # extract last bar of each n min period
    sampled = m.loc[m["bar_num"] % n == n - 1].copy()

    # 2. Extract the time part as a string (e.g., '09:29', '09:59')
    sampled["time"] = sampled["Date"].dt.strftime("%H:%M")

    pivot = sampled.pivot(index=["bar_num", "time"], columns="day", values="cum_vol")

    # 4. Sort by bar_num (which will also sort time in session order)
    pivot = pivot.sort_index(level="bar_num")

    # 5. Drop the bar_num level, keeping only time as the index
    pivot.index = pivot.index.droplevel("bar_num")

    # 6. Calculate the row-wise average, ignoring NaNs
    row_avg = pivot.mean(axis=1, skipna=True)

    # 7. Update each value to be the % of average volume for that row
    return (pivot.div(row_avg, axis=0) * 100).round(2)


def calc_ema(df: pd.DataFrame, length: int) -> pd.Series:
    """calculate ema of close prices."""
    return df["close"].ewm(span=length, adjust=False).mean().round(2)


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """calculate vwap anchored to each globex session open"""
    # 1) mark where the 1-minute gap occurs
    gap = df.index.to_series().diff().ne(pd.Timedelta(minutes=1))
    # 2) build a group label that increments at each gap
    grp = gap.cumsum()

    # 3) running numerator and denominator per group
    num = df["wap"] * df["volume"]
    cum_num = num.groupby(grp).cumsum()
    cum_vol = df["volume"].groupby(grp).cumsum()

    # 4) VWAP is running numerator / running volume
    return (cum_num / cum_vol).round(2)


def calc_strat(df: pd.DataFrame) -> pd.Series:
    """return a series categorising bar by its strat bar type 0 - inside, 1 up, 2 down, 3 outside"""
    hs = df["high"].diff().gt(0)
    ls = df["low"].diff().lt(0)
    return hs.astype(int) + ls * 2


def calc_standardized_volume(df: pd.DataFrame, n: int) -> pd.Series:
    # Calculate the normalized volume (z-score for volume)
    # ((volume - rolling_mean) / rolling_std) * 100, rounded to 0 decimal places
    nvol = (((df["volume"] - df["volume"].rolling(window=n).mean()) / df["volume"].rolling(window=n).std()) * 100).round()
    return nvol.fillna(0).astype(int)


def add_indicators(df: pd.DataFrame, ema_length: int = 88, nvol_window: int = 20) -> None:
    df["vwap"] = calc_vwap(df)
    df["ema"] = df["close"].ewm(span=ema_length, adjust=False).mean().round(2)
    df["strat"] = calc_strat(df)
    df["nvol"] = calc_standardized_volume(df, nvol_window)


def single_day(df: pd.DataFrame, di: pd.DataFrame, dt: str | date, rth_only: bool = True) -> pd.DataFrame:
    timestamp = pd.to_datetime(dt)
    return df[di.at[timestamp, "rth_first"] : di.at[timestamp, "rth_last"]] if rth_only else df[di.at[timestamp, "first"] : di.at[timestamp, "last"]]


def local_extremes(df: pd.DataFrame, n: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    hs = df["high"].rolling(2 * n + 1, center=True, min_periods=1).max()
    ls = df["low"].rolling(2 * n + 1, center=True, min_periods=1).min()
    return df[df["high"] == hs].index.to_list(), df[df["low"] == ls].index.to_list()


def around(df: pd.DataFrame, tmstmp: pd.Timestamp | str, n: int = 9) -> pd.DataFrame:
    """return a slice of df around a timestamp of 2n+2 minutes. If str assume hh:mm"""
    offset = pd.Timedelta(minutes=n)
    offset1 = pd.Timedelta(minutes=n + 1)
    if isinstance(tmstmp, str):
        tm = pd.Timestamp(tmstmp)
        return df.between_time((tm - offset).time(), (tm + offset1).time())

    return df.loc[tmstmp - offset : tmstmp + offset1]


def calc_atr(df: pd.DataFrame, n: int) -> pd.Series:
    rng = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    df2 = pd.DataFrame({"tm": df.index.time, "rng": rng}, index=rng.index)
    return df2.groupby("tm").rng.agg("mean")


def to_date(timestmp: pd.Timestamp) -> date:
    return timestmp.to_pydatetime().date()


def make_filename(fname: str) -> Path:
    return Path.home() / "Documents" / "data" / fname


def load_files_as_dict(p: Path, spec: str) -> dict[Path, pd.DataFrame]:
    """return a dict of paths and data_frames"""
    return {x: load_file(x) for x in sorted(p.glob(spec, case_sensitive=False))}


def load_files(p: Path, spec: str) -> pd.DataFrame | None:
    """load all files matching specified pattern and return a single dataframe"""
    d = load_files_as_dict(p, spec)
    if len(d):
        df = pd.concat(d.values())
        if not df.index.is_monotonic_increasing:
            raise ValueError("index not monotonic increasing")
        return df
    return None


# Path('c:/temp/ultra').
def load_overlapping_files(p: Path, spec: str) -> pd.DataFrame:
    """concatenates files into one dataframe skipping duplicate index entries. this only works when minute bars are complete because it picks the first occurence of a time"""
    comb = None
    for df in load_files_as_dict(p, spec).values():
        comb = df if comb is None else pd.concat([comb, df.loc[~df.index.isin(comb.index)]], axis=0, join="outer")
    if not comb.index.is_monotonic_increasing:
        raise ValueError("index not monotonic increasing")
    return comb


def load_file(fname: Path | str) -> pd.DataFrame:
    """load csv skipping first col and convert Date to index"""
    df = pd.read_csv(fname, parse_dates=["Date"], usecols=lambda col: not col.startswith("Unnamed"), index_col="Date")
    df.columns = df.columns.str.lower()
    print(f"loaded {fname} {df.shape[0]} {df.shape[1]}")
    return df


def save_df(df: pd.DataFrame, symbol: str) -> None:
    """save dataframe to csv in original format"""
    idx = day_index(df)
    print(idx)
    fout = make_filename(f"{symbol} {idx.index[0].strftime('%Y%m%d')}.csv")
    print(f"saving {fout}")
    # reverse the parsing operations so the saved csv is the same format
    # drop the first col which is 0,1,2 then make the index date time a standard col
    #    df2 = df.drop(df.columns[0], axis=1)
    df.reset_index(names="Date", inplace=True)
    df["Date"] = df["Date"].dt.strftime("%Y%m%d %H:%M:%S")
    df.to_csv(fout)


def create_volume_profile(df: pd.DataFrame, prominence: float | int = 40, smoothing_period: int = 1) -> pd.DataFrame:
    """return df of price,volume, peak flag"""
    mx = df["high"].max()
    mn = df["low"].min()

    def num_bins(hi: float, lo: float) -> int:
        return int((hi - lo) * 4 + 1)

    def to_bin(p: float) -> int:
        return int((p - mn) * 4)

    def to_price(b: int) -> float:
        return b / 4 + mn

    bins = num_bins(mx, mn)
    #    print(df.index[0], df.index[-1], mn, mx, bins)

    xs = np.zeros(bins)
    for i, r in df.iterrows():
        x = r["low"]
        y = r["high"]
        xs[to_bin(x) : to_bin(y) + 1] += r["volume"] / num_bins(y, x)

    if smoothing_period > 1:
        # sma smoothing
        sma_kernel = np.full(smoothing_period, 1 / smoothing_period)
        ys = np.convolve(xs, sma_kernel, "same")
    else:
        ys = xs

    peak_indicies, peak_info = find_peaks(ys, prominence=np.percentile(ys, prominence))
    # pprint(peak_indicies)
    # pprint(to_price(peak_indicies))
    # pprint(peak_info)
    mxs = np.zeros(bins, dtype=bool)
    mxs[peak_indicies] = True

    return pd.DataFrame({"volume": xs, "is_peak": mxs}, index=to_price(np.arange(bins)))


def display(df: pd.DataFrame) -> None:
    """
    Displays a DataFrame in a Rich table with conditional cell highlighting.

    Args:
        df_input (pd.DataFrame): The input DataFrame with a DatetimeIndex and expected columns.
    """
    title = f"ES 1 min {df.index[0].strftime('%d-%m')}"
    table = Table(title=title, title_style="yellow", show_header=True, header_style="bold cyan", box=None)

    # --- 1. Data Preparation ---
    df_display = df.copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate previous high and low for comparison
    # We do this on the original df_input to get correct previous values even for the first row of the slice
    # then select the relevant slice.
    df_temp_for_comparison = df[["high", "low"]].copy()
    df_temp_for_comparison["prev_high"] = df_temp_for_comparison["high"].shift(1)
    df_temp_for_comparison["prev_low"] = df_temp_for_comparison["low"].shift(1)

    # Add these shifted columns to our display slice
    df_display["prev_high"] = df_temp_for_comparison["prev_high"]
    df_display["prev_low"] = df_temp_for_comparison["prev_low"]
    df_display["rng"] = (df_display["high"] - df_display["low"]) * 4

    # Define columns to display and their headers for the Rich table
    # Order matters here for table.add_column and later for row_data.append
    columns_to_render = {"time": "time", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "vwap": "vwap", "ema": "ema", "strat": "strat", "rng": "rng", "nvol": "nvol"}

    for header in columns_to_render.values():
        if header in ["volume", "nvol"]:
            table.add_column(header, justify="right")
        else:
            table.add_column(header)

    # Color mapping for 'strat' column
    strat_colors = {0: "yellow", 1: "green", 2: "red", 3: "purple"}

    is_first_row = True
    # --- 2. Iterate and Build Table Rows ---
    for index, row in df_display.iterrows():
        row_data_styled = []
        if index.minute % 5 == 0 and not is_first_row:
            table.add_row(*[""] * len(columns_to_render))
        is_first_row = False
        # Time (hh:mm)
        time_str = index.strftime("%H:%M") if isinstance(index, pd.Timestamp) else str(index)
        row_data_styled.append(time_str)

        # Open
        row_data_styled.append(f"{row['open']:.2f}")  # Assuming 2 decimal places for price

        # High (conditional color)
        high_val = row["high"]
        prev_high_val = row["prev_high"]
        high_display = f"{high_val:.2f}"
        if pd.notna(prev_high_val) and high_val > prev_high_val:
            high_display = f"[green]{high_display}[/]"
        row_data_styled.append(high_display)

        # Low (conditional color)
        low_val = row["low"]
        prev_low_val = row["prev_low"]
        low_display = f"{low_val:.2f}"
        if pd.notna(prev_low_val) and low_val < prev_low_val:
            low_display = f"[red]{low_display}[/]"
        row_data_styled.append(low_display)

        # Close
        row_data_styled.append(f"{row['close']:.2f}")

        # Volume
        row_data_styled.append(str(int(row["volume"])))

        # VWAP
        row_data_styled.append(f"{row['vwap']:.2f}")  # VWAP often has more precision

        # EMA
        row_data_styled.append(f"{row['ema']:.2f}")  # EMA also

        # Strat (conditional color)
        strat_val = int(row["strat"])
        strat_display = str(strat_val)
        color = strat_colors.get(strat_val)  # Safely get color
        if color:
            strat_display = f"[{color}]{strat_display}[/]"
        row_data_styled.append(strat_display)

        row_data_styled.append(str(int(row["rng"])))
        # NVol (conditional color)
        nvol_val = int(row["nvol"])
        nvol_display = str(nvol_val)
        if nvol_val > 99:
            nvol_display = f"[yellow]{nvol_display}[/]"
        row_data_styled.append(nvol_display)

        table.add_row(*row_data_styled)

    console.print(table)


@dataclass
class Block:
    dt: datetime  # close dt of bar
    open: float
    close: float


def blocks_to_df(blks: list[Block]) -> pd.DataFrame:
    df = pd.DataFrame(map(asdict, blks))
    df.set_index("dt", inplace=True)
    return df


def calc_tlb(xs: pd.Series, n: int) -> list | tuple[pd.DataFrame, float]:
    """takes a series and a number of blocks for a reversal and returns a DF and the reversal price."""
    if len(xs) < 2:
        return []
    blks = [Block(xs.index[1], xs.iloc[0], xs.iloc[1])]
    # queue of last n+1 closes
    q = deque(xs[0:2], maxlen=n + 1)
    dirn = 1 if xs.iloc[1] > xs.iloc[0] else -1
    for dt, x in xs[2:].items():
        last_cl = q[-1]
        rev = q[0]
        if (dirn == 1 and x > last_cl) or (dirn == -1 and x < last_cl):
            blks.append(Block(dt, last_cl, x))
            q.append(x)
        elif (dirn == 1 and x < rev) or (dirn == -1 and x > rev):
            # print(f'rev {x} {q}')
            prev_cl = q[-2]
            blks.append(Block(dt, prev_cl, x))
            q.clear()
            q.append(prev_cl)
            q.append(x)
            dirn *= -1
    print(f"{'uptrend' if dirn == 1 else 'downtrend'} reversal price {q[0]}")

    return blocks_to_df(blks), q[0]


class LineBreak:
    def __init__(self, n: int):
        self.reversalBocksLength = n
        self.dirn = 0
        self.lines = deque(maxlen=n + 1)
        self.blocks = []

    def append(self, x: float, dt: pd.Timestamp | datetime) -> None:
        if len(self.lines) < self.reversalBocksLength:
            self._append_block(x, dt)
        else:
            high = max(self.lines)
            low = min(self.lines)
            if x > high or x < low:
                # if reversal add prior close to queue of lines
                if (x > high and self.dirn == -1) or (x < low and self.dirn == 1):
                    # print(f'reversal adding {self.lines[-2]}')
                    self.lines.append(self.lines[-2])
                self._append_block(x, dt)

    # def asDataFrame(self):
    #     return pd.DataFrame(self.blocks)

    # add closing price to lines queue and if there is at least 1 prior line add a block
    # update direction of the last block in self.dirn
    def _append_block(self, x: float, dt: pd.Timestamp | datetime) -> None:
        if len(self.lines) > 0:
            last = self.lines[-1]
            self.dirn = 1 if x > last else -1
            self.lines.append(x)
            block = {}
            block["date"] = dt
            block["open"] = last
            block["close"] = x
            block["dirn"] = self.dirn
            self.blocks.append(block)
        else:
            self.lines.append(x)

    # first block is wrong
    def test(self) -> None:
        cls = [135, 132, 128, 133, 130, 130, 132, 134, 139, 137, 145, 158, 147, 143, 150, 149, 160, 164, 167, 156, 165, 168, 171, 173, 169, 177, 180, 176, 170, 175, 179, 173, 170, 170, 168, 165, 171, 175, 179, 175]
        for c in cls:
            self.append(c)
            print(f"{c} {self.lines}")
        # df = self.asDataFrame()
        # print(df)
