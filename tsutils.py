#!/usr/bin/python3
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# import tsutils as ts
# df = ts.load_file(ts.make_filename('esu1 20210705.csv'))

# Time series utility functions


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


def count_back(xs: pd.Series, i: int) -> int:
    current = xs.iloc[i]
    c = 0
    for k in range(i - 1, -1, -1):
        prev = xs.iloc[k]
        if c > 0:
            if current >= prev:
                c += 1
            else:
                break
        elif c < 0:
            if current <= prev:
                c -= 1
            else:
                break
        else:
            c = 1 if current >= prev else -1

    return c


def calc_hilo(ser: pd.Series) -> pd.Series:
    cs = []
    cs.append(0)
    for i in range(1, ser.size):
        cs.append(count_back(ser, i))
    return pd.Series(cs, ser.index)


def day_index(df: pd.DataFrame) -> pd.DataFrame:
    """create a df [date, first, last, rth_first, rth_last] for each trading day based on gaps in index"""
    first_bar_selector = df.index.diff() != timedelta(minutes=1)
    last_bar_selector = np.roll(first_bar_selector, -1)
    # contiguous time ranges
    idx = pd.DataFrame({"first": df.index[first_bar_selector], "last": df.index[last_bar_selector]})
    # calculate rth start as offset from open
    rth_start = idx["first"] + timedelta(hours=15, minutes=30)
    # mask out rth_start if it is after last bar of day and calculate rth_end propogating NaT values
    idx["rth_first"] = rth_start.mask(idx["last"] < rth_start)
    idx["rth_last"] = np.minimum(idx["last"], idx["rth_first"] + timedelta(hours=6, minutes=29))
    idx["duration"] = ((idx["last"] - idx["first"]).dt.total_seconds()) / 60 + 1
    # assume trade date is date of last bar
    idx.set_index(pd.to_datetime(idx["last"].dt.date), inplace=True)
    idx.index.name = "date"
    return idx


def create_day_summary(df: pd.DataFrame, df_di: pd.DataFrame) -> pd.DataFrame:
    xs = []
    for i, r in df_di.iterrows():
        open_time = r["first"]
        rth_open = r["rth_first"]
        eu_close = min(r["last"], rth_open - pd.Timedelta(minutes=1))
        glbx_df = df[open_time:eu_close]
        rth_hi, rth_lo, rth_hi_tm, rth_lo_tm, rth_open_price, rth_close, rth_fhi, rth_flo = pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
        if not pd.isnull(rth_open):
            rth_df = df[rth_open : r["rth_last"]]
            rth_hi = rth_df["high"].max()
            rth_hi_tm = rth_df["high"].idxmax()
            rth_lo = rth_df["low"].min()
            rth_lo_tm = rth_df["low"].idxmin()
            rth_open_price = rth_df.iat[0, rth_df.columns.get_loc("open")]
            rth_close = rth_df.iat[-1, rth_df.columns.get_loc("close")]

            rth_h1_last = rth_open + pd.Timedelta(minutes=59)
            rth_h1_df = df[rth_open:rth_h1_last]
            rth_fhi = rth_h1_df["high"].max()
            rth_flo = rth_h1_df["low"].min()

        xs.append(
            {
                "date": i,
                "glbx_high": glbx_df["high"].max(),
                "glbx_low": glbx_df["low"].min(),
                "rth_open": rth_open_price,
                "rth_high": rth_hi,
                "rth_low": rth_lo,
                "close": rth_close,
                "rth_high_tm": rth_hi_tm,
                "rth_low_tm": rth_lo_tm,
                "rth_h1_high": rth_fhi,
                "rth_h1_low": rth_flo,
            }
        )

    day_summary_df = pd.DataFrame(xs)
    day_summary_df.set_index("date", inplace=True)
    return day_summary_df


# create a new DF which aggregates bars using a daily index
def aggregate_daily_bars(df: pd.DataFrame, daily_index: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    rows = []
    for i, r in daily_index.dropna(subset=[start_col, end_col]).iterrows():
        rows.append(aggregate_bars(df, pd.to_datetime(i), r[start_col], r[end_col]))

    daily = pd.DataFrame(rows)
    daily.set_index("date", inplace=True)
    daily["change"] = daily.close.diff()
    daily["gap"] = daily.open - daily.close.shift()
    daily["day_chg"] = daily.close - daily.open
    daily["range"] = daily.high - daily.low
    daily["strat"] = calc_strat(daily)
    return daily[daily.volume > 10000]


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


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    is_first_bar = df.index.to_series().diff() != timedelta(minutes=1)
    xs = []
    start = 0
    for i, r in df.iterrows():
        if is_first_bar.loc[i]:
            start = i
        v = np.average(df["wap"].loc[start:i], weights=df["volume"].loc[start:i])
        xs.append(round(v, 2))
    return pd.Series(xs, df.index)


def calc_strat(df: pd.DataFrame) -> pd.Series:
    """return a series categorising bar by its strat bar type 0 - inside, 1 up, 2 down, 3 outside"""
    hs = df["high"].diff().gt(0)
    ls = df["low"].diff().lt(0)
    return hs.astype(int) + ls * 2


def calc_atr(df: pd.DataFrame, n: int) -> pd.Series:
    rng = df.high.rolling(n).max() - df.low.rolling(n).min()
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
    df = pd.read_csv(
        fname,
        parse_dates=["Date"],
        usecols=lambda col: not col.startswith("Unnamed"),
        index_col="Date"
    )
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
