#!/usr/bin/python3
import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint

import tsutils as ts
from mdbutils import load_price_history

console = Console()
STRAT_BAR_COLOUR: str = "strat"


@dataclass
class MinVolDay:
    trade_dt: date
    start_tm: datetime
    start_bar: int
    eu_start_bar: int
    eu_end_bar: int
    rth_start_bar: int
    rth_end_bar: int


@dataclass
class PlotData:
    """Container for all data required by the volume bar plotting function."""

    df_min_vol: pd.DataFrame
    day_index: pd.DataFrame
    day_summary: pd.DataFrame
    skip_first: bool


def samp():
    open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    dates = [datetime(year=2013, month=10, day=10), datetime(year=2013, month=11, day=10), datetime(year=2013, month=12, day=10), datetime(year=2014, month=1, day=10), datetime(year=2014, month=2, day=10)]

    fig = go.Figure(data=[go.Candlestick(x=dates, open=open_data, high=high_data, low=low_data, close=close_data)])

    fig.show()


def samp_3lb():
    df = pd.read_csv("d:/3lb.csv", delimiter="\\s+", converters={"date": lambda e: datetime.strptime(e, "%Y-%m-%d")})
    colours = df["dirn"].map({-1: "red", 1: "green"})
    xs = df["date"].dt.strftime("%m-%d")
    fig = go.Figure(data=[go.Bar(x=xs, y=df["close"] - df["open"], base=df["open"], marker=dict(color=colours))])
    fig.update_xaxes(type="category")
    fig.show()


def plot_3lb(fname):
    df = pd.read_csv(ts.make_filename(fname), parse_dates=["dt"], index_col=0)
    df["colour"] = np.where(df["close"] > df["open"], "green", "red")
    df["text"] = df.apply(lambda row: f"{row['open']}<br>{row['close']}" if row["colour"] == "red" else f"{row['close']}<br>{row['open']}", axis=1)
    df["xlabel"] = df.index.strftime("%d %b")
    fig = go.Figure(data=[go.Bar(x=df["xlabel"], y=df["close"] - df["open"], base=df["open"], marker=dict(color=df["colour"]), text=df["text"])])
    fig.show()


def _create_plot_data(df: pd.DataFrame, min_vol: int) -> PlotData:
    """Transform raw m1 data into structures needed for plotting.

    Args:
        df: Raw 1-minute OHLCV data with vwap and ema columns
        min_vol: Minimum volume threshold for bar aggregation
    """
    day_index = ts.day_index(df)
    day_summary = ts.create_day_summary(df, day_index)
    num_days = day_index.shape[0]
    ts.print_day_summary(day_summary)
    # Skip first day if we have multiple days (first day used for prior hi-lo)
    skip_first = num_days > 1
    if skip_first:
        slice_df = df[day_index.iat[1, 0] : day_index.iat[num_days - 1, 1]]
    else:
        slice_df = df

    df_min_vol = ts.aggregate_min_volume(slice_df, min_vol)

    return PlotData(
        df_min_vol=df_min_vol,
        day_summary=day_summary,
        day_index=day_index,
        skip_first=skip_first,
    )


#        color="LightSeaGreen",
def draw_daily_lines(df, fig, tms, idxs):
    for op, cl in idxs:
        #        print(f'op {tms.iloc[op]} cl {tms.iloc[cl]}')
        fig.add_vline(x=tms.iloc[op], line_width=1, line_dash="dash", line_color="blue")
        fig.add_vline(x=tms.iloc[cl], line_width=1, line_dash="dash", line_color="grey")
        y = df.Open.iloc[op]
        fig.add_shape(type="line", x0=tms.iloc[op], y0=y, x1=tms.iloc[cl], y1=y, line=dict(color="LightSeaGreen", dash="dot"))


def high_lows(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Return tuple of (local highs, local lows) as Series of original values with local extrema."""

    def find_extrema(values, rolling) -> pd.Series:
        local_extreme = values[values == rolling]
        non_adjacent = local_extreme.diff().ne(0)
        return values.loc[non_adjacent[non_adjacent].index]

    high_col = df["high"]
    low_col = df["low"]

    rolling_high = high_col.rolling(window, center=True).max()
    rolling_low = low_col.rolling(window, center=True).min()

    return find_extrema(high_col, rolling_high), find_extrema(low_col, rolling_low)


def add_hilo_labels(df, fig) -> None:
    hs, ls = high_lows(df, 21)

    fig.add_trace(go.Scatter(x=[tm for tm in df.loc[hs.index].tm], y=hs.add(1), text=[f"{p:.2f}" for p in df.loc[hs.index].high], mode="text", textposition="top center", name="local high"))
    fig.add_trace(go.Scatter(x=[tm for tm in df.loc[ls.index].tm], y=ls.sub(1), text=[f"{p:.2f}" for p in df.loc[ls.index].low], mode="text", textposition="bottom center", name="local low"))


def bar_containing(df, dt):
    return (df["date"] <= dt) & (df["dateCl"] > dt)


# return a high and low range to nearest multiple of n
def make_yrange(df, op, cl, n):
    hi = df["high"][op:cl].max() + n // 2
    lo = df["low"][op:cl].min() - n // 2
    return (lo // n) * n, ((hi // n) + 1) * n


# pair of start_index:end_index suitable for use with iloc[s:e]
def make_day_index(df):
    # filter by hour > 21 since holidays can have low volume
    is_first_bar = (df.index.diff().fillna(pd.Timedelta(hours=1)) > timedelta(minutes=59)) & (df.index.hour > 21)
    xs = df[is_first_bar].index.to_list()
    # add index after final bar
    xs.append(df.shape[0])
    return [(op, cl) for op, cl in zip(xs, xs[1:])]


# return a list of tuples of the op,cl indexes
def make_rth_index(df, day_index):
    is_first_bar = (df.index.diff().fillna(pd.Timedelta(hours=1)) > timedelta(minutes=59)) & (df.index.hour > 21)
    rth_opens = df[is_first_bar].index.map(lambda e: e + np.timedelta64(930 - e.minute, "m"))
    rth_closes = df[is_first_bar].index.map(lambda e: e + np.timedelta64(1320 - e.minute, "m"))

    # select rows matching time, convert index to a normal col, add col which is date
    ops = df[df.index.isin(rth_opens)]
    ops2 = ops.reset_index()
    ops2["dt"] = ops2["date"].dt.date

    cls = df[df.index.isin(rth_closes)]
    cls2 = cls.reset_index()
    cls2["dt"] = cls2["date"].dt.date

    # join dfs on date ie include only days that have open+close
    mrg = pd.merge(ops2, cls2, how="inner", on="dt")
    return [(x, y) for x, y in zip(mrg["date_x"], mrg["date_y"])]


def plot_minvol_price_chart(data: PlotData) -> None:
    """Plot a minimum volume price chart with session markers and hi-lo labels."""
    df_min_vol = data.df_min_vol
    day_summary_df = data.day_summary
    idx = data.day_index
    skip_first = data.skip_first

    # Create X-axis labels
    tm = df_min_vol.index.strftime("%d/%m %H:%M")

    # Create figure with colored bars
    fig = color_bars(df_min_vol, tm, STRAT_BAR_COLOUR)
    add_hilo_labels(df_min_vol, fig)

    # Create min-vol day index for session markers
    mvds = create_min_vol_index(df_min_vol, idx)

    def add_h_line(x_start, x_end, y_value, prefix="", color="Gray", dash="dot"):
        fig.add_shape(
            type="line",
            x0=x_start,
            y0=y_value,
            x1=x_end,
            y1=y_value,
            line=dict(color=color, dash=dash),
        )
        if prefix:
            label = f"{prefix} {y_value:.2f}"
            fig.add_annotation(text=label, x=x_end, y=y_value, showarrow=False)

    for i in mvds[1:] if skip_first else mvds:
        eu_open = df_min_vol.at[df_min_vol.index[i.eu_start_bar], "open"]
        add_h_line(tm[i.eu_start_bar], tm[i.eu_end_bar], eu_open, "", color="LightSeaGreen")

        xstart = tm[i.start_bar]
        xend = tm[i.rth_end_bar]

        if i.rth_start_bar > 0:
            xstart_rth = tm[i.rth_start_bar]
            fig.add_vline(x=xstart_rth, line_width=1, line_dash="dash", line_color="blue")

            if i.rth_end_bar > 0:
                fig.add_vline(x=xend, line_width=1, line_dash="dash", line_color="blue")

            rth_open = day_summary_df.at[i.trade_dt, "rth_open"]
            add_h_line(xstart_rth, xend, rth_open, "open", color="LightSeaGreen")

            glbx_hi = day_summary_df.at[i.trade_dt, "glbx_high"]
            add_h_line(xstart_rth, xend, glbx_hi, "glbx hi")

            glbx_lo = day_summary_df.at[i.trade_dt, "glbx_low"]
            add_h_line(xstart_rth, xend, glbx_lo, "glbx lo")

        # Add first hour hi-lo
        h1_hi = day_summary_df.at[i.trade_dt, "rth_h1_high"]
        if pd.notna(h1_hi):
            xstart_rth = tm[i.rth_start_bar]
            add_h_line(xstart_rth, xend, h1_hi, "h1 h")

            h1_lo = day_summary_df.at[i.trade_dt, "rth_h1_low"]
            add_h_line(xstart_rth, xend, h1_lo, "h1 l")

        # Add previous day rth hi-lo-close
        ix = day_summary_df.index.searchsorted(i.trade_dt)
        if ix > 0:
            prev_day = day_summary_df.iloc[ix - 1]
            add_h_line(xstart, xend, prev_day.rth_low, "yl", color="chocolate")
            add_h_line(xstart, xend, prev_day.rth_high, "yh", color="chocolate")
            add_h_line(xstart, xend, prev_day.close, "cl", color="cyan")

    fig.show()


def _make_date_range(idx: pd.Index, dt_str: str, n: int) -> tuple[int, int]:
    """Resolve dt and n to inclusive positional index range.

    Args:
        idx: DatetimeIndex of trade dates
        dt_str: Index value (e.g., "-1") or date string (e.g., "20250615")
        n: Number of days. Positive means dt is start, negative means dt is end.
           n=0 is treated as n=1.

    Returns:
        Tuple of (start_pos, end_pos) as inclusive indexes into idx
    """
    length = len(idx)
    pos = None

    # Try as integer index
    try:
        i = int(dt_str)
        if -length <= i < length:
            pos = i if i >= 0 else length + i
    except ValueError:
        pass

    # Try as date string if not resolved
    if pos is None:
        try:
            target = pd.to_datetime(dt_str)
            pos = idx.searchsorted(target, side="right") - 1
            pos = max(0, pos)
        except ValueError:
            pos = length - 1  # Default to last

    # Normalize n=0 to n=1
    if n == 0:
        n = 1

    # Calculate range based on sign of n
    if n > 0:
        # dt is start, range extends forward
        start_pos = pos
        end_pos = min(length - 1, pos + n - 1)
    else:
        # dt is end, range extends backward
        start_pos = max(0, pos + n + 1)
        end_pos = pos
    breakpoint()
    return start_pos, end_pos


def _slice_trading_days(df: pd.DataFrame, dt: str, n: int) -> pd.DataFrame:
    """Slice trading days from df including prior day for reference.

    Args:
        df: Full 1-minute OHLCV data
        dt: Date string or index value
        n: Number of days. Positive: dt is start. Negative: dt is end.

    Returns:
        Sliced DataFrame with requested days + 1 prior day for reference lines
    """
    idx = ts.day_index(df)
    console.print(
        f"Loaded {len(idx)} trading days from {idx.index[0].date()} to {idx.index[-1].date()}"
    )

    start_pos, end_pos = _make_date_range(idx.index, dt, n)

    # Include prior day for reference (hi-lo-close lines)
    start_pos = max(0, start_pos - 1)

    start_ts = idx.iat[start_pos, 0]  # 'first' column
    end_ts = idx.iat[end_pos, 1]  # 'last' column
    console.print(f"Slicing data from {start_ts} to {end_ts} ({end_pos - start_pos + 1} days)")

    return df[start_ts:end_ts]

def load_timeseries_from_csv(data_dir: Path, pattern: str, dt: str, n: int) -> PlotData:
    """Load and augment data from CSV files. Returns slice of data for plotting.

    Args:
        data_dir: Directory containing CSV files
        pattern: Glob pattern for files (e.g., "esu5*.csv")
        dt: Date in yyyymmdd format or an index value
        n: Number of days to plot
    """
    df_all = ts.load_overlapping_files(data_dir, pattern)

    # calc vwap and ema if not present
    if "wap" in df_all.columns and "vwap" not in df_all.columns:
        df_all["vwap"] = ts.calc_vwap(df_all)
    if "ema" not in df_all.columns:
        # ema(87) for m1 bars approx ema(20) for m5 bars
        df_all["ema"] = ts.calc_ema(df_all, 87)

    df = _slice_trading_days(df_all, dt, n)

    return _create_plot_data(df, min_vol=2500 if abs(n) < 3 else 5000)


def load_timeseries_from_mongo(symbol: str, dt_str: str, n: int) -> PlotData:
    """Load and transform data from MongoDB."""
    df = load_price_history(symbol, dt_str, n)
    return _create_plot_data(df, min_vol=2500 if abs(n) < 3 else 5000)


def plot(dt: str, n: int) -> None:
    """Plot n days of price history from CSV files.

    Args:
        dt: Date in yyyymmdd format or an index value (e.g., "-1" for last day)
        n: Number of days to plot
    """
    data = load_timeseries_from_csv(Path.home() / "Documents" / "data", "esu5*.csv", dt, n)
    plot_minvol_price_chart(data)


def plot_mongo(symbol: str, dt_str: str, n: int) -> None:
    """Plot n days of price history for symbol starting with dt.

    Args:
        symbol: Futures symbol (e.g., "esu5")
        dt: Date in yyyymmdd format or an index value
        n: Number of days to plot
    """
    data = load_timeseries_from_mongo(symbol, dt_str, n)
    plot_minvol_price_chart(data)


def color_bars(df, tm, bar_colour: str):
    if bar_colour == STRAT_BAR_COLOUR:
        df["tm"] = tm
        df["btype"] = ts.calc_strat(df)
        df_inside = df.loc[df["btype"] == 0]
        df_up = df.loc[df["btype"] == 1]
        df_down = df.loc[df["btype"] == 2]
        df_outside = df.loc[df["btype"] == 3]

        fig = go.Figure(data=[go.Scatter(x=tm, y=df["vwap"], line=dict(color="orange"), name="vwap")])
        if "ema" in df:
            fig.add_trace(go.Scatter(x=tm, y=df["ema"], line=dict(color="yellow"), name="ema"))
        fig.add_trace(go.Ohlc(x=df_inside["tm"], open=df_inside["open"], high=df_inside["high"], low=df_inside["low"], close=df_inside["close"], name="inside"))
        fig.add_trace(go.Ohlc(x=df_up["tm"], open=df_up["open"], high=df_up["high"], low=df_up["low"], close=df_up["close"], name="up"))
        fig.add_trace(go.Ohlc(x=df_down["tm"], open=df_down["open"], high=df_down["high"], low=df_down["low"], close=df_down["close"], name="down"))
        fig.add_trace(go.Ohlc(x=df_outside["tm"], open=df_outside["open"], high=df_outside["high"], low=df_outside["low"], close=df_outside["close"], name="outside"))

        fig.data[2].increasing.line.color = "yellow"
        fig.data[2].decreasing.line.color = "yellow"
        fig.data[3].increasing.line.color = "green"
        fig.data[3].decreasing.line.color = "green"
        fig.data[4].increasing.line.color = "red"
        fig.data[4].decreasing.line.color = "red"
        # TODO why does 5 not work??
        try:
            fig.data[5].increasing.line.color = "purple"
            fig.data[5].decreasing.line.color = "purple"
        except IndexError:
            pass
        return fig
    return go.Figure(data=[go.Candlestick(x=tm, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="ES"), go.Scatter(x=tm, y=df["vwap"], line=dict(color="orange"), name="vwap")])


def plot_atr(n: int):
    df = ts.load_files(Path("/temp/ultra"), "zesm5*.csv")
    atr = ts.calc_atr(df, n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=atr.index, y=atr, mode="lines", name="ATR5"))
    fig.layout.title = f"rolling {n} minute ATR"
    fig.show()


def plot_cumulative_volume():
    # df = ts.load_files(Path("~/documents/data").expanduser(), "esm5*.csv")
    df = ts.load_overlapping_files(Path("/temp/ultra"), "zesm5*.csv")
    di = ts.day_index(df)
    df_pivot = ts.pivot_cum_vol_avg_by_day(df, di)
    fig = plot_cumulative_volume_by_day(df_pivot)
    fig.show()


def plot_tick(days: int):
    """display the last n days"""
    df = ts.load_files(ts.make_filename("zTICK-NYSE*.csv"))
    #  last n days from a dataframe with a datetime index
    idx = ts.day_index(df)
    filtered = df[df.index >= idx.first.iloc[-days]]
    hi = filtered["high"].quantile(0.95)
    lo = filtered["low"].quantile(0.05)
    mid = (filtered["high"] + filtered["low"]) / 2
    console.print(f"tick percentiles 5,95 {lo:.2f} and {hi:.2f}")
    tm = filtered.index.strftime("%d/%m %H:%M")
    fig = px.bar(x=tm, y=filtered.high, base=filtered.low)
    fig.add_trace(go.Scatter(x=tm, y=mid, line=dict(color="green"), name="ema"))
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=tm, y=df['High'], mode='lines', name='high') )
    # fig.add_trace(go.Scatter(x=tm, y=df['Low'], mode='lines', name='low') )
    fig.add_hline(y=hi, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=lo, line_width=1, line_dash="dash", line_color="grey")
    fig.show()


def create_min_vol_index(df_min_vol, day_index) -> list[MinVolDay]:
    """create list of MinVolDay from day_index and a df containing volume aggregated bars"""
    # first bar will either be at 23:00 most of the time but 22:00 when US/UK clocks change at different dates
    def floor_index(tm):
        """return index of interval that includes tm"""
        return df_min_vol.index.searchsorted(tm, side="right") - 1

    start_tm = df_min_vol.index[0]
    last = df_min_vol.shape[0] - 1
    xs = []
    for i, r in day_index.iterrows():
        start_tm = r["first"]
        #    for startTm in day_index.openTime:
        start_bar = floor_index(start_tm)
        eu_start = start_tm + timedelta(minutes=540)
        eu_start_bar = floor_index(eu_start)
        eu_end_bar, rth_start_bar, rth_end_bar = -1, -1, -1
        if eu_start_bar < last:
            eu_end = start_tm + timedelta(minutes=929)
            eu_end_bar = floor_index(eu_end)
            if eu_end_bar < last:
                rth_start = start_tm + timedelta(minutes=930)
                rth_start_bar = floor_index(rth_start)
                rth_end = start_tm + timedelta(minutes=1320)
                rth_end_bar = floor_index(rth_end)
        xs.append(MinVolDay(i, start_tm, start_bar, eu_start_bar, eu_end_bar, rth_start_bar, rth_end_bar))
    return xs


def plot_volp(symbol: str, dt: str, n: int):
    df = load_price_history(symbol, dt, n)
    idx = ts.day_index(df)
    # day_summary_df = ts.create_day_summary(df, idx)
    # num_days = idx.shape[0]
    s, e = (idx.iat[0, 0], idx.iat[n - 1, 1]) if n > 0 else (idx.iat[n, 0], idx.iat[-1, 1])
    title = f"volume profile from {s} to {e}"
    # loaded an additional day for hi-lo info but create minVol for display skipping first day
    df_day = df[s:e]
    df_min_vol = ts.aggregate_min_volume(df_day, 2500)
    profile_df = ts.create_volume_profile(df_day, 25, 5)
    peaks = profile_df[profile_df["is_peak"]]

    # plot the unsmoothed volume profile but use smoothed one for peaks
    bar_graph = go.Bar(y=profile_df.index, x=profile_df["volume"], orientation="h")
    annots = go.Scatter(x=peaks["volume"], y=peaks.index, text=peaks.index, mode="markers+text", textposition="bottom center")
    tm = df_min_vol.index.strftime("%d/%m %H:%M")
    price_chart = go.Scatter(y=df_min_vol["close"], x=tm)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
    fig.add_trace(bar_graph, row=1, col=1)
    fig.add_trace(annots, row=1, col=1)
    fig.add_trace(price_chart, row=1, col=2)
    # Customize the layout (optional)
    fig.update_layout(
        title=title,
        xaxis_title="volume",
        yaxis_title="price",
        # autosize=False,
        # width=600,
        # height=600
    )

    fig.show()


def plot_cumulative_volume_by_day(pivot_df: pd.DataFrame) -> go.Figure:
    """
    Plot cumulative volume percentage lines for each trading day.
    Highlights the most recent day's points for easy identification.

    Args:
        pivot_df: DataFrame with time as index, trading dates as columns, cum_vol_avg as values

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Get column names (trading dates) and sort them
    trading_dates = sorted(pivot_df.columns)

    # Get the most recent date (rightmost column)
    most_recent_date = trading_dates[-1] if trading_dates else None

    # Define colors - use a color scale for better distinction
    colors = px.colors.qualitative.Prism
    if len(trading_dates) > len(colors):
        # If we have more days than colors, cycle through them
        colors = colors * (len(trading_dates) // len(colors) + 1)

    # Plot each trading day
    for i, dt in enumerate(trading_dates):
        # Get data for this day, drop NaN values
        day_data = pivot_df[dt].dropna()

        if len(day_data) == 0:
            continue

        # Determine if this is the most recent day
        is_most_recent = dt == most_recent_date

        # Format date for legend
        date_str = dt.strftime("%Y-%m-%d")
        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=day_data.index,  # Time values
                y=day_data.values,  # cum_vol_avg values
                mode="lines+markers" if is_most_recent else "lines",
                name=date_str,
                line=dict(color=colors[i % len(colors)], width=3 if is_most_recent else 2),
                marker=dict(size=8 if is_most_recent else 4, color=colors[i % len(colors)], symbol="circle") if is_most_recent else dict(size=4),
                hovertemplate=f"<b>{date_str}</b><br>" + "Time: %{x}<br>" + "Cum Vol %: %{y:.1f}%<br>" + "<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title={"text": "Cumulative Volume Percentage by Trading Day", "x": 0.5, "xanchor": "center", "font": {"size": 18}},
        xaxis_title="Time",
        yaxis_title="Cumulative Volume (%)",
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        width=1000,
        height=600,
        template="plotly_white",
    )

    # Add horizontal reference lines at 75%, 100%, 125%
    for level in [90, 110]:
        fig.add_hline(y=level, line_dash="dash", line_color="gray", line_width=1, opacity=0.7, annotation_text=f"{level}%", annotation_position="right", annotation_font_size=10, annotation_font_color="gray")

    # Format x-axis to show times nicely
    fig.update_xaxes(
        tickangle=45,
        tickmode="array",
        tickvals=pivot_df.index[:: max(1, len(pivot_df.index) // 10)],  # Show every nth tick to avoid crowding
        showgrid=True,
        gridcolor="lightgray",
    )

    # Format y-axis with fixed range
    fig.update_yaxes(
        showgrid=True,
        gridcolor="blue",
        ticksuffix="%",
        range=[50, 175],  # Fixed y-axis range from 50% to 150%
        dtick=25,  # Show ticks every 25%
    )
    return fig


def load_trading_days(collection, symbol, min_vol):
    """return dataframe of complete trading days [date, bar-count, volume, normalised-volume]"""
    cursor = collection.aggregate(
        [{"$match": {"symbol": symbol}}, {"$group": {"_id": {"$dateTrunc": {"date": "$timestamp", "unit": "day"}}, "count": {"$sum": 1}, "volume": {"$sum": "$volume"}}}, {"$match": {"volume": {"$gte": min_vol}}}, {"$sort": {"_id": 1}}]
    )
    df = pd.DataFrame(list(cursor))
    v = df.volume
    df["stdvol"] = (v - v.mean()) / v.std()
    df.set_index("_id", inplace=True)
    df.index.rename("date", inplace=True)
    return df


def map_to_date_range(xs, x):
    """given an array and a tuple of indexes, return a tuple containing the start end datetime."""
    return datetime.combine(xs[x[0]] - timedelta(days=1), time(22, 0)), datetime.combine(xs[x[1]], time(22, 0))


def convert_to_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        return date.today()


def plot_stdvol(df: pd.DataFrame) -> None:
    fig = px.bar(df, x=df.index, y="stdvol")
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="Plot daily chart")
    parser.add_argument("--dt", type=str, default="-1", help="Start date (yyyymmdd) or index")
    parser.add_argument("--index", type=int, default=-1, help="Index of day to plot e.g. -1 for last")
    parser.add_argument("--tlb", type=str, default="", help="Display three line break [fname]")
    parser.add_argument("--volp", action="store_true", help="Display volume profile for day")
    parser.add_argument("--mdb", type=str, default="", help="Load from MongoDB [yyyymmdd]")
    parser.add_argument("--atr", action="store_true", help="Display ATR")
    parser.add_argument("--cumvol", action="store_true", help="plot relative cumulative volume")
    parser.add_argument("--tick", action="store_true", help="Display tick")
    parser.add_argument("--days", type=int, default=1, help="Number of days")
    parser.add_argument("--sym", type=str, default="esu5", help="Index symbol")

    argv = parser.parse_args()
    console.print(argv)
    if len(argv.tlb) > 0:
        plot_3lb(argv.tlb)
    elif argv.volp and len(argv.mdb) > 0:
        plot_volp(argv.sym, argv.dt, argv.days)
    elif len(argv.mdb) > 0:
        plot_mongo(argv.sym, argv.dt, argv.days)
    elif argv.atr:
        plot_atr(5)
    elif argv.cumvol:
        plot_cumulative_volume()
    elif argv.tick:
        plot_tick(argv.days)
    else:
        plot(argv.dt, argv.days)


# plot_atr()
# hilo(df)
# samp_3lb()
if __name__ == "__main__":
    main()
    # compare_emas()
