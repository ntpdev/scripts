#!/usr/bin/python3
import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdbutils import load_price_history
import tsutils as ts

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
    rth_opens = df[is_first_bar].index.apply(lambda e: e + np.timedelta64(930 - e.minute, "m"))
    rth_closes = df[is_first_bar].index.apply(lambda e: e + np.timedelta64(1320 - e.minute, "m"))

    # select rows matching time, convert index to a normal col, add col which is date
    ops = df[df.Date.isin(rth_opens)]
    ops2 = ops.reset_index()
    ops2["dt"] = ops2.Date.dt.date

    cls = df[df.Date.isin(rth_closes)]
    cls2 = cls.reset_index()
    cls2["dt"] = cls2.Date.dt.date

    # join dfs on date ie include only days that have open+close
    mrg = pd.merge(ops2, cls2, how="inner", on="dt")
    return [(x, y) for x, y in zip(mrg["index_x"], mrg["index_y"])]


def plot(index):
    df = pd.read_csv(ts.make_filename("es-minvol.csv"), parse_dates=["date", "dateCl"], index_col=0)

    # create a string for X labels
    tm = df.index.strftime("%d/%m %H:%M")
    fig = color_bars(df, tm, STRAT_BAR_COLOUR)
    #    fig = go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ES'),
    #                        go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])
    xs = make_day_index(df)
    rths = make_rth_index(df, xs)
    draw_daily_lines(df, fig, tm, rths)
    add_hilo_labels(df, fig)

    op, cl = xs[index]
    fig.layout.xaxis.range = [op, cl]
    lo, hi = make_yrange(df, op, cl, 4)
    fig.layout.yaxis.range = [lo, hi]
    fig.show()


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
        fig.data[5].increasing.line.color = "purple"
        fig.data[5].decreasing.line.color = "purple"
        return fig
    return go.Figure(data=[go.Candlestick(x=tm, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="ES"), go.Scatter(x=tm, y=df["vwap"], line=dict(color="orange"), name="vwap")])


def plot_atr():
    df = ts.load_files(ts.make_filename("esu4*.csv"))
    atr = ts.calc_atr(df, 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=atr.index, y=atr, mode="lines", name="ATR5"))
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
    print(f"tick percentiles 5,95 {lo:.2f} and {hi:.2f}")
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
    # first bar will either be at 23:00 most of the time but 22:00 when US/UK clocks change at different dates
    start_tm = df_min_vol.index[0]
    last = df_min_vol.shape[0] - 1
    xs = []
    for i, r in day_index.iterrows():
        start_tm = r["first"]
        #    for startTm in day_index.openTime:
        start_bar = floor_index(df_min_vol, start_tm)
        print(start_tm, start_bar)
        eu_start = start_tm + timedelta(minutes=540)
        eu_start_bar = floor_index(df_min_vol, eu_start)
        eu_end_bar, rth_start_bar, rth_end_bar = -1, -1, -1
        if eu_start_bar < last:
            eu_end = start_tm + timedelta(minutes=929)
            eu_end_bar = floor_index(df_min_vol, eu_end)
            if eu_end_bar < last:
                rth_start = start_tm + timedelta(minutes=930)
                rth_start_bar = floor_index(df_min_vol, rth_start)
                rth_end = start_tm + timedelta(minutes=1320)
                rth_end_bar = floor_index(df_min_vol, rth_end)
        xs.append(MinVolDay(i, start_tm, start_bar, eu_start_bar, eu_end_bar, rth_start_bar, rth_end_bar))
    return xs


def plot_mongo(symbol: str, dt: str, n: int):
    """plot n days of price history for symbol starting with dt
    dt is either a date in yyyymmdd format or a index value"""
    df = load_price_history(symbol, dt, n)
    idx = ts.day_index(df)
    day_summary_df = ts.create_day_summary(df, idx)
    num_days = idx.shape[0]
    # loaded an additional day for hi-lo info but create minVol for display skipping first day
    skip_first = num_days > 1
    slice = df[idx.iat[1, 0] : idx.iat[num_days - 1, 1]] if skip_first else df
    df_min_vol = ts.aggregate_min_volume(slice, 1500)

    # create a string for X labels
    tm = df_min_vol.index.strftime("%d/%m %H:%M")
    fig = color_bars(df_min_vol, tm, STRAT_BAR_COLOUR)
    add_hilo_labels(df_min_vol, fig)
    mvds = create_min_vol_index(df_min_vol, idx)

    def add_h_line(x_start, x_end, y_value, prefix="", color="Gray", dash="dot"):
        fig.add_shape(type="line", x0=x_start, y0=y_value, x1=x_end, y1=y_value, line=dict(color=color, dash=dash))
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

        # add first hour hi-lo
        h1_hi = day_summary_df.at[i.trade_dt, "rth_h1_high"]
        if pd.notna(h1_hi):
            xstart_rth = tm[i.rth_start_bar]
            add_h_line(xstart_rth, xend, h1_hi, "h1 h")
            
            h1_lo = day_summary_df.at[i.trade_dt, "rth_h1_low"]
            add_h_line(xstart_rth, xend, h1_lo, "h1 l")

        # add previous day rth hi-lo-close
        ix = day_summary_df.index.searchsorted(i.trade_dt)
        if ix > 0:
            prev_day = day_summary_df.iloc[ix - 1]
            
            add_h_line(xstart, xend, prev_day.rth_low, "yl", color="chocolate")
            add_h_line(xstart, xend, prev_day.rth_high, "yh", color="chocolate")
            add_h_line(xstart, xend, prev_day.close, "cl", color="cyan")

    fig.show()


def plot_volp(symbol, dt, n):
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


def floor_index(df, tm):
    """return index of interval that includes tm"""
    return df.index.searchsorted(tm, side="right") - 1


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
    parser.add_argument("--index", type=int, default=-1, help="Index of day to plot e.g. -1 for last")
    parser.add_argument("--tlb", type=str, default="", help="Display three line break [fname]")
    parser.add_argument("--volp", action="store_true", help="Display volume profile for day")
    parser.add_argument("--mdb", type=str, default="", help="Load from MongoDB [yyyymmdd]")
    parser.add_argument("--atr", action="store_true", help="Display ATR")
    parser.add_argument("--tick", action="store_true", help="Display tick")
    parser.add_argument("--days", type=int, default=1, help="Number of days")
    parser.add_argument("--sym", type=str, default="esm5", help="Index symbol")

    argv = parser.parse_args()
    print(argv)
    if len(argv.tlb) > 0:
        plot_3lb(argv.tlb)
    elif argv.volp and len(argv.mdb) > 0:
        plot_volp(argv.sym, convert_to_date(argv.mdb), argv.days)
    elif len(argv.mdb) > 0:
        plot_mongo(argv.sym, argv.mdb, argv.days)
    elif argv.atr:
        plot_atr()
    elif argv.tick:
        plot_tick(argv.days)
    else:
        plot(argv.index)


# plot_atr()
# hilo(df)
# samp_3lb()
if __name__ == "__main__":
    main()
    # compare_emas()
