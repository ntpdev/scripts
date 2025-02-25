#!/usr/bin/python3
# python -m pip install pymongo
import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mdbutils as md
import tsutils as ts


@dataclass
class MinVolDay:
    tradeDt: date
    startTm: datetime
    startBar: int
    euStartBar: int
    euEndBar: int
    rthStartBar: int
    rthEndBar: int


def samp():
    open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    dates = [datetime(year=2013, month=10, day=10), datetime(year=2013, month=11, day=10), datetime(year=2013, month=12, day=10), datetime(year=2014, month=1, day=10), datetime(year=2014, month=2, day=10)]

    fig = go.Figure(data=[go.Candlestick(x=dates, open=open_data, high=high_data, low=low_data, close=close_data)])

    fig.show()


def samp3LB():
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


def highs(df, window):
    hs = df["high"].rolling(window, center=True).max()
    hsv = df["high"][np.equal(df["high"], hs)]
    t = pd.Series.diff(hsv)
    # remove 0 elements
    return df.high[t[t.ne(0)].index]


def lows(df, window):
    hs = df["low"].rolling(window, center=True).min()
    hsv = df["low"][np.equal(df["low"], hs)]
    # remove adjacent values which are the same by differencing and removing 0
    t = pd.Series.diff(hsv)
    return df.low[t[t.ne(0)].index]


def add_hilo_labels(df, tms, fig):
    hs = highs(df, 21)
    ls = lows(df, 21)

    fig.add_trace(go.Scatter(x=[tm for tm in df.loc[hs.index].tm], y=hs.add(1), text=["%.2f" % p for p in df.loc[hs.index].high], mode="text", textposition="top center", name="local high"))

    fig.add_trace(go.Scatter(x=[tm for tm in df.loc[ls.index].tm], y=ls.sub(1), text=["%.2f" % p for p in df.loc[ls.index].low], mode="text", textposition="bottom center", name="local low"))


def bar_containing(df, dt):
    return (df["date"] <= dt) & (df["dateCl"] > dt)


# return a high and low range to nearest multiple of n
def make_yrange(df, op, cl, n):
    h = df["high"][op:cl].max() + n // 2
    l = df["low"][op:cl].min() - n // 2
    return (l // n) * n, ((h // n) + 1) * n


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
    fig = color_bars(df, tm, "strat")
    #    fig = go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ES'),
    #                        go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])
    xs = make_day_index(df)
    rths = make_rth_index(df, xs)
    draw_daily_lines(df, fig, tm, rths)
    add_hilo_labels(df, tm, fig)

    op, cl = xs[index]
    fig.layout.xaxis.range = [op, cl]
    l, h = make_yrange(df, op, cl, 4)
    fig.layout.yaxis.range = [l, h]
    fig.show()


def color_bars(df, tm, opt):
    if opt == "strat":
        df["tm"] = tm
        df["btype"] = ts.calc_strat(df)
        dfInside = df.loc[df["btype"] == 0]
        dfUp = df.loc[df["btype"] == 1]
        dfDown = df.loc[df["btype"] == 2]
        dfOutside = df.loc[df["btype"] == 3]

        fig = go.Figure(data=[go.Scatter(x=tm, y=df["vwap"], line=dict(color="orange"), name="vwap")])
        if "ema" in df:
            fig.add_trace(go.Scatter(x=tm, y=df["ema"], line=dict(color="yellow"), name="ema"))
        fig.add_trace(go.Ohlc(x=dfInside["tm"], open=dfInside["open"], high=dfInside["high"], low=dfInside["low"], close=dfInside["close"], name="inside"))
        fig.add_trace(go.Ohlc(x=dfUp["tm"], open=dfUp["open"], high=dfUp["high"], low=dfUp["low"], close=dfUp["close"], name="up"))
        fig.add_trace(go.Ohlc(x=dfDown["tm"], open=dfDown["open"], high=dfDown["high"], low=dfDown["low"], close=dfDown["close"], name="down"))
        fig.add_trace(go.Ohlc(x=dfOutside["tm"], open=dfOutside["open"], high=dfOutside["high"], low=dfOutside["low"], close=dfOutside["close"], name="outside"))

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


def create_minVol_index(dfMinVol, day_index) -> list[MinVolDay]:
    # first bar will either be at 23:00 most of the time but 22:00 when US/UK clocks change at different dates
    startTm = dfMinVol.index[0]
    last = dfMinVol.shape[0] - 1
    xs = []
    for i, r in day_index.iterrows():
        startTm = r["first"]
        #    for startTm in day_index.openTime:
        startBar = floor_index(dfMinVol, startTm)
        print(startTm, startBar)
        euStart = startTm + timedelta(minutes=540)
        euStartBar = floor_index(dfMinVol, euStart)
        euEndBar, rthStartBar, rthEndBar = -1, -1, -1
        if euStartBar < last:
            euEnd = startTm + timedelta(minutes=929)
            euEndBar = floor_index(dfMinVol, euEnd)
            if euEndBar < last:
                rthStart = startTm + timedelta(minutes=930)
                rthStartBar = floor_index(dfMinVol, rthStart)
                rthEnd = startTm + timedelta(minutes=1320)
                rthEndBar = floor_index(dfMinVol, rthEnd)
        xs.append(MinVolDay(i, startTm, startBar, euStartBar, euEndBar, rthStartBar, rthEndBar))
    return xs


# def create_day_summary(df, day_index):
#     xs = []
#     for i,r in day_index.iterrows():
#         euClose = r.openTime + pd.Timedelta(minutes=929)
#         rthOpen = r.openTime + pd.Timedelta(minutes=930)
#         rthClose = r.openTime + pd.Timedelta(minutes=1319)
#         xs.append({'date' : i,
#                    'glbx_high': df['High'][r.openTime:euClose].max(),
#                    'glbx_low': df['Low'][r.openTime:euClose].min(),
#                    'rth_hi': df['High'][rthOpen:rthClose].max(),
#                    'rth_lo': df['Low'][rthOpen:rthClose].min(),
#                    'close': df['Close'][rthClose]
#                    })
#     day_summary_df = pd.DataFrame(xs)
#     day_summary_df.set_index('date', inplace=True)
#     return day_summary_df


def plot_mongo(symbol, dt, n):
    df = md.load_price_history(symbol, dt, n)
    idx = ts.day_index(df)
    day_summary_df = ts.create_day_summary(df, idx)
    num_days = idx.shape[0]
    # loaded an additional day for hi-lo info but create minVol for display skipping first day
    slice = df[idx.iat[1, 0] : idx.iat[num_days - 1, 1]] if num_days > 1 else df
    dfMinVol = ts.aggregate_min_volume(slice, 1500)

    # create a string for X labels
    tm = dfMinVol.index.strftime("%d/%m %H:%M")
    fig = color_bars(dfMinVol, tm, "strat")
    add_hilo_labels(dfMinVol, tm, fig)
    mvds = create_minVol_index(dfMinVol, idx)

    for i in mvds[1:]:
        euOpen = dfMinVol.at[dfMinVol.index[i.euStartBar], "open"]
        fig.add_shape(type="line", x0=tm[i.euStartBar], y0=euOpen, x1=tm[i.euEndBar], y1=euOpen, line=dict(color="LightSeaGreen", dash="dot"))
        xstart = tm[i.startBar]
        xend = tm[i.rthEndBar]
        if i.rthStartBar > 0:
            xstart_rth = tm[i.rthStartBar]
            fig.add_vline(x=xstart_rth, line_width=1, line_dash="dash", line_color="blue")
            if i.rthEndBar > 0:
                fig.add_vline(x=xend, line_width=1, line_dash="dash", line_color="blue")
            rthOpen = day_summary_df.at[i.tradeDt, "rth_open"]
            fig.add_shape(type="line", x0=xstart_rth, y0=rthOpen, x1=xend, y1=rthOpen, line=dict(color="LightSeaGreen", dash="dot"))
            fig.add_annotation(text=f"open {rthOpen:.2f}", x=xend, y=rthOpen, showarrow=False)
            # fig.add_annotation(x=tm[i.rthStartBar], y=rthOpen, text=f"open {rthOpen:.2f}", showarrow=False, font=dict(color="LightSeaGreen"))
            glbxHi = day_summary_df.at[i.tradeDt, "glbx_high"]
            fig.add_shape(type="line", x0=xstart_rth, y0=glbxHi, x1=xend, y1=glbxHi, line=dict(color="Gray", dash="dot"))
            fig.add_annotation(text=f"glbx h h {glbxHi:.2f}", x=xend, y=glbxHi, showarrow=False)
            glbxLo = day_summary_df.at[i.tradeDt, "glbx_low"]
            fig.add_shape(type="line", x0=xstart_rth, y0=glbxLo, x1=xend, y1=glbxLo, line=dict(color="Gray", dash="dot"))
            fig.add_annotation(text=f"glbx lo {glbxLo:.2f}", x=xend, y=glbxLo, showarrow=False)

        # add first hour hi-lo
        h1Hi = day_summary_df.at[i.tradeDt, "rth_h1_high"]
        if pd.notna(h1Hi):
            xstart_rth = tm[i.rthStartBar]
            fig.add_shape(type="line", x0=xstart_rth, y0=h1Hi, x1=xend, y1=h1Hi, line=dict(color="Gray", dash="dot"))
            h1Lo = day_summary_df.at[i.tradeDt, "rth_h1_low"]
            fig.add_shape(type="line", x0=xstart_rth, y0=h1Lo, x1=xend, y1=h1Lo, line=dict(color="Gray", dash="dot"))
            fig.add_annotation(text=f"h1 h {h1Hi:.2f}", x=xend, y=h1Hi, showarrow=False)
            fig.add_annotation(text=f"h1 l {h1Lo:.2f}", x=xend, y=h1Lo, showarrow=False)

        # add previous day rth hi-lo-close
        ix = day_summary_df.index.searchsorted(i.tradeDt)
        if ix > 0:
            x = day_summary_df.iloc[ix - 1]
            prevRthHi = x.rth_high
            prevRthLo = x.rth_low
            prevRthClose = x.close
            glbxLo = day_summary_df.at[i.tradeDt, "glbx_low"]
            if prevRthLo > glbxLo:
                fig.add_shape(type="line", x0=xstart, y0=prevRthLo, x1=xend, y1=prevRthLo, line=dict(color="chocolate", dash="dot"))
                fig.add_annotation(text=f"yl {prevRthLo:.2f}", x=xend, y=prevRthLo, showarrow=False)

            fig.add_shape(type="line", x0=xstart, y0=prevRthHi, x1=xend, y1=prevRthHi, line=dict(color="chocolate", dash="dot"))
            fig.add_shape(type="line", x0=xstart, y0=prevRthClose, x1=xend, y1=prevRthClose, line=dict(color="cyan", dash="dot"))
            fig.add_annotation(text=f"yh {prevRthHi:.2f}", x=xend, y=prevRthHi, showarrow=False)
            fig.add_annotation(text=f"cl {prevRthClose:.2f}", x=xend, y=prevRthClose, showarrow=False)

    fig.show()


def plot_volp(symbol, dt, n):
    df = md.load_price_history(symbol, dt, n)
    idx = ts.day_index(df)
    day_summary_df = ts.create_day_summary(df, idx)
    num_days = idx.shape[0]
    s, e = (idx.iat[0, 0], idx.iat[n - 1, 1]) if n > 0 else (idx.iat[n, 0], idx.iat[-1, 1])
    title = f"volume profile from {s} to {e}"
    # loaded an additional day for hi-lo info but create minVol for display skipping first day
    df_day = df[s:e]
    dfMinVol = ts.aggregateMinVolume(df_day, 2500)
    profile_df = ts.create_volume_profile(df_day, 25, 5)
    peaks = profile_df[profile_df["is_peak"] == True]

    # plot the unsmooted volume profile but use smoothed one for peaks
    bar_graph = go.Bar(y=profile_df.index, x=profile_df["volume"], orientation="h")
    annots = go.Scatter(x=peaks["volume"], y=peaks.index, text=peaks.index, mode="markers+text", textposition="bottom center")
    tm = dfMinVol.index.strftime("%d/%m %H:%M")
    price_chart = go.Scatter(y=dfMinVol["close"], x=tm)

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


def load_trading_days(collection, symbol, minVol):
    """return dataframe of complete trading days [date, bar-count, volume, normalised-volume]"""
    cursor = collection.aggregate(
        [{"$match": {"symbol": symbol}}, {"$group": {"_id": {"$dateTrunc": {"date": "$timestamp", "unit": "day"}}, "count": {"$sum": 1}, "volume": {"$sum": "$volume"}}}, {"$match": {"volume": {"$gte": minVol}}}, {"$sort": {"_id": 1}}]
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


def parse_isodate(s):
    try:
        return date.fromisoformat(s)
    except ValueError:
        return datetime.now().date()


def plot_stdvol(df):
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
    parser.add_argument("--sym", type=str, default="esz4", help="Index symbol")

    argv = parser.parse_args()
    print(argv)
    if len(argv.tlb) > 0:
        plot_3lb(argv.tlb)
    elif argv.volp and len(argv.mdb) > 0:
        plot_volp(argv.sym, parse_isodate(argv.mdb), argv.days)
    elif len(argv.mdb) > 0:
        plot_mongo(argv.sym, parse_isodate(argv.mdb), argv.days)
    elif argv.atr:
        plot_atr()
    elif argv.tick:
        plot_tick(argv.days)
    else:
        plot(argv.index)


# plot_atr()
# hilo(df)
# samp3LB()
if __name__ == "__main__":
    main()
    # compare_emas()
