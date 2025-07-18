#!/usr/bin/env python3
# Note chmod +x *.py
# ensure Unix style line endings

import argparse
import itertools
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subp
import requests as req
from lightweight_charts import Chart
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from twelvedata import TDClient
from xlsxwriter.utility import xl_range

import tsutils

# pip install twelvedata
BASEDIR = "Downloads"
APIKEY = os.environ["TW_API_KEY"]

tdc = TDClient(apikey=APIKEY)
console = Console()


def make_fullpath(fn: str) -> Path:
    return Path.home() / BASEDIR / fn


def make_filename(symbol: str, dt: date) -> Path:
    return make_fullpath(f"{symbol} {dt.isoformat()}.csv")


def list_cached_files(symbol: str):
    """list files most recent first"""
    p = Path.home() / BASEDIR
    return sorted(p.glob(symbol + " 202*.csv"), reverse=True)


def load_file(fname: str):
    df = pd.read_csv(make_fullpath(fname), parse_dates=["datetime"], index_col="datetime", engine="python")
    console.print(f"loaded {fname} {df.shape[0]} {df.shape[1]}\n", style="green")
    return df


def json_to_df(objs):
    return pd.DataFrame(objs["values"])


def plot(symbol: str, df: pd.DataFrame, mas: list[str], n: int = 250) -> None:
    dfc = df.iloc[-n:].copy()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=dfc.index, open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name=symbol))
    for ma in mas:
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc[ma], name=ma))

    # add hi lo markers
    markers = dfc[dfc["hilo"] > 19]
    fig.add_trace(go.Scatter(x=markers.index, y=markers["high"] * 1.01, mode="markers", name="20d high", marker=dict(color="green", symbol="triangle-up")))
    markers = dfc[dfc["hilo"] < -19]
    fig.add_trace(go.Scatter(x=markers.index, y=markers["low"] * 0.99, mode="markers", name="20d low", marker=dict(color="red", symbol="triangle-down")))

    # remove weekends / holidays
    all_dates = pd.date_range(dfc.index[0], dfc.index[-1])
    missing_dates = all_dates.difference(dfc.index)
    fig.update_xaxes(rangebreaks=[dict(values=missing_dates)])

    fig.show()


def plot_tv(symbol: str, df: pd.DataFrame, mas: list[str]) -> None:
    """plot using lightweight charts"""
    chart = Chart(toolbox=True)
    chart.watermark(symbol, color="rgba(180, 180, 240, 0.7)")
    chart.set(df)

    # add p/b limits
    last = df[-20:]["close"].max()
    for y in [0.975, 0.95, 0.9]:
        chart.horizontal_line(last * y, text=f"{(1 - y) * 100:.1f}%", color="rgba(133,153,0,0.5)")

    # add ma lines
    colours = ["rgba(42,161,152,0.6)", "rgba(203, 75, 22, 0.6)", "rgba(211,54,130,0.6)"]
    ic = itertools.cycle(colours)
    for ma in mas:
        line = chart.create_line(name=ma, color=next(ic))
        line.set(df[[ma]])  # requires a df with the column

    # hi lo markers - added in time order
    n = 19
    markers = df[(df["hilo"] > n) | (df["hilo"] < -n)]
    for i, r in markers.iterrows():
        if r["hilo"] > n:
            chart.marker(i, "above", "arrow_up", "rgb(133,153,0)")
        else:  # r['hilo'] < -19
            chart.marker(i, "above", "arrow_down", "rgb(220,50,47)")

    chart.show(block=True)


def plot_3lb(symbol, df):
    tlb, rev = tsutils.calc_tlb(df.close, 3)
    # tlb = tlb2[-100:]
    tlb["height"] = tlb["close"] - tlb["open"]
    tlb["dirn"] = np.sign(tlb["height"])
    colours = tlb["dirn"].map({-1: "red", 1: "green"})
    xs = tlb.index.strftime("%Y-%m-%d")
    fig = subp.make_subplots(rows=1, cols=2, subplot_titles=([symbol + " 3LB", symbol + " close"]))
    f1 = go.Bar(x=xs.values, y=tlb["height"], base=tlb["open"], name=symbol, marker=dict(color=colours))
    f2 = go.Scatter(x=df.index[-100:], y=df[-100:]["close"], mode="lines", name=symbol, marker=dict(color="blue"))

    fig.add_trace(f1, row=1, col=1)
    fig.add_trace(f2, row=1, col=2)
    c = "green" if df.loc[df.index[-1], "close"] < rev else "red"
    fig.add_hline(y=rev, line_width=1, line_color=c, line_dash="dash", row=1, col=1)
    fig.add_hline(y=rev, line_width=1, line_color=c, line_dash="dash", row=1, col=2)
    fig.add_annotation(text=f"reversal {rev:.2f}", x=df.index[-100], y=rev * 1.01, font=dict(size=12, color=c), showarrow=False, row=1, col=2)
    fig.update_layout(xaxis_type="category")
    fig.show()


def plot_cumulative(df):
    # Create a barchart using the 'perc' column from the same dataframe
    fig = subp.make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["dtOpen"], y=df["perc"], name="perc"))
    fig.add_trace(go.Scatter(x=df["dtOpen"], y=df["cumulative"], mode="lines"), secondary_y=True)
    # Show the final figure
    fig.show()


def plot_heatmap(df):
    d = []
    for i, r in df.iterrows():
        s = r["system"].split(",")
        entry = s[0]
        ex = s[1]
        v = r["cumulative"]
        # v = r['maxdd']
        d.append({"entry": entry, "exit": ex, "value": v})
    map = pd.DataFrame(d)
    # breakpoint()
    # Add a heatmap trace to the figure
    fig = go.Figure(data=go.Heatmap(x=map["entry"], y=map["exit"], z=map["value"], colorscale="Viridis", showscale=True))

    # Set the title and axis labels
    # fig.update_layout(
    #     title='Heatmap',
    #     xaxis_title='Entry',
    #     yaxis_title='Exit',
    #     zaxis_title='Value'
    # )

    # Show the plot
    fig.show()


# standardise but use *100 so +1 std is 100
def normalise_as_perc(series: pd.Series, n: int = 20) -> pd.Series:
    return (100 * (series - series.rolling(window=n).mean()) / series.rolling(window=n).std()).fillna(0).round(0).astype(int)


def strat(hs, ls):
    """return a series categorising bar by its strat bar type 0 - inside, 1 up, 2 down, 3 outside"""
    x = hs.diff().gt(0)
    y = ls.diff().lt(0)
    return x.astype(int) + y * 2


def print_range_table(df, xs):
    last = df["close"].iat[-1]

    headers = ["Range", "High", "Low", "Last", "% Drawdown", "Volatility", "% Range"]
    tbl = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    for n in xs:
        mx_cl = df["close"].iloc[-n:].max()
        mx = df["high"].iloc[-n:].max()
        mn = df["low"].iloc[-n:].min()
        rng = 100.0 * (last - mn) / (mx - mn)
        volatility = log_returns.rolling(window=n).std() * np.sqrt(252)
        tbl += f"| {n}d | {mx:.2f} | {mn:.2f} | {last:.2f} | {(last / mx_cl - 1) * 100:.2f} | {100 * volatility.iloc[-1]:.1f} | {rng:.1f} |\n"

    console.print("\n--- ranges", style="yellow")
    console.print(Markdown(tbl), style="cyan")


# returns {'datetime': '1993-01-29', 'unix_time': 728317800} for SPY
def load_earliest_date(symbol):
    s = f"https://api.twelvedata.com/earliest_timestamp?symbol={symbol}&interval=1day&apikey={APIKEY}"
    response = req.get(url=s)
    if response.status_code == 200:
        objs = response.json()
        print(objs)


def load_twelve_data(symbol, days=255):
    print(f"loading {symbol}")
    # ts = tdc.time_series(symbol=symbol, interval='1day', start_date='2007-01-01', outputsize=5000, dp=2, order='ASC')
    ts = tdc.time_series(symbol=symbol, interval="1day", outputsize=days, dp=2, order="ASC")
    df = ts.with_ma(ma_type="SMA", time_period=150).with_ma(ma_type="SMA", time_period=50).with_ma(ma_type="EMA", time_period=19).as_pandas()

    df.rename(columns={"ma1": "sma150", "ma2": "sma50", "ma3": "ema19"}, inplace=True)
    print(df.tail())
    fname = make_filename(symbol, df.index[-1].date())
    df["change"] = pd.Series.diff(df.close).round(2)
    df["pct_chg"] = (pd.Series.pct_change(df.close) * 100).round(2)
    df["voln"] = normalise_as_perc(df.volume)
    # df['perc'] = percFromMin(df.close)
    df["hilo"] = tsutils.calc_hilo(df["close"])
    df["strat"] = strat(df.high, df.low)
    df["tlb"] = three_line_break(df.close)
    df.to_csv(fname)
    console.print(f"saved {symbol} {df.index[0].date()} to {df.index[-1].date()} shape={df.shape}", style="green")
    console.print(fname, style="green")
    return df


def load_twelve_data_raw(symbols):
    dt = (datetime.today().date() - timedelta(days=365)).isoformat()
    url = f"https://api.twelvedata.com/time_series?apikey={APIKEY}&interval=1day&start_date={dt}&symbol={symbols}&type=etf&format=JSON&dp=2&order=ASC"
    print(url)
    response = req.get(url=url)
    if response.status_code == 200:
        objs = response.json()
        df = json_to_df(objs)
        fn = f"c:\\users\\niroo\\downloads\\{symbols} {dt.date()}.csv"
        df.set_index("datetime", inplace=True)
        df.to_csv(fn)
        print("saved " + fn)


def scan(df, entry_hi, exit_lo, stop_perc, target_perc):
    xs = []
    state = 0
    stop = None
    for i, row in df.iterrows():
        if state == 0 and row["hilo"] > entry_hi:
            entry = i
            state = 1
            stop = row["close"] * stop_perc
            target = row["close"] * target_perc
        elif state == 1 and (row["close"] < stop or row["hilo"] < exit_lo):
            # elif state == 1 and row['hilo'] < -19:
            xs.append((entry, i))
            state = 0
        elif state == 1 and row["close"] > target:
            xs.append((entry, i))
            state = 0

    ys = []
    for ent, ex in xs:
        d = {"dtOpen": ent, "dtClose": ex, "open": df.at[ent, "close"], "close": df.at[ex, "close"]}
        ys.append(d)
    df2 = pd.DataFrame(ys)
    df2["points"] = df2["close"] - df2["open"]
    df2["perc"] = (df2["points"] / df2["open"]) * 100.0
    df2["cumulative"] = (df2["close"] / df2["open"]).cumprod()
    df2["drawdown"] = df2["cumulative"] - df2["cumulative"].expanding().max()
    print(df2)
    pts = df2["points"]
    wins = pts > 0
    losses = pts < 0
    return {
        "system": f"{entry_hi},{exit_lo},{stop_perc},{target_perc}",
        "pts": pts.sum(),
        "cumulative": df2["cumulative"].iat[-1],
        "maxdd": round(df2["drawdown"].min() * 100.0, 2),
        "winC": pts[wins].count(),
        "winT": pts[wins].sum(),
        "lossC": pts[losses].count(),
        "lossT": pts[losses].sum(),
    }


def plot_swings(df):
    #        fig = px.line(x=swings.index, y=swings['close'])
    fig = px.bar(x=df.index, y=df["change"])
    fig.update_layout(xaxis_type="category")  # treat datetime as category
    fig.show()


def add_historic_volatility(df: pd.DataFrame, period: int, col_name: str = "pct_chg") -> pd.DataFrame:
    """
    Calculate historic volatility over a given period and add it to the dataframe.
    Also prints 10th, 50th, and 90th percentile values in a markdown table.

    Parameters:
        df (pd.DataFrame): Input dataframe with price data
        period (int): Lookback period for volatility calculation
        col_name (str): Column name to use for volatility calculation (default 'pct_chg')

    Returns:
        pd.DataFrame: DataFrame with added 'hvol' column
    """
    # Calculate rolling standard deviation of percentage changes
    rolling_std = df[col_name].rolling(window=period).std()

    # Annualize the volatility (assuming 252 trading days)
    df["hvol"] = rolling_std * np.sqrt(252)

    # Calculate percentiles
    percentiles = df["hvol"].quantile([0.1, 0.5, 0.9])

    # Create and print markdown table
    md_text = dedent(f"""\
        ## Historic Volatility Percentiles (Period: {period} days)
        | Percentile | Value |
        |------------|-------|
        | 10th       | {percentiles[0.1]:.4f} |
        | 50th (median) | {percentiles[0.5]:.4f} |
        | 90th       | {percentiles[0.9]:.4f} |
    """)
    console.print(Markdown(md_text))

    return df


def plot_with_indicator(symbol: str, df: pd.DataFrame, indicator_name: str, overlays: list[str] = None, shaded_areas: list[str] = None) -> go.Figure:
    """
    Create a candlestick chart with specified overlays and a subplot of the
    specified indicator, including horizontal percentile/level lines

    Parameters:
        df (pd.DataFrame): Input dataframe with OHLC data and indicators.
                           The DataFrame index should be datetime-like for
                           proper time-series plotting.
                           Must contain 'open', 'high', 'low', 'close' columns.
        indicator_name (str): Name of the indicator column to plot in the
                              subplot.
        overlay_cols (list[str], optional): List of column names in df to
                                            plot as overlays on the main
                                            candlestick chart.
                                            Defaults to None or an empty list.

    Returns:
        go.Figure: Plotly figure object.
    """
    if overlays is None:
        overlays = []
    if shaded_areas is None:
        shaded_areas = []

    indicator_name = ""
    if indicator_name in df.columns:
        rows = 2
        heights = [0.7, 0.3]
    else:
        rows = 1
        heights = [1]

    # Create figure with subplots
    fig = subp.make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=heights,
    )

    # --- Plot 1: Candlestick and Overlays (Row 1) ---
    candlestick = go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red",
    )
    fig.add_trace(candlestick, row=1, col=1)

    # Add overlay traces
    overlay_trace_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, col_name in enumerate(overlays):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                name=col_name,
                line=dict(
                    color=overlay_trace_colors[i % len(overlay_trace_colors)],
                    width=1.2,
                ),
            ),
            row=1,
            col=1,
        )

    # Add horizontal lines at -10% and -20% of max close
    if "close" in df.columns and not df["close"].empty:
        max_close = df["close"].max()
        if pd.notna(max_close) and np.isfinite(max_close):
            last = df["close"].iat[-1]
            last_perc = (last / max_close - 1) * 100
            level_minus_10 = max_close * 0.90
            level_minus_20 = max_close * 0.80

            fig.add_hline(y=level_minus_10, line_dash="dash", line_color="salmon", annotation_text=f"-10% {level_minus_10:.2f}", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="salmon", row=1, col=1)
            fig.add_hline(y=level_minus_20, line_dash="dash", line_color="lightcoral", annotation_text=f"-20% {level_minus_20:.2f}", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="lightcoral", row=1, col=1)
            fig.add_hline(y=last, line_dash="dash", line_color="lightcoral", annotation_text=f"{last_perc:.1f}% {level_minus_20:.2f}", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="lightcoral", row=1, col=1)

    # --- Plot 2: Indicator (Row 2) ---
    if indicator_name in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[indicator_name],
                name=indicator_name,
                line=dict(color="teal", width=1.5),
            ),
            row=2,
            col=1,
        )

        # Add horizontal lines for 10, 50, 90 percentile values
        # Ensure data is numeric and finite for percentile calculation
        indicator_data = df[indicator_name].dropna()
        indicator_data = indicator_data[np.isfinite(indicator_data)]

        if not indicator_data.empty:
            try:
                percentiles = indicator_data.quantile([0.10, 0.50, 0.90])
                p_labels = {0.10: "10th Pctl", 0.50: "Median (50th Pctl)", 0.90: "90th Pctl"}
                p_colors = {0.10: "blueviolet", 0.50: "darkcyan", 0.90: "blueviolet"}
                p_pos = {0.10: "bottom left", 0.50: "top left", 0.90: "top left"}

                for p_val, p_label_val in p_labels.items():
                    value = percentiles.loc[p_val]
                    if pd.notna(value):
                        fig.add_hline(y=value, line_dash="dot", line_color=p_colors[p_val], annotation_text=f"{p_label_val} ({value:.2f})", annotation_position=p_pos[p_val], annotation_font_size=10, annotation_font_color=p_colors[p_val], row=2, col=1)
            except Exception as e:
                print(f"Could not calculate/plot percentiles for {indicator_name}: {e}")
        else:
            print(f"Not enough valid data to calculate percentiles for {indicator_name}.")

    else:
        print(f"Warning: Indicator column '{indicator_name}' not found or empty.")

    # # --- Layout Updates ---
    # fig.update_layout(
    #     title_text=f"Stock Price Analysis: Overlays & {indicator_name}",
    #     height=800,
    #     legend_title_text='Legend',
    # )

    # # Update y-axis titles
    # fig.update_yaxes(title_text="Price / Overlays", row=1, col=1)
    # fig.update_yaxes(title_text=indicator_name, row=2, col=1)

    # Remove gaps for non-trading days work if DatetimeIndex
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    # # Hide the X-axis title for the top subplot as it's shared
    # fig.update_xaxes(title_text=None, row=1, col=1)
    # # Set X-axis title for the bottom subplot (which is the shared one)
    # fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.show()

    return fig


def calculate_relative_performance(symbol1: str, df1: pd.DataFrame, symbol2: str, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative performance of two stocks based on their closing prices.

    Args:
        symbol1: Name/ticker of first stock
        df1: DataFrame containing OHLC data for first stock with 'close' column
        symbol2: Name/ticker of second stock
        df2: DataFrame containing OHLC data for second stock with 'close' column

    Returns:
        DataFrame with percentage change from first close for each stock,
        indexed by date and with columns named after the symbols.
    """
    # Calculate percentage change from first close for each dataframe
    perc_change1 = 100 * (df1["close"] / df1["close"].iloc[0] - 1)
    perc_change2 = 100 * (df2["close"] / df2["close"].iloc[0] - 1)

    # Create new dataframe combining both series aligned by date
    rel_df = pd.DataFrame({symbol1: perc_change1, symbol2: perc_change2})

    return rel_df.dropna()


def plot_relative_performance(df: pd.DataFrame, show_indicator: str = "hist") -> None:
    rel_df = df.iloc[-200:]
    all_dates = pd.date_range(rel_df.index[0], rel_df.index[-1])
    missing_dates = all_dates.difference(rel_df.index)
    """
    Plot relative performance of stocks over time using Plotly,
    including a histogram showing the difference between the first two columns.
    
    Args:
        rel_df: DataFrame containing percentage change values for each stock,
                typically output from calculate_relative_performance()
                
    Returns:
        None (displays interactive plot)
    """
    if len(rel_df.columns) < 2:
        raise ValueError("DataFrame must have at least two columns to compare")

    # Create figure with subplots
    fig = subp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7, 0.3], subplot_titles=("Performance", "Performance Difference"))

    # Add performance lines to top subplot
    for col in rel_df.columns[:2]:  # Only show first two columns if there are more
        fig.add_trace(go.Scatter(x=rel_df.index, y=rel_df[col], name=col, mode="lines", connectgaps=False), row=1, col=1)

    # Calculate differences between first two columns and add histogram to bottom subplot
    diff = rel_df.iloc[:, 1] - rel_df.iloc[:, 0]
    if show_indicator == "hist":
        # Calculate some statistics for annotation
        mean_diff = diff.mean()
        median_diff = diff.median()

        fig.add_trace(go.Histogram(x=diff, name=f"{rel_df.columns[0]} - {rel_df.columns[1]}", marker_color="#FFA15A", opacity=0.75, nbinsx=50, hoverinfo="x+y"), row=2, col=1)

        # Add vertical lines for mean and median differences
        fig.add_vline(x=mean_diff, line_dash="dot", annotation_text=f"Mean: {mean_diff:.2f}", annotation_position="top right", line_color="red", row=2, col=1)

        fig.add_vline(x=median_diff, line_dash="dash", annotation_text=f"Median: {median_diff:.2f}", annotation_position="bottom right", line_color="blue", row=2, col=1)

        # Customize layout
        fig.update_layout(
            title_text=f"Relative Performance Comparison & Difference Analysis<br>{rel_df.columns[0]} vs {rel_df.columns[1]}",
            height=800,
            showlegend=True,
            bargap=0.05,  # gap between bars in histogram
        )
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text=f"Difference ({rel_df.columns[0]} - {rel_df.columns[1]})", row=2, col=16)
    elif show_indicator == "bar":
        # Show the difference rel_df.columns[1]} - {rel_df.columns[0] as a barchart
        fig.add_trace(go.Bar(x=diff.index, y=diff.values, name=f"{rel_df.columns[1]} - {rel_df.columns[0]}", marker_color="teal", opacity=0.75, hoverinfo="x+y"), row=2, col=1)
        percentiles = diff.quantile([0.1, 0.5, 0.9])
        # Add percentile lines
        for p, value in percentiles.items():
            fig.add_hline(y=value, line_dash="dash", line_color="salmon", annotation_text=f"{p * 100:.0f}% {value:.2f}", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="salmon", row=2, col=1)

        # Customize layout
        fig.update_layout(
            title_text=f"Relative Performance Comparison & Difference Analysis<br>{rel_df.columns[0]} vs {rel_df.columns[1]}",
            height=800,
            showlegend=True,
        )
        # fig.update_xaxes(
        #     type='category',
        #     tickangle=45,
        #     tickmode='array',
        #     tickvals=diff.index[::21],  # Show ~every 21 trading days (monthly)
        #     ticktext=diff.index[::21].strftime('%b %d, %Y'),
        #     nticks=12
        # )

    else:
        raise ValueError("show_indicator must be either 'hist' or 'bar'")

    # Update axis titles
    fig.update_yaxes(title_text="Percentage Change (%)", row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(values=missing_dates)], row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(values=missing_dates)], row=2, col=1)
    # fig.update_xaxes(title_text="Date", row=1, col=1)

    fig.show()


def plot_tr_pct(df, n):
    # Calculate previous day's close
    df["prev_close"] = df["close"].shift(1)

    # Calculate True Range components
    df["true_high"] = df[["high", "prev_close"]].max(axis=1)
    df["true_low"] = df[["low", "prev_close"]].min(axis=1)

    # Compute True Range and percentage
    df["TR"] = df["true_high"] - df["true_low"]
    df["TR_pct"] = (df["TR"] / df["close"]) * 100
    df["diff"] = (df["close"] / df["ema19"] - 1.0) * 100
    plot_daily_bar(df.copy(), "diff", "ratio close ema19", n)


def plot_daily_bar(df: pd.DataFrame, colname: str, collabel: str, n: int):
    # Clean data
    df = df.dropna(subset=[colname]).copy()
    idx = df.index
    all_dates = pd.date_range(idx[0], idx[-1])
    missing_dates = all_dates.difference(idx)

    # Calculate moving average
    ma = df[colname].rolling(n).mean()

    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Bar(x=df.index, y=df[colname], name=colname, marker_color="#636EFA"))

    fig.add_trace(go.Scatter(x=df.index, y=ma, mode="lines", name=f"{n}-Period MA", line=dict(color="#FFA15A", width=2)))

    # Calculate percentiles in a df [value colour label] index on percentile
    qs = [0.1, 0.5, 0.9]
    quant_df = df[colname].quantile(qs).round(2).to_frame(name="value").assign(color=["#00CC96", "#AB63FA", "#EF553B"], label=lambda x: (x.index * 100).astype(int).astype(str) + "% " + x["value"].astype(str))

    # Add horizontal lines
    for _, row in quant_df.iterrows():
        fig.add_hline(y=row["value"], line_dash="dot", line_color=row["color"], annotation_text=row["label"], annotation_position="bottom left")

    # Configure layout
    fig.update_layout(
        title=f"{collabel} with {n} ma",
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(211, 211, 211, 0.3)",
            rangebreaks=[dict(values=missing_dates)],
        ),
        yaxis=dict(title=colname, range=[df[colname].min(), df[colname].max()], gridcolor="rgba(211, 211, 211, 0.3)"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
    return fig


def three_line_break(xs):
    """input a series of closes. return a series with the 3LB value after the current close."""
    tlb, rev = tsutils.calc_tlb(xs, 3)
    ys = (tlb.close - tlb.open).apply(lambda e: 1 if e > 0 else -1)
    ys.rename("tlb", inplace=True)
    df2 = pd.merge(xs, ys, how="left", left_index=True, right_index=True)
    # df2.tlb.ffill(inplace=True)
    # df2.tlb.fillna(0, inplace=True)
    # forward fill missing values, replace initial nan with 0
    return df2.tlb.ffill().fillna(0).astype("int32")


def print_summary_information(symbol: str, df: pd.DataFrame, mas: list[str]):
    # Summary Information Table
    close = df["close"]
    date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    trading_days = len(df)

    high_idx = close.idxmax()
    high_value = close.max()
    high_date = high_idx.strftime("%Y-%m-%d")

    low_idx = close.idxmin()
    low_value = close.min()
    low_date = low_idx.strftime("%Y-%m-%d")

    last_value = close.iloc[-1]
    avg_vol = int(df["volume"].iloc[-20:].mean())

    pct_in_range = round(((last_value - low_value) / (high_value - low_value)) * 100)
    pct_drawdown = round(((last_value / high_value) - 1) * 100, 2)
    pct_off_low = round((last_value / low_value - 1) * 100, 2)

    # Create summary table
    summary_table = Table(title="Summary Information", style="white")

    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Symbol", symbol)
    summary_table.add_row("Range", date_range)
    summary_table.add_row("Trading Days", str(trading_days))
    summary_table.add_row("First", f"{close.iloc[0]:.2f}")
    summary_table.add_row("High", f"{high_date} {high_value:.2f}")
    summary_table.add_row("Low", f"{low_date} {low_value:.2f}")
    summary_table.add_row("Last", f"{last_value:.2f}")
    summary_table.add_row("Volume Avg", f"{avg_vol:,d}")
    summary_table.add_row("% in range", f"{pct_in_range}%")
    summary_table.add_row("% drawdown", f"{pct_drawdown}%")
    summary_table.add_row("% off low", f"{pct_off_low}%")

    # Current price info calculations
    price_data = {"name": ["high", "low", "last"] + mas, "value": [high_value, low_value, last_value] + [df[ma].iloc[-1] for ma in mas]}

    # Create DataFrame
    price_df = pd.DataFrame(price_data)

    price_df["pct_diff"] = ((price_df["value"] - last_value) / last_value) * 100

    # Formatting functions as vectorized operations
    def format_pct_diff(row):
        if row["name"] == "last":
            return "-"
        if abs(row["pct_diff"]) < 0.01:
            return "~0.00%"

        colour = "[white]"
        arrow = ""
        if row["pct_diff"] > 0:
            colour = "[green]"
            arrow = " ▲"
        elif row["pct_diff"] < 0:
            colour = "[red]"
            arrow = " ▼"

        return f"{colour}{row['pct_diff']:>8.2f}%{arrow}[/]"

    price_df["formatted_value"] = price_df["value"].apply(lambda x: f"{x:.2f}")
    price_df["formatted_pct_diff"] = price_df.apply(format_pct_diff, axis=1)

    # Sort by value descending
    price_df.sort_values("value", ascending=False, inplace=True)
    # Create current price table (assuming you're using Rich Table)
    current_price_table = Table(title="Current Price Information", style="white")
    current_price_table.add_column("Metric", style="cyan")
    current_price_table.add_column("Value", justify="right")
    current_price_table.add_column("% Difference", justify="right")

    for _, row in price_df.iterrows():
        current_price_table.add_row(row["name"], row["formatted_value"], row["formatted_pct_diff"])

    # Print both tables using rich
    console.print(summary_table)
    console.print(current_price_table)


def process(symbol: str, df: pd.DataFrame, sw_perc: float = 5.0):
    print_summary_information(symbol, df, ["ema19", "sma50", "sma150"])
    print(df.iloc[-20:])
    console.print("\n-- 3 line break", style="yellow")
    tlb, rev = tsutils.calc_tlb(df.close, 3)
    print(tlb[-5:])
    console.print(f"\n-- swings {sw_perc}", style="yellow")
    swings = tsutils.find_swings(df.close, sw_perc)
    print(swings)
    # print drawdowns if in upswing
    r = swings.iloc[-1]
    if r.change > 0:
        print(f"\n--- p/b limits\nhigh {r.end}\n2%   {r.end * 0.98:.2f}")
        print(f"5%   {r.end * 0.95:.2f}\n10%  {r.end * 0.9:.2f}")
    # plot_swings(swings)
    # plot('spy', df)
    print_range_table(df, [5, 10, 20, 50, 100, 250])


def plot_latest(symbol: str) -> pd.DataFrame | None:
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        #        plot(symbol, df, ['ema19', 'sma50', 'sma150'], 500)
        plot_tv(symbol, df, ["ema19", "sma50", "sma150"])
        return df
    return None


def plot_latest_3lb(symbol: str) -> pd.DataFrame | None:
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        plot_3lb(symbol, df)
        return df
    return None


def plot_latest_ind(symbol: str) -> pd.DataFrame | None:
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        add_historic_volatility(df, 20)
        df["above"] = df["close"] > df["ema19"]
        # plot_with_indicator(symbol, df, 'voln', ['sma150', 'sma50', 'ema19'], "above")
        plot_tr_pct(df, 10)
        return df
    return None


def plot_relative(symbol: str):
    xs = list_cached_files("spy")
    ys = list_cached_files(symbol)
    if len(xs) > 0 and len(ys) > 0:
        df = load_file(xs[0])
        df2 = load_file(ys[0])
        rel = calculate_relative_performance("spy", df, symbol, df2)
        plot_relative_performance(rel, "bar")


def view(symbol: str) -> pd.DataFrame | None:
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        process(symbol, df, 5.0 if (symbol == "spy" or symbol == "qqq") else 10.0)
        save_excel(symbol, df)
        return df
    return None


def load(symbol: str):
    load_twelve_data(symbol, 520)
    return view(symbol)


def list_cached(symbol: str):
    print(f"cached files for symbol {symbol}")
    for p in list_cached_files(symbol):
        print(str(p))


def save_excel(symbol: str, df: pd.DataFrame):
    fn = make_fullpath(symbol + ".xlsx")
    writer = pd.ExcelWriter(fn, engine="xlsxwriter")
    df.to_excel(writer, sheet_name=symbol)
    workbook = writer.book
    worksheet = writer.sheets[symbol]
    hilo = df.columns.get_loc("hilo") + 1  # index of col
    rng = xl_range(1, hilo, df.shape[0], hilo)  # 0 indexed

    fmt_decimal = workbook.add_format({"num_format": "#,##0.00"})
    fmt_date = workbook.add_format({"num_format": "dd/mm/yy", "align": "left"})
    # Add a format. Light red fill with dark red text.
    fmt_red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
    # Add a format. Green fill with dark green text.
    fmt_green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})

    worksheet.conditional_format(rng, {"type": "cell", "criteria": ">", "value": 19, "format": fmt_green})
    worksheet.conditional_format(rng, {"type": "cell", "criteria": "<", "value": -19, "format": fmt_red})
    worksheet.set_column("A:A", None, fmt_date)
    worksheet.set_column("B:E", None, fmt_decimal)
    worksheet.set_column("G:K", None, fmt_decimal)
    # worksheet.conditional_format('B2:B65', {'type': '3_color_scale'})

    writer.close()
    console.print(f"\nsaved {fn}", style="green")


def main_concat():
    xs = list_cached_files("spy")
    if len(xs) > 0:
        df = load_file("c:\\users\\niroo\\downloads\\spy 2023-12-15.csv")
        df2 = load_file(xs[0])
        df_updated = pd.concat([df, df2[-30:]])
        df_updated.drop_duplicates(inplace=True)
        df_updated.to_csv(make_fullpath("spy-lt.csv"))
        print(df_updated)


def concat(filename1, filename2, output_name):
    with open(filename1) as f1, open(filename2) as f2, open(output_name, "w") as f3:
        for line in f1:
            f3.write(line)
        for line in f2:
            if not line.startswith("datetime"):
                f3.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load EoD data from twelvedata.com")
    parser.add_argument("action", type=str, help="The action to perform [load|view|list|earliest|plot|plot3lb|plotind|plotrel]")
    parser.add_argument("symbol", type=str, help="The symbol to use in the action")
    args = parser.parse_args()
    if args.action == "load":
        load(args.symbol)
    elif args.action == "view":
        df = view(args.symbol)
        xs = scan(df, 19, -39, 0.95, 1.4)
        print(xs)
    elif args.action == "earliest":
        load_earliest_date(args.symbol)
    elif args.action == "list":
        list_cached(args.symbol)
    elif args.action == "plot":
        plot_latest(args.symbol)
    elif args.action == "plot3lb":
        plot_latest_3lb(args.symbol)
    elif args.action == "plotind":
        plot_latest_ind(args.symbol)
    elif args.action == "plotrel":
        plot_relative(args.symbol)

    # load_earliest_date('spy')
    # df = load_file('c:\\users\\niroo\\downloads\\spy 2023-12-15.csv')
    # plot('spy', df)
    # scan(df, 19, -35, .975)
    # xs = []
    # for x in range(10, 25):
    #     for y in range(19,50):
    #         d = scan(df, x, -y, .975)
    #         xs.append(d)
    # df2 = pd.DataFrame(xs)
    # print(df2)
    # plot_heatmap(df2)
