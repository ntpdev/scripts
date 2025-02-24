#!/usr/bin/env python3
# Note chmod +x *.py
# ensure Unix style line endings

from datetime import date, datetime, timedelta
from pathlib import Path
from twelvedata import TDClient
import argparse
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subp
import requests as req
from xlsxwriter.utility import xl_range
import tsutils

# pip install twelvedata
BASEDIR = 'Downloads'
APIKEY = os.environ['TW_API_KEY']

tdc = TDClient(apikey=APIKEY)

def make_fullpath(fn: str) -> Path:
    return Path.home() / BASEDIR / fn


def make_filename(symbol: str, dt: date) -> Path:
    return make_fullpath(f'{symbol} {dt.isoformat()}.csv')


def list_cached_files(symbol: str):
    """list files most recent first"""
    p = Path.home() / BASEDIR
    return sorted(p.glob(symbol + ' 202*.csv'), reverse=True)


def load_file(fname: str):
    df = pd.read_csv(make_fullpath(fname), parse_dates=['datetime'], index_col='datetime', engine='python')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df


def json_to_df(objs):
    return pd.DataFrame(objs['values'])


def plot(symbol, df):
    pts = -250
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[pts:], y=df[pts:]['close'], mode='lines', name=symbol))
    fig.add_trace(go.Scatter(x=df.index[pts:], y=df[pts:]['ema19'], mode='lines', name='ema19'))
    # fig1 = px.line(df[pts:], x=df.index[pts:], y=['close', 'ema19'], title=symbol, template='plotly_dark')
    x = df[pts:]
    x = x[x['hilo'] > 19]
    # print(x)
    fig.add_trace(go.Scatter(x=x.index, y=x['close'], mode='markers', name='20d high', marker=dict(color='green')))
    x = df[pts:]
    x = x[x['hilo'] < -19]
    fig.add_trace(go.Scatter(x=x.index, y=x['close'], mode='markers', name='20d low', marker=dict(color='red')))
    # fig2 = px.scatter(x, x=x.index, y='close', color='green')

    # fig3 = px.Figure(data=fig1.data + fig2.data)
    # fig3.show()
    fig.show()


def plot_3lb(symbol, df):
    tlb2, rev = tsutils.calc_tlb(df.close, 3)
    tlb = tlb2[-100:]
    tlb['height'] = tlb['close']-tlb['open']
    tlb['dirn'] = np.sign(tlb['height'])
    colours = tlb['dirn'].map({-1: "red", 1: "green"})
    xs = tlb.index.strftime('%Y-%m-%d')
    fig = subp.make_subplots(rows=1, cols=2, subplot_titles=([symbol + ' 3LB', symbol + ' close']))
    f1 = go.Bar(x = xs.values, y = tlb['height'], base = tlb['open'], name=symbol, marker=dict(color = colours))
    f2 = go.Scatter(x = df.index[-100:], y = df[-100:]['close'], mode='lines', name=symbol, marker=dict(color = 'blue'))

    fig.add_trace(f1, row=1, col=1)
    fig.add_trace(f2, row=1, col=2)
    c = 'green' if df.loc[df.index[-1], 'close'] < rev else 'red'
    fig.add_hline(y=rev, line_width=1, line_color=c, line_dash="dash", row=1, col=2)
    fig.add_annotation(text=f"reversal {rev:.2f}", x=df.index[-100], y=rev * 1.01,
                     font=dict(size=12, color=c),
                     showarrow=False, row=1, col=2)
    fig.update_layout(xaxis_type='category')
    fig.show()


def plot_cumulative(df):
    # Create a barchart using the 'perc' column from the same dataframe
    fig = subp.make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(x=df['dtOpen'], y=df['perc'], name='perc'))
    fig.add_trace(go.Scatter(x=df['dtOpen'], y=df['cumulative'], mode='lines'), secondary_y=True)
    # Show the final figure
    fig.show()


def plot_heatmap(df):
    d = []
    for i,r in df.iterrows():
        s = r['system'].split(',')
        entry = s[0]
        ex = s[1]
        v = r['cumulative']
        # v = r['maxdd']
        d.append({'entry':entry, 'exit':ex, 'value':v})
    map = pd.DataFrame(d)
    # breakpoint()
    # Add a heatmap trace to the figure
    fig = go.Figure(
        data=go.Heatmap(
        x=map['entry'],
        y=map['exit'],
        z=map['value'],
        colorscale='Viridis',
        showscale=True))

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
def normalise_as_perc(v, n=20):
    return (100 * (v - v.rolling(window=n).mean())/v.rolling(window=n).std()).fillna(0).round(0).astype(int)


def strat(hs, ls):
    '''return a series categorising bar by its strat bar type 0 - inside, 1 up, 2 down, 3 outside'''
    x = hs.diff().gt(0)
    y = ls.diff().lt(0)
    return x.astype(int) + y * 2


def calc_range(df, xs):
    last = df['close'].iloc[-1]
    print('\n--- ranges')
    for n in xs:
        mx = df['high'].iloc[-n:].max()
        mn = df['low'].iloc[-n].min()
        rng = 100. * (last - mn) / (mx - mn)
        print(f'range {n:2d} {mx} {mn} {last} {rng:.1f}')


# returns {'datetime': '1993-01-29', 'unix_time': 728317800} for SPY
def load_earliest_date(symbol):
    s = f'https://api.twelvedata.com/earliest_timestamp?symbol={symbol}&interval=1day&apikey={APIKEY}'
    response = req.get(url=s)
    if response.status_code == 200:
        objs = response.json()
        print(objs)


def load_twelve_data(symbol, days=255):
    print(f'loading {symbol}')
    # ts = td.time_series(symbol=symbol, interval='1day', start_date='2010-01-01', outputsize=5000, dp=2, order='ASC')
    ts = tdc.time_series(symbol=symbol, interval='1day', outputsize=days, dp=2, order='ASC')
    df = ts.with_ma(ma_type='SMA', time_period=150).with_ma(ma_type='SMA', time_period=50).with_ma(ma_type='EMA', time_period=19).as_pandas()

    df.rename(columns={'ma1':'sma150', 'ma2':'sma50', 'ma3':'ema19'}, inplace=True)
    print(df.tail())
    fname = make_filename(symbol, df.index[-1].date())
    df['change'] = pd.Series.diff(df.close).round(2)
    df['pct_chg'] = (pd.Series.pct_change(df.close) * 100).round(2)
    df['voln'] = normalise_as_perc(df.volume)
    # df['perc'] = percFromMin(df.close)
    df['hilo'] = tsutils.calc_hilo(df.close)
    df['strat'] = strat(df.high, df.low)
    df['tlb'] = three_line_break(df.close)
    df.to_csv(fname)
    print(f'saved {symbol} {df.index[0].date()} to {df.index[-1].date()} shape={df.shape}')
    print(fname)
    return df


def load_twelve_data_raw(symbols):
    dt = (datetime.today().date() - timedelta(days=365)).isoformat()
    url = f'https://api.twelvedata.com/time_series?apikey={APIKEY}&interval=1day&start_date={dt}&symbol={symbols}&type=etf&format=JSON&dp=2&order=ASC'
    print(url)
    response = req.get(url=url)
    if response.status_code == 200:
        objs = response.json()
        df = json_to_df(objs)
        fn = f'c:\\users\\niroo\\downloads\\{symbols} {dt.date()}.csv'
        df.set_index('datetime', inplace=True)
        df.to_csv(fn)
        print('saved ' + fn)


def scan(df, entry_hi, exit_lo, stop_perc, target_perc):
    xs = []
    state = 0
    stop = None
    for i, row in df.iterrows():
        if state == 0 and row['hilo'] > entry_hi:
            entry = i
            state = 1
            stop = row['close'] * stop_perc
            target = row['close'] * target_perc
        elif state == 1 and (row['close'] < stop or row['hilo'] < exit_lo):
        # elif state == 1 and row['hilo'] < -19:
           xs.append((entry, i))
           state = 0
        elif state == 1 and row['close'] > target:
            xs.append((entry, i))
            state = 0

    ys = []
    for ent, ex in xs:
        d = {'dtOpen': ent,
             'dtClose': ex,
             'open': df.at[ent, 'close'],
             'close': df.at[ex, 'close']}
        ys.append(d)
    df2 = pd.DataFrame(ys)
    df2['points'] = df2['close'] - df2['open']
    df2['perc'] = (df2['points'] / df2['open']) * 100.
    df2['cumulative'] = (df2['close'] / df2['open']).cumprod()
    df2['drawdown'] = df2['cumulative'] - df2['cumulative'].expanding().max() 
    print(df2)
    pts = df2['points']
    wins = pts > 0
    losses = pts < 0
    return {'system': f'{entry_hi},{exit_lo},{stop_perc},{target_perc}',
            'pts': pts.sum(),
            'cumulative': df2['cumulative'].iat[-1],
            'maxdd': round(df2['drawdown'].min() * 100.,2),
            'winC': pts[wins].count(),
            'winT': pts[wins].sum(),
            'lossC':pts[losses].count(),
            'lossT':pts[losses].sum()
            }


def plot_swings(df):
    #        fig = px.line(x=swings.index, y=swings['close'])
    fig = px.bar(x=df.index, y=df['change'])
    fig.update_layout(xaxis_type='category') # treat datetime as category
    fig.show()


def three_line_break(xs):
    '''input a series of closes. return a series with the 3LB value after the current close.'''
    tlb, rev = tsutils.calc_tlb(xs, 3)
    ys = (tlb.close - tlb.open).apply(lambda e: 1 if e > 0 else -1)
    ys.rename('tlb', inplace=True)
    df2 = pd.merge(xs, ys, how='left', left_index=True, right_index=True)
    # df2.tlb.ffill(inplace=True)
    # df2.tlb.fillna(0, inplace=True)
    # forward fill missing values, replace initial nan with 0
    return df2.tlb.ffill().fillna(0).astype('int32')


def process(df, sw_perc = 5.0):
    print(df[-20:])
    print('\n-- 3 line break')
    tlb, rev = tsutils.calc_tlb(df.close, 3)
    print(tlb[-5:])
    print(f'\n-- swings {sw_perc}')
    swings = tsutils.find_swings(df.close, sw_perc)
    print(swings)
    # print drawdowns if in upswing
    r = swings.iloc[-1]
    if r.change > 0:
        print(f'\n--- p/b limits\nhigh {r.end}\n2%   {r.end * .98:.2f}')
        print(f'5%   {r.end * .95:.2f}\n10%  {r.end * .9:.2f}')
    #plot_swings(swings)
    # plot('spy', df)
    calc_range(df, [5,20,50])


def plot_latest(symbol: str):
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        plot(symbol, df)
        return df
    return None


def plot_latest_3lb(symbol: str):
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        plot_3lb(symbol, df)
        return df
    return None


def view(symbol: str):
    xs = list_cached_files(symbol)
    if len(xs) > 0:
        df = load_file(xs[0])
        process(df, 5.0 if (symbol == 'spy' or symbol == 'qqq') else 10.0)
        save_excel(symbol, df)
        return df
    return None


def load(symbol: str):
    load_twelve_data(symbol, 520)
    return view(symbol)


def list_cached(symbol: str):
    print(f'cached files for symbol {symbol}')
    for p in list_cached_files(symbol):
        print(str(p))


def save_excel(symbol: str, df: pd.DataFrame):
    fn = make_fullpath(symbol + '.xlsx')
    writer = pd.ExcelWriter(fn, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=symbol)
    workbook = writer.book
    worksheet = writer.sheets[symbol]
    hilo = df.columns.get_loc('hilo') + 1 # index of col
    rng = xl_range(1, hilo, df.shape[0], hilo) # 0 indexed

    fmt_decimal = workbook.add_format({'num_format': '#,##0.00'})
    fmt_date = workbook.add_format({'num_format': 'dd/mm/yy', 'align': 'left'})
    # Add a format. Light red fill with dark red text.
    fmt_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    # Add a format. Green fill with dark green text.
    fmt_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})

    worksheet.conditional_format(rng, {'type': 'cell', 'criteria': '>', 'value': 19, 'format': fmt_green}) 
    worksheet.conditional_format(rng, {'type': 'cell', 'criteria': '<', 'value': -19, 'format': fmt_red}) 
    worksheet.set_column('A:A', None, fmt_date)
    worksheet.set_column('B:E', None, fmt_decimal)
    worksheet.set_column('G:K', None, fmt_decimal)
    #worksheet.conditional_format('B2:B65', {'type': '3_color_scale'})

    writer.close()
    print(f'saved {fn}')


def main_concat():
    xs = list_cached_files('spy')
    if len(xs) > 0:
        df = load_file('c:\\users\\niroo\\downloads\\spy 2023-12-15.csv')
        df2 = load_file(xs[0])
        df_updated = pd.concat([df, df2[-30:]])
        df_updated.drop_duplicates(inplace=True)
        df_updated.to_csv(make_fullpath('spy-lt.csv'))
        print(df_updated)


def concat(filename1, filename2, output_name):
    with open(filename1, 'r') as f1, open(filename2, 'r') as f2, open(output_name, 'w') as f3:
        for line in f1:
            f3.write(line)
        for line in f2:
            if not line.startswith('datetime'):
                f3.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load EoD data from twelvedata.com')
    parser.add_argument('action', type=str, help='The action to perform [load|view|list|earliest|plot|plot3lb]')
    parser.add_argument('symbol', type=str, help='The symbol to use in the action')
    args = parser.parse_args()
    if args.action == 'load':
        load(args.symbol)
    elif args.action == 'view':
        df = view(args.symbol)
        xs = scan(df, 19, -39, .95, 1.4)
        print(xs)
    elif args.action == 'earliest':
        load_earliest_date(args.symbol)
    elif args.action == 'list':
        list_cached(args.symbol)
    elif args.action == 'plot':
        plot_latest(args.symbol)
    elif args.action == 'plot3lb':
        plot_latest_3lb(args.symbol)

    #load_earliest_date('spy')
    #df = load_file('c:\\users\\niroo\\downloads\\spy 2023-12-15.csv')
    #plot('spy', df)
    #scan(df, 19, -35, .975)
    # xs = []
    # for x in range(10, 25):
    #     for y in range(19,50):
    #         d = scan(df, x, -y, .975)
    #         xs.append(d)
    # df2 = pd.DataFrame(xs)
    # print(df2)
    # plot_heatmap(df2)
