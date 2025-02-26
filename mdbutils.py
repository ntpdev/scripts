#!/usr/bin/python3

from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pymongo.mongo_client import MongoClient
from rich.console import Console
from rich.table import Table

import tsutils as ts

console = Console()


class SymbolSummary(BaseModel):
    """Model representing a summary for a single symbol."""
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(alias="_id")
    count: int
    start: datetime
    end: datetime


def print_symbol_summary_table(symbol_summaries: list[SymbolSummary]):
    # Create a table instance
    table = Table(title="Symbol Summaries")

    # Add columns to the table
    table.add_column("Symbol", justify="left", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Start", justify="center", style="green")
    table.add_column("End", justify="center", style="green")

    # Add rows to the table
    for summary in symbol_summaries:
        table.add_row(summary.symbol, str(summary.count), summary.start.strftime("%Y-%m-%d %H:%M:%S"), summary.end.strftime("%Y-%m-%d %H:%M:%S"))

    # Print the table to the console
    console.print(table)


def load_summary(collection: MongoClient):
    """Return a summary of all symbols in the collection as a DataFrame with columns [count, start, end], indexed by symbol."""

    pipeline = [{"$group": {"_id": "$symbol", "count": {"$sum": 1}, "start": {"$min": "$timestamp"}, "end": {"$max": "$timestamp"}}}, {"$sort": {"_id": 1}}]

    df = pd.DataFrame(collection.aggregate(pipeline))
    df.rename(columns={"_id": "symbol"}, inplace=True)
    return df.set_index("symbol")


def load_summary_ex(collection: MongoClient) -> list[SymbolSummary]:
    """Return a summary of all symbols in the collection as a pydantic model."""

    pipeline = [{"$group": {"_id": "$symbol", "count": {"$sum": 1}, "start": {"$min": "$timestamp"}, "end": {"$max": "$timestamp"}}}, {"$sort": {"_id": 1}}]
    return [SymbolSummary(**doc) for doc in collection.aggregate(pipeline)]


def load_timeseries(collection, symbol, tm_start, tm_end):
    """Load time series data for a specific symbol within a time range with exclusive end and calculate EMA."""
    print(f"loading {symbol} {tm_start} to {tm_end}")

    cursor = collection.find(filter={"symbol": symbol, "timestamp": {"$gte": tm_start, "$lt": tm_end}}, sort=["timestamp"])

    df = pd.DataFrame(map(lambda d: {"date": d["timestamp"], "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"], "volume": d["volume"], "vwap": d["vwap"]}, cursor)).set_index("date")

    # Calculate EMA and add it to the DataFrame
    df["ema"] = df.close.ewm(span=87, adjust=False).mean()

    return df


def load_trading_days(collection, symbol, min_vol):
    """return df of complete trading days. the bars are aggregated by calendar date [date, bar-count, volume, standardised-volume]"""
    pipeline = [{"$match": {"symbol": symbol}}, {"$group": {"_id": {"$dateTrunc": {"date": "$timestamp", "unit": "day"}}, "count": {"$sum": 1}, "volume": {"$sum": "$volume"}}}, {"$match": {"volume": {"$gte": min_vol}}}, {"$sort": {"_id": 1}}]
    df = pd.DataFrame(collection.aggregate(pipeline))
    v = df.volume
    df["stdvol"] = (v - v.mean()) / v.std()
    df.set_index("_id", inplace=True)
    df.index.rename("date", inplace=True)
    return df


def load_gaps(collection, symbol, gap_mins):
    """return table of gaps in m1 time series. last bar of day and first bar of next day [last_bar, first_bar, gap]"""
    cursor = collection.aggregate(
        [
            {
                "$match": {"symbol": symbol},
            },
            {
                "$setWindowFields": {
                    "sortBy": {"timestamp": 1},
                    "output": {
                        "lastTm": {
                            "$shift": {
                                "output": "$timestamp",
                                "by": -1,
                            },
                        },
                    },
                },
            },
            {
                "$project": {
                    "timestamp": 1,
                    "lastTm": 1,
                    "gap": {
                        "$dateDiff": {
                            "startDate": "$lastTm",
                            "endDate": "$timestamp",
                            "unit": "minute",
                        },
                    },
                },
            },
            {"$match": {"gap": {"$gte": gap_mins}}},
        ]
    )
    return pd.DataFrame(map(lambda r: {"last_bar": r["lastTm"], "first_bar": r["timestamp"], "gap": r["gap"]}, cursor))


def find_datetime_range(df, dt, n):
    """find start,end interval for n days beginning or ending with dt"""
    n = n if abs(n) > 1 else 1
    d = min(dt, df.index[-1])
    df_range = df[df.index >= d][:n] if n > 0 else df[df.index <= d][n:]
    # return start of first row and end of last row
    return df_range.iat[0, 0], df_range.iat[-1, 1]


def make_trade_dates(tm_start, tm_end, df_gaps):
    """build df of [trade_date, start, end, rth_start] where range is [start, end). rth_start may be NaT"""
    e = pd.concat([df_gaps["last_bar"], pd.Series(tm_end)], ignore_index=True)
    s = pd.concat([pd.Series(tm_start), df_gaps["first_bar"]], ignore_index=True)
    rs = s + pd.Timedelta(minutes=930)

    # mask rth_start values where rth_start is after end
    df = pd.DataFrame({"start": s, "end": e + pd.Timedelta(minutes=1), "rth_start": rs.mask(e < rs)})
    df.set_index(e.dt.date, inplace=True)
    df.index.name = "date"
    return df


def calculate_trading_hours(df_trade_days, dt, range_name):
    """return start and end inclusive datetime for a given date and range name. range_name is 'rth' or 'glbx' or 'day'."""
    try:
        st = df_trade_days.at[dt, "start"]
        end = df_trade_days.at[dt, "end"]  # df_trade_days has exclusive end
        if range_name == "rth":
            return st + pd.Timedelta(minutes=930), st + pd.Timedelta(minutes=1319)
        if range_name == "glbx":
            return st, st + pd.Timedelta(minutes=929)
        return st, end
    except KeyError:
        console.print(f"KeyError: {dt} not found in dataframe", style="red")
    return None


def load_price_history(symbol, dt, n=1):
    """return m1 bars for n days. if n is negative dt is the last day loaded"""
    client = MongoClient("localhost", 27017)
    collection = client["futures"].m1
    df_summary = load_summary(collection)
    df_gaps = load_gaps(collection, symbol, 30)
    df_trade_days = make_trade_dates(df_summary.at[symbol, "start"], df_summary.at[symbol, "end"], df_gaps)
    s, e = find_datetime_range(df_trade_days, dt, n)
    return load_timeseries(collection, symbol, s, e)


def main(symbol: str):
    client = MongoClient("localhost", 27017)
    collection = client["futures"].m1

    df_summary = load_summary(collection)
    console.print("--- summary of collection", style="yellow")
    console.print(df_summary)

    console.print("\n\n--- summary of collection from pydantic", style="yellow")
    print_symbol_summary_table(load_summary_ex(collection))

    df_days = load_trading_days(collection, symbol, 100000)
    console.print(f"\n\n--- trading days for {symbol}", style="yellow")
    console.print(df_days)

    df_gaps = load_gaps(collection, symbol, 30)
    console.print(f"\n\n--- gaps for {symbol}", style="yellow")
    console.print(df_gaps)

    # this is like day_index but uses the gaps mdb query
    df_trade_days = make_trade_dates(df_summary.at[symbol, "start"], df_summary.at[symbol, "end"], df_gaps)
    console.print(f"\n\n--- trade date index for {symbol}", style="yellow")
    console.print(df_trade_days)

    tms, tme = find_datetime_range(df_trade_days, date.today(), -5)
    df = load_timeseries(collection, symbol, tms, tme)
    tms, tme = calculate_trading_hours(df_trade_days, tme.date(), "rth")
    console.print(f"\n\n--- m1 bars from {tms} to {tme}", style="yellow")
    rows = df[tms:tme]
    if not rows.empty:
        console.print(rows.head())
        console.print(rows.tail())

    df_di = ts.day_index(df)
    console.print("\n\n--- day index", style="yellow")
    console.print(df_di)

    console.print("\n\n--- day summary", style="yellow")
    summ = ts.create_day_summary(df, df_di)
    console.print(summ)
    console.print(f"\n\n--- last row {summ.index[-1]}", style="yellow")
    console.print(summ.iloc[-1])


if __name__ == "__main__":
    main("esh5")
