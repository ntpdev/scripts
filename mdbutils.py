#!/usr/bin/python3

from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pymongo.mongo_client import MongoClient
from rich.console import Console
from rich.table import Table

import tsutils as ts

console = Console()


class SummaryResults(BaseModel):
    """Model representing a summary for a single symbol."""
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(alias="_id")
    count: int
    start: datetime
    end: datetime


def print_symbol_summary_table(symbol_summaries: list[SummaryResults]):
    # Create a table instance
    table = Table(title="symbol summaries")

    # Add columns to the table
    table.add_column("symbol", justify="left", style="cyan")
    table.add_column("count", justify="right", style="magenta")
    table.add_column("start", justify="center", style="green")
    table.add_column("end", justify="center", style="green")

    # Add rows to the table
    for s in symbol_summaries:
        table.add_row(s.symbol, str(s.count), s.start.strftime("%Y-%m-%d %H:%M:%S"), s.end.strftime("%Y-%m-%d %H:%M:%S"))

    # Print the table to the console
    console.print(table)


class MDBRepository:
    def __init__(self, host:str, database: str):
        self.host = host
        self.db = database
        self.coll = None
        self.client = None
    
    def collection(self):
        if self.client is None: 
            self.client = MongoClient(self.host, 27017)
            self.coll = self.client[self.db].m1
        return self.coll

    def load_summary(self):
        """Return a summary of all symbols in the collection as a DataFrame with columns [count, start, end], indexed by symbol."""

        pipeline = [ {
                "$group": {
                    "_id": "$symbol",
                    "count": {"$sum": 1},
                    "start": {"$min": "$timestamp"},
                    "end": {"$max": "$timestamp"}
                }
            },{
                "$sort": {"_id": 1}
        } ]

        df = pd.DataFrame(self.collection().aggregate(pipeline))
        df.rename(columns={"_id": "symbol"}, inplace=True)
        return df.set_index("symbol")


    def load_summary_ex(self) -> list[SummaryResults]:
        """Return a summary of all symbols in the collection as a pydantic model."""
        pipeline = [ {
                "$group": {
                    "_id": "$symbol",
                    "count": {"$sum": 1},
                    "start": {"$min": "$timestamp"},
                    "end": {"$max": "$timestamp"}
                }
            },{
                "$sort": {"_id": 1}
        } ]
        return [SummaryResults(**doc) for doc in self.collection().aggregate(pipeline)]


    def load_timeseries(self, symbol: str, tm_start: pd.Timestamp, tm_end: pd.Timestamp) -> pd.DataFrame:
        """Load time series data for a specific symbol within a time range with exclusive end and calculate EMA."""
        print(f"loading {symbol} {tm_start} to {tm_end}")

        cursor = self.collection().find(filter={"symbol": symbol, "timestamp": {"$gte": tm_start, "$lt": tm_end}}, sort=["timestamp"])

        df = pd.DataFrame(map(lambda d: {"date": d["timestamp"], "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"], "volume": d["volume"], "vwap": d["vwap"]}, cursor)).set_index("date")

        # Calculate EMA and add it to the DataFrame
        df["ema"] = df.close.ewm(span=87, adjust=False).mean()

        return df


    def load_trading_days(self, symbol: str, min_vol: int) -> pd.DataFrame:
        """return df of complete trading days. the bars are aggregated by calendar date [date, bar-count, volume, standardised-volume]"""
        cursor = self.collection().aggregate( [ {
                    "$match": { "symbol": symbol }
                },{
                    "$group": {
                        "_id": { "$dateTrunc": {"date": "$timestamp", "unit": "day"} },
                        "count": { "$sum": 1 },
                        "volume": { "$sum": "$volume" } }
                },{
                    "$match": {
                        "volume": { "$gte": min_vol } }
                },{
                    "$sort": { "_id": 1 }
                } ] )
        df = pd.DataFrame(cursor)
        v = df.volume
        df["stdvol"] = (v - v.mean()) / v.std()
        df.set_index("_id", inplace=True)
        df.index.rename("date", inplace=True)
        return df


    def load_gaps(self, symbol: str, gap_mins: int) -> pd.DataFrame:
        """return table of gaps in m1 time series. last bar of day and first bar of next day [last_bar, first_bar, gap]"""
        cursor = self.collection().aggregate(
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


def lookup_trade_date(idx: pd.Index, dt_input: str) -> pd.Timestamp:
    """return a date that exists in the index. dt is an index value or a date typically in yyyymmdd format."""
    try:
        pos = int(dt_input)
        n = len(idx)
        if -n <= pos < n:
            return idx[pos]
    except ValueError:
        pass

    try:
        return asof_index(idx, pd.to_datetime(dt_input))
    except ValueError:
        pass

    return idx[-1]


def asof_index(idx: pd.Index, target: pd.Timestamp) -> pd.Timestamp:
    pos = idx.searchsorted(target, side="right") - 1
    return idx[pos] if pos >= 0 else idx[0]
            

def find_datetime_range(df: pd.DataFrame, dt: pd.Timestamp, n: int):
    """find start,end interval for n days beginning or ending with dt.
    dt must be a date in the index
    """
    n = n if abs(n) > 1 else 1
    d = min(dt, df.index[-1])
    df_range = df[df.index >= d][:n] if n > 0 else df[df.index <= d][n:]
    # return start of first row and end of last row
    return df_range.iat[0, 0], df_range.iat[-1, 1]


def make_trade_dates(tm_start, tm_end, df_gaps) -> pd.DataFrame:
    """return df of [trade_date, start, end, rth_start] where range is [start, end). rth_start may be NaT"""
    e = pd.concat([df_gaps["last_bar"], pd.Series(tm_end)], ignore_index=True)
    s = pd.concat([pd.Series(tm_start), df_gaps["first_bar"]], ignore_index=True)
    rs = s + pd.Timedelta(minutes=930)

    # mask rth_start values where rth_start is after end
    # use to_numpy() to remove indexes and allow the datetime to be used as an index
    df = pd.DataFrame({
        "start": s.to_numpy(),
        "end": (e + pd.Timedelta(minutes=1)).to_numpy(),
        "rth_start": rs.mask(e < rs).to_numpy()},
        index = e.dt.normalize())
    df.index.name = "date"
    return df


def calculate_trading_hours(df_trade_days: pd.DataFrame , dt: pd.Timestamp | date | str, range_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """return start and end inclusive datetime for a given date and range name. range_name is 'rth' or 'glbx' or 'day'."""   
    try:
        dt = dt if isinstance(dt, pd.Timestamp) else pd.to_datetime(dt)
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


def load_price_history(symbol:str, dt_str:str, n:int = 1) -> pd.DataFrame:
    """return m1 bars for n days. dt_str is either a date or and index value. if n is negative dt_str is the last day loaded"""
    mdb = MDBRepository("localhost", "futures")
    df_summary = mdb.load_summary()
    df_gaps = mdb.load_gaps(symbol, 30)
    df_trade_days = make_trade_dates(df_summary.at[symbol, "start"], df_summary.at[symbol, "end"], df_gaps)
    s, e = find_datetime_range(df_trade_days,  lookup_trade_date(df_trade_days.index, dt_str), n)
    return mdb.load_timeseries(symbol, s, e)


def print_summary_row(df: pd.DataFrame, row_idx: int | str | pd.Timestamp) -> None:
    """
    Print a single row from the DataFrame in the specified format using rich.
    
    Parameters:
        df: DataFrame with a DatetimeIndex and price columns.
        row_idx: The index (date or integer) of the row to print.
    """
    # Get the row as a Series
    row = df.loc[row_idx] if not isinstance(row_idx, int) else df.iloc[row_idx]
    
    # Filter only float columns (price columns)
    price_cols = [
        col for col in df.columns 
        if pd.api.types.is_float_dtype(df[col])
    ]
    
    # Build a mapping: price -> list of column names (labels)
    price_to_labels = {}
    for col in price_cols:
        price = row[col]
        if pd.isna(price):
            continue
        price_to_labels.setdefault(price, []).append(col)
    
    # Sort prices descending
    sorted_prices = sorted(price_to_labels.keys(), reverse=True)
    
    # Prepare rich console and table
    trade_date = (
        row.name.strftime("%Y-%m-%d")
        if isinstance(row.name, (pd.Timestamp, pd.DatetimeIndex))
        else str(row.name)
    )
    console.print(f"\n[bold]Trade date {trade_date}[/bold]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("price", justify="right")
    table.add_column("change", justify="right")
    table.add_column("name", justify="left")
    
    rth_open = row.get("rth_open", None)
    
    for price in sorted_prices:
        labels = ", ".join(sorted(price_to_labels[price]))
        if pd.notna(rth_open):
            change = price - rth_open
            change_str = f"{change:+.2f}" if change != 0 else ""
        else:
            change_str = ""
        table.add_row(f"{price:.2f}", change_str, labels)
    
    console.print(table)

        # --- Summary line ---
    # Get high/low prices and times
    rth_high = row.get("rth_high", None)
    rth_low = row.get("rth_low", None)
    rth_high_tm = row.get("rth_high_tm", None)
    rth_low_tm = row.get("rth_low_tm", None)
    
    # Only print summary if all values are present
    if (pd.notna(rth_high) and pd.notna(rth_low) and pd.notna(rth_high_tm) and pd.notna(rth_low_tm)):
        # Format times
        high_time_str = pd.to_datetime(rth_high_tm).strftime("%H:%M")
        low_time_str = pd.to_datetime(rth_low_tm).strftime("%H:%M")
        
        # Determine direction and compute change/time
        if rth_high_tm < rth_low_tm:
            # High to low
            direction = "traded from high to low"
            start_price, start_time = rth_high, high_time_str
            end_price, end_time = rth_low, low_time_str
            change = rth_low - rth_high
            time_delta = pd.to_datetime(rth_low_tm) - pd.to_datetime(rth_high_tm)
        else:
            # Low to high
            direction = "traded from low to high"
            start_price, start_time = rth_low, low_time_str
            end_price, end_time = rth_high, high_time_str
            change = rth_high - rth_low
            time_delta = pd.to_datetime(rth_high_tm) - pd.to_datetime(rth_low_tm)
        
        # Format time delta
        hours, remainder = divmod(time_delta.seconds, 3600)
        minutes = remainder // 60
        
        # Format change with sign
        change_str = f"{change:+.2f}"
        
        summary = (
            f"day {direction} {start_price:.2f} ({start_time}) "
            f"to {end_price:.2f} ({end_time}) {change_str} "
            f"{hours} hours {minutes} minutes"
        )
        console.print(summary)


def main(symbol: str):
    mdb = MDBRepository("localhost", "futures")

    df_summary = mdb.load_summary()
    console.print("--- summary of collection", style="yellow")
    console.print(df_summary)

    console.print("\n\n--- summary of collection from pydantic", style="yellow")
    print_symbol_summary_table(mdb.load_summary_ex())

    min_vol = 100000
    df_days = mdb.load_trading_days(symbol, min_vol)
    console.print(f"\n\n--- trading days for {symbol} min volume {min_vol}", style="yellow")
    console.print(df_days)

    df_gaps = mdb.load_gaps(symbol, 30)
    console.print(f"\n\n--- gaps for {symbol}", style="yellow")
    console.print(df_gaps)

    # this is like day_index but uses the gaps mdb query
    df_trade_days = make_trade_dates(df_summary.at[symbol, "start"], df_summary.at[symbol, "end"], df_gaps)
    console.print(f"\n\n--- trade date index for {symbol}", style="yellow")
    console.print(df_trade_days)

    tms, tme = find_datetime_range(df_trade_days, lookup_trade_date(df_trade_days.index, "-1"), -5)
    console.print(f"\nloaded 5 days from {tms} to {tme}", style="cyan")
    df = mdb.load_timeseries(symbol, tms, tme)
    tms, tme = calculate_trading_hours(df_trade_days, tme.normalize(), "rth")
    console.print(f"\n\n--- m1 rth bars from {tms} to {tme}", style="yellow")
    rows = df.loc[tms:tme]
    if not rows.empty:
        console.print(rows.head())
        console.print(rows.tail())

    df_di = ts.day_index(df)
    console.print("\n\n--- day index", style="yellow")
    console.print(df_di)

    console.print("\n\n--- day summary", style="yellow")
    summ = ts.create_day_summary(df, df_di)
    console.print(summ)
    console.print(f"\n\n--- last rows", style="yellow")
    print_summary_row(summ, -2)
    print_summary_row(summ, -1)


if __name__ == "__main__":
    main("esu5")
