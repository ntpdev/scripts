#!/usr/bin/python3
import argparse
import glob
import re
from pathlib import Path
from textwrap import dedent

import pandas as pd
from rich.console import Console

console = Console()

MULTIPLIER = {"MES": 5, "MNQ": 2, "MYM": 0.5}


class Blotter:
    def __init__(self):
        self.openPositions = []
        self.nextSeqNo = 0
        self.seqDict = {}
        self.trades = []

    def process_trade_log(self, df: pd.DataFrame, skip_rows: int) -> pd.DataFrame:
        for i, r in df.iterrows():
            if i < skip_rows:
                continue
            for _ in range(r.Quantity):
                self.process_single_contract(r)
        return pd.DataFrame(self.trades)

    def process_single_contract(self, r):
        sym = r["Symbol"]
        found = self.find_matching(sym, r["Action"])
        if found == -1:
            if sym not in self.seqDict:
                self.seqDict[sym] = self.nextSeqNo
                self.nextSeqNo += 1
            self.openPositions.append(r)
        else:
            op = self.openPositions.pop(found)
            self.record_trade(op, r)
            if not any(x["Symbol"] == sym for x in self.openPositions):
                del self.seqDict[sym]

    def record_trade(self, op, cl):
        trade = {}
        trade["OpTm"] = op.Timestamp
        #        trade['ClTm'] = cl.Timestamp
        trade["Seq"] = self.seqDict[op["Symbol"]]
        trade["Symbol"] = op["Symbol"]
        trade["Action"] = op.Action
        trade["Open"] = op.Price
        trade["Close"] = cl.Price
        pts = (cl.Price - op.Price) * (1 if op.Action == "BOT" else -1)
        prf = calc_profit(op["Fin Instrument"][:3], pts)
        trade["Points"] = pts
        trade["Profit"] = prf
        trade["Comm"] = 1.24
        trade["Net"] = prf - 1.24
        self.trades.append(trade)

    def find_matching(self, symbol: str, action: str) -> int:
        """FIFO match on open positions"""
        opening_action = "SLD" if action == "BOT" else "BOT"
        found = -1
        for i, v in enumerate(self.openPositions):
            if v["Symbol"] == symbol and v["Action"] == opening_action:
                found = i
                break
        return found


def calc_profit(symbol: str, pts: float) -> float:
    return pts * MULTIPLIER.get(symbol, 1)


def print_trade_stats(trades: pd.Series, commissions: float = 0) -> None:
    def safe_div(n: float, d: float) -> float:
        return n / d if abs(d) > 1e-6 else 0

    # Basic trade metrics
    min_sz = 3
    wins_mask = trades > min_sz
    loses_mask = trades < -min_sz

    count_wins = wins_mask.sum()
    sum_wins = trades[wins_mask].sum()
    count_loses = loses_mask.sum()
    sum_loses = trades[loses_mask].sum()
    count_trades = count_wins + count_loses

    win_perc = safe_div(100 * count_wins, count_trades)
    avg_win = safe_div(sum_wins, count_wins)
    avg_loss = safe_div(sum_loses, count_loses)
    ratio = safe_div(avg_win, -avg_loss)

    # Kelly Criterion calculation
    p = win_perc / 100  # Win probability (decimal)
    b = ratio  # Win/loss ratio
    kelly = safe_div((p * (b + 1) - 1), b)

    # Financial metrics
    profit = sum(trades)
    net_profit = profit - commissions

    # Print results
    console.print("\n--- trade stats ---", style="yellow")
    s = dedent(f"""\
    contracts: {len(trades)}  net profit: ${net_profit:.2f}  gross: ${profit:.2f}  commissions: ${commissions:.2f}
    wins: {count_wins} ({sum_wins:.2f})  losses: {count_loses} ({sum_loses:.2f})  win%: {win_perc:.0f}%
    avg win: {avg_win:.1f}  avg loss: {avg_loss:.1f}  W/L ratio: {ratio:.1f}
    Kelly criterion: {kelly:.2%} (suggested max risk per trade)
    """)

    # Optional: Risk of Ruin estimation
    if count_trades > 0:
        risk_of_ruin = ((1 - p) / p) ** (net_profit / abs(avg_loss)) if p > 0.5 else 1
        s += f"estimated risk of ruin: {risk_of_ruin:.2%}"
    console.print(s, style="green")


def load_file(fname: str) -> pd.DataFrame:
    #    df = pd.read_csv(Path.home() / "OneDrive" / "Documents" / fname, usecols=[0, 1, 2, 3, 4, 5, 6], parse_dates={"Timestamp": [5, 6]})
    df = pd.read_csv(Path("c:\\temp") / fname, usecols=[0, 1, 2, 3, 4, 5, 6], parse_dates={"Timestamp": [5, 6]})
    print(f"loaded {fname} {df.shape[0]} {df.shape[1]}")
    return df


def read_trades(filepath: str) -> pd.DataFrame:
    """Read trades from CSV file and return processed DataFrame"""
    # Only read the columns we need
    usecols = ["Fin Instrument", "Symbol", "Action", "Quantity", "Price", "Time", "Date"]

    # Optimize data types for memory efficiency
    dtype = {"Fin Instrument": "string", "Symbol": "string", "Action": "string", "Quantity": "int32", "Price": "float64", "Date": "string", "Time": "string"}

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype)

    # Create timestamp more efficiently
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y%m%d %H:%M:%S")

    # Drop the original Date and Time columns since we have Timestamp
    df = df.drop(columns=["Date", "Time"])

    console.print(f"loaded {Path(filepath).name} {len(df)} rows", style="green")

    # Return with desired column order
    return df[["Timestamp", "Fin Instrument", "Symbol", "Action", "Quantity", "Price"]]


def aggregrate_by_sequence(df):
    return df.groupby(["Seq"]).agg(Symbol=pd.NamedAgg(column="Symbol", aggfunc="first"), Action=pd.NamedAgg(column="Action", aggfunc="first"), Num=pd.NamedAgg(column="Symbol", aggfunc="count"), Profit=pd.NamedAgg(column="Profit", aggfunc="sum"))


def extract_date_from_filename(filepath: str) -> str:
    """Extract date from filename for sorting purposes"""
    filename = Path(filepath).name
    # Look for YYMMDD pattern in filename
    match = re.search(r"(\d{6})", filename)
    return match.group(1) if match else filename


def load_trades_from_input(input_spec: str) -> pd.DataFrame:
    """Load trades from either a single file or file specification"""
    if "*" in input_spec or "?" in input_spec:
        # Handle file specification with wildcards
        matching_files = glob.glob(input_spec)

        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {input_spec}")

        # Sort files by date extracted from filename
        matching_files.sort(key=extract_date_from_filename)

        dataframes = []
        for filepath in matching_files:
            df = read_trades(filepath)
            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True)
    # Handle single file
    return read_trades(input_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IB trade logs")
    parser.add_argument("--skip", metavar="skip", default=0, type=int, help="number of rows to skip")
    parser.add_argument("--input", metavar="input", default="", help="file name or file specification")
    args = parser.parse_args()

    if args.input:
        df = load_trades_from_input(args.input)
    else:
        # Default behavior - load specific files
        base_path = Path(r"c:\temp")
        default_files = [base_path / "trades-250606.csv", base_path / "trades-250613.csv", base_path / "trades-250620.csv"]
        dataframes = [read_trades(str(f)) for f in default_files]
        df = pd.concat(dataframes, ignore_index=True)

    b = Blotter()
    trades = b.process_trade_log(df, args.skip)
    console.print("\n--- trades ---", style="yellow")
    console.print(trades, style="cyan")
    c = len(b.openPositions)
    if c > 0:
        console.print(f"\nOpen contracts {c}", style="yellow")
        for p in b.openPositions:
            console.print(f"{p['Symbol']} {p['Action']} {p['Price']}", style="cyan")
    else:
        console.print("All contracts matched", style="yellow")
    print_trade_stats(trades["Profit"], trades["Comm"].sum())

    console.print("\n--- sequences ---", style="yellow")
    seqs = aggregrate_by_sequence(trades)
    console.print(seqs, style="cyan")
    print_trade_stats(seqs["Profit"], trades["Comm"].sum())
