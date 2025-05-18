#!/usr/bin/python3
import argparse
from textwrap import dedent
import pandas as pd
from rich.console import Console
from rich.pretty import pprint
from pathlib import Path

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

    def find_matching(self, symbol: str, action: str):
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


def print_trade_stats(trades):
    # Basic trade metrics
    min_sz = 3
    wins = trades.Profit[trades.Profit > min_sz].count()
    sum_wins = trades.Profit[trades.Profit > min_sz].sum()
    loses = trades.Profit[trades.Profit < -min_sz].count()
    sum_loses = trades.Profit[trades.Profit < -min_sz].sum()
    win_perc = 100 * wins / (wins + loses) if (wins + loses) > 0 else 0
    avg_win = sum_wins / wins if wins > 0 else 0
    avg_loss = sum_loses / loses if loses > 0 else 0
    ratio = avg_win / -avg_loss if avg_loss < 0 else 0

    # Kelly Criterion calculation
    p = win_perc / 100  # Win probability (decimal)
    b = ratio  # Win/loss ratio
    kelly = (p * (b + 1) - 1) / b if b > 0 else 0

    # Financial metrics
    profit = trades.Profit.sum()
    commisions = trades.Comm.sum()
    net_profit = profit - commisions

    # Print results
    console.print("\n--- trade stats ---", style="yellow")
    s = dedent(f"""\
    contracts: {len(trades)}  net profit: ${net_profit:.2f}  gross: ${profit:.2f}  commissions: ${commisions:.2f}
    wins: {wins} ({sum_wins:.2f})  losses: {loses} ({sum_loses:.2f})  win%: {win_perc:.0f}%
    avg win: {avg_win:.1f}  avg loss: {avg_loss:.1f}  W/L ratio: {ratio:.1f}
    Kelly criterion: {kelly:.2%} (suggested max risk per trade)
    """)

    # Optional: Risk of Ruin estimation
    if wins > 0 and loses > 0:
        risk_of_ruin = ((1 - p) / p) ** (net_profit / abs(avg_loss)) if p > 0.5 else 1
        s += f"estimated risk of ruin: {risk_of_ruin:.2%}"
    console.print(s, style="green")


def load_file(fname: str) -> pd.DataFrame:
    df = pd.read_csv(Path.home() / "OneDrive" / "Documents" / fname, usecols=[0, 1, 2, 3, 4, 5, 6], parse_dates={"Timestamp": [5, 6]})
    print(f"loaded {fname} {df.shape[0]} {df.shape[1]}")
    return df


def read_trades(filepath: str) -> pd.DataFrame:
    usecols = ["Fin Instrument", "Symbol", "Action", "Quantity", "Price", "Time", "Date"]
    dtype = {"Quantity": "int32", "Price": "float64", "Date": "str", "Time": "str"}

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype)

    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y%m%d %H:%M:%S")
    console.print(f"loaded {filepath} {df.shape[0]} {df.shape[1]}", style="green")

    final_cols = ["Timestamp", "Fin Instrument", "Symbol", "Action", "Quantity", "Price"]
    return df[final_cols]


def aggregrate_by_sequence(df):
    return df.groupby(["Seq"]).agg(Symbol=pd.NamedAgg(column="Symbol", aggfunc="first"), Action=pd.NamedAgg(column="Action", aggfunc="first"), Num=pd.NamedAgg(column="Symbol", aggfunc="count"), Profit=pd.NamedAgg(column="Profit", aggfunc="sum"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IB trade logs")
    parser.add_argument("--skip", metavar="skip", default=0, type=int, help="number of rows to skip")
    parser.add_argument("--input", metavar="input", default="", help="file name")
    args = parser.parse_args()

    if args.input:
        df = read_trades(args.input)
    else:
        df = pd.concat([load_file("trades-0214.csv"), load_file("trades.20220222.csv"), load_file("trades.20220223.csv")])

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
    console.print("\n--- sequences ---", style="yellow")
    console.print(aggregrate_by_sequence(trades), style="cyan")
    print_trade_stats(trades)
