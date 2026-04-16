"""Fetch historical stock price data from Yahoo Finance using Playwright."""

import argparse
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import Page, sync_playwright
from rich.console import Console
from rich.table import Table

import tsutils

console = Console()

OUTPUT_DIR = Path.home() / "Downloads"


@dataclass(slots=True)
class SymbolData:
    symbol: str
    df: pd.DataFrame = field(repr=False)
    long_name: str = ""
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def __str__(self) -> str:
        start_date = self.df.index[0].strftime("%Y-%m-%d")
        end_date = self.df.index[-1].strftime("%Y-%m-%d")
        return f"symbol: {self.symbol}, name: {self.long_name}, rows: {len(self.df)}, data: from {start_date} to {end_date}"


def accept_consent(page: Page) -> None:
    page.goto("https://uk.finance.yahoo.com/", wait_until="domcontentloaded")
    page.locator("#didomi-notice-agree-button").click(timeout=10_000)


def fetch_rest(page: Page, symbols: list[str], start: date, end: date) -> list[SymbolData]:
    results: list[SymbolData] = []
    for symbol in symbols:
        api_url = make_data_url(symbol, start, end)
        console.print(f"javascript fetch {api_url}")
        data = page.evaluate(
            """
            async (url) => {
                const resp = await fetch(url);
                return await resp.json();
            }
            """,
            api_url,
        )
        results.append(parse_json(data, symbol))
    return results


def scrape_data(symbols: list[str], start: date, end: date, headless: bool = True, use_html_table: bool = False) -> list[SymbolData]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        accept_consent(page)
        if use_html_table:
            results = fetch_html_scrape(page, symbols)
        else:
            results = fetch_rest(page, symbols, start, end)
        context.close()
        browser.close()
    return results


def clean_header(text: str) -> str:
    return (
        text.replace("Closing price adjusted for splits.", "")
        .replace(
            "Adjusted closing price adjusted for splits and dividend and/or capital gain distributions.",
            "",
        )
        .strip()
    )


def fetch_html_scrape(page: Page, symbols: list[str]) -> list[SymbolData]:
    results: list[SymbolData] = []

    for symbol in symbols:
        page.goto(
            f"https://uk.finance.yahoo.com/quote/{symbol}/history/",
            wait_until="domcontentloaded",
        )
        table = page.locator('[data-testid="history-table"] table')
        table.wait_for(timeout=30_000)
        table_html = table.evaluate("el => el.outerHTML")
        table_tag = BeautifulSoup(table_html, "html.parser").find("table")
        df, dividends = parse_table(table_tag)
        results.append(SymbolData(symbol=symbol, df=df, dividends=dividends))

    return results


def parse_table(table: Tag) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = table.find_all("tr")
    header = [clean_header(th.get_text(strip=True)) for th in rows[0].find_all("th")]

    records: list[list[str]] = []
    dividend_records: list[dict[str, str | float]] = []

    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]

        if len(cells) == 7:
            records.append(cells)
        elif len(cells) == 2 and (m := re.match(r"([\d.]+)\s*Dividend", cells[1])):
            dividend_records.append({"date": cells[0], "dividend": float(m.group(1))})

    df = pd.DataFrame(records, columns=header)

    df = df.rename(
        columns={
            "Close*": "close",
            "Adj Close**": "adj close",
        }
    )
    df.columns = df.columns.str.lower()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", "", regex=False),
            errors="coerce",
        )

    df["date"] = pd.to_datetime(
        df["date"].str.replace("Sept", "Sep", regex=False),
        format="%d %b %Y",
    )

    df = df.set_index("date")
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]

    dividends = pd.DataFrame(dividend_records)
    if dividends.empty:
        dividends = pd.DataFrame(columns=["dividend"])
        dividends.index.name = "date"
    else:
        dividends["date"] = pd.to_datetime(
            dividends["date"].str.replace("Sept", "Sep", regex=False),
            format="%d %b %Y",
        )
        dividends = dividends.set_index("date")
        dividends.index.name = "date"

    return df, dividends


def parse_json(data: dict, symbol: str) -> SymbolData:
    def to_index(timestamps) -> pd.DatetimeIndex:
        ts = np.array(timestamps, dtype="int64")
        return pd.to_datetime(ts, unit="s", utc=True).normalize().tz_localize(None)

    def quote_col(key: str) -> np.ndarray:
        return np.round(np.array(quote[key], dtype="float64"), 2)

    result = data["chart"]["result"][0]
    long_name = result["meta"].get("longName", "")
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote_col("open"),
            "high": quote_col("high"),
            "low": quote_col("low"),
            "close": quote_col("close"),
            "volume": pd.array(quote["volume"], dtype="Int64"),
        },
        index=to_index(result["timestamp"]),
    )
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    divs = result.get("events", {}).get("dividends", {})
    if divs:
        div_records = sorted(divs.values(), key=lambda d: d["date"])
        dividends = pd.DataFrame(
            {"dividend": [d["amount"] for d in div_records]},
            index=to_index([d["date"] for d in div_records]),
        )
        dividends.index.name = "date"
    else:
        dividends = pd.DataFrame()
    return SymbolData(symbol=result["meta"].get("symbol", symbol), df=df, long_name=long_name, dividends=dividends)


def make_data_url(symbol: str, start: date, end: date) -> str:
    def to_timestamp(d: date) -> int:
        return int(datetime.combine(d, time.min, UTC).timestamp())

    params = {
        "events": "capitalGain|div|split",
        "formatted": "true",
        "includeAdjustedClose": "true",
        "interval": "1d",
        "period1": to_timestamp(start),
        "period2": to_timestamp(end + timedelta(days=1)),
        "symbol": symbol,
        "userYfid": "true",
        "lang": "en-GB",
        "region": "GB",
    }
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{urlencode(params)}"


def save_data(items: list[SymbolData]) -> None:
    for item in items:
        cleaned = item.symbol.replace(".", "-")
        prices_path = OUTPUT_DIR / f"{cleaned}.csv"
        item.df.to_csv(prices_path)
        console.print(f"Saved {prices_path}", style="cyan")
        if not item.dividends.empty:
            dividends_path = OUTPUT_DIR / f"{cleaned}-dividends.csv"
            item.dividends.to_csv(dividends_path)
            console.print(f"Saved {dividends_path}", style="cyan")


def augment_data(items: list[SymbolData]) -> None:
    for item in items:
        _augment_data(item.df)


def _augment_data(df: pd.DataFrame) -> None:
    close = df["close"]
    df["sma150"] = close.rolling(window=150).mean().round(2)
    df["sma50"] = close.rolling(window=50).mean().round(2)
    df["ema19"] = close.ewm(span=19, adjust=False).mean().round(2)
    df["change"] = close.diff().round(2)
    df["pct_chg"] = (close.pct_change() * 100).round(2)
    df["ddown"] = (close / close.cummax() - 1).round(4)
    df["voln"] = (100 * (df["volume"] - df["volume"].rolling(window=20).mean()) / df["volume"].rolling(window=20).std()).fillna(0).round(0).astype(int)
    df["hilo"] = tsutils.calc_hilo(close)
    df["strat"] = df["high"].diff().gt(0).astype(int) + df["low"].diff().lt(0) * 2
    tlb, _ = tsutils.calc_tlb(close, 3)
    ys = (tlb.close - tlb.open).apply(lambda e: 1 if e > 0 else -1)
    ys.rename("tlb", inplace=True)
    merged = pd.merge(close, ys, how="left", left_index=True, right_index=True)
    df["tlb"] = merged.tlb.ffill().fillna(0).astype("int32")


#     print_summary_information(symbol, df, ["ema19", "sma50", "sma150"])
def print_summary_information(symbolData: SymbolData, mas: list[str]):
    # Summary Information Table
    df = symbolData.df
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

    def calculate_returns(start_date, end_date):
        close_start = df.loc[start_date, "close"]
        close_end = df.loc[end_date, "close"]
        num_days = (end_date - start_date).days
        total_return = close_end / close_start - 1
        return total_return, (1 + total_return) ** (365 / num_days) - 1 if num_days > 0 else 0

    _, ann_whole = calculate_returns(df.index[0], df.index[-1])
    ann_whole_pct = round(ann_whole * 100, 2)

    if high_idx < low_idx:
        _, ann_hl = calculate_returns(high_idx, low_idx)
    else:
        _, ann_hl = calculate_returns(low_idx, high_idx)
    ann_hl_pct = round(ann_hl * 100, 2)

    # Monthly investment returns
    total_invested, current_value = calculate_monthly_investment_returns(df)
    dca_return_pct = round(((current_value / total_invested - 1) * 100), 2)
    dca_annualized = round(100 * ((current_value / total_invested) ** (365 / (df.index[-1] - df.index[0]).days) - 1), 2)

    # Create summary table
    summary_table = Table(title="Summary Information", style="white")

    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Symbol", symbolData.symbol)
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
    summary_table.add_row("Ann %", f"{ann_whole_pct}%")
    summary_table.add_row("Ann H-L %", f"{ann_hl_pct}%")
    summary_table.add_row("DCA (Monthly $100)", f"${total_invested:.2f}")
    summary_table.add_row("Current Value", f"${current_value:.2f}")
    summary_table.add_row("DCA Return", f"{dca_return_pct}%")
    summary_table.add_row("DCA Ann", f"{dca_annualized}%")

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

    # Bad data detection via robust z-score on High/Low ratio
    ratio = df["high"] / df["low"]
    median = ratio.median()
    mad = (ratio - median).abs().median()
    robust_z = 0.6745 * (ratio - median) / mad
    z_threshold = 10
    flagged = robust_z.abs() > z_threshold
    if flagged.any():
        flag_table = Table(title=f"Potential Bad Data (robust |z| > {z_threshold})", style="red")
        flag_table.add_column("Date", style="cyan")
        flag_table.add_column("High", justify="right")
        flag_table.add_column("Low", justify="right")
        flag_table.add_column("H/L Ratio", justify="right")
        flag_table.add_column("Robust Z", justify="right")
        for idx in flagged[flagged].index:
            flag_table.add_row(
                idx.strftime("%Y-%m-%d"),
                f"{df.loc[idx, 'high']:.2f}",
                f"{df.loc[idx, 'low']:.2f}",
                f"{ratio.loc[idx]:.4f}",
                f"{robust_z.loc[idx]:.2f}",
            )
        console.print(flag_table)


def print_range_table(sd: SymbolData, xs):
    df = sd.df
    last = df["close"].iat[-1]

    headers = ["Range", "High", "Low", "Last", "% Ddown", "% HVol", "% Range"]
    tbl = Table(title="Range / Volatilty data", style="cyan")
    for h in headers:
        tbl.add_column(h, justify="right")

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    for n in xs:
        mx_cl = df["close"].iloc[-n:].max()
        mx = df["high"].iloc[-n:].max()
        mn = df["low"].iloc[-n:].min()
        rng = 100.0 * (last - mn) / (mx - mn)
        volatility = log_returns.rolling(window=n).std() * np.sqrt(252)
        tbl.add_row(f"{n}d", f"{mx:.2f}", f"{mn:.2f}", f"{last:.2f}", f"{(last / mx_cl - 1) * 100:.2f}", f"{100 * volatility.iloc[-1]:.1f}", f"{rng:.1f}")

    console.print(tbl)


def print_tlb(sd: SymbolData):
    df = sd.df
    tlb, rev = tsutils.calc_tlb(df.close, 3)
    # tlb = tlb2[-100:]
    tlb["height"] = tlb["close"] - tlb["open"]
    tlb["dirn"] = np.sign(tlb["height"])
    last_dirn = tlb.iat[-1, 3]

    trend = "uptrend" if last_dirn > 0 else "downtrend"
    console.print(f"\n--- 3 Line Break\n{trend}, reversal {rev}", style="yellow")
    console.print(tlb[-5:])


def calculate_monthly_investment_returns(df: pd.DataFrame) -> tuple[float, float]:
    """Calculate returns from investing $100 on the first trading day of each month.

    Assumes fractional shares can be held.

    Args:
        df: DataFrame with datetime index and 'close' column.

    Returns:
        Tuple of (total_invested, current_value)
    """
    df = df.copy()
    df["month"] = df.index.to_period("M")
    # Get the first trading day of each month
    first_days = df.groupby("month").head(1).index
    first_closes = df.loc[first_days, "close"]
    # Shares bought each month
    shares_bought = 100 / first_closes
    total_shares = shares_bought.sum()
    total_invested = 100 * len(first_closes)
    last_close = df["close"].iloc[-1]
    current_value = total_shares * last_close
    return total_invested, current_value


def load_from_csv(symbols: list[str]) -> list[SymbolData]:
    results: list[SymbolData] = []
    for symbol in symbols:
        cleaned = symbol.replace(".", "-")
        prices_path = OUTPUT_DIR / f"{cleaned}.csv"
        df = pd.read_csv(prices_path, index_col="date", parse_dates=True)
        dividends_path = OUTPUT_DIR / f"{cleaned}-dividends.csv"
        dividends = pd.read_csv(dividends_path, index_col="date", parse_dates=True) if dividends_path.exists() else pd.DataFrame()
        results.append(SymbolData(symbol=symbol, df=df, dividends=dividends))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical stock price data from Yahoo Finance")
    parser.add_argument("action", choices=["load", "view"], help="Action to perform")
    parser.add_argument("symbols", nargs="+", help="One or more stock symbols")
    parser.add_argument("--years", type=int, default=2, help="Number of years of history to fetch (default: 2)")
    args = parser.parse_args()

    if args.action == "load":
        today = date.today()
        start = today.replace(year=today.year - args.years)
        end = today
        data = scrape_data(args.symbols, start, end, headless=False)
        augment_data(data)
        save_data(data)
    elif args.action == "view":
        data = load_from_csv(args.symbols)
    else:
        parser.error(f"unsupported action: {args.action}")

    for item in data:
        console.print(str(item))
        print_summary_information(item, ["ema19", "sma50", "sma150"])
        print_range_table(item, [5, 10, 20, 50, 200])
        print_tlb(item)


if __name__ == "__main__":
    main()
