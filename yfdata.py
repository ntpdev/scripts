"""Fetch historical stock price data from Yahoo Finance using Playwright."""

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

console = Console()

OUTPUT_DIR = Path.home() / "Downloads"


@dataclass(slots=True)
class SymbolData:
    symbol: str
    df: pd.DataFrame = field(repr=False)
    long_name: str = ""
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def __repr__(self) -> str:
        start_date = self.df.index[0].strftime("%Y-%m-%d")
        end_date = self.df.index[-1].strftime("%Y-%m-%d")
        return f"symbol: {self.symbol}, name: {self.long_name}, rows: {len(self.df)}, data: from {start_date} to {end_date}"


def accept_consent(page: Page) -> None:
    page.goto("https://uk.finance.yahoo.com/", wait_until="domcontentloaded")
    page.locator("#didomi-notice-agree-button").click(timeout=10_000)


def fetch_rest(page: Page, symbols: list[str], start: date, end: date) -> list[SymbolData]:
    results: list[SymbolData] = []
    accept_consent(page)
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
    accept_consent(page)
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
            dividend_records.append({"Date": cells[0], "dividend": float(m.group(1))})

    df = pd.DataFrame(records, columns=header)

    df = df.rename(
        columns={
            "Close*": "Close",
            "Adj Close**": "Adj Close",
        }
    )

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", "", regex=False),
            errors="coerce",
        )

    df["Date"] = pd.to_datetime(
        df["Date"].str.replace("Sept", "Sep", regex=False),
        format="%d %b %Y",
    )

    df = df.set_index("Date")
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    dividends = pd.DataFrame(dividend_records)
    if dividends.empty:
        dividends = pd.DataFrame(columns=["dividend"])
        dividends.index.name = "Date"
    else:
        dividends["Date"] = pd.to_datetime(
            dividends["Date"].str.replace("Sept", "Sep", regex=False),
            format="%d %b %Y",
        )
        dividends = dividends.set_index("Date")
        dividends.index.name = "Date"

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
            "Open": quote_col("open"),
            "High": quote_col("high"),
            "Low": quote_col("low"),
            "Close": quote_col("close"),
            "Volume": pd.array(quote["volume"], dtype="Int64"),
        },
        index=to_index(result["timestamp"]),
    )
    df.index.name = "Date"
    df = df.dropna(subset=["Close"])
    divs = result.get("events", {}).get("dividends", {})
    if divs:
        div_records = sorted(divs.values(), key=lambda d: d["date"])
        dividends = pd.DataFrame(
            {"dividend": [d["amount"] for d in div_records]},
            index=to_index([d["date"] for d in div_records]),
        )
        dividends.index.name = "Date"
    else:
        dividends = pd.DataFrame()
    return SymbolData(symbol=symbol, df=df, long_name=long_name, dividends=dividends)


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in items:
        cleaned = item.symbol.replace(".", "-")
        prices_path = OUTPUT_DIR / f"{cleaned}.csv"
        item.df.to_csv(prices_path)
        console.print(f"Saved {prices_path}", style="cyan")
        if not item.dividends.empty:
            dividends_path = OUTPUT_DIR / f"{cleaned}-dividends.csv"
            item.dividends.to_csv(dividends_path)
            console.print(f"Saved {dividends_path}", style="cyan")


def main() -> None:
    start = date(2025, 4, 14)
    end = date(2026, 4, 10)
    data = scrape_data(["ISF.L"], start, end, headless=False)
    save_data(data)
    for item in data:
        console.print(item)


if __name__ == "__main__":
    main()
