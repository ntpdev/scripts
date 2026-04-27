"""
Fetch OHLCV data from StockCharts.com and store as Parquet.

Usage:
    python scdata.py
"""

import argparse
import json
import logging
import random
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TypeVar

import pandas as pd
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright
from rich.console import Console
from rich.markdown import Markdown

import tsutils as ts

console = Console()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOLS_RAW = """
spy qqq
xlb xlc xle xlf xli xlk xlp xlre xlu xlv xly
eem pbw xbi xop xme gld
isf.l vod.l pct.l mtro.l
"""

# JavaScript injected into the browser page to call the StockCharts data API.
# __SYMBOLS__ is replaced at runtime with a comma-separated list of tickers.
_FETCH_JS = """\
(async (symbols) => {
    const encoded = encodeURIComponent(symbols);
    const url = `https://stockcharts.com/json/data?cmd=get-daily-data&startDate=1999-01-01\
&dateAligned=true&src=freecharts-perf&symbols=${encoded}&r=${Date.now()}`;
    const response = await fetch(url);
    const data = await response.json();
    return JSON.stringify(data);
})('__SYMBOLS__')
"""

# ---------------------------------------------------------------------------
# Pure helpers — no browser, no I/O
# ---------------------------------------------------------------------------


def default_symbols() -> list[str]:
    """Return a sorted list of tickers from the default watchlist."""
    return sorted(_SYMBOLS_RAW.split())


T = TypeVar("T")


def iter_batches(items: Sequence[T], max_batch: int) -> Iterator[Sequence[T]]:
    """
    Yield evenly-sized slices of *items*, each with at most *max_batch* elements.

    >>> list(iter_batches(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], 4))
    [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i', 'j']]
    """
    total = len(items)
    if total <= 0 or max_batch < 1:
        return
    n_batches = (total + max_batch - 1) // max_batch  # ceil division
    base, extra = divmod(total, n_batches)
    offset = 0
    for size in [base + 1] * extra + [base] * (n_batches - extra):
        yield items[offset : offset + size]
        offset += size


def _float_or_none(value: object) -> float | None:
    """Return float, or None when the value is missing / empty."""
    match value:
        case None | "":
            return None
        case _:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None


_COLUMN_DTYPES = {
    "symbol": "category",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "Int64",
    "sctr_reg": "float32",
    "sctr_snp": "float32",
    "sctr_etf": "float32",
}

_COLUMNS = ["symbol", "date", *(_COLUMN_DTYPES.keys())]


def json_to_dataframe(json_str: str) -> pd.DataFrame:
    """
    Parse the StockCharts OHLCV JSON payload and return a typed DataFrame.

    The JSON structure is::

        {
          "symbols": {
            "AAPL": {
              "dailyData": [
                { "date": "2024-01-02",
                  "openPrice": 185.0, "highPrice": 187.0, "lowPrice": 184.0,
                  "closePrice": 186.0, "volume": 12345678,
                  "sctrReg": 72.1, "sctrSnp": 68.4, "sctrEtf": null },
                ...
              ]
            },
            ...
          }
        }

    Returns a DataFrame with plain columns.
    """
    payload: dict = json.loads(json_str)
    symbols_data: dict = payload.get("symbols", {})

    records = [
        {
            "symbol": symbol,
            "date": entry.get("date"),
            "open": _float_or_none(entry.get("openPrice")),
            "high": _float_or_none(entry.get("highPrice")),
            "low": _float_or_none(entry.get("lowPrice")),
            "close": _float_or_none(entry.get("closePrice")),
            "volume": entry.get("volume"),
            "sctr_reg": _float_or_none(entry.get("sctrReg")),
            "sctr_snp": _float_or_none(entry.get("sctrSnp")),
            "sctr_etf": _float_or_none(entry.get("sctrEtf")),
        }
        for symbol, body in symbols_data.items()
        for entry in body.get("dailyData", [])
    ]

    if not records:
        return pd.DataFrame(columns=_COLUMNS)

    return (
        pd.DataFrame.from_records(records)
        .assign(
            date=lambda d: pd.to_datetime(d["date"]),
            volume=lambda d: pd.to_numeric(d["volume"], errors="coerce"),
        )
        .astype(_COLUMN_DTYPES)
        .reset_index(drop=True)
    )


def save(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Write *df* to *output_dir* as a Parquet file using the PyArrow engine.
    """
    dest = output_dir / f"stockcharts-{date.today().strftime('%y%m%d')}.parquet"

    df.to_parquet(dest, engine="pyarrow", compression="snappy")
    log.info("Saved %d rows → %s", len(df), dest)
    return dest


def find_latest(directory: Path) -> Path | None:
    if files := list(directory.glob("stockcharts-*.parquet")):
        return max(files, key=lambda p: p.stem)
    return None


def find_latest_sctr(directory: Path) -> Path | None:
    if files := list(directory.glob("stockcharts-sctr-*.csv")):
        return max(files, key=lambda p: p.stem)
    return None


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a per-symbol summary table showing row count, first date, and last date.
    """
    summary = df.groupby("symbol", observed=True).agg(rows=("date", "size"), start=("date", "min"), end=("date", "max")).reset_index()
    col_widths = {
        "symbol": max(summary["symbol"].str.len().max(), 6),
        "rows": max(summary["rows"].astype(str).str.len().max(), 4),
        "start": 10,
        "end": 10,
    }
    header = f"{'symbol':<{col_widths['symbol']}}  {'rows':>{col_widths['rows']}}  {'start':>{col_widths['start']}}  {'end':>{col_widths['end']}}"
    print(header)
    print("-" * len(header))
    for row in summary.itertuples(index=False):
        print(f"{row.symbol:<{col_widths['symbol']}}  {row.rows:>{col_widths['rows']}}  {str(row.start):>{col_widths['start']}}  {str(row.end):>{col_widths['end']}}")
    print(df.info())


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------


def _dismiss_shadow_dom_popup(page: Page, host_selector: str) -> bool:
    """
    Click the consent/accept button inside a shadow-DOM host element.

    Returns ``True`` if dismissed, ``False`` if not found within 1 second.
    """
    try:
        button = page.locator(f"{host_selector} #cmpbntyestxt")
        button.wait_for(state="visible", timeout=1_000)
        button.click()
        log.info("Dismissed popup: %s", host_selector)
        return True
    except Exception:
        log.debug("Popup not found: %s", host_selector)
        return False


def _fetch_batch_json(page: Page, symbols: list[str]) -> str:
    """
    Inject JavaScript into *page* to call the StockCharts data API for
    *symbols* and return the raw JSON response string.
    """
    ticker_str = ",".join(s.upper() for s in sorted(symbols))
    js = _FETCH_JS.replace("__SYMBOLS__", ticker_str)
    result: str = page.evaluate(js)
    log.info("Fetched JSON for %d symbols (%d chars)", len(symbols), len(result))
    return result


# ---------------------------------------------------------------------------
# Browser session context manager
# ---------------------------------------------------------------------------


@dataclass
class _BrowserSession:
    """
    Thin wrapper that owns the Playwright lifecycle.

    Use as a context manager::

        with _BrowserSession() as session:
            session.page.navigate("https://example.com")
    """

    _pw: Playwright = field(init=False, repr=False)
    _browser: Browser = field(init=False, repr=False)
    _context: BrowserContext = field(init=False, repr=False)
    page: Page = field(init=False, repr=False)

    def __enter__(self) -> "_BrowserSession":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=False,
        )
        self._context = self._browser.new_context(viewport=None)
        self._install_request_blocking()
        self.page = self._context.new_page()
        return self

    def __exit__(self, *_: object) -> None:
        for obj in (self.page, self._context, self._browser, self._pw):
            try:
                obj.close()  # type: ignore[union-attr]
            except Exception as exc:
                log.warning("Error closing %s: %s", type(obj).__name__, exc)
        try:
            self._pw.stop()
        except Exception as exc:
            log.warning("Error stopping Playwright: %s", exc)

    def _install_request_blocking(self) -> None:
        """Abort known third-party tracker / ad / consent CDN requests."""
        blocked = [
            "**/cdn.cookielaw.org/**",
            "**/cookiebot.com/**",
            "**/usercentrics.eu/**",
            "**/didomi.io/**",
            "**/trustarc.com/**",
            "**/quantcast.mgr.consensu.org/**",
            "**/consent.cookiefirst.com/**",
            "**/widget.intercom.io/**",
            "**/js.driftt.com/**",
            "**/static.zdassets.com/**",
            "**/pagead2.googlesyndication.com/**",
        ]
        for pattern in blocked:
            self._context.route(pattern, lambda route, _: route.abort())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_stock_data(
    symbols: list[str],
    output_dir: Path,
    *,
    batch_max: int = 10,
) -> pd.DataFrame:
    """
    Launch a browser, navigate to StockCharts, dismiss the cookie popup, then
    fetch OHLCV + SCTR data for *symbols* in batches.

    Each batch result is accumulated into a single DataFrame which is saved as
    ``stockcharts-YYMMDD.parquet`` inside *output_dir* before being returned.

    Parameters
    ----------
    symbols:
        Ticker strings, e.g. ``["AAPL", "MSFT"]``.
    output_dir:
        Directory where the Parquet file is written.
    batch_max:
        Maximum number of symbols per API call (default 10).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with plain columns.
    """
    frames: list[pd.DataFrame] = []

    with _BrowserSession() as session:
        session.page.goto("https://stockcharts.com/")
        session.page.wait_for_load_state("domcontentloaded")
        _dismiss_shadow_dom_popup(session.page, "#cmpwrapper")

        for batch in iter_batches(symbols, batch_max):
            time.sleep(random.uniform(2.0, 5.0))
            raw_json = _fetch_batch_json(session.page, list(batch))
            df_batch = json_to_dataframe(raw_json)
            frames.append(df_batch)
            log.info("Batch done: %d rows accumulated", sum(len(f) for f in frames))

    if not frames:
        log.warning("No data retrieved.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    save(combined, output_dir)

    return combined


# ---------------------------------------------------------------------------
# SCTR data fetcher
# ---------------------------------------------------------------------------

_FETCH_SCTR_JS = """\
(async () => {
    const url = `https://stockcharts.com/j-sum/sum?cmd=sctr&view=L&timeframe=W&r=${Date.now()}`;
    const response = await fetch(url);
    const data = await response.json();
    return JSON.stringify(data);
})()
"""


def fetch_sctr_data(output_dir: Path) -> pd.DataFrame:
    """
    Launch a browser, navigate to StockCharts, dismiss the cookie popup, then
    fetch SCTR data and save to CSV.

    The CSV file is named ``stockcharts-sctr-YYMMDD.csv`` where the date comes
    from the first element of the JSON response.

    Parameters
    ----------
    output_dir:
        Directory where the CSV file is written.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [symbol, close, volume, mcap, SCTR, delta, name, industry, sector].
    """
    with _BrowserSession() as session:
        session.page.goto("https://stockcharts.com/")
        session.page.wait_for_load_state("domcontentloaded")
        _dismiss_shadow_dom_popup(session.page, "#cmpwrapper")

        result: str = session.page.evaluate(_FETCH_SCTR_JS)
        log.info("Fetched SCTR JSON (%d chars)", len(result))

    payload: list[dict] = json.loads(result)
    if not payload:
        log.warning("No SCTR data retrieved.")
        return pd.DataFrame()

    # Extract date from the first object
    date_str = payload[0].get("date", "")
    try:
        file_date = pd.to_datetime(date_str)
        date_label = file_date.strftime("%y%m%d")
    except (ValueError, TypeError):
        log.warning("Could not parse date '%s', using today.", date_str)
        date_label = date.today().strftime("%y%m%d")

    records = []
    for entry in payload[1:]:
        market_cap = _float_or_none(entry.get("marketCap"))
        mcap = round(market_cap / 1e9, 2) if market_cap is not None else None
        records.append(
            {
                "symbol": entry.get("symbol"),
                "close": _float_or_none(entry.get("close")),
                "volume": _float_or_none(entry.get("vol")),
                "mcap": mcap,
                "SCTR": entry.get("SCTR"),
                "delta": _float_or_none(entry.get("delta")),
                "name": entry.get("name"),
                "industry": entry.get("industry"),
                "sector": entry.get("sector"),
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        log.warning("No SCTR records parsed.")
        return df

    df = df[["symbol", "close", "volume", "mcap", "SCTR", "delta", "name", "industry", "sector"]]

    dest = output_dir / f"stockcharts-sctr-{date_label}.csv"
    df.to_csv(dest, index=False)
    log.info("Saved %d rows → %s", len(df), dest)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path.home() / "Downloads"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data from StockCharts.com and store as Parquet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    load = sub.add_parser("load", help="Fetch data from StockCharts and save to Parquet")
    load.add_argument("symbols", nargs="*", default=None, help="Ticker symbols (default: built-in watchlist)")

    view = sub.add_parser("view", help="Load the latest Parquet file and print summary info")
    view.add_argument("symbol", nargs="?", default=None, help="Optional single ticker to print tail for")

    sub.add_parser("sctr", help="Fetch SCTR data from StockCharts and save to CSV")

    sub.add_parser("rank", help="Load the latest SCTR CSV and print the top 12 rows")

    return parser


def _cmd_load(args: argparse.Namespace) -> None:
    """
    columns [symbol, date, open, high, low, close, volume, sctr_reg, sctr_snp, sctr_etf]
    """
    syms = sorted(args.symbols) if args.symbols else default_symbols()
    log.info("Fetching data for %d symbols → %s", len(syms), _OUTPUT_DIR)
    result = fetch_stock_data(syms, _OUTPUT_DIR)
    log.info("Done. DataFrame shape: %s", result.shape)


def _cmd_sctr(_args: argparse.Namespace) -> None:
    log.info("Fetching SCTR data → %s", _OUTPUT_DIR)
    result = fetch_sctr_data(_OUTPUT_DIR)
    log.info("Done. DataFrame shape: %s", result.shape)


def _cmd_rank(_args: argparse.Namespace) -> None:
    latest = find_latest_sctr(_OUTPUT_DIR)
    if latest is None:
        log.error("No stockcharts-sctr-*.csv files found in %s", _OUTPUT_DIR)
        sys.exit(1)
    log.info("Loading %s", latest)
    df = pd.read_csv(latest)
    print(f"File: {latest}")
    print(f"Shape: {df.shape}")
    print()

    # Compute display columns for all rows so both tables can select from them.
    display = df.copy()
    display["dtv"] = ((display["close"] * display["volume"]) / 1e9).round(2)
    display["volume"] = (display["volume"] / 1e6).round(2)
    display_cols = ["symbol", "close", "volume", "dtv", "mcap", "SCTR", "delta", "name", "industry", "sector"]
    display = display[display_cols]

    def _to_markdown(subset: pd.DataFrame) -> str:
        headers = list(subset.columns)
        tbl = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
        for row in subset.itertuples(index=False):
            tbl += f"| {' | '.join(str(v) for v in row)} |\n"
        return tbl

    # Table 1 – top 12 by SCTR (already ordered in the file)
    console.print(Markdown("## Top 12 SCTR stocks"))
    console.print(Markdown(_to_markdown(display.head(12))))

    # Table 2 – top 12 by DTV, then sorted by SCTR
    top_dtv = display.sort_values("dtv", ascending=False).head(12).sort_values("SCTR", ascending=False)
    console.print(Markdown("## Top 12 DTV stocks sorted by SCTR"))
    console.print(Markdown(_to_markdown(top_dtv)))


def _cmd_view(args: argparse.Namespace) -> None:
    latest = find_latest(_OUTPUT_DIR)
    if latest is None:
        log.error("No stockcharts-*.parquet files found in %s", _OUTPUT_DIR)
        sys.exit(1)
    log.info("Loading %s", latest)
    df = pd.read_parquet(latest, engine="pyarrow").astype({"symbol": "category"})
    print(f"File: {latest}")
    print(f"Shape: {df.shape}")
    print()

    if args.symbol:
        sym = args.symbol.upper()
        subset = df[df["symbol"] == sym]
        if subset.empty:
            log.error("Symbol %s not found in %s", sym, latest)
            sys.exit(1)
        print(f"Symbol: {sym}  Rows: {len(subset)}")
        subset = ts.augment_data(subset)
        print(subset.tail().to_string(index=False))
    else:
        print_summary(df)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    match args.command:
        case "load":
            _cmd_load(args)
        case "view":
            _cmd_view(args)
        case "sctr":
            _cmd_sctr(args)
        case "rank":
            _cmd_rank(args)
        case _:
            log.error("Unknown command: %s", args.command)
            sys.exit(1)
