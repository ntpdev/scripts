"""
Fetch OHLCV data from StockCharts.com and store as Parquet.

Usage:
    python stockcharts_fetcher.py
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

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
eem pbw xbi xop xme
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


def batch_sizes(total: int, max_batch: int) -> list[int]:
    """
    Split *total* into the fewest evenly-sized batches each ≤ *max_batch*.

    >>> batch_sizes(10, 4)
    [3, 3, 4]
    >>> batch_sizes(12, 4)
    [4, 4, 4]
    """
    if total <= 0 or max_batch <= 0:
        return []
    n_batches = (total + max_batch - 1) // max_batch  # ceil division
    base, extra = divmod(total, n_batches)
    return [base + (1 if i < extra else 0) for i in range(n_batches)]


def _float_or_none(value: Any) -> float | None:
    """Return float, or None when the value is missing / empty."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    Returns a DataFrame with plain columns sorted by ``(symbol, date)``.
    """
    payload: dict = json.loads(json_str)
    symbols_data: dict = payload.get("symbols", {})

    rows: list[dict] = []
    for symbol, body in symbols_data.items():
        for entry in body.get("dailyData", []):
            rows.append(
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
            )

    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "sctr_reg", "sctr_snp", "sctr_etf"])

    return (
        pd.DataFrame(rows)
        .assign(
            date=lambda d: pd.to_datetime(d["date"]),
            volume=lambda d: pd.to_numeric(d["volume"], errors="coerce"),
        )
        .astype({"symbol": "category", "open": "float32", "high": "float32", "low": "float32", "close": "float32", "volume": "Int64", "sctr_reg": "float32", "sctr_snp": "float32", "sctr_etf": "float32"})
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def save_dataframe(df: pd.DataFrame, dest: Path) -> None:
    """
    Write *df* to *dest* as a Parquet file using the PyArrow engine.

    Snappy compression is used for a good size/speed trade-off.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, engine="pyarrow", compression="snappy")
    log.info("Saved %d rows → %s", len(df), dest)


def find_latest_parquet(directory: Path) -> Path | None:
    """Find the most recent ``stockcharts-YYMMDD.parquet`` file in *directory*."""
    pattern = re.compile(r"^stockcharts-(\d{6})\.parquet$")
    best: Path | None = None
    best_date = ""
    for p in directory.glob("stockcharts-*.parquet"):
        m = pattern.match(p.name)
        if m and m.group(1) > best_date:
            best_date = m.group(1)
            best = p
    return best


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
        Combined DataFrame with plain columns sorted by ``(symbol, date)``.
    """
    sizes = batch_sizes(len(symbols), batch_max)
    frames: list[pd.DataFrame] = []
    offset = 0

    with _BrowserSession() as session:
        session.page.goto("https://stockcharts.com/")
        session.page.wait_for_load_state("domcontentloaded")
        _dismiss_shadow_dom_popup(session.page, "#cmpwrapper")

        for size in sizes:
            batch = symbols[offset : offset + size]
            offset += size
            time.sleep(random.uniform(2.0, 5.0))
            raw_json = _fetch_batch_json(session.page, batch)
            df_batch = json_to_dataframe(raw_json)
            frames.append(df_batch)
            log.info("Batch done: %d rows accumulated", sum(len(f) for f in frames))

    if not frames:
        log.warning("No data retrieved.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    date_str = date.today().strftime("%y%m%d")
    dest = output_dir / f"stockcharts-{date_str}.parquet"
    save_dataframe(combined, dest)

    return combined


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

    sub.add_parser("view", help="Load the latest Parquet file and print summary info")

    return parser


def _cmd_load(args: argparse.Namespace) -> None:
    """
    columns [symbol, date, open, high, low, close, volume, sctr_reg, sctr_snp, sctr_etf]
    """
    syms = sorted(args.symbols) if args.symbols else default_symbols()
    log.info("Fetching data for %d symbols → %s", len(syms), _OUTPUT_DIR)
    result = fetch_stock_data(syms, _OUTPUT_DIR)
    log.info("Done. DataFrame shape: %s", result.shape)


def _cmd_view(_args: argparse.Namespace) -> None:
    latest = find_latest_parquet(_OUTPUT_DIR)
    if latest is None:
        log.error("No stockcharts-*.parquet files found in %s", _OUTPUT_DIR)
        sys.exit(1)
    log.info("Loading %s", latest)
    df = pd.read_parquet(latest, engine="pyarrow").astype({"symbol": "category"})
    print(f"File: {latest}")
    print(f"Shape: {df.shape}")
    print()
    print_summary(df)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    {"load": _cmd_load, "view": _cmd_view}[args.command](args)
