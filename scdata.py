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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright
from plotly.subplots import make_subplots
from rich.console import Console
from rich.markdown import Markdown

import tsutils as ts

T = TypeVar("T")

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
spy qqq spsm rsp
xlb xlc xle xlf xlk xly
eem pbw xop xme gld smh
"""
# isf.l vod.l pct.l mtro.l


# _SYMBOLS_RAW = "spy qqq"

_OUTPUT_DIR = Path.home() / "Downloads"

# JavaScript injected into the browser page to call the StockCharts data API.
# __SYMBOLS__ is replaced at runtime with a comma-separated list of tickers.
# supports upto 10 symbols. The date range returned will the same for all symbols.
_FETCH_DAILY_JS = """\
(async (symbols) => {
    const encoded = encodeURIComponent(symbols);
    const url = `https://stockcharts.com/json/data?cmd=get-daily-data&startDate=1999-01-01\
&dateAligned=true&src=freecharts-perf&symbols=${encoded}&r=${Date.now()}`;
    const response = await fetch(url);
    const data = await response.json();
    return JSON.stringify(data);
})('__SYMBOLS__')
"""

_FETCH_SCTR_JS = """\
(async () => {
    const url = `https://stockcharts.com/j-sum/sum?cmd=sctr&view=L&timeframe=W&r=${Date.now()}`;
    const response = await fetch(url);
    const data = await response.json();
    return JSON.stringify(data);
})()
"""

# ---------------------------------------------------------------------------
# Pure helpers — no browser, no I/O
# ---------------------------------------------------------------------------


def default_symbols() -> list[str]:
    """Return a sorted list of tickers from the default watchlist."""
    return sorted(_SYMBOLS_RAW.split())


def iter_batches(items: Sequence[T], max_batch: int) -> Iterator[Sequence[T]]:
    """
    Split items in evenly sized batches, each with at most *max_batch* elements.

    in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], 4
    out [['a', 'b', 'c', 'd'], ['e', 'f', 'g'], ['h', 'i', 'j']]
    """
    n = len(items)
    if n <= 0 or max_batch < 1:
        return

    n_batches = (n + max_batch - 1) // max_batch
    base, extra = divmod(n, n_batches)

    offset = 0

    for i in range(n_batches):
        size = base + 1 if i < extra else base
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


def calc_volume_rank(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """rank volume in the last n days as a 0–100 score"""
    nvol = df["volume"].rolling(n).rank(pct=True) * 100
    return nvol.round().fillna(0).astype(int)


def calc_standardized_volume(df: pd.DataFrame, n: int) -> pd.Series:
    vol = df["volume"]

    rolling_median = vol.rolling(window=n).median()
    rolling_mad = (vol - rolling_median).abs().rolling(window=n).median()

    # 1.4826 rescales MAD to be consistent with normal σ
    nvol = ((vol - rolling_median) / (1.4826 * rolling_mad)) * 100

    return nvol.round().fillna(0).astype(int)


def calc_ewm_log_volume_score(
    df: pd.DataFrame,
    span: int = 100,
    scale: float = 100.0,
    eps: float = 1e-9,
) -> pd.Series:
    x_now = np.log(df["volume"])

    # Shift to avoid using the current bar in the benchmark.
    x_hist = x_now.shift(1)

    mean = x_hist.ewm(span=span, adjust=False).mean()
    std = x_hist.ewm(span=span, adjust=False).std()

    z = (x_now - mean) / (std + eps)

    return (z * scale).fillna(0).round().astype(int)


_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sctr_reg",
    "sctr_snp",
    "sctr_etf",
]


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


def daily_json_to_dataframe(json_str: str) -> pd.DataFrame:
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


def save_daily_data(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Write daily OHLCV data to *output_dir* as a Parquet file.
    """
    dest = output_dir / f"stockcharts-{date.today().strftime('%y%m%d')}.parquet"

    df.to_parquet(dest, engine="pyarrow", compression="snappy")
    log.info("Saved %d daily rows → %s", len(df), dest)
    return dest


def find_latest_daily(directory: Path) -> Path | None:
    if files := list(directory.glob("stockcharts-*.parquet")):
        return max(files, key=lambda p: p.stem)
    return None


def find_latest_sctr(directory: Path) -> Path | None:
    if files := list(directory.glob("stockcharts-sctr-*.csv")):
        return max(files, key=lambda p: p.stem)
    return None


def clean_files(directory: Path, spec: str) -> None:
    """
    Scan `directory` for files matching `spec`, sort them by name, and
    keep only the first and last.
    """
    files = sorted(directory.glob(spec))
    log.info("Found %d file(s) matching %s", len(files), spec)

    if len(files) <= 2:
        log.info("No files deleted")
        return

    to_remove = files[1:-1]
    for f in to_remove:
        f.unlink(missing_ok=True)
    log.info("Deleted %d file(s)", len(to_remove))


def _load_latest_parquet(dtype_overrides: dict | None = None) -> tuple[pd.DataFrame, Path]:
    latest = find_latest_daily(_OUTPUT_DIR)
    if latest is None:
        log.error("No stockcharts-*.parquet files found in %s", _OUTPUT_DIR)
        sys.exit(1)
    log.info("Loading %s", latest)
    df = pd.read_parquet(latest, engine="pyarrow")
    if dtype_overrides:
        df = df.astype(dtype_overrides)
    return df, latest


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
# Request / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCFetchRequest:
    """
    Request options for fetching StockCharts data.

    daily:
        Whether to fetch daily OHLCV data.
    sctr:
        Whether to fetch SCTR ranking data.
    symbols:
        Ticker symbols for daily data. Ignored when daily is False.
        If None and daily is True, the built-in default watchlist is used.
    output_dir:
        Directory where output files are written.
    batch_max:
        Maximum number of symbols per daily-data API request.
    """

    daily: bool = True
    sctr: bool = True
    symbols: Sequence[str] | None = None
    output_dir: Path = field(default_factory=lambda: Path.home() / "Downloads")
    batch_max: int = 10

    def resolved_symbols(self) -> list[str]:
        if self.symbols is None:
            return default_symbols()

        return sorted({symbol for raw in self.symbols if (symbol := str(raw).strip().upper())})


@dataclass
class SCFetchResult:
    """
    Result from fetching StockCharts data.

    daily:
        Daily OHLCV DataFrame, if requested and retrieved.
    daily_path:
        Path to saved daily Parquet file, if written.
    sctr:
        SCTR DataFrame, if requested and retrieved.
    sctr_path:
        Path to saved SCTR CSV file, if written.
    """

    daily: pd.DataFrame | None = None
    daily_path: Path | None = None
    sctr: pd.DataFrame | None = None
    sctr_path: Path | None = None


# ---------------------------------------------------------------------------
# SCTR parser
# ---------------------------------------------------------------------------


def sctr_json_to_dataframe(json_str: str) -> tuple[pd.DataFrame, str]:
    """
    Parse the StockCharts SCTR JSON payload and return a DataFrame plus
    a YYMMDD date label used for the filename.

    The JSON structure is a list where the first element contains the date
    and subsequent elements are per-symbol records.
    """
    payload: list[dict] = json.loads(json_str)

    if not payload:
        log.warning("No SCTR data retrieved.")
        return pd.DataFrame(), date.today().strftime("%y%m%d")

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
        return df, date_label

    df = df[
        [
            "symbol",
            "close",
            "volume",
            "mcap",
            "SCTR",
            "delta",
            "name",
            "industry",
            "sector",
        ]
    ]

    return df, date_label


def save_sctr_data(df: pd.DataFrame, output_dir: Path, date_label: str) -> Path:
    """
    Write SCTR data to *output_dir* as a CSV file.
    """
    dest = output_dir / f"stockcharts-sctr-{date_label}.csv"
    df.to_csv(dest, index=False)
    log.info("Saved %d SCTR rows → %s", len(df), dest)
    return dest


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


def _fetch_daily_json(page: Page, symbols: Sequence[str]) -> str:
    """
    Inject JavaScript into *page* to call the StockCharts daily data API for
    *symbols* and return the raw JSON response string.
    """
    ticker_str = ",".join(s.upper() for s in symbols)
    js = _FETCH_DAILY_JS.replace("__SYMBOLS__", ticker_str)
    result: str = page.evaluate(js)
    log.info("Fetched daily JSON for %d symbols (%d chars)", len(symbols), len(result))
    return result


def _fetch_sctr_json(page: Page) -> str:
    """
    Inject JavaScript into *page* to call the StockCharts SCTR API and return
    the raw JSON response string.
    """
    result: str = page.evaluate(_FETCH_SCTR_JS)
    log.info("Fetched SCTR JSON (%d chars)", len(result))
    return result


def _prepare_stockcharts_page(page: Page) -> None:
    """
    Navigate to StockCharts and dismiss the consent popup if present.
    """
    page.goto("https://stockcharts.com/")
    page.wait_for_load_state("domcontentloaded")
    _dismiss_shadow_dom_popup(page, "#cmpwrapper")


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
        for obj in (self.page, self._context, self._browser):
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
# Public fetch orchestration
# ---------------------------------------------------------------------------


def fetch_stockcharts(request: SCFetchRequest) -> SCFetchResult:
    """
    Fetch daily data, SCTR data, or both using a single browser session.

    The browser is launched once, StockCharts is opened once, and the requested
    API calls are made from the same page context.
    """
    if not request.daily and not request.sctr:
        raise ValueError("At least one of request.daily or request.sctr must be True")

    result = SCFetchResult()

    with _BrowserSession() as session:
        _prepare_stockcharts_page(session.page)

        if request.daily:
            symbols = request.resolved_symbols()
            frames: list[pd.DataFrame] = []

            log.info(
                "Fetching daily data for %d symbols → %s",
                len(symbols),
                request.output_dir,
            )

            for batch in iter_batches(symbols, request.batch_max):
                time.sleep(random.uniform(2.0, 5.0))
                raw_json = _fetch_daily_json(session.page, batch)
                df_batch = daily_json_to_dataframe(raw_json)
                frames.append(df_batch)

                log.info(
                    "Daily batch done: %d rows accumulated",
                    sum(len(f) for f in frames),
                )

            if not frames:
                log.warning("No daily data retrieved.")
            else:
                daily = pd.concat(frames, ignore_index=True).reset_index(drop=True)

                if daily.empty:
                    log.warning("No daily data retrieved.")
                else:
                    result.daily = daily
                    result.daily_path = save_daily_data(daily, request.output_dir)

        if request.sctr:
            log.info("Fetching SCTR data → %s", request.output_dir)

            raw_json = _fetch_sctr_json(session.page)
            sctr, date_label = sctr_json_to_dataframe(raw_json)

            if not sctr.empty:
                sctr_path = save_sctr_data(sctr, request.output_dir, date_label)

                result.sctr = sctr
                result.sctr_path = sctr_path
            else:
                log.warning("No SCTR data retrieved.")

    return result


def fetch_daily_data(
    symbols: list[str],
    output_dir: Path,
    *,
    batch_max: int = 10,
) -> pd.DataFrame:
    """
    Fetch only daily OHLCV data.

    Compatibility wrapper around fetch_stockcharts().
    """
    result = fetch_stockcharts(
        SCFetchRequest(
            daily=True,
            sctr=False,
            symbols=symbols,
            output_dir=output_dir,
            batch_max=batch_max,
        )
    )

    return result.daily if result.daily is not None else pd.DataFrame()


def fetch_sctr_data(output_dir: Path) -> pd.DataFrame:
    """
    Fetch only SCTR data.

    Compatibility wrapper around fetch_stockcharts().
    """
    result = fetch_stockcharts(
        SCFetchRequest(
            daily=False,
            sctr=True,
            output_dir=output_dir,
        )
    )

    return result.sctr if result.sctr is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _missing_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    all_dates = pd.date_range(dates.min(), dates.max())
    return list(all_dates.difference(dates))


def _cmd_load(args: argparse.Namespace) -> None:
    """
    returns daily data from stockcharts as a dataframe
    df [symbol, date, open, high, low, close, volume, sctr_reg, sctr_snp, sctr_etf]
    """
    syms = sorted(args.symbols) if args.symbols else default_symbols()
    log.info("Fetching daily data for %d symbols → %s", len(syms), _OUTPUT_DIR)
    result = fetch_daily_data(syms, _OUTPUT_DIR)
    log.info("Done. DataFrame shape: %s", result.shape)


def _cmd_fetch(args: argparse.Namespace) -> None:
    """
    Fetch daily and/or SCTR data using a single browser session.
    """
    daily = args.daily
    sctr = args.sctr

    if not daily and not sctr:
        daily = True
        sctr = True

    symbols = sorted(args.symbols) if args.symbols else None

    if symbols and not daily:
        log.warning("Symbols were supplied but --daily not set, so symbols are ignored.")

    request = SCFetchRequest(
        daily=daily,
        sctr=sctr,
        symbols=symbols,
        output_dir=_OUTPUT_DIR,
    )

    log.info(
        "Fetching StockCharts data: daily=%s sctr=%s → %s",
        daily,
        sctr,
        _OUTPUT_DIR,
    )

    result = fetch_stockcharts(request)

    if result.daily is not None:
        log.info("Daily DataFrame shape: %s", result.daily.shape)
        log.info("Daily file: %s", result.daily_path)

    if result.sctr is not None:
        log.info("SCTR DataFrame shape: %s", result.sctr.shape)
        log.info("SCTR file: %s", result.sctr_path)


def _cmd_clean(_args: argparse.Namespace) -> None:
    """
    Clean up downloaded files.
    """
    clean_files(_OUTPUT_DIR, "stockcharts-*.parquet")
    clean_files(_OUTPUT_DIR, "stockcharts-sctr-*.csv")


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
    n = 20
    console.print(Markdown(f"## Top {n} SCTR stocks"))
    console.print(Markdown(_to_markdown(display.head(n))))

    # Table 2 – top 12 by DTV, then sorted by SCTR
    top_dtv = display.nlargest(20, ["dtv"]).head(n).sort_values("SCTR", ascending=False)
    console.print(Markdown(f"## Top {n} DTV stocks sorted by SCTR"))
    console.print(Markdown(_to_markdown(top_dtv)))

    # Table 3 – largest weekly change by SCTR
    top_chg = display[(display["delta"] > 8) & (display["dtv"] > 0.5) & (display["SCTR"] > 60)].head(n)
    console.print(Markdown(f"## Top {n} Large positive change"))
    console.print(Markdown(_to_markdown(top_chg)))


def _cmd_view(args: argparse.Namespace) -> None:
    df, latest = _load_latest_parquet(dtype_overrides={"symbol": "category"})
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
        subset = subset.drop(columns=["symbol"]).set_index("date")
        subset = ts.augment_data(subset)
        print(subset.tail())
    else:
        print_summary(df)


def _cmd_plotrel(args: argparse.Namespace) -> None:
    df, latest = _load_latest_parquet()

    spy_df = df[df["symbol"] == "SPY"].sort_values("date")
    if spy_df.empty:
        log.error("SPY not found in %s", latest)
        sys.exit(1)

    last_100 = spy_df.tail(100)
    min_row = last_100.loc[last_100["close"].idxmin()]
    start_date = pd.Timestamp(min_row["date"])
    log.info("SPY minimum close in last 100 rows: %.2f on %s", min_row["close"], start_date.date())

    filtered = df[df["date"] >= start_date].copy()
    if args.symbols:
        filtered = filtered[filtered["symbol"].isin([s.upper() for s in args.symbols])]
    wide = filtered.pivot(index="date", columns="symbol", values="close")
    normalized = (wide / wide.iloc[0] - 1) * 100

    fig = go.Figure()
    for symbol in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[symbol],
                mode="lines",
                name=symbol,
            )
        )

    title = f"Relative performance since {start_date.strftime('%Y-%m-%d')}"
    missing = _missing_dates(normalized.index)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="% Change",
        hovermode="x unified",
        xaxis_rangebreaks=[dict(values=missing)],
    )

    _print_relative_performance_table(normalized)

    if args.out:
        fig.write_html(args.out)
        log.info("Saved plot → %s", args.out)
    else:
        fig.show()


def _cmd_rs(args: argparse.Namespace) -> None:
    if len(args.symbols) < 2:
        log.error("rs command requires at least 2 symbols")
        sys.exit(1)

    df, _ = _load_latest_parquet()

    symbols_upper = [s.upper() for s in args.symbols]
    baseline = symbols_upper[0]
    all_symbols = symbols_upper

    two_years_ago = pd.Timestamp.now() - pd.DateOffset(years=2)

    filtered = df[df["symbol"].str.upper().isin(all_symbols) & (df["date"] >= two_years_ago)].copy()
    if filtered.empty:
        log.error("No data found for specified symbols in the last 2 years")
        sys.exit(1)

    wide = filtered.pivot(index="date", columns="symbol", values="close")
    wide = wide.dropna()

    if baseline not in wide.columns:
        log.error("Baseline symbol %s not found in data", baseline)
        sys.exit(1)

    rs_raw = wide.div(wide[baseline], axis=0)
    rs_raw = rs_raw.drop(columns=[baseline])
    rs = rs_raw / rs_raw.iloc[0]

    has_roc = args.roc is not None and args.roc > 0
    if has_roc:
        fig = make_subplots(rows=2, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    else:
        fig = go.Figure()

    for symbol in rs.columns:
        trace = go.Scatter(
            x=rs.index,
            y=rs[symbol],
            mode="lines",
            name=f"{symbol}/{baseline}",
        )
        if has_roc:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    if has_roc:
        roc = rs.diff(args.roc)
        for symbol in roc.columns:
            fig.add_trace(
                go.Scatter(
                    x=roc.index,
                    y=roc[symbol],
                    mode="lines",
                    name=f"{symbol}/{baseline} {args.roc}d Δ",
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="RS (normalized to 1.0)", row=1, col=1)
        fig.update_yaxes(title_text=f"{args.roc}d Δ", row=2, col=1)
    else:
        fig.update_yaxes(title_text="RS (normalized to 1.0)")

    missing = _missing_dates(rs.index)
    if has_roc:
        fig.update_xaxes(rangebreaks=[dict(values=missing)], row=1, col=1)
        fig.update_xaxes(rangebreaks=[dict(values=missing)], row=2, col=1)
    else:
        fig.update_xaxes(rangebreaks=[dict(values=missing)])

    fig.update_layout(
        title=f"Relative Strength vs {baseline} (last 2 years)",
        hovermode="x unified",
        showlegend=True,
    )

    if args.out:
        fig.write_html(args.out)
        log.info("Saved plot → %s", args.out)
    else:
        fig.show()


@dataclass
class VolumeRankIndicator:
    """Calculate volume rank indicator series for plotting."""

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)
    marker_extremes: tuple[float, float] | None = field(init=False)

    def __init__(self, df: pd.DataFrame, window: int = 20, threshold: float = 90.0) -> None:
        self.indicator = calc_volume_rank(df, n=window)
        self.marker = self.indicator >= threshold
        self.indicator_label = f"VolRank% {window}"
        self.marker_label = f"Vol ≥ {threshold}"
        self.marker_extremes = None


@dataclass
class BollingerBandPositionIndicator:
    """Calculate position within Bollinger Band indicator series for plotting."""

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)
    marker_extremes: tuple[float, float] | None = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 20,
        multiplier: float = 2.0,
        threshold: float = 0.0,
    ) -> None:
        close = df["close"]
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * multiplier)
        lower_band = rolling_mean - (rolling_std * multiplier)

        band_range = upper_band - lower_band
        self.indicator = (close - lower_band) / band_range
        self.marker = self.indicator <= threshold
        self.indicator_label = f"BB position {window}/{multiplier}"
        self.marker_label = f"BB pos <= {threshold}"
        self.marker_extremes = (
            self.indicator.quantile(0.05),
            self.indicator.quantile(0.95),
        )


@dataclass
class RateOfChangeIndicator:
    """Percentage Rate of Change"""

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 10,
        threshold: float = 5.0,
    ) -> None:
        close_n = df["close"].shift(window)
        self.indicator = 100.0 * (df["close"] - close_n) / close_n
        self.indicator_label = f"ROC {window}"
        if threshold > 0:
            self.marker = self.indicator >= threshold
            self.marker_label = f"ROC ≥ {threshold:.0%}"
        else:
            self.marker = self.indicator <= threshold
            self.marker_label = f"ROC <= {threshold:.0%}"
        self.marker_extremes = (
            self.indicator.quantile(0.05),
            self.indicator.quantile(0.95),
        )


@dataclass
class LogReturnIndicator:
    """period-normalised log-return"""

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 10,
        threshold: float = 0.10,
    ) -> None:
        close_n = df["close"].shift(window)
        self.indicator = pd.Series(np.log(df["close"] / close_n), index=df.index)
        self.indicator_label = f"PLR {window}"
        if threshold > 0:
            self.marker = self.indicator >= threshold
            self.marker_label = f"PLR ≥ {threshold:.2%}"
        else:
            self.marker = self.indicator <= threshold
            self.marker_label = f"PLR ≤ {threshold:.2%}"


@dataclass
class LogReturnRankIndicator:
    """
    Calculate and hold period-normalised log-return series.
    Marker: rolling k-day rank (lowest in last k days).
    """

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 10,
        minmax: bool = True,
        rank_window: int = 21,
    ) -> None:
        # ---- indicator --------------------------------------------------------
        close_n = df["close"].shift(window)
        self.indicator = pd.Series(np.log(df["close"] / close_n), index=df.index)
        self.indicator_label = f"LogRet {window}"

        # ---- marker: rolling k-day rank (lowest) ----------------------------
        if minmax:
            self.marker = self.indicator.rolling(rank_window, min_periods=1).apply(lambda x: x.iloc[-1] == x.min(), raw=False).astype(bool)
            self.marker_label = f"Lowest {rank_window}-day rank"
        else:
            self.marker = self.indicator.rolling(rank_window, min_periods=1).apply(lambda x: x.iloc[-1] == x.max(), raw=False).astype(bool)
            self.marker_label = f"Highest {rank_window}-day rank"

        self.marker_extremes = (
            self.indicator.quantile(0.05),
            self.indicator.quantile(0.95),
        )


@dataclass
class AtrPercentIndicator:
    """ATR as percentage of close"""

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 14,
        threshold: float = 5.0,
    ) -> None:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        # True Range is the max of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR via simple moving average (Wilder's style uses EWM, but SMA is common)
        atr = true_range.rolling(window=window).mean()

        # ATR% = (ATR / low) * 100
        self.indicator = pd.Series((atr / close) * 100, index=df.index)
        self.indicator_label = f"ATR% {window}"

        if threshold > 0:
            self.marker = self.indicator >= threshold
            self.marker_label = f"ATR% ≥ {threshold:.2f}"
        else:
            self.marker = self.indicator <= threshold
            self.marker_label = f"ATR% ≤ {threshold:.2f}"

        self.marker_extremes = (
            self.indicator.quantile(0.05),
            self.indicator.quantile(0.95),
        )


@dataclass
class AtrPercentRankIndicator:
    """ATR as percentage of close, marker flags max over last k days"""

    indicator: pd.Series = field(init=False)
    marker: pd.Series = field(init=False)
    indicator_label: str = field(init=False)
    marker_label: str = field(init=False)
    marker_extremes: tuple[float, float] = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 14,
        rank_window: int = 20,
    ) -> None:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        # True Range is the max of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR via simple moving average
        atr = true_range.rolling(window=window).mean()

        # ATR% = (ATR / close) * 100
        self.indicator = pd.Series((atr / close) * 100, index=df.index)
        self.indicator_label = f"ATR% {window}"

        # Marker: True when indicator equals rolling max of last rank_window days
        rolling_max = self.indicator.rolling(window=rank_window, min_periods=1).max()
        self.marker = self.indicator == rolling_max
        self.marker_label = f"ATR% max {rank_window}d"

        self.marker_extremes = (
            self.indicator.quantile(0.05),
            self.indicator.quantile(0.95),
        )


def _cmd_plot(args: argparse.Namespace) -> None:
    df, _ = _load_latest_parquet()

    sym = args.symbol.upper()
    subset = df[df["symbol"].str.upper() == sym].sort_values("date")
    if subset.empty:
        log.error("Symbol %s not found in data", sym)
        sys.exit(1)

    one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
    warmup_start = one_year_ago - pd.Timedelta(days=80)
    subset = subset[subset["date"] >= warmup_start].copy()

#    indicator = LogReturnRankIndicator(subset, window=2, minmax=True)
    indicator = AtrPercentRankIndicator(subset, window=10, rank_window=21)
    subset = subset[subset["date"] >= one_year_ago]

    fig = make_subplots(rows=2, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["close"],
            mode="lines",
            name=sym,
        ),
        row=1,
        col=1,
    )

    markers = subset[indicator.marker[subset.index]]
    fig.add_trace(
        go.Scatter(
            x=markers["date"],
            y=markers["close"],
            mode="markers",
            marker=dict(size=6, color="red", symbol="triangle-up"),
            name=indicator.marker_label,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=subset["date"],
            y=indicator.indicator.loc[subset.index],
            name=indicator.indicator_label,
            marker_color="rgba(100,149,237,0.6)",
        ),
        row=2,
        col=1,
    )

    if indicator.marker_extremes:
        lower_pct, upper_pct = indicator.marker_extremes

        fig.add_hline(
            y=lower_pct,
            line_dash="dash",
            line_color="gray",
            opacity=0.6,
            annotation_text=f"5th percentile {lower_pct:.4f}",
            annotation_position="bottom left",
            row=2,
            col=1,
        )

        fig.add_hline(
            y=upper_pct,
            line_dash="dash",
            line_color="gray",
            opacity=0.6,
            annotation_text=f"95th percentile {upper_pct:.4f}",
            annotation_position="bottom left",
            row=2,
            col=1,
        )

    missing = _missing_dates(subset["date"])
    fig.update_xaxes(
        rangebreaks=[dict(values=missing)],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        rangebreaks=[dict(values=missing)],
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_yaxes(title_text=indicator.indicator_label, row=2, col=1)
    fig.update_layout(
        title=f"{sym} — last 1 year",
        hovermode="x unified",
        showlegend=True,
    )

    if args.out:
        fig.write_html(args.out)
        log.info("Saved plot → %s", args.out)
    else:
        fig.show()


def _print_relative_performance_table(normalized: pd.DataFrame) -> None:
    """Print the latest percentage change per symbol as a sorted console table."""
    latest = normalized.iloc[-1].dropna()
    table = pd.DataFrame({"symbol": latest.index, "%change": latest.values})
    table["%change"] = pd.to_numeric(table["%change"], errors="coerce").round(2)
    table = table.sort_values("%change", ascending=False).reset_index(drop=True)

    col_widths = {
        "symbol": max(table["symbol"].astype(str).str.len().max(), 6),
        "%change": max(table["%change"].astype(str).str.len().max(), 8),
    }
    header = f"{'symbol':<{col_widths['symbol']}}  {'%change':>{col_widths['%change']}}"
    print()
    print(header)
    print("-" * len(header))
    for row in table.itertuples(index=False):
        print(f"{row.symbol:<{col_widths['symbol']}}  {row._1:>{col_widths['%change']}.2f}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data from StockCharts.com and store as Parquet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    load = sub.add_parser("load", help="Fetch data from StockCharts and save to Parquet")
    load.add_argument("symbols", nargs="*", default=None, help="Ticker symbols (default: built-in watchlist)")

    fetch = sub.add_parser("fetch", help="Fetch daily, SCTR, or both using one browser session")
    fetch.add_argument("symbols", nargs="*", default=None, help="Ticker symbols for daily data, default: built-in watchlist")
    fetch.add_argument("--daily", action="store_true", help="Fetch daily OHLCV data")
    fetch.add_argument("--sctr", action="store_true", help="Fetch SCTR data")

    view = sub.add_parser("view", help="Load the latest Parquet file and print summary info")
    view.add_argument("symbol", nargs="?", default=None, help="Optional single ticker to print tail for")

    sub.add_parser("sctr", help="Fetch SCTR data from StockCharts and save to CSV")

    sub.add_parser("rank", help="Load the latest SCTR CSV and print the top 12 rows")

    sub.add_parser("clean", help="Clean up old Parquet files")

    plotrel = sub.add_parser("plotrel", help="Plot relative performance since SPY's lowest close in last 100 days")
    plotrel.add_argument("--out", default=None, help="Optional path to save HTML instead of opening browser")
    plotrel.add_argument("symbols", nargs="*", default=None, help="Ticker symbols (default: built-in watchlist)")

    rs_parser = sub.add_parser("rs", help="Plot relative strength of 2+ stocks over the last 2 years")
    rs_parser.add_argument("symbols", nargs="+", help="Ticker symbols: first is the baseline, rest are compared against it")
    rs_parser.add_argument("--out", default=None, help="Optional path to save HTML instead of opening browser")
    rs_parser.add_argument("--roc", type=int, default=None, help="Plot n-day rate of change of RS on a subplot")

    plot_parser = sub.add_parser("plot", help="Plot closing price and standardized volume for a symbol")
    plot_parser.add_argument("symbol", help="Ticker symbol to plot")
    plot_parser.add_argument("--out", default=None, help="Optional path to save HTML instead of opening browser")

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    match args.command:
        case "clean":
            _cmd_clean(args)
        case "load":
            _cmd_load(args)
        case "fetch":
            _cmd_fetch(args)
        case "view":
            _cmd_view(args)
        case "sctr":
            _cmd_sctr(args)
        case "rank":
            _cmd_rank(args)
        case "plotrel":
            _cmd_plotrel(args)
        case "rs":
            _cmd_rs(args)
        case "plot":
            _cmd_plot(args)
        case _:
            log.error("Unknown command: %s", args.command)
            sys.exit(1)
