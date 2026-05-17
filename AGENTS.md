# Coding Guidelines AI Coding agents working in the scripts project

## Project Overview

This folder contains a number of utility Python scripts. Here are some key files:
- chat.py - a command line chat application using the OpenAI responses API
- chat2.py - a simple command line chat application using the OpenAI chat API
- twdata.py - a utility script for working with stock data from Twelve Data API
- yfdata.py - a utility script for working with stock data from Yahoo Finance
- scdata.py - a utility script for working with stock data from Stockcharts

**Technology Stack:**
- Python 3.13
- Package Manager: uv
- Database: MongoDB (async client 5.6)
- HTTP Client: httpx with HTTP/2 support
- Charts Graphing: Plotly
- Web Scraping: BeautifulSoup4
- Linting: ruff

**Project Structure:**
- Source layout is a flat directory structure
- Tests in `tests/` directory


## Tool
Use ripgrep to find text. example find the function "plot_dashboard" in any python file
```bash
rg "plot_dashboard" -t py -C 3
```
## Build, Lint & Test Commands

### Linting & Formatting
```bash
# Check code with ruff (linting) and auto-fix
uv run ruff check --fix twdata.py

# Format code with ruff
uv run ruff format twdata.py

```

### Running the Application
```bash
python twdata.py
```

# Style Guidelines for Coding Agent

## 1. Type Annotations — Signatures and Boundaries
Annotate all function signatures. Use built-in generics (`list[str]`, `dict[str, int]`). Use the most generic type possible so prefer Sequence over list etc. Import `TypeVar`, `Iterator`, `Callable` etc. from `collections.abc`, never `typing`. Do not annotate local variables unless the type is non-obvious.

examples
```
s = "hello" # type obvious
config: dict[str, Any] = json.loads(raw) # Non-obvious add annotation
def fn(xs: list[str]) -> str | None:
def first[T](items: list[T]) -> T
type Point = tuple[float, float]
```

## 2. Dataclasses and Value Objects
Use `@dataclass` for mutable stateful objects. Use `NamedTuple` for small immutable records passed between functions. Use `field(init=False, repr=False)` for internally-managed attributes. Use plain `dict` only for short-lived accumulators that never leave a function.

## 3. Dispatch and Branching
if / elif chains can be used, but consider if match / case would be more readable.

## 4. Logging
Use the `logging` module for all diagnostic output. Configure once at module level. Use `log.debug` liberally — it costs nothing when disabled. Never use `print()` for diagnostics in any code that could be imported as a module. In CLI-only scripts, `print()` is acceptable for structured user-facing output only; prefer Rich for anything tabular.

## 5. Rich for Terminal Output
This codebase uses Rich. Use `rich.console.Console` and `rich.table.Table` / `rich.markdown.Markdown` for user-facing terminal output. Instantiate a single module-level `console = Console()`. Do not mix Rich and bare `print()` for the same output type.

## 6. Pure Functions vs I/O
Keep transformation logic in pure functions with no side effects. I/O and orchestration belong in `_cmd_*` CLI handlers or dedicated fetch/save functions. Pure helpers are defined above I/O code in the file. Pure functions should be directly unit-testable; I/O functions do not need to be.

## 7. File System
Use `pathlib` for file system paths and avoid exclusively. Use `/` for path joining. Use `.glob()` for discovery. Return `Path | None` when a file may not exist. Never use `os.path`, string concatenation for paths, or bare `open()` with string literals.

## 8. Pandas Style
Prefer method chaining with `.assign()`, `.astype()`, `.reset_index()` for constructing new DataFrames. Use `lambda d:` inside `.assign()` for inter-column derivations. Mutation via `df["col"] = ...` is acceptable after an explicit `.copy()`. Always `.copy()` before mutating a DataFrame received as a parameter or produced by a slice. Avoid `iterrows()`; use `itertuples()` for read-only row iteration, or vectorised operations.

## 9. NumPy vs Pandas
Use NumPy for numerical array operations — z-scores, rolling stats, bin indexing, boolean masks. Use Pandas for anything with a date index, groupby, or destined for display or storage. Move between them explicitly with `.values` or `np.array()`. Only pull in `scipy` for operations with no NumPy equivalent (e.g. `find_peaks`, `curve_fit`).

## 10. Error Handling
In `_cmd_*` entry points, signal fatal errors with `log.error(...)` then `sys.exit(1)`. Do not raise exceptions for expected user-facing failures. Use bare `except Exception` only in teardown/cleanup paths, always logging the exception with context. Never silently swallow exceptions.

## 11. CLI Structure
Use `argparse` with `add_subparsers(dest="command", required=True)`. Build the parser in a standalone `_build_parser() -> argparse.ArgumentParser` function. Each subcommand maps 1-to-1 to a `_cmd_<name>(args: argparse.Namespace) -> None` function. For larger CLIs or new projects, prefer Typer or Click over argparse.

## 12. Constants
Define string literals, URL templates, column lists, dtype dicts, and default paths as module-level constants in `UPPER_SNAKE_CASE` with a leading underscore for module-private values. Constants belong only in module scope or `__main__` blocks — not inside functions, and not imported by other modules as a side-channel for configuration.

### Workflow
Work systematically in the following order.
- make code and test changes
- run unit tests and ensure new tests pass (ignore pre-existing problems)

Once the functional changes are complete then run lint and format
- `ruff check --fix [file]`
- `ruff format [file]`

### Naming Conventions
Follow PEP 8 guidelines. Function names should normally start with a verb / action but can be a noun if the function returns a value, is idempotent and makes no visible data updates (caching, logging, printing are not visible data updates).

