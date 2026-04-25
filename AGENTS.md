# Coding Guidelines AI Coding agents working in the scripts project

## Project Overview

This folder contains a number of utility Python scripts. Here are some key files:
- cat.py - a command line chat application using the OpenAI responses API
- chat2.py - a simple command line chat application using the OpenAI chat API
- twdata.py - a utility script for working with stock data from Twelve Data API
- yfdata.py - a utility script for working with stock data from Yahoo Finance

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

## Code Style Guidelines

### General Principles
- Use a modern python coding style with types hints for function parameters and return types
- use list | 


### Workflow
Work systematically in the following order.
- make code and test changes
- run unit tests and ensure new tests pass (ignore pre-existing problems)

Once the functional changes are complete then
- run lint `ruff check --fix`
- run format `ruff --format`


### Naming Conventions
Follow PEP 8 guidelines. Function names should normally start with a verb / action but can be a noun if the function returns a value, is idempotent and makes no visible data updates (caching, logging, printing are not visible data updates).

