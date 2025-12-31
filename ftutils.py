#!/usr/bin/python3
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

# from chatutils import execute_python_script
from rich.pretty import pprint

from htm2md import html_to_markdown

console = Console()


class ErrorInfo(BaseModel):
    error: bool
    type: str
    message: str
    url: str


class ArticleLink(BaseModel):
    headline: str
    summary: str = Field(default="")
    url: str


class ArticleList(BaseModel):
    timestamp_retrieved: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: str
    articles: list[ArticleLink]

    def to_markdown(self) -> str:
        """Convert the ArticleList to a Markdown formatted string."""
        markdown = f"## Articles from {self.source}\n\n"
        for i, a in enumerate(self.articles, start=1):
            if len(a.summary):
                markdown += f"{i}. {a.headline}\n   - {a.summary}\n   - {a.url}\n"
            else:
                markdown += f"{i}. {a.headline}\n   - {a.url}\n"
        markdown += f"\nretrieved {self.timestamp_retrieved}\n"
        return markdown


def merge(*article_lists: ArticleList) -> ArticleList:
    """Merge article lists together, merging duplicate URLs."""
    # Create a dictionary to track URLs that have already been added
    merged = {}
    for article_list in article_lists:
        for article in article_list.articles:
            if article.url in merged:
                # Replace if the summary in the current article is longer
                if len(article.summary) > len(merged[article.url].summary):
                    merged[article.url] = article
            else:
                merged[article.url] = article

    return ArticleList(timestamp_retrieved=article_lists[0].timestamp_retrieved, source=article_lists[0].source, articles=list(merged.values()))


class Citation(BaseModel):
    title: str
    url: str
    date: str
    archiveurl: str
    archivedate: str


class QuoteList(BaseModel):
    timestamp_retrieved: str
    quotes: list[dict]


retrieve_headlines_fn = {
    "type": "function",
    "function": {
        "name": "retrieve_headlines",
        "description": "Retrieve headlines from a news web site",
        "parameters": {"type": "object", "properties": {"source": {"type": "string", "enum": ["bbc", "bloomberg", "ft", "nyt", "wsj"], "description": "the name of the news web site"}}},
        "required": ["source"],
    },
}

retrieve_article_fn = {
    "type": "function",
    "function": {
        "name": "retrieve_article",
        "description": "Downloads the text content of a news article from the URL",
        "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "URL of the article to download"}}, "required": ["url"]},
    },
}

retrieve_stock_quotes_fn = {
    "type": "function",
    "function": {
        "name": "retrieve_stock_quotes",
        "description": "Retrieve stock quotes for a list of symbols from Bloomberg",
        "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}, "description": "A list of stock symbols"}}, "required": ["symbols"]},
    },
}

evaluate_expression_fn = {
    "type": "function",
    "function": {
        "name": "eval",
        "description": "Evaluates a mathematical or Python expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "the expression",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def evaluate_expression_impl(expression: str) -> Any:
    # Split into individual lines removing blank lines but preserving indents
    parts = [e for e in re.split(r"; |\n", expression) if e.strip()]
    if not parts:
        return None  # Empty input

    parts = ["import math", "import datetime"] + parts
    # Separate final expression
    *statements, last_part = parts

    # Create a namespace dictionary to store variables
    namespace = {}

    # Execute all statements updating the namespace as necessary
    if statements:
        exec("\n".join(statements), namespace)

    # Evaluate result of final expression
    return eval(last_part.strip(), namespace)


def evaluate_expression(expression: str) -> str:
    result = ""
    if expression:
        try:
            console.print("eval: " + expression, style="yellow")
            result = evaluate_expression_impl(expression)
            console.print("result: " + str(result), style="yellow")
        except Exception as e:
            result = f"ERROR: {e.__class__.__name__}: {e}"
            console.print(result, style="red")
    else:
        result = "ERROR: no expression found"
        console.print(result, style="red")
    return str(result)


def retrieve_article(url: str) -> str:
    if not url:
        return ErrorInfo(error=True, type="invalid argument", message="url required", url="")

    if "www.ft.com" in url:
        return retrieve_ft_article(url)
    if "www.wsj.com" in url:
        return retrieve_wsj_article(url)
    if "www.bloomberg.com" in url:
        return retrieve_bloomberg_article(url)
    return retrieve_bbc_article(url)


def retrieve_stock_quotes(symbols: list[str]) -> QuoteList | ErrorInfo:
    """Retrieves historical stock quotes for the given symbols.

    Args:
        symbols: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOG']).

    Returns:
        a list of quotes and details about the symbol

    """

    if not symbols:
        return ErrorInfo(error=True, type="invalid argument", message="symbols required", url="")

    def make_url(ticker: str) -> str:
        t = ticker.upper()
        path = f"{t}/" if ":" in ticker else f"{t}:UN/"
        return "https://www.bnnbloomberg.ca/stock/" + path

    d = {make_url(e): process_bnn_stock_page for e in symbols}
    result = retrieve_using_playwright(d)

    headers = {"Accept": "application/json", "Accept-Encoding": "gzip, deflate", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0"}

    with httpx.Client(http2=True, headers=headers) as client:
        for r in result.values():
            r.update(get_bnnbloomberg_quote(client, r["symbol"]))
    return result if isinstance(result, ErrorInfo) else QuoteList(timestamp_retrieved=datetime.now().isoformat(), quotes=list(result.values()))


@cache
def ftutils_functions() -> dict[str, dict[str, Any]]:
    """Returns a dictionary mapping function names to their definitions and a callable."""

    def name(d):
        return d["function"]["name"]

    return {
        name(evaluate_expression_fn): {"defn": evaluate_expression_fn, "fn": evaluate_expression},
        name(retrieve_headlines_fn): {"defn": retrieve_headlines_fn, "fn": retrieve_headlines},
        name(retrieve_article_fn): {"defn": retrieve_article_fn, "fn": retrieve_article},
        name(retrieve_stock_quotes_fn): {"defn": retrieve_stock_quotes_fn, "fn": retrieve_stock_quotes},
    }


def make_clean_filename(text: str) -> str:
    words = re.sub(r"[\\\.\/[\]<>'\",:*?|]", " ", text.lower()).split()
    return "_".join(words[:5])


def save_soup(soup: BeautifulSoup, fname: Path):
    console.print(f"saving file {str(fname)}")
    with fname.open("w", encoding="utf-8") as f:
        f.write(soup.prettify())


def load_soup(fname: Path) -> BeautifulSoup:
    with fname.open(encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "html.parser")


def save_markdown_article(title: str, text: str) -> Path | None:
    if not title:
        return None

    filename = make_clean_filename(title) + ".md"
    p = Path.home() / "Documents" / "chats" / filename
    pprint(f"saved {p}")
    p.write_text(text, encoding="utf-8")
    return p


def text_between(content: str, start_tag: str, end_tag: str) -> str:
    return "" if (start := content.find(start_tag)) < 0 or (end := content.find(end_tag, start + len(start_tag))) < 0 else content[start + len(start_tag) : end]


def add_citation(text: str, cite: Citation) -> str:
    pprint(cite)
    s = f"# {cite.title}\n\n**source:** {cite.url}\n\n**published:** {cite.date}\n\n"
    return s + text


def get_bnnbloomberg_quote(client: httpx.Client, symbol: str) -> dict:
    """
    Fetches symbol summary info and company info via REST calls

    Returns:
        dict: The JSON response as a dictionary, or None if an error occurs.
    """
    result = {}

    def merge_from(d):
        for k, v in d.items():
            if v is not None:
                if isinstance(v, str):
                    try:
                        result[k] = float(v)
                    except:
                        result[k] = v
                else:
                    result[k] = v

    try:
        url = "https://bnn.stats.bellmedia.ca/bnn/api/stock/Scorecard?symbol=" + symbol
        console.print(f"get quote scorecard from {url}")
        response = client.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        scorecard = response.json()
        url = "https://bnn.stats.bellmedia.ca/bnn/api/stock/companyInfo?brand=bnn&lang=en&symbol=" + symbol
        response = client.get(url)
        response.raise_for_status()
        company_info = response.json()

        merge_from(scorecard)
        merge_from(company_info)
    except Exception as e:
        console.print(f"An unexpected error occurred: {e}", style="red")
    return result


def process_bnn_stock_page(page) -> dict | None:
    ticker_head = page.locator("h1.c-heading")
    name = page.locator("h2.bmw-market-status__title")
    close = page.locator("span.bmw-market-status__info__price")
    d = {"symbol": ticker_head.inner_text(), "name": name.inner_text(), "close": float(close.inner_text())}
    console.print("scorecard found " + d["symbol"], style="yellow")
    return d


def retrieve_bbc_most_read() -> ArticleList:
    """
    find h2  with id=mostRead-label, from parent div find all child anchor tags
    """

    with httpx.Client(http2=True) as client:
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-UK,en;q=0.5",
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
        }
        response = client.get("https://www.bbc.co.uk/news", headers=headers)
        response.raise_for_status()
        content = response.text

    soup = BeautifulSoup(content, "html.parser")
    #    save_soup(soup, Path.home() / "Downloads" / "bbc-news.html")
    xs = []
    if mrs := soup.find("h2", id="mostRead-label"):
        xs = [ArticleLink(headline=anchor.get_text(strip=True), url=urljoin("https://www.bbc.co.uk", anchor["href"])) for anchor in mrs.find_parent("div").find_all("a")]
    return ArticleList(source="BBC News", articles=xs)


def get_bbc_article_contents(url: str) -> BeautifulSoup | None:
    console.print(f"navigate {url}", style="yellow")

    def dismiss_maybe_later_popup() -> bool:
        try:
            overlay = page.locator('[data-testid="test-overlay"]')
            overlay.wait_for(state="visible", timeout=5000)
            page.locator('button:has-text("Maybe later")').click()
            overlay.wait_for(state="hidden")
            return True
        except PlaywrightTimeoutError:
            return False

    def reject_cookies() -> bool:
        try:
            btn = page.locator('[data-testid="reject-button"]')
            btn.wait_for(state="visible", timeout=5000)
            btn.click()
            page.locator('section[aria-labelledby="consent-banner-title"]').wait_for(state="hidden")
            return True
        except PlaywrightTimeoutError:
            return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(url)

            dismiss_maybe_later_popup()
            reject_cookies()

            locator = page.locator("#main-heading")
            console.print(f"retrieved - {locator.text_content()}")

            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            # save_soup(soup, Path("~/Downloads/z.htm").expanduser())
            return soup
        except Exception as e:
            console.print(f"error retrieving {e}", style="red")
            return None
        finally:
            browser.close()


def retrieve_bbc_article(url: str) -> str:
    soup = get_bbc_article_contents(url)
    md = html_to_markdown(soup=soup.find(id="main-content"), href_base="https://bbc.co.uk")
    if headline := soup.find("title"):
        console.print(f"title: {headline.text}")
    if time_tag := soup.find("time", attrs={"datetime": True}):
        datetime_str = time_tag["datetime"]
        d = datetime.fromisoformat(datetime_str)
        console.print(f"published: {d.strftime('%A %Y-%m-%d')}")

    cite = Citation(title=headline.text.strip(), url=url, date=d.date().isoformat(), archiveurl="", archivedate="")
    md = add_citation(md, cite)
    save_markdown_article(cite.title, md)
    return md


def retrieve_wsj_article(url: str) -> str:
    content, cite = retrieve_archive(url)
    soup = BeautifulSoup(content, "html.parser")
    save_soup(soup, Path("~/Downloads/temp.html").expanduser())
    # remove some divs before extracting text
    if divs := soup.find_all("div"):
        xs = [d for d in divs if "background-position:/*x=*/0% /*y=*/0%;" in d.get("style")]
        console.print(f"removing divs with style {len(xs)}", style="red")
        for d in xs:
            d.decompose()

    md = html_to_markdown(soup=soup.find("section"), href_base="https://www.wsj.com/")
    md = add_citation(md, cite)
    save_markdown_article(cite.title, md)
    return md


def retrieve_bloomberg_article(url: str) -> str:
    content, cite = retrieve_archive(url)
    soup = BeautifulSoup(content, "html.parser").find("article")
    breakpoint()

    # remove some divs before extracting text
    if divs := soup.find_all("div", style=lambda x: x and "align-items" in x):
        #    if divs := soup.find_all("div"):
        # xs = [d for d in divs if "background-position:/*x=*/0% /*y=*/0%;" in d.get("style")]
        xs = list(divs)
        console.print(f"removing divs with style {len(xs)}", style="red")
        for d in xs:
            d.decompose()

    md = html_to_markdown(soup=soup, href_base="https://www.bloomberg.com/")
    md = add_citation(md, cite)
    save_markdown_article(cite.title, md)
    return md


def retrieve_using_playwright(url_dict: dict[str, Callable], headless: bool = False) -> dict[str, Any]:
    """Generic wrapper for parsing multiple web pages."""
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        mobile = p.devices["iPhone 15 Plus"]
        context = browser.new_context(**mobile)
        try:
            page = context.new_page()
            for url, parse_function in url_dict.items():
                try:
                    console.print(f"fetching {url}", style="yellow")
                    page.goto(url)
                    page.wait_for_load_state("domcontentloaded")
                    try:
                        iframe_locator = page.frame_locator("iframe[id^='sp_message_iframe']")
                        iframe_locator.locator("button[aria-label='YES, I AGREE']").click(timeout=5000)
                    except PlaywrightTimeoutError:
                        console.print("iframe not found")

                    # Call the parsing function for this page
                    results[url] = parse_function(page)

                except Exception as e:
                    console.print(f"error retrieving {url}: {e}", style="red")  # Improved error message
                    results[url] = ErrorInfo(error=True, type=type(e).__name__, message=str(e), url=url)
            page.close()
        finally:
            browser.close()
    pprint(results)
    return results


class Cookie(TypedDict, total=False):
    """Type hint for cookie structure."""

    name: str
    value: str
    domain: str
    expiry: float


@dataclass
class SiteConfig:
    """Configuration for a specific news site."""

    name: str
    url: str
    cookie_domain: str
    extract_function: Callable[[BeautifulSoup], ArticleList]
    filename_stem: str


    def _get_dir(self) -> Path:
        """Get the download directory path."""
        return Path.home() / "Downloads"

    def make_page_filename(self) -> Path:
        return self._get_dir() / f"{self.filename_stem}.html"

    def make_cookie_filename(self) -> Path:
        return self._get_dir() / f"{self.filename_stem}-cookies.json"

SESSION_COOKIE_PATTERNS = frozenset(
    {
        # WSJ
        "s_cc",
        "s_ppv",
        "s_tp",
        "_dj_ses.",
        "_uetsid",
        "__tbc",
        "xbc",
        "_lr_retry_request",
        # NYTimes
        "nyt-b-sid",  # Browser session ID
        "nyt-traceid",  # Session tracing (no expiry)
        "_dd_s",  # DataDog session
        "_cb_svref",  # ChartBeat session reference
        "nyt-jkidd",  # User session data
        "nyt-purr",  # Session tracking
        "nyt-gdpr",  # GDPR session consent
        # FT.com patterns
        "FtComEntryPoint",  # Session entry point
        "OriginalReferer",  # Session referrer
        # bloomberg
        "_pxde",
        "_px2",
        "_pxhd",
        "_reg-csrf",
        "_reg-csrf-token",
        "pxcts",  # PerimeterX timestamp
        "_last-refresh",  # Page timestamp (1 hour)
        "AF_SYNC",  # Sync timestamp
    }
)

ONE_DAY_SECONDS = 86400  # 24 * 3600


def is_session_cookie(cookie: Cookie) -> bool:
    """
    Determine if a cookie is a session cookie.

    Session cookies are identified by:
    - No expiry field
    - Expiry within 1 day
    - Matching known session cookie patterns
    """
    expiry = cookie.get("expiry")

    # No expiry = session cookie
    if not expiry:
        return True

    # httpOnly cookies are typically session/auth tokens
    if cookie.get("httpOnly"):
        return True

    # Short expiry = session cookie
    if expiry - time.time() <= ONE_DAY_SECONDS:
        return True

    # Known session cookie pattern
    cookie_name = cookie.get("name", "")
    return any(pattern in cookie_name for pattern in SESSION_COOKIE_PATTERNS)


def load_cookies_from_file(filepath: Path, domain_suffix: str) -> dict[str, str]:
    """Load persistent cookies from a JSON file and return as dict {name: value}."""
    if not filepath.exists():
        raise FileNotFoundError(f"Cookies file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        cookies_list = json.load(f)

    return {cookie["name"]: cookie["value"] for cookie in cookies_list if cookie.get("domain", "").endswith(domain_suffix) and not is_session_cookie(cookie)}


def bloomberg_extract_headlines(soup: BeautifulSoup) -> list[ArticleLink]:
    def build_url(path: str) -> str:
        base = "https://bloomberg.com"
        url = urljoin(base, path)
        parsed = urlparse(url)
        return urlunparse(parsed._replace(query="", fragment=""))

    root = soup.find("section", {"data-zoneid": "Above the Fold"})
    if not root:
        return []

    results = {}  # store in dict keyed by url to filter duplicates
    for headline_elem in root.find_all("div", {"data-component": "headline"}):
        # Find the enclosing <a> tag
        link_tag = headline_elem.find_parent("a")
        if not link_tag:
            continue

        # Extract URL
        url = link_tag.get("href")
        if not url or url in results:
            continue

        # Extract headline text (from span inside headline)
        headline_span = headline_elem.find("span")
        headline = headline_span.get_text(strip=True) if headline_span else headline_elem.get_text(strip=True)

        # Try to find optional summary with data-testid="summary"
        summary_elem = link_tag.find("section", {"data-component": "summary"})
        summary = summary_elem.get_text(strip=True) if summary_elem else ""

        results[url] = ArticleLink(headline=headline, summary=summary, url=build_url(url))

    return list(results.values())


def ft_most_read(soup: BeautifulSoup) -> list[str]:
    """Extract URLs from the most-read section"""
    most_read_urls = []

    # Find all most-read items using data-id="most-read-id"
    most_read_items = soup.find_all("div", attrs={"data-id": "most-read-id"})

    for item in most_read_items:
        # Find the heading link within most-read items
        heading_elem = item.select_one('a[data-trackable="heading-link"]')
        if not heading_elem:
            continue

        url_path = heading_elem.get("href")
        if not url_path:
            continue

        # Fully qualify the URL
        url = urljoin("https://www.ft.com", url_path)

        most_read_urls.append(url)

    return most_read_urls


def ft_extract_headlines(soup: BeautifulSoup) -> list[ArticleLink]:
    """Extract FT story cards with headings and optional subheads"""
    root = soup.find("main")
    if not root:
        return []

    # Find all story group containers using the data-trackable attribute
    story_groups = root.find_all("div", attrs={"data-trackable": lambda x: x and "storyGroupTitle" in x})

    articles = {}

    for story_group in story_groups:
        # Find all elements with data-trackable that contain stories
        stories = story_group.find_all(attrs={"data-trackable": True})

        for story in stories:
            story_trackable = story.get("data-trackable", "")
            if not story_trackable.startswith("story:"):
                continue  # Skip non-story elements

            # Extract headline and URL
            heading_elem = story.select_one('a[data-trackable="heading-link"]')
            if not heading_elem:
                continue

            # Get URL
            url_path = heading_elem.get("href")
            if not url_path:
                continue
            url = urljoin("https://www.ft.com", url_path)

            # Get headline text - more robust approach
            headline = ""

            # Try to find the main headline span (not the indicator)
            headline_spans = heading_elem.find_all("span.text")
            for span in headline_spans:
                text = span.get_text(strip=True)
                # Skip indicator text like "Lex." or "The Big Read."
                if text and not any(indicator in text for indicator in ["Lex.", "The Big Read.", "opinion content."]):
                    headline = text
                    break

            # Fallback: get all text and clean it up
            if not headline:
                all_text = heading_elem.get_text(strip=True)
                # Remove common indicator texts
                for indicator in ["Lex.", "The Big Read."]:
                    all_text = all_text.replace(indicator, "").strip()
                headline = all_text

            if not headline:
                continue

            # Try to find summary/standfirst (optional)
            summary = ""
            # Look for standfirst in multiple possible locations
            summary_elem = story.select_one('.standfirst a[data-trackable="standfirst-link"] span') or story.select_one(".standfirst span") or story.select_one(".featured-story-content .standfirst span")
            if summary_elem:
                summary = summary_elem.get_text(strip=True)

            if url not in articles:
                articles[url] = ArticleLink(headline=headline, summary=summary, url=url)

    # Separate most-read articles from others
    most_read_articles = []
    other_articles = []
    most_read = ft_most_read(root)

    for article in articles.values():
        if article.url in most_read:
            most_read_articles.append(article)
        else:
            other_articles.append(article)
    # Return most-read articles first, then others
    return most_read_articles + other_articles


def nyt_extract_headlines(soup: BeautifulSoup) -> list[ArticleLink]:
    """Extract NYT story cards with headings and optional subheads"""
    root = soup.find("main")
    if not root:
        return []

    # Find all story wrappers
    stories = root.select('.story-wrapper[data-tpl="sli"]')

    articles = []

    for story in stories:
        # Find heading and URL together (same anchor element)
        heading = None
        url = None

        # Get the anchor element
        link_elem = story.select_one('[data-tpl="h"] a[data-tpl="l"]')
        if not link_elem:
            continue  # Skip if no link found

        url = link_elem.get("href")
        if not url:
            continue

        # Get heading text from within the anchor
        heading_elem = link_elem.select_one("p.indicate-hover")
        if heading_elem:
            heading = heading_elem.get_text(strip=True)

        if not heading or not url:
            continue

        # Try to find subhead (optional)
        summary = ""
        subhead_elem = story.select_one('[data-tpl="bo"] p.summary-class')
        if subhead_elem:
            summary = subhead_elem.get_text(strip=True)

        articles.append(ArticleLink(headline=heading, summary=summary, url=url))

    return articles


def wsj_extract_headlines(soup: BeautifulSoup) -> list[ArticleLink]:
    """Extract all mobile homepage cards with headings and subheads"""
    root = soup.find("main")
    if not root:
        return []

    # Find all elements with data-parsely-slot starting with "mobile-homepage-card-"
    cards = root.find_all(attrs={"data-parsely-slot": lambda x: x and x.startswith("mobile-homepage-card-")})

    articles = []
    for card in cards:
        # Try to find heading
        heading_elem = card.select_one('[data-testid="flexcard-headline"] .e1sf124z8')
        if not heading_elem:
            continue  # Skip this card if no heading

        heading = heading_elem.get_text(strip=True)

        # Find the parent anchor tag for the URL
        link_elem = heading_elem.find_parent("a")
        if not link_elem or not link_elem.get("href"):
            continue  # Skip if no link found

        url = link_elem["href"]

        # Optional subhead
        summary = ""
        subhead_elem = card.select_one('[data-testid="flexcard-text"]')
        if subhead_elem:
            summary = subhead_elem.get_text(strip=True)

        articles.append(ArticleLink(headline=heading, summary=summary, url=url))

    return articles


def scrape_site(site_config: SiteConfig) -> ArticleList | ErrorInfo:
    """Scrape a single site using its configuration."""
    # Load persistent cookies
    try:
        cookie_file = site_config.make_cookie_filename()
        cookies = load_cookies_from_file(cookie_file, site_config.cookie_domain)
        print(f"🍪 Loaded {len(cookies)} cookies from {cookie_file}")
    except Exception as e:
        return ErrorInfo(error=True, type=e.__class__.__name__, message=f"❌ Failed to load cookies: {e}", url="")

    # Set headers to mimic a mobile browser
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6.2 Mobile/15E148 Safari/604.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": site_config.url,
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    }

    # Make request
    with httpx.Client(cookies=cookies, headers=headers, follow_redirects=True, timeout=30.0, http2=True) as client:
        try:
            print(f"🌐 Fetching {site_config.url}...")
            response = client.get(site_config.url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return ErrorInfo(error=True, type=e.__class__.__name__, message=f"❌ HTTP error occurred: {e}", url="")
        except Exception as e:
            return ErrorInfo(error=True, type=e.__class__.__name__, message=f"❌ Request failed: {e}", url="")

    print("parsing response")
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract articles using site-specific function
    return ArticleList(source=site_config.name, articles=site_config.extract_function(soup))


def retrieve_headlines(source: str) -> ArticleList | ErrorInfo:
    site = source.lower()
    if site not in SITE_CONFIGS:
        msg = f"❌ Unknown site: {site}. Available sites: {list(SITE_CONFIGS.keys())}"
        return ErrorInfo(error=True, type="invalid argument", message=msg, url="")

    try:
        site_config = SITE_CONFIGS[site]
        return scrape_site(site_config)
    except Exception as e:
        return ErrorInfo(error=True, type=e.__class__.__name__, message=str(e), url=source)


def print_most_read_table(most_read: ArticleList):
    """Prints the list of ArticleLink objects from an ArticleList object as a markdown numbered list."""
    md = ""

    for i, article in enumerate(most_read.articles):
        # Add headline as the main item
        md += f"{i + 1}. **{article.headline}**\n"

        # Add summary if not blank
        if len(article.summary):
            md += f"   {article.summary}\n"

        # Add URL
        md += f"   {article.url}\n\n"

    # Add footer information
    md += f"from {most_read.source} retrieved {most_read.timestamp_retrieved}"

    # Print as markdown
    console.print(Markdown(md))


def parse_citation(text: str) -> Citation | None:
    """
    .. {{cite web
    | title       = UK and France aim for new Ukraine peace deal involving initial 1-mont…
    | url         = https://www.ft.com/content/603ba62c-73b2-4e14-846d-e3825c79bf56
    | date        = 2025-03-03
    | archiveurl  = http://archive.today/cml1e
    | archivedate = 2025-03-03 }} ..."
    """
    cite = text_between(text, "{{cite web", "}}")

    # Regular expression to extract key-value pairs
    pattern = r"\|\s*(\w+)\s*=\s*(.*)"
    if matches := re.findall(pattern, cite):
        data = {key.lower(): value.strip() for key, value in matches}
        return Citation(**data)
    return None


def retrieve_archive(url: str):
    console.print(f"fetching {url} from archive", style="yellow")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # go to archive home page and search for URL
        page.goto("https://archive.is/")

        input_locator = page.locator("#q")
        input_locator.wait_for(state="visible")
        if input := input_locator.element_handle():
            input.fill(url)
            page.click('input[type="submit"][value="search"]')

        # Wait until at least one <img> is present inside the thumbs container
        # page.locator("#row0 .THUMBS-BLOCK img").wait_for()

        # must wait for a unique element then can click the last <img> inside the thumbs block
        page.locator("#row0 .THUMBS-BLOCK img >> nth=0").wait_for(state="attached")
        page.locator("#row0 .THUMBS-BLOCK img >> nth=-1").click()

        # wait for target page
        page.wait_for_load_state("domcontentloaded")
        content = page.content()

        browser.close()

    return content, parse_citation(content)


def tool_call_handler(fnname: str, args: dict) -> str:
    pass


def retrieve_ft_article(url: str) -> str:
    content, citation = retrieve_archive(url)
    pprint(citation)

    text = ""
    if content:
        soup = BeautifulSoup(content, "html.parser")
        save_soup(soup, Path("~/Downloads/temp.html").expanduser())

        # Extract the first <h1> text (if present)
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        if title:
            citation.title = title  # citation can contain abbreviated title so replace

        # Find the article container
        article = soup.find(id="article-body")

        # Extract non-empty div texts and join with double newlines
        output = "\n\n".join(text for div in article.find_all("div", recursive=False) if (text := div.get_text(strip=True, separator=" ")))

    output = add_citation(output, citation)
    save_markdown_article(citation.title, output)
    return output

# Site configurations
SITE_CONFIGS: dict[str, SiteConfig] = {
    "bloomberg": SiteConfig(
        name="Bloomberg UK",
        url="https://www.bloomberg.com/uk",
        cookie_domain=".bloomberg.com",
        extract_function=bloomberg_extract_headlines,
        filename_stem="bloomberg",
    ),
    "ft": SiteConfig(
        name="Financial Times",
        url="https://www.ft.com/",
        cookie_domain=".ft.com",
        extract_function=ft_extract_headlines,
        filename_stem="ft",
    ),
    "nyt": SiteConfig(
        name="New York Times",
        url="https://www.nytimes.com/",
        cookie_domain=".nytimes.com",
        extract_function=nyt_extract_headlines,
        filename_stem="nyt",
    ),
    "wsj": SiteConfig(
        name="Wall Street Journal",
        url="https://www.wsj.com/",
        cookie_domain=".wsj.com",
        extract_function=wsj_extract_headlines,
        filename_stem="wsj",
    ),
}


def test_eval():
    s = "10 - (7 * .52 + 4 * .47)"
    print(evaluate_expression(s))
    print(evaluate_expression("'strawberry'.count('r')"))
    s = "apple_price_last_year = 0.26 / 1.10; orange_price_last_year = 0.79 / 1.10; apple_price_today = apple_price_last_year * 1.10; orange_price_today = orange_price_last_year * 1.10; total_cost_apples = 12 * apple_price_today; total_cost_oranges = 7 * orange_price_today; total_cost = total_cost_apples + total_cost_oranges; change = 10 - total_cost; change"
    print(evaluate_expression(s))
    s = "apple_price_last_year = 0.26 / 1.10\norange_price_last_year = 0.79 / 1.10\napple_price_today = apple_price_last_year * 1.10\norange_price_today = orange_price_last_year * 1.10\ntotal_cost_apples = 12 * apple_price_today\ntotal_cost_oranges = 7 * orange_price_today\ntotal_cost = total_cost_apples + total_cost_oranges\nchange = 10 - total_cost\nround(change,2)"
    print(evaluate_expression(s))
    # test imports
    s = "import sympy; x = sympy.symbols('x'); expr= x**2 - 2*x - 3; sympy.solve(expr, x)"
    print(evaluate_expression(s))
    print(evaluate_expression("(datetime.date.today() - datetime.date(1969, 7, 20)).days"))
    # test error handling
    print(evaluate_expression("x = 3; y = 2; y / (x - 3)"))


if __name__ == "__main__":
    pprint(f"force import of module {math.pi}")
    # pprint(retrieve_stock_quotes(["JNK", "TLT", "SPY", "PBW"]))

    # for site in SITE_CONFIGS.keys():
    for site in ["nyt"]:
        items = retrieve_headlines(site)
        if isinstance(items, ArticleList):
            print_most_read_table(items)
        else:
            console.print(items, style="red")
