#!/usr/bin/python3
import re
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import httpx
import math
from bs4 import BeautifulSoup
from functools import cache
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
# from chatutils import execute_python_script
from rich.pretty import pprint
from rich.table import Table

from htm2md import html_to_markdown

NYT_URL = "https://www.nytimes.com/"
WSJ_URL = "https://www-wsj-com.translate.goog/?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp&_x_tr_hist=true"
BLOOMBERG_URL = "https://www.bloomberg.com/uk"
# cookies from a bloomberg request
BLOOMBERG_COOKIES = """
exp_pref=UK; country_code=GB; _pxhd=hybJB-QwokZf4leNclckCnRVuIpFClaKV1llOjmL0jtP37YQSo3OvB3LQomq-H5WHuzFBOtLOYP6-94lFNOnYA==:xcFLr9qTPJw8sGEzPYHXuJc/hPFuxaP3C/CQ9l6KTqWEzP/bEwlbm11lGlIPx9iZG7/zyzfnN5LXSk2J/ImKNFYpsCly7dMgTOUK5jxixZY=; session_id=eb8c34d4-17dc-4f1f-8aa1-96c51d642f0f; _session_id_backup=eb8c34d4-17dc-4f1f-8aa1-96c51d642f0f; agent_id=7b759df2-41ab-48f2-8d18-e8f4c830e74f; session_key=05e93f9def0810c2a5f8f184dab9e72b2a664249; _reg-csrf-token=vUgp4QCG-ZhI6oKN1fp1d1IRVQcaDZdJojN8; _reg-csrf=s%3A_t_6ZTpMq4aJfacc4id6cG6T.N3kayACM%2BeU1su9PG3W38wIn0F5hjdkbfPezAkk%2Beck; _sp_krux=true; gatehouse_id=39936fa4-25a4-4ca0-b34b-6b0745e9aaaa; geo_info=%7B%22countryCode%22%3A%22GB%22%2C%22country%22%3A%22GB%22%2C%22field_n%22%3A%22hf%22%2C%22trackingRegion%22%3A%22Europe%22%2C%22cacheExpiredTime%22%3A1743438724334%2C%22region%22%3A%22Europe%22%2C%22fieldN%22%3A%22hf%22%7D%7C1743438724334; geo_info={%22country%22:%22GB%22%2C%22region%22:%22Europe%22%2C%22fieldN%22:%22hf%22}|1743438723668; consentUUID=44d6a9b5-52bd-46da-8b60-dacd569ab812_42; consentDate=2025-03-24T16:32:09.533Z; bbgconsentstring=req1fun1pad1; bdfpc=004.2124236125.1742833928898; _ga_GQ1PBLXZCT=GS1.1.1743025085.3.1.1743025481.0.0.0; _ga=GA1.1.2038307183.1742833929; _gcl_au=1.1.465814927.1742833929; usnatUUID=5e7f4c49-db91-48fa-95bc-fbd68e8b7c26; pxcts=890d621c-08cd-11f0-9cbe-2c6d2c51ccd5; _pxvid=840a3f28-08cd-11f0-a082-8df59c8c78b6; _user-data=%7B%22status%22%3A%22anonymous%22%7D; _last-refresh=2025-3-26%2021%3A38; _pxde=e0adf1db95f603de3ae8f5ccb5a02c822502d8210417fd7f4170bce2642f3ea3:eyJ0aW1lc3RhbXAiOjE3NDMwMjUzMjgxNDIsImZfa2IiOjAsImlwY19pZCI6W119; _px2=eyJ1IjoiOTlmOGY2NDAtMGE4YS0xMWYwLWFhZWItYTcwMjM0ZjFmN2U4IiwidiI6Ijg0MGEzZjI4LTA4Y2QtMTFmMC1hMDgyLThkZjU5YzhjNzhiNiIsInQiOjE3NDMwMjU2MjgxNDIsImgiOiIyOWQwNzNmMDU4NThlMjA1ZGUwNTllYTYxYjdmODIxYzQ1NTc1Mzg4ODFmMDNjMjQyMTJkOThhMWMwNzUxMTliIn0=
"""

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
        "name": "evaluate_expression",
        "description": "Evaluates a mathematical or Python expression provided as a string. Types and constants from standard python libraries like math and datetime are available.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to be evaluated.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        },
        "strict": True
    },
}


def evaluate_expression_impl(expression: str) -> Any | None:
    # Split into individual parts and remove whitespace
    parts = [p.strip() for p in re.split(r'[;\n]', expression)]
    if not parts:
        return None  # Empty input
    
    parts = ["import math", "import datetime"] + parts
    # Separate final expression
    *statements, last_part = parts
    
    # Create a namespace dictionary to store variables
    namespace = {}
    
    # Execute all statements updating the namespace as necessary
    if statements:
        exec('\n'.join(statements), namespace)
    
    # Evaluate and return the final expression
    return eval(last_part, namespace)


def evaluate_expression(expression: str) -> str:
    result = ""
    if expression:
        try:
            console.print("eval: " + expression, style="yellow")           
            result = evaluate_expression_impl(expression)
            console.print("result: " + str(result), style="yellow")
        except Exception as e:
            result = "ERROR: " + str(e)
            console.print(result, style="red")
    else:
        result = "ERROR: no expression found"
        console.print(result, style="red")
    return str(result)


def retrieve_headlines(source: str) -> ArticleList:
    if not source:
        return ErrorInfo(error=True, type="invalid argument", message= "source required", url="")
    try:
        s = source.lower()
        if s == "ft":
            return retrieve_ft_headlines()
        if s == "nyt":
            xs = retrieve_cached_headlines()[NYT_URL]
            return ArticleList(source="New York Times", articles=xs)
        if s == "wsj":
            xs = retrieve_cached_headlines()[WSJ_URL]
            return ArticleList(source="Wall Street Journal", articles=xs)
        if s == "bloomberg":
            xs = retrieve_bloomberg_home_page()
            return ArticleList(source="Bloomberg UK", articles=xs)
        return retrieve_bbc_most_read()
    except Exception as e:
        return ErrorInfo(error=True, type=e.__class__.__name__, message=str(e), url=source)


def retrieve_article(url: str) -> str:
    if not url:
        return ErrorInfo(error=True, type="invalid argument", message= "url required", url="")

    if "www.ft.com" in url:
        return retrieve_ft_article(url)
    if "www.wsj.com" in url:
        return retrieve_wsj_article(url)
    return retrieve_bbc_article(url)


def retrieve_stock_quotes(symbols: list[str]) -> QuoteList | ErrorInfo:
    """Retrieves historical stock quotes for the given symbols.

    Args:
        symbols: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOG']).

    Returns:
        a list of quotes and details about the symbol

    """

    if not symbols:
        return ErrorInfo(error=True, type="invalid argument", message= "symbols required", url="")

    def make_url(ticker: str) -> str:
        t = ticker.upper()
        return f"https://www.bnnbloomberg.ca/stock/{t}/" if ticker.find(":") >= 0 else f"https://www.bnnbloomberg.ca/stock/{t}:UN/"

    d = {make_url(e): process_bnn_stock_page for e in symbols}
    result = retrieve_using_playwright(d)
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0"
    }

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
        return BeautifulSoup(f.read(), 'html.parser')


def save_markdown_article(title: str, text: str) -> Path | None:
    if not title:
        return None

    filename = make_clean_filename(title) + ".md"
    p = Path.home() / "Documents" / "chats" / filename
    pprint(f"saved {p}")
    p.write_text(text, encoding="utf-8")
    return p


def text_between(content: str, start_tag: str, end_tag: str) -> str:
    r = ""
    start = content.find(start_tag)
    if start >= 0:
        end = content.find(end_tag, start)
        if end > start:
            r = content[start + len(start_tag) : end]
    return r


def add_citation(text: str, cite: Citation) -> str:
    pprint(cite)
    s = f"# {cite.title}\n\n**source:** {cite.url}\n\n**published:** {cite.date}\n\n"
    return s + text


def get_bnnbloomberg_quote(client: httpx.Client, symbol: str) -> dict:
    """
    Fetches JNK:UN summary info and company info

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
        print(f"An unexpected error occurred: {e}")
    return result


def process_bnn_stock_page(page) -> dict | None:
    ticker_head = page.locator("h1.c-heading")
    name = page.locator("h2.bmw-market-status__title")
    close = page.locator("span.bmw-market-status__info__price")
    d = {
        "symbol": ticker_head.inner_text(),
        "name": name.inner_text(),
        "close": float(close.inner_text())}
    console.print("scorecard found " + d["symbol"], style="yellow")
    return d


def retrieve_bloomberg_home_page() -> list[ArticleList]:
    """use httpx to get uk homepage and parse headlines. need a valid cookie string"""

    def find_parent_prop(element, tagname, attr):
        """Traverse up the DOM tree to find the nearest parent <a> tag and return its href."""
        elem = element
        while (elem := elem.parent) is not None:
            if elem.name == tagname and elem.has_attr(attr):
                return elem[attr]
        return None

    def find_sibling_with_component(element, prop, component_value):
        """Find the nearest sibling element with the specified data-component attribute."""
        sibling = element
        while (sibling := sibling.find_next_sibling()) is not None:
            if sibling.has_attr(prop) and sibling[prop] == component_value:
                return sibling.get_text(strip=True)
        return ""

    with httpx.Client(http2=True) as client:
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-UK,en;q=0.5",
            "Connection": "keep-alive",
            "Cookie": BLOOMBERG_COOKIES,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0"
        }
        response = client.get(BLOOMBERG_URL, headers=headers)
        response.raise_for_status()
        content = response.text

    soup = BeautifulSoup(content, 'html.parser')
    save_soup(soup, Path.home() / "Downloads" / "bloomberg-uk.html")
    section = soup.find('section', {'data-zoneid': 'Above the Fold'})
    headlines = section.find_all('div', {'data-testid': 'headline'})
    xs = []
    for div in headlines:
        href = find_parent_prop(div, "a", "href")
        summary = find_sibling_with_component(div, "data-component", "summary")
        if href:
            xs.append(ArticleLink(headline=div.get_text(strip=True), summary = summary, url="https://www.bloomberg.com" + href.split('?')[0]))
    return xs


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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0"
        }
        response = client.get("https://www.bbc.co.uk/news", headers=headers)
        response.raise_for_status()
        content = response.text

    soup = BeautifulSoup(content, 'html.parser')
#    save_soup(soup, Path.home() / "Downloads" / "bbc-news.html")
    xs = []
    if mrs := soup.find("h2", id="mostRead-label"):
        xs = [ArticleLink(headline=anchor.get_text(strip=True), url="https://bbc.co.uk" + anchor["href"]) for anchor in mrs.find_parent("div").find_all("a")]
    return ArticleList(source="BBC News", articles=xs)


def get_bbc_article_contents(url: str) -> BeautifulSoup | None:
    console.print(f"navigate {url}", style="yellow")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(url)

            if cookie_button := page.wait_for_selector('button[data-testid="reject-button"]', timeout=5000):
                cookie_button.click()
            # find the h1 by id
            page.wait_for_timeout(1000)
            locator = page.locator("#main-heading")
            console.print(f"retrieved - {locator.text_content()}")

            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            save_soup(soup, Path("c://temp/z.htm"))
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
    if divs := soup.find_all('div', style=lambda x: x and 'align-items' in x):
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
        iphone_15 = p.devices["iPhone 15"]
        context = browser.new_context(**iphone_15)
        try:
            page = context.new_page()
            for url, parse_function in url_dict.items():
                try:
                    console.print(f"fetching {url}", style="yellow")
                    page.goto(url)
                    page.wait_for_load_state("domcontentloaded")
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

def parse_bloomberg_homepage(page) -> list[ArticleLink]:
    links = []
    for anchor in page.query_selector_all('a[data-component="story-link"]'):
        headline_element = anchor.query_selector('div[data-component="headline"]')
        summary_element = anchor.query_selector('section[data-component="summary"]')
        
        if headline_element and summary_element:
            headline = headline_element.inner_text()
            summary = summary_element.inner_text()
            href = anchor.get_attribute('href')
            links.append(ArticleLink(headline=headline, summary=summary, url=href))

    return links

def parse_nyt_homepage(page) -> list[ArticleLink]:
    """retrieve first 10 headlines from nyt mobile home page"""

    def is_valid(text: str) -> bool:
        return text is not None and text != "LIVE" and text != "BREAKING" and "min read" not in text

    links = []
    # accept cookies
    if button := page.wait_for_selector('button[data-testid="Accept all-btn"]', timeout=4000):
        button.click()

    # accept compliance
    if div := page.wait_for_selector("div#complianceOverlay", timeout=1000):
        div.wait_for_selector("button", timeout=100).click()

    if element := page.locator('span[data-testid="todays-date"]'):
        date_string = element.inner_text()
        try:
            # Parse the date string, ignoring the day
            pubdate = datetime.strptime(date_string, "%A, %B %d, %Y").date()
            console.print(f"published {pubdate}", style="yellow")
        except ValueError:
            console.print(f"Error: Could not parse date string: {date_string}", style="red")

    story_elements = page.query_selector_all("section.story-wrapper")
    for section in story_elements:
        anchor = section.query_selector(" > a")
        href = anchor.get_attribute("href") if anchor else ""

        child_elements = section.query_selector_all("p")  # Get all p elements.
        texts = [e.text_content().strip() for e in child_elements if is_valid(e.text_content())]

        title = texts[0] if len(texts) > 0 else ""
        subtitle = texts[1] if len(texts) > 1 else ""

        if len(title):
            links.append(ArticleLink(headline=title, summary=subtitle, url=href))
            if len(links) >= 10:
                break

    return links


def parse_wsj_homepage(page) -> list[ArticleLink]:
    """retrieve first 10 headlines from wsj mobile home page"""
    xs = []
    if cards := page.locator(".e1u7xa1g1.css-1x52dtc-CardLayoutItem"):
        for i in range(10):
            card = cards.nth(i)
            h3 = card.locator("h3")
            title = h3.text_content()
            anchor = h3.locator("a")
            subtitle = card.locator("p").first.text_content().strip()
            url = anchor.get_attribute("href")
            xs.append(ArticleLink(headline=title, summary=subtitle, url=url[: url.find("?")]))

    return xs

def retrieve_bloomberg_home_page_playright() -> ArticleList | None:
    xs = retrieve_using_playwright({BLOOMBERG_URL: parse_bloomberg_homepage})
    return ArticleList(source="Bloomberg", articles=list(xs.values()))

def retrieve_nyt_home_page() -> ArticleList | None:
    xs = retrieve_using_playwright({NYT_URL: parse_nyt_homepage})
    return ArticleList(source="New York Times", articles=list(xs.values()))


def retrieve_nyt_home_page_ex() -> ArticleList:
    """retrieve first 10 headlines from wsj mobile home page"""

    def is_valid(text: str) -> bool:
        return text is not None and text != "LIVE" and "min read" not in text

    xs = []
    with sync_playwright() as p:
        console.print("fetching nytimes", style="yellow")
        try:
            browser = p.chromium.launch(headless=False)

            iphone_15 = p.devices["iPhone 15"]
            context = browser.new_context(**iphone_15)
            page = context.new_page()

            page.goto("https://www.nytimes.com/")
            # accept cookies
            if button := page.wait_for_selector('button[data-testid="Accept all-btn"]', timeout=4000):
                button.click()

            # accept compliance
            if div := page.wait_for_selector("div#complianceOverlay", timeout=1000):
                div.wait_for_selector("button", timeout=100).click()

            if element := page.locator('span[data-testid="todays-date"]'):
                date_string = element.inner_text()
                try:
                    # Parse the date string, ignoring the day
                    pubdate = datetime.strptime(date_string, "%A, %B %d, %Y").date()
                    console.print(f"published {pubdate}", style="yellow")
                except ValueError:
                    console.print(f"Error: Could not parse date string: {date_string}", style="red")
            story_elements = page.query_selector_all("section.story-wrapper")

            for i, story_element in enumerate(story_elements):
                if i >= 10:
                    break  # Stop after the first 10 elements

                anchor = story_element.query_selector(" > a")
                href = anchor.get_attribute("href") if anchor else ""

                child_elements = story_element.query_selector_all("p")  # Get all p elements.
                texts = [e.text_content().strip() for e in child_elements if is_valid(e.text_content())]

                title = texts[0] if len(texts) > 0 else ""
                subtitle = texts[1] if len(texts) > 1 else ""

                xs.append(ArticleLink(headline=title, summary=subtitle, url=href))
        except Exception as e:
            console.print(f"error retrieving {e}", style="red")
            return None
        finally:
            browser.close()

    return ArticleList(source="New York Times", articles=xs)


def retrieve_wsj_home_page() -> ArticleList:
    xs = retrieve_using_playwright({WSJ_URL: parse_wsj_homepage})
    return ArticleList(source="Wall Street Journal", articles=xs)


def retrieve_wsj_home_page_ex() -> ArticleList | None:
    """retrieve first 10 headlines from wsj mobile home page"""
    xs = []
    with sync_playwright() as p:
        console.print("fetching wsj.com via google translate", style="yellow")
        try:
            browser = p.chromium.launch(headless=False)

            iphone_15 = p.devices["iPhone 15"]
            context = browser.new_context(**iphone_15)
            page = context.new_page()

            page.goto(WSJ_URL)

            if cards := page.locator(".e1u7xa1g1.css-1x52dtc-CardLayoutItem"):
                for i in range(10):
                    card = cards.nth(i)
                    h3 = card.locator("h3")
                    title = h3.text_content()
                    anchor = h3.locator("a")
                    subtitle = card.locator("p").first.text_content().strip()
                    url = anchor.get_attribute("href")
                    # console.print(f"Title: {title}, Subtitle: {subtitle}, URL: {href}")
                    xs.append(ArticleLink(headline=title, summary=subtitle, url=url[: url.find("?")]))

        except Exception as e:
            console.print(f"error retrieving {e}", style="red")
            return None
        finally:
            browser.close()

    return ArticleList(source="Wall Street Journal", articles=xs)


def print_most_read_table(most_read: ArticleList):
    """Prints the list of ArticleLink objects from an ArticleList object in a table format."""
    table = Table(show_header=True, header_style="bold yellow", box=None)
    table.add_column("Num", style="cyan")
    table.add_column("Headline", style="cyan", width=60)
    table.add_column("Teaser / Article id", style="cyan")

    for i, article in enumerate(most_read.articles):
        table.add_row(str(i + 1), article.headline, article.summary if len(article.summary) else article.url[-36:])

    console.print(table)
    console.print(f"from {most_read.source} retrieved {most_read.timestamp_retrieved}\n", style="yellow")


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

        if input := page.wait_for_selector("#q"):
            input.fill(url)
            page.click('input[type="submit"][value="search"]')

        # find list of page snapshots and click latest one
        if anchor := page.locator("div.TEXT-BLOCK").locator("a").first:
            anchor.click()

        # wait for target page
        page.wait_for_load_state("domcontentloaded")
        content = page.content()

        browser.close()

    return content, parse_citation(content)


def tool_call_handler(fnname: str, args: dict) -> str:
    pass


def parse_ft_most_read_section(soup) -> ArticleList:
    """return most read articles. retrieve articles from most-read list and augment with teasers"""

    articles = {}
    if most_read := soup.find("div", class_="o-teaser-collection--numbered"):
        for title_anchor in most_read.find_all("a"):
            title = title_anchor.text.strip()
            url = title_anchor["href"]
            articles[url] = ArticleLink(headline=title, url="https://www.ft.com" + url)

    # find all teaser blocks so we can get subtitle and add to most read
    for div in soup.find_all("div", class_="o-teaser__content"):
        # Retrieve the anchor (assuming these are always present)
        url = div.select_one(".o-teaser__heading a")["href"]

        # Retrieve the subtitle, handling the case where it might not exist
        subtitle_element = div.select_one(".o-teaser__standfirst a")
        subtitle = subtitle_element.text.strip() if subtitle_element else ""
        if url in articles:
            if len(articles[url].summary) < len(subtitle):
                articles[url].summary = subtitle

    return ArticleList(source="Financial Times", articles=articles.values())


def parse_ft_most_read(soup) -> ArticleList:

    def extract_article(div) -> ArticleLink:
    # get "data-content-id" or find child with read attribute
        id = div.get("data-content-id")
        content_id = id if id else div.find("div", class_="headline").get("data-content-id")
        # read child span with class="text" and get content
        headline_text = div.find("span", class_="text").get_text(strip=True)
        return ArticleLink(headline=headline_text, url="https://www.ft.com/content/" + content_id)

    xs = [extract_article(d) for d in soup.find_all("div", class_="headline--scale-7")]
    xs.extend(extract_article(d) for d in soup.find_all("div", {"data-id": "most-read-id"}))

    return ArticleList(source="Financial Times", articles=xs)


def retrieve_ft_headlines() -> ArticleList:
    with httpx.Client(http2=True) as client:
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-UK,en;q=0.5",
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0"
        }

        xs = []
        response = client.get("https://www.ft.com/", headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        xs.append(parse_ft_most_read(soup))

        response = client.get("https://www.ft.com/technology", headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        xs.append(parse_ft_most_read_section(soup))

        response = client.get("https://www.ft.com/markets", headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        xs.append(parse_ft_most_read_section(soup))

        response = client.get("https://www.ft.com/world-uk", headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        xs.append(parse_ft_most_read_section(soup))

    return merge(*xs)


def retrieve_ft_article(url: str) -> str:
    content, citation = retrieve_archive(url)
    pprint(citation)

    text = ""

    if content:
        if title := text_between(content, "<title>", "</title>"):
            citation.title = title  # citation can contain abbreviated title so replace

        # the archive version of ft articles contains the extracted text as well as the html
        start = content.find("articleBody")
        end = content.find("wordCount", start)
        if start > 0 and end > start:
            s = content[start + 14 : end - 3]
            s = s.replace("\\n", "\n")
            text += s

    text = add_citation(text, citation)
    save_markdown_article(citation.title, text)
    return text


@cache
def retrieve_cached_headlines():
    """retrieve from both sites as we need to web scrape. results are cached"""
    return retrieve_using_playwright({NYT_URL: parse_nyt_homepage, WSJ_URL: parse_wsj_homepage})


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
    retrieve_bloomberg_article("https://www.bloomberg.com/news/features/2025-04-24/price-alwaleed-s-elon-musk-ties-are-boosting-his-comeback-bid")
    # get_bnnbloomberg_quote("JNK:UN")
    # exit(0)
    # pprint(retrieve_stock_quotes(["JNK", "TLT", "SPY", "PBW"]))
    # exit(0)
    # items = retrieve_headlines("nyt")
    # print_most_read_table(items)

    # items = retrieve_headlines("wsj")
    # print_most_read_table(items)

    # items = retrieve_headlines("bloomberg")
    # print_most_read_table(items)

    # items = retrieve_headlines("ft")

    # items2 = retrieve_ft_most_read_section("https://www.ft.com/markets")
    # print_most_read_table(items)
    # md = retrieve_wsj_article('https://www.wsj.com/politics/elections/democrat-party-strategy-progressive-moderates-13e8df10')
    # markdown = Markdown(md, style="cyan", code_theme="monokai")
    # console.print(markdown, width=80)
    items = retrieve_bbc_most_read()
    # items = ArticleList(source="Bloomberg UK", articles=retrieve_bloomberg_home_page())
    print_most_read_table(items)
    test_eval()

    # items = retrieve_wsj_home_page()
    # print_most_read_table(items)
    # console.print(Markdown(items.to_markdown(), style="cyan"), width=80)
    # print_most_read_table(items2)
    # m = merge(items, items2)
    # print_most_read_table(m)
    # pprint(items)
