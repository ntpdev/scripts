#!/usr/bin/python3
import re
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import requests
from bs4 import BeautifulSoup
from functools import cache
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from rich.table import Table

from htm2md import html_to_markdown

NYT_URL = "https://www.nytimes.com/"
WSJ_URL = "https://www-wsj-com.translate.goog/?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp&_x_tr_hist=true"

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
    timestamp_retrieved: str
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
        "parameters": {"type": "object", "properties": {"source": {"type": "string", "enum": ["bbc", "ft", "nyt", "wsj"], "description": "the name of the news web site"}}},
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
        "description": "Retrieve stock quotes for a list of tickers from Bloomberg",
        "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}, "description": "A list of stock tickers"}}, "required": ["tickers"]},
    },
}


@cache
def ftutils_functions() -> dict[str, dict[str, Any]]:
    """Returns a dictionary mapping function names to their definitions and a callable."""

    def name(d):
        return d["function"]["name"]

    return {
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


def get_bnnbloomberg_quote(symbol: str) -> dict:
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
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        scorecard = response.json()
        url = "https://bnn.stats.bellmedia.ca/bnn/api/stock/companyInfo?brand=bnn&lang=en&symbol=" + symbol
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        company_info = response.json()

        merge_from(scorecard)
        merge_from(company_info)
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # catch any other errors
    return result


def process_bnn_stock_page(page) -> dict | None:
    # url = f"https://www.bnnbloomberg.ca/stock/{ticker}/"
    # console.print("navigating to {url}", style="yellow")
    # page.goto(url)

    # the table is loaded on demand via a rest call when it is in view
    # alternatively use page.keyboard.press("PageDown")
    # <h2 id="bmw-scorecard--r86513c" class="bmw-header__title">Scorecard</h2>
    # breakpoint()
    page.wait_for_timeout(1000)
    # page.keyboard.press("PageDown")
    # page.wait_for_timeout(125)
    # page.get_by_text("Scorecard").scroll_into_view_if_needed(timeout=9000)

    ticker_head = page.locator("h1.c-heading")
    name = page.locator("h2.bmw-market-status__title")
    close = page.locator("span.bmw-market-status__info__price")
    d = {}
    d["symbol"] = ticker_head.inner_text()
    d["name"] = name.inner_text()
    d["close"] = float(close.inner_text())
    console.print("scorecard found " + d["symbol"], style="yellow")
    # Find all list items within the scorecard
    # listitems = page.locator("section.bmw-scorecard > ul > li")
    # for item in listitems.all():
    #     label = item.locator("span").inner_text().lower()
    #     value = item.locator("strong").inner_text()
    #     if value != "-":
    #         try:
    #             v = float(value)
    #         except ValueError:
    #             v = value
    #         d[label] = v

    # page.get_by_text("Company Info").scroll_into_view_if_needed(timeout=9000)
    # info = page.locator("section.bmw-company-info div.bmw-company-info__tab div.bmw-company-info__tab__content > p")
    # d["description"] = info.inner_text()

    return d


def extract_bbc_most_read() -> ArticleList:
    """
    find h2  with id=mostRead-label, from parent div find all child anchor tags
    """
    r = requests.get("https://www.bbc.co.uk/news")
    soup = BeautifulSoup(r.text, "html.parser")
    xs = []
    if mrs := soup.find("h2", id="mostRead-label"):
        xs = [ArticleLink(headline=anchor.get_text(strip=True), url="https://bbc.co.uk" + anchor["href"]) for anchor in mrs.find_parent("div").find_all("a")]
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="BBC News", articles=xs)


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


def retrieve_using_playwright(url_dict: dict[str, Callable], headless: bool = False) -> dict[str, Any]:
    """Generic wrapper for calling a parsing function on a web page."""
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


def parse_nyt_homepage(page) -> list[ArticleLink]:
    """retrieve first 10 headlines from nyt mobile home page"""

    def is_valid(text: str) -> bool:
        return text is not None and text != "LIVE" and "min read" not in text

    xs = []
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

    return xs


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


def retrieve_nyt_home_page() -> ArticleList:
    xs = retrieve_using_playwright("https://www.nytimes.com/", parse_nytimes_homepage)
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="New York Times", articles=list(xs.values()))


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

    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="New York Times", articles=xs)


def retrieve_wsj_home_page() -> ArticleList:
    xs = retrieve_using_playwright("https://www-wsj-com.translate.goog/?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp&_x_tr_hist=true", parse_wsj_homepage)
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Wall Street Journal", articles=xs)


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

            page.goto("https://www-wsj-com.translate.goog/?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp&_x_tr_hist=true")

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

    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Wall Street Journal", articles=xs)


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


def extract_article(div) -> ArticleLink:
    # get "data-content-id" or find child with read attribute
    id = div.get("data-content-id")
    content_id = id if id else div.find("div", class_="headline").get("data-content-id")
    # read child span with class="text" and get content
    headline_text = div.find("span", class_="text").get_text(strip=True)
    return ArticleLink(headline=headline_text, url="https://www.ft.com/content/" + content_id)


def retrieve_ft_most_read_section(url: str) -> ArticleList:
    """return most read articles. retrieve articles from most-read list and augment with teasers"""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

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

    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Financial Times", articles=articles.values())


def retrieve_ft_most_read() -> ArticleList:
    r = requests.get("https://www.ft.com/")
    soup = BeautifulSoup(r.text, "html.parser")

    xs = [extract_article(d) for d in soup.find_all("div", class_="headline--scale-7")]
    xs.extend(extract_article(d) for d in soup.find_all("div", {"data-id": "most-read-id"}))

    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Financial Times", articles=xs)


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


def retrieve_headlines(source: str = "") -> ArticleList:
    s = source.lower()
    if s == "ft":
        return merge(retrieve_ft_most_read(), retrieve_ft_most_read_section("https://www.ft.com/technology"), retrieve_ft_most_read_section("https://www.ft.com/markets"))
    if s == "nyt":
        xs = retrieve_cached_headlines()[NYT_URL]
        return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="New York Times", articles=xs)
    if s == "wsj":
        xs = retrieve_cached_headlines()[WSJ_URL]
        return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Wall Street Journal", articles=xs)
    return extract_bbc_most_read()


def retrieve_article(url: str = "") -> str:
    if "www.ft.com" in url:
        return retrieve_ft_article(url)
    if "www.wsj.com" in url:
        return retrieve_wsj_article(url)
    return retrieve_bbc_article(url)


def retrieve_stock_quotes(tickers: list[str]) -> QuoteList | None:
    "retrieve the closing price from the web page and other fields via REST calls"

    def make_url(ticker: str) -> str:
        t = ticker.upper()
        return f"https://www.bnnbloomberg.ca/stock/{t}/" if ticker.find(":") >= 0 else f"https://www.bnnbloomberg.ca/stock/{t}:UN/"

    d = {make_url(e): process_bnn_stock_page for e in tickers}
    result = retrieve_using_playwright(d)
    for r in result.values():
        r.update(get_bnnbloomberg_quote(r["symbol"]))
    return result if isinstance(result, ErrorInfo) else QuoteList(timestamp_retrieved=datetime.now().isoformat(), quotes=list(result.values()))


if __name__ == "__main__":
    # get_bnnbloomberg_quote("JNK:UN")
    # exit(0)
    # pprint(retrieve_stock_quotes(["JNK", "TLT", "SPY", "PBW"]))
    # exit(0)
    items = retrieve_headlines("nyt")
    print_most_read_table(items)

    items = retrieve_headlines("wsj")
    print_most_read_table(items)
    # items = retrieve_ft_most_read()
    # items2 = retrieve_ft_most_read_section("https://www.ft.com/markets")
    # print_most_read_table(items)
    # md = retrieve_wsj_article('https://www.wsj.com/politics/elections/democrat-party-strategy-progressive-moderates-13e8df10')
    # markdown = Markdown(md, style="cyan", code_theme="monokai")
    # console.print(markdown, width=80)
    # items = extract_bbc_most_read()
    # print_most_read_table(items)

    # items = retrieve_wsj_home_page()
    # print_most_read_table(items)
    # console.print(Markdown(items.to_markdown(), style="cyan"), width=80)
    # print_most_read_table(items2)
    # m = merge(items, items2)
    # print_most_read_table(m)
    # pprint(items)
