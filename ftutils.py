#!/usr/bin/python3
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table

from htm2md import html_to_markdown

console = Console()


class ArticleLink(BaseModel):
    headline: str
    summary: str = Field(default="")
    url: str


class ArticleList(BaseModel):
    timestamp_retrieved: str
    source: str
    articles: list[ArticleLink]


class Citation(BaseModel):
    title: str
    url: str
    date: str
    archiveurl: str
    archivedate: str


retrieve_headlines_fn = {
    "type": "function",
    "function": {
        "name": "retrieve_headlines",
        "description": "Retrieve headlines from a news web site",
        "parameters": {"type": "object", "properties": {"source": {"type": "string", "enum": ["bbc", "ft", "wsj"], "description": "the name of the news web site"}}},
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


def get_function_definitions():
    return [retrieve_headlines_fn, retrieve_article_fn]


def make_clean_filename(text: str) -> str:
    words = re.sub(r"[\\\.\/[\]<>'\",:*?|]", " ", text.lower()).split()
    return "_".join(words[:5])


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


def extract_bbc_most_read() -> ArticleList:
    """
    find h2  with id=mostRead-label, from parent div find all child anchor tags
    """
    r = requests.get("https://www.bbc.co.uk/news")
    soup = BeautifulSoup(r.text, "html.parser")
    mrs = soup.find("h2", id="mostRead-label")
    xs = []
    if mrs:
        for e in mrs.find_parent("div").find_all("a"):
            xs.append(ArticleLink(headline=e.get_text(strip=True), url="https://bbc.co.uk" + e.get("href")))
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="BBC News", articles=xs)


def retrieve_wsj_article(url: str) -> str:
    content, citation = retrieve_archive(url)
    pprint(citation)
    soup = BeautifulSoup(content, "html.parser")

    # remove some divs before extracting text
    divs = soup.find_all("div")
    if divs:
        xs = [d for d in divs if "background-position:/*x=*/0% /*y=*/0%;" in d.get("style")]
        console.print(f"removing divs with style {len(xs)}", style="red")
        for d in xs:
            d.decompose()

    md = html_to_markdown(soup=soup.find("section"), title=soup.find("h1").get_text().strip(), subtitle=soup.find("h2").get_text().strip())
    save_markdown_article(citation.title, md)
    return md


def retrieve_wsj_home_page() -> ArticleList:
    """retrieve first 10 headlines from wsj mobile home page"""
    xs = []
    with sync_playwright() as p:
        console.print("fetching wsj.com via google translate", style="yellow")

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

        browser.close()

    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Wall Street Journal", articles=xs)


def print_most_read_table(most_read: ArticleList):
    """Prints the list of ArticleHeadline objects from an FtMostRead object in a table format."""
    table = Table(show_header=True, header_style="bold yellow", box=None)
    table.add_column("Num", style="cyan")
    table.add_column("Headline", style="cyan", width=72)
    table.add_column("Article id", style="cyan")

    for i, article in enumerate(most_read.articles):
        table.add_row(str(i), article.headline, article.url[-36:])

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


def retrieve_ft_most_read() -> ArticleList:
    r = requests.get("https://www.ft.com/")
    soup = BeautifulSoup(r.text, "html.parser")

    xs = [extract_article(d) for d in soup.find_all("div", class_="headline--scale-7")]
    xs.extend(extract_article(d) for d in soup.find_all("div", {"data-id": "most-read-id"}))

    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Financial Times", articles=xs)


def retrieve_ft_article(url: str) -> str:
    content, citation = retrieve_archive(url)
    pprint(citation)

    buffer = ""

    if content:
        if title := text_between(content, "<title>", "</title>"):
            citation.title = title  # citation can contain abbreviated title so replace
        buffer = f"## {citation.title}\n\n**source:** {citation.url}\n\n**date:** {citation.date}\n\n"

        # the archive version of ft articles contains the extracted text as well as the html
        start = content.find("articleBody")
        end = content.find("wordCount", start)
        if start > 0 and end > start:
            s = content[start + 14 : end - 3]
            s = s.replace("\\n", "\n")
            buffer += s

    save_markdown_article(citation.title, buffer)
    return buffer


def retrieve_headlines(source: str = "") -> ArticleList:
    s = source.lower()
    if s == "ft":
        return retrieve_ft_most_read()
    if s == "wsj":
        return retrieve_wsj_home_page()
    return extract_bbc_most_read()


def retrieve_article(url: str = "") -> str:
    if "www.ft.com" in url:
        return retrieve_ft_article(url)
    if "www.wsj.com" in url:
        return retrieve_wsj_article(url)
    return ""  # retrieve_bbc_article(url)


if __name__ == "__main__":
    items = retrieve_ft_most_read()
    print_most_read_table(items)
    # md = retrieve_wsj_article('https://www.wsj.com/politics/elections/democrat-party-strategy-progressive-moderates-13e8df10')
    # markdown = Markdown(md, style="cyan", code_theme="monokai")
    # console.print(markdown, width=80)
    items = extract_bbc_most_read()
    print_most_read_table(items)

    items = retrieve_wsj_home_page()
    print_most_read_table(items)
    pprint(items)
