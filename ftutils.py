#!/usr/bin/python3
from bs4 import BeautifulSoup
from datetime import date, datetime
import requests
from playwright.sync_api import sync_playwright
import re
from pydantic import BaseModel
from pathlib import Path
from rich.pretty import pprint
from rich.table import Table
from rich.markdown import Markdown
from rich.console import Console

console = Console()

class ArticleLink(BaseModel):
    headline: str
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

ft_most_read_fn = {
            "type": "function",
            "function": {
                "name": "retrieve_ft_most_read",
                "description": "Retrieves the most read articles from the Financial Times",
            } }

ft_article_fn = {
            "type": "function",
            "function": {
                "name": "retrieve_ft_article",
                "description": "Downloads the content of a Financial Times article given its URL",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the article to download"
                        }
                    },
                    "required": ["url"]
                }
            } }

def get_function_definitions():
    return [ft_most_read_fn, ft_article_fn]


def make_clean_filename(text: str) -> str:
    words = re.sub(r"[\\\.\/[\]<>'\":*?|]", " ", text.lower()).split()
    return "_".join(words[:5])


def text_between(content: str, start_tag: str, end_tag: str) -> str:
    r = ""
    start = content.find(start_tag)
    if start >= 0:
        end = content.find(end_tag, start)
        if end > start:
            r = content[start + len(start_tag):end]
    return r


def extract_bbc_most_read() -> ArticleList:
    """
    find h2  with id=mostRead-label, from parent div find all child anchor tags
    """
    r = requests.get("https://www.bbc.co.uk/news")
    soup = BeautifulSoup(r.text, "html.parser")
    mrs = soup.find('h2', id='mostRead-label')
    xs = []
    if mrs:
        for e in mrs.find_parent('div').find_all('a'):
            xs.append(ArticleLink(headline= e.get_text(strip=True), url= 'https://bbc.co.uk' + e.get('href')))
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="BBC News", articles=xs)


def retrieve_wsj_home_page(url: str) -> str:
    content = ""
    with sync_playwright() as p:
        console.print(f"fetching {url}", style='yellow')

        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
        page = context.new_page() #     page = browser.new_page()

        page.goto(url)
        page.wait_for_load_state('domcontentloaded')

        # Check for and click the cookie consent button (if it exists)
        if cookie_button := page.query_selector(".agree-btn"):
            cookie_button.click()
            # Optionally, wait for the cookie popup to disappear
            # page.wait_for_selector(".agree-btn", state="detached", timeout=5000)

        page.wait_for_selector(".subscribe")

        content = page.content()
        browser.close()
    return content


def retrieve_wsj_most_read() -> ArticleList:
    """
    find most popular news stories
    """
    content = retrieve_wsj_home_page('http://www.wsj.com/')
    soup = BeautifulSoup(content, "html.parser")

    divs = soup.find_all('div', {'data-layout-type': 'most-popular-news'})
    xs = []
    for h3 in divs[0].find_all('h3'):
        text = h3.get_text(strip=True)
        anchor = h3.find('a')
        url = anchor.get('href')
        xs.append(ArticleLink(headline= text, url= url[:url.find('?')]))
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="Wall Street Journal", articles=xs)


def print_most_read_table(most_read: ArticleList):
    """Prints the list of ArticleHeadline objects from an FtMostRead object in a table format."""
    table = Table(show_header=True, header_style="bold yellow", box=None)
    table.add_column("Num", style="cyan")
    table.add_column("Headline", style="cyan", width=72)
    table.add_column("Article id", style="cyan")

    for i,article in enumerate(most_read.articles):
        table.add_row(str(i), article.headline, article.url[-36:])

    console.print(table)
    console.print(f'from {most_read.source} retrieved {most_read.timestamp_retrieved}\n', style='yellow')


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
    console.print(f"fetching {url} from archive", style='yellow')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # go to archive home page and search for URL
        page.goto('https://archive.is/')

        page.fill('#q', url)
        page.click('input[type="submit"][value="search"]')

        # find list of page snapshots and click latest one
        if anchor := page.locator('div.TEXT-BLOCK').locator('a').first:
            anchor.click()

        # wait for target page
        page.wait_for_load_state('domcontentloaded')
        content = page.content()

        browser.close()

    return content


def tool_call_handler(messages: list, tool_calls: list) -> list:
    pass


def extract_article(div) -> ArticleLink:
    # get "data-content-id" or find child with read attribute
    id = div.get("data-content-id")
    content_id = id if id else div.find("div", class_ = "headline").get("data-content-id")
    # read child span with class="text" and get content
    headline_text = div.find("span", class_ = "text").get_text(strip=True)
    return ArticleLink(headline= headline_text, url= "https://www.ft.com/content/" + content_id)


def retrieve_ft_most_read() -> ArticleList:
    r = requests.get("https://www.ft.com/")
    soup = BeautifulSoup(r.text, "html.parser")

    xs = [extract_article(d) for d in soup.find_all("div", class_="headline--scale-7")]
    xs.extend(extract_article(d) for d in soup.find_all("div", {"data-id": "most-read-id"}))

    return ArticleList(timestamp_retrieved= datetime.now().isoformat(), source="Financial Times",articles= xs)


def retrieve_ft_article(url: str) -> str:
    content = retrieve_archive(url)
    buffer = ''

    if content:
        citation = parse_citation(content)
        pprint(citation)
        if title := text_between(content, "<title>", "</title>"):
            citation.title = title # citation can contain abbreviated title so replace
        buffer = f"## {citation.title}\n\n**source:** {citation.url}\n\n**date:** {citation.date}\n\n"

        filename = make_clean_filename(citation.title) + '.md'

        # the archive version of ft articles contains the extracted text as well as the html
        start = content.find("articleBody")
        end = content.find("wordCount", start)
        if start > 0 and end > start:
            s = content[start+14:end-3]
            s = s.replace('\\n', '\n')
            buffer += s
    
    p = Path.home() / 'Documents' / 'chats' / filename
    pprint(f'saved {p}')
    p.write_text(buffer, encoding="utf-8")
    return buffer


if __name__ == "__main__":
    items = retrieve_ft_most_read()
    print_most_read_table(items)
    # s = retrieve_ft_article(items.articles[2].url)
    # markdown = Markdown(s, style="cyan", code_theme="monokai")
    # console.print(markdown, width=80)
    items = extract_bbc_most_read()
    print_most_read_table(items)

    items = retrieve_wsj_most_read()
    print_most_read_table(items)
    # pprint(items)
