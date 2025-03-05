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

def extract_bbc_most_read() -> ArticleList:
    """
    find h2  with id=mostRead-label, find parent div and all all child anchor tags
    """
    r = requests.get("https://www.bbc.co.uk/news")
    soup = BeautifulSoup(r.text, "html.parser")
    mrs = soup.find('h2', id='mostRead-label')
    xs = []
    if mrs:
        for e in mrs.find_parent('div').find_all('a'):
            xs.append(ArticleLink(headline= e.get_text(strip=True), url= 'https://bbc.co.uk/news' + e.get('href')))
    return ArticleList(timestamp_retrieved=datetime.now().isoformat(), source="BBC News", articles=xs)


def retrieve_wsj_home_page(url: str) -> str:
    content = ""
    with sync_playwright() as p:
        console.print(f"fetching {url}", style='yellow')

        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url)
        page.wait_for_load_state('domcontentloaded')
        content = page.content()
        browser.close()
    return content


def retrieve_wsj_most_read() -> ArticleList:
    """
    find h2  with id=mostRead-label, find parent div and all all child anchor tags
    """
    soup = BeautifulSoup(retrieve_wsj_home_page('https://www.wsj.com/'), "html.parser")

    divs = soup.find_all('div', {'data-layout-type': 'most-popular-news'})
    xs = []
    for div in divs[0].find_all('h3'):
        anchor = div.find('a')
        url = anchor.get('href')
        d = anchor.find('div')
        text = d.get_text(strip=True)
        # find anchor tag and extract href
        # find child div and extract text content
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


# Function to parse the citation block
def parse_citation(text: str) -> Citation:
    # Regular expression to extract key-value pairs
    pattern = r"\|\s*(\w+)\s*=\s*(.*)"
    matches = re.findall(pattern, text)

    data = {key.lower(): value.strip() for key, value in matches}

    # Create and return a Citation object
    return Citation(
        title=data.get("title", ""),
        url=data.get("url", ""),
        date=data.get("date", ""),
        archiveurl=data.get("archiveurl", ""),
        archivedate=data.get("archivedate", ""),
    )

def find_citation(text: str) -> Citation:
    # text = """... {{cite web
    # | title       = UK and France aim for new Ukraine peace deal involving initial 1-mont…
    # | url         = https://www.ft.com/content/603ba62c-73b2-4e14-846d-e3825c79bf56
    # | date        = 2025-03-03
    # | archiveurl  = http://archive.today/cml1e
    # | archivedate = 2025-03-03 }} ..."""

    # Extract the citation block using regex
    start = text.find("{{cite web")
    if start > 0:
        start += 10
        end = text.find("}}", start)
        return parse_citation(text[start:end])

    return None


def retrieve_archive(url: str):
    console.print(f"fetching {url} from archive", style='yellow')
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto('https://archive.is/')

        page.fill('#q', url)

        # Click the button with id "search"
        page.click('input[type="submit"][value="search"]')

        page.wait_for_load_state('domcontentloaded') 

        anchor = page.locator('div.TEXT-BLOCK').locator('a').first

        anchor.click()

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
    """Downloads the most read articles from the Financial Times website and returns them as an FtMostRead object.

    Returns:
        FtMostRead: An object containing the timestamp of when the data was retrieved and a list of ArticleHeadline objects.
    """
    r = requests.get("https://www.ft.com/")
    soup = BeautifulSoup(r.text, "html.parser")

    xs = [extract_article(d) for d in soup.find_all("div", class_="headline--scale-7")]
    xs.extend(extract_article(d) for d in soup.find_all("div", {"data-id": "most-read-id"}))

    return ArticleList(timestamp_retrieved= datetime.now().isoformat(), source="Financial Times",articles= xs)


def retrieve_ft_article(url: str) -> str:
    content = retrieve_archive(url)
    buffer = ''

    if content:
        citation = find_citation(content)
        pprint(citation)
        start = content.find("<title>")
        end = content.find("</title>", start)
        if start > 0 and end > start:
            citation.title = content[start+7:end] # citation can contain abbreviated title so replace
            buffer = f"## {citation.title}\n\n**source:** {citation.url}\n\n**date:** {citation.date}\n\n"

        title_words = citation.title.split()
        filename = '_'.join(title_words[:5]) + '.md'

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
    # s = retrieve_ft_article(items.articles[0].url)
    # markdown = Markdown(s, style="cyan", code_theme="monokai")
    # console.print(markdown, width=80)
    items = extract_bbc_most_read()
    print_most_read_table(items)

    items = retrieve_wsj_most_read()
    print_most_read_table(items)
