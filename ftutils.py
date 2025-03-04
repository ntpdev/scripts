#!/usr/bin/python3
from bs4 import BeautifulSoup
from datetime import date, datetime
import requests
from playwright.sync_api import sync_playwright
import re
from pydantic import BaseModel
from pathlib import Path
from rich.pretty import pprint

class ArticleHeadline(BaseModel):
    headline: str
    url: str

class FtMostRead(BaseModel):
    date_retrieved: str
    articles: list[ArticleHeadline]

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
    # text = """xyz {{cite web
    # | title       = UK and France aim for new Ukraine peace deal involving initial 1-mont…
    # | url         = https://www.ft.com/content/603ba62c-73b2-4e14-846d-e3825c79bf56
    # | date        = 2025-03-03
    # | archiveurl  = http://archive.today/cml1e
    # | archivedate = 2025-03-03 }} xyz"""

    # Extract the citation block using regex
    start = text.find("{{cite web")
    if start > 0:
        start += 10
        end = text.find("}}", start)
        return parse_citation(text[start:end])

    return None


def retrieve_ft_archive(url: str):
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto('https://archive.is/')

        page.fill('#q', url)

        # Click the button with id "search"
        page.click('input[type="submit"][value="search"]')

        page.wait_for_load_state() 

        div = page.locator('div.TEXT-BLOCK')
        anchor = div.locator('a').first

        anchor.click()

        page.wait_for_load_state('networkidle')
        content = page.content()

        # await asyncio.sleep(30)

        browser.close()

        return content


def tool_call_handler(messages: list, tool_calls: list) -> list:
    pass

def retrieve_ft_most_read() -> FtMostRead:
    """Downloads the most read articles from the Financial Times website and returns them as an FtMostRead object.

    Returns:
        FtMostRead: An object containing the timestamp of when the data was retrieved and a list of ArticleHeadline objects.
    """
    r = requests.get("https://www.ft.com/")
    soup = BeautifulSoup(r.text, "html.parser")

    most_read_divs = soup.find_all("div", {"data-id": "most-read-id"})
    xs = []
    for div in most_read_divs:
        # read child div with class headline and get attribute "data-content-id"
        content_id = div.find("div", class_ = "headline").get("data-content-id")
        # read child span with class="text" and get content
        headline_text = div.find("span", class_ = "text").get_text(strip=True)
        xs.append(ArticleHeadline(headline= headline_text, url= "https://www.ft.com/content/" + content_id))
    return FtMostRead(date_retrieved= datetime.now().isoformat(), articles= xs)


def retrieve_ft_article(url: str) -> str:
    # return f"**from:** {url}\n\n The US is suspending military aid to Ukraine as President Donald Trump seeks to increase pressure on President Volodymyr Zelenskyy to make concessions just days after the two leaders publicly sparred in the White House over peace talks with Russia."
    content = retrieve_ft_archive(url)
    buffer = ''

    if content:
        citation = find_citation(content)
        print(citation)
        start = content.find("<title>")
        end = content.find("</title>", start)
        if start > 0 and end > start:
            citation.title = content[start+7:end] # citation can contain abbreviated title so replace
            buffer = f"## {citation.title}\n\n**source:** {citation.url}\n**date:** {citation.date}\n\n"

        title_words = citation.title.split()
        filename = '_'.join(title_words[:5]) + '.md'

        start = content.find("articleBody")
        end = content.find("wordCount", start)
        if start > 0 and end > start:
            s = content[start+14:end-3]
            s = s.replace('\\n', '\n')
            buffer += s
    
    p = Path.home() / 'Downloads' / filename
    print(f'saved {p}')
    p.write_text(buffer, encoding="utf-8")
    return buffer


if __name__ == "__main__":
    items = retrieve_ft_most_read()
    pprint(items)
    retrieve_ft_article(items.articles[0].url)
