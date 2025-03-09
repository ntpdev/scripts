import requests
from bs4 import BeautifulSoup, NavigableString, Comment
from playwright.sync_api import sync_playwright
import re
from datetime import datetime
from rich.markdown import Markdown
from rich.console import Console
from rich.pretty import pprint
from pathlib import Path
from dataclasses import dataclass

console = Console()

@dataclass
class anchor_tag:
    text: str
    href: str


def test():
    # open file c://test/z.htm
    # load into BeautifulSoup
    # find all h3 tags and print content
    try:
        with open('c://temp/z.htm', 'r', encoding='utf-8') as file:
            content = file.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        section = soup.find_all('div', {'data-layout-type': 'most-popular-news'})
        # find all h3 tags and print content
        h3_tags = section[0].find_all('h3')
        for h3 in h3_tags:
            text = h3.get_text(strip=True) 
            link = h3.find('a')['href']
            print(f"{text} - {link}")
    except Exception as e:
        console.print(f"Error reading or parsing the file: {e}", style='red')

def get_bnnbloomberg_quote(symbol: str) -> dict:
    """
    Fetches JNK:UN the rest request does not contain the last price as that is already on the page

    Returns:
        dict: The JSON response as a dictionary, or None if an error occurs.
    """
    url = "https://bnn.stats.bellmedia.ca/bnn/api/stock/Scorecard?symbol=" + symbol
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}") #catch any other errors
        return None


def get_wsj_html(url: str) -> str | None:
    console.print(f'retrieve {url}', style='yellow')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        page = browser.new_page()
        try:
            # search on archive for url
            page.goto('https://archive.is/')
            page.fill('#q', url)
            page.click('input[type="submit"][value="search"]')

            # click which version of archived page we want
            anchor = page.locator('div.TEXT-BLOCK').locator('a').first
            anchor.click()
            page.wait_for_load_state('domcontentloaded')
            with Path('c://temp/z.htm').open('w', encoding='utf-8') as f:
                f.write(page.content())

            main_tag = page.locator('main')
            h1_tag = page.locator('h1')
            return h1_tag.text_content(), main_tag.inner_html()
        except Exception as e:
            console.print(f"error retrieving {e}", style='red')
            return None
        finally:
            browser.close()


def get_bbc_html(url: str) -> BeautifulSoup | None:
    console.print(f'navigate {url}', style='yellow')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(url)

            # find the h1 by id
            locator = page.locator('#main-heading')
            console.print(f'retrieved - {locator.text_content()}')

            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            save_soup(soup, Path('c://temp/z.htm'))
            return soup
        except Exception as e:
            console.print(f"error retrieving {e}", style='red')
            return None
        finally:
            browser.close()


def html_to_markdown(url=None, html_content=None, soup=None, href_base=None):
    """
    Convert HTML to Markdown, preserving document structure and formatting. Anchors are replaced
    with bold and repeated as links at the end
    
    Args:
        url (str, optional): URL of the webpage to extract text from.
        html_content (str, optional): HTML content as a string.
        
    Returns:
        str: Markdown-formatted text from the webpage with preserved structure.
    
    Raises:
        ValueError: If neither url nor html_content is provided.
        RuntimeError: If there's an issue fetching the URL.
    """
    # Initialize BeautifulSoup
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            soup = BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL: {str(e)}")
    elif html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
    # else:
    #     raise ValueError("Either url or html_content must be provided")
    
    # Remove non-content elements
    for tag in ['script', 'style', 'meta', 'noscript', 'head', 'iframe', 'svg', 'nav', 'footer', 'header']:
        for element in soup.find_all(tag):
            element.decompose()
    
    # Remove comments using proper Comment class
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    anchors = []
    
    def process_children(node):
        """Recursively process all child nodes, preserving their structure."""
        result = ""
        for child in node.children:
            if isinstance(child, NavigableString):
                if not child.strip():
                    continue
                result += child
            else:
                result += node_to_markdown(child)
        return result
    
    def node_to_markdown(node):
        """Convert an HTML node to markdown with proper formatting."""
        if isinstance(node, NavigableString):
            return str(node)
        
        tag_name = node.name.lower() if node.name else ""
        
        # Skip empty nodes and newlines
        if not tag_name or not node.contents:
            return ""
        
        # Process contents recursively to maintain formatting in child elements
        children_md = process_children(node)
        
        # Handle block elements
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(tag_name[1])
            return f"\n\n{'#' * level} {children_md.strip()}\n\n"
        
        elif tag_name == 'p':
            return f"\n\n{children_md.strip()}\n\n"
        
        elif tag_name in ['div', 'section', 'article']:
            return f"\n{children_md.strip()}\n"
        
        elif tag_name == 'blockquote':
            # Add > prefix to each line
            lines = children_md.strip().split("\n")
            blockquote_text = "\n".join(f"> {line}" for line in lines)
            return f"\n\n{blockquote_text}\n\n"
        
        elif tag_name == 'pre':
            # For code blocks with pre tags
            code_content = children_md.strip()
            code_tag = node.find('code')
            language = code_tag.get('class', [''])[0].replace('language-', '') if code_tag else ''
            return f"\n\n```{language}\n{code_content}\n```\n\n"
        
        elif tag_name == 'br':
            return "\n"
        
        elif tag_name == 'hr':
            return "\n\n---\n\n"
        
        # Handle lists with proper nesting
        elif tag_name == 'ul':
            list_items = []
            for li in node.find_all('li', recursive=False):
                # Process li contents recursively
                item_content = process_children(li).strip()
                # Handle nested lists - indent by 2 spaces
                item_content = item_content.replace('\n', '\n  ')
                list_items.append(f"* {item_content}")
            return "\n" + "\n".join(list_items) + "\n\n"
        
        elif tag_name == 'ol':
            list_items = []
            for i, li in enumerate(node.find_all('li', recursive=False), 1):
                # Process li contents recursively
                item_content = process_children(li).strip()
                # Handle nested lists - indent by 3 spaces (for number alignment)
                item_content = item_content.replace('\n', '\n   ')
                list_items.append(f"{i}. {item_content}")
            return "\n" + "\n".join(list_items) + "\n\n"
        
        # Handle tables
        elif tag_name == 'table':
            # Extract headers
            headers = []
            header_row = node.find('tr')
            if header_row and header_row.find('th'):
                headers = [process_children(th).strip() for th in header_row.find_all('th')]
            
            # Extract rows
            rows = []
            for tr in node.find_all('tr'):
                # Skip rows that are header rows
                if tr.find('th'):
                    continue
                
                # Process cells
                cells = [process_children(td).strip().replace('\n', ' ') for td in tr.find_all('td')]
                if cells:
                    rows.append(cells)
            
            # Build markdown table
            if headers or rows:
                table_md = []
                
                # Handle case with no explicit headers but has rows
                if not headers and rows:
                    headers = [''] * len(rows[0])
                
                # Add header row
                if headers:
                    table_md.append(f"| {' | '.join(headers)} |")
                    table_md.append(f"| {' | '.join(['---'] * len(headers))} |")
                
                # Add data rows
                for row in rows:
                    # Ensure row has same number of cells as headers
                    while len(row) < len(headers):
                        row.append('')
                    table_md.append(f"| {' | '.join(row)} |")
                
                return f"\n\n{chr(10).join(table_md)}\n\n"
            return ""
        
        # Handle inline elements
        elif tag_name in ['b', 'strong']:
            return f"**{children_md.strip()}**"
        
        elif tag_name in ['i', 'em']:
            return f"*{children_md.strip()}*"
        
        elif tag_name == 'code' and node.parent.name != 'pre':
            # Inline code (not inside a pre block)
            return f"`{children_md}`"
        
        elif tag_name == 'a':
            href = node.get('href', '')
            if href:
                text = re.sub('\n', '', children_md.strip())
                if 'archive.is' in href:
                    href = href[href.find('https', 5):]
                anchors.append(anchor_tag(text, href))
                return f"*{text}*"
            return children_md
        
        elif tag_name == 'img':
            alt = node.get('alt', 'image')
            src = node.get('src', '')
            if src:
                return f"![{alt}]({src})"
            return alt
        
        # Default case - just return the processed contents
        return children_md
    
    # Process the body recursively to maintain order
    # body = soup.body or soup
    # mn = body.find('main', id='main-content')
    markdown_text = process_children(soup)
    xs = [f"{str(i+1)}. {a.text} {a.href if a.href.startswith("http") else href_base + a.href}" for i,a in enumerate(anchors)]
    markdown_text += "\n### links\n\n" + "\n\n".join(xs)
    
    
    # Clean up whitespace
    #markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    #markdown_text = re.sub(r'^\s+|\s+$', '', markdown_text, flags=re.MULTILINE)
    
    return markdown_text.strip()

def test_soup(soup):
    # directly walk tree NB because soup treats any name that is not a method as a selector
    # soup.tag => soup.find['tag'] and soup['attr'] => soup.get('attr')
    e = soup.body.h1.get_text()
    # find find_all to search
    e = soup.find(id="list") # find by id
    xs = [e.name for e in soup.find_all(class_="list-item")]
    pprint(xs)
    xs = list(soup.find(id="list").stripped_strings) # retrieve all text under id=list
    pprint(xs)
    xs = [e['href'] for e in soup.find_all("a")]
    pprint(xs)
    if e := soup.find("div", attrs={"data-id": "animal"}):
        pprint(e)

    # select to find by css selectors which is slower but more powerful
    # soup.select -> list
    # soup.css.iselect -> generator
    xs = [e.name for e in  soup.select("li.list-item")]
    pprint(xs)
    xs = list(soup.select("li > a")) # directly inside
    pprint(xs)
    xs = list(soup.select("ul a")) # contained anywhere
    pprint(xs)
    xs = [e.name for e in soup.css.iselect("#list li")] # contained anywhere
    pprint(xs)


# Example usage
def main2():
    # Example with HTML string including complex structures
    html_content = """
    <html>
        <body>
            <h1>Main Heading with <em>emphasis</em></h1>
            <p>This is a paragraph with <b>bold text</b> and <a href="https://example.com">a link</a>.</p>
            <h2>Subheading</h2>
            <div id="list">
                <ul>
                    <li class="list-item">List item 1</li>
                    <li class="list-item">List item <strong>2</strong>
                        <ul>
                            <li>Nested list item with <a href="https://example.org">link</a></li>
                            <li>Another nested item</li>
                        </ul>
                    </li>
                </ul>
            </div>
            <p>Another paragraph with <i>italic text</i> and <code>inline code</code>.</p>
            <div>the quick brown</div>
            <div data-id="animal">fox jumped</div>
            <pre><code class="language-python">
def hello_world():
    print("Hello, world!")
            </code></pre>
            <h3>Another Subheading</h3>
            <table>
                <tr>
                    <th>Header 1</th>
                    <th>Header 2</th>
                </tr>
                <tr>
                    <td>Data 1 with <b>bold</b></td>
                    <td>Data 2</td>
                </tr>
                <tr>
                    <td>Data 3</td>
                    <td>Data 4 with <a href="#">link</a></td>
                </tr>
            </table>
            <blockquote>
                This is a blockquote with <a href="https://example.org">a link</a>.
                <br>
                And a second line.
            </blockquote>
            <img src="example.jpg" alt="Example image">
        </body>
    </html>
    """
    test_soup(BeautifulSoup(html_content, 'html.parser'))

    markdown_output = html_to_markdown(html_content=html_content)
    markdown = Markdown(markdown_output, style="cyan", code_theme="monokai")
    # console.print(markdown, width=80)
    
    # Example with URL
    try:
        # url = "https://www.bbc.co.uk/news/articles/c3w14gw3wwlo"
        # title, content = get_bbc_html(url)
        # content = html_to_markdown(html_content= content, title= title)
        # markdown = Markdown(content, style="cyan", code_theme="monokai")
        # console.print(markdown, width=80)
        title, content = get_wsj_html('https://www.wsj.com/politics/elections/democrat-party-strategy-progressive-moderates-13e8df10')
        # 'https://www.wsj.com/world/china/china-trump-trade-war-worries-0c2fa146')
        #'https://www.wsj.com/world/middle-east/trump-slams-door-on-arab-plan-for-gaza-with-resorts-8cb744a1')
        p = Path('c://temp/z.htm')
        # soup = BeautifulSoup(content, 'html.parser')
        # breakpoint()
        # section = soup.find('main')
        # save_soup(soup, p)
        soup = load_soup(p)

        # remove some divs before extracting text
        divs = soup.find_all('div')
        if divs:
            xs = [d for d in divs if "background-position:/*x=*/0% /*y=*/0%;" in d.get('style')]
            console.print(f'removing divs with style {len(xs)}', style="red")
            for d in xs:
                d.decompose()

        md = html_to_markdown(soup = soup.find('section'), title= soup.find('h1').get_text().strip(), subtitle= soup.find('h2').get_text().strip())
        with Path('c://temp/z.md').open('w', encoding='utf-8') as f:
            f.write(md)
        markdown = Markdown(md, style="cyan", code_theme="monokai")
        console.print(markdown, width=80)

    except Exception as e:
        print(f"Error: {e}")


def save_soup(soup: BeautifulSoup, fname: Path):
        with fname.open('w', encoding='utf-8') as f:
            f.write(soup.prettify())

def load_soup(fname: Path) -> BeautifulSoup:
    with fname.open('r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), 'html.parser')

def print_markdown(fname: Path):
    # it seems rich-cli is not using utf-8 decoding and fails to parse the right double quote character \xe2\x80\x9d
    # there is an unreleased fix with it in
    # well below market expectations “This reflects the payback from earlier exports front-loading and Trump’s faster and broad-based tariff hikes,” economists at Barclays said in a note.
    with fname.open('r', encoding='utf-8') as f:
        console.print(Markdown(f.read(), style="cyan"), width=80)


if __name__ == "__main__":
    main2()
    # soup = get_bbc_html('https://www.bbc.co.uk/news/articles/cd7eyz3yn5do')
    #  #find time tags with a datetime attribute.
    # if headline := soup.find(id="main-heading"):
    #     console.print(f"title: {headline.text}")
    # if time_tag:= soup.find('time', attrs={'datetime': True}):
    #     datetime_str = time_tag['datetime']
    #     d = datetime.fromisoformat(datetime_str)
    #     console.print(f"published: {d.strftime("%A %Y-%m-%d")}")
    # md = html_to_markdown(soup = soup.find(id='main-content'), href_base = "https://bbc.co.uk")
    # console.print(Markdown(md, style="cyan"), width=80)
    scrape_stock_data(["SPY:UN", "XOM:UN", "TLT:UN", "JNK:UN"])

    #print_markdown(Path.home() / 'Documents' / 'chats' / 'china_struggles_to_shake_off.md')