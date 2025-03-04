#!/usr/bin/python3
from dataclasses import dataclass
from bs4 import BeautifulSoup
from datetime import date, datetime
import requests
import pandas as pd
import re
import itertools
from pathlib import Path
from rich.pretty import pprint


# extract table of stocks from motley fool disclosure page
# using requests http library https://requests.readthedocs.io/en/latest/
# and BeautifulSoup html parsing library https://beautiful-soup-4.readthedocs.io/en/latest/index.html

# parse string 'Apple - NASDAQ:AAPL\n  AAPL'
# 1 AAPL
# 2 NASDAQ:AAPL
# 3 AAPL


def parse_fool_disclosure(cols: list[str]) -> dict[str, str | int]:
    # ['1', 'Apple - NASDAQ:AAPL\nAAPL', 'AAPL', '313']
    d = {}
    d["rank"] = int(cols[0])
    parts = re.split(r"[-:\n]", cols[1])
    if len(parts) > 2:
        d["company"] = parts[0].strip()
        d["exchange"] = parts[1].strip()
    else:
        d["company"] = cols[1]
        d["exchange"] = "NA"
    d["ticker"] = cols[2]
    d["held"] = int(cols[3])
    return d


def extract_fool_stocks(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find("table")
    table_body = table.find("tbody")

    data = []
    rows = table_body.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        xs = [e.text.strip() for e in cols]
        data.append(parse_fool_disclosure(xs))
    return pd.DataFrame(data).astype({"company": "string", "exchange": "string", "ticker": "string"})


def download_fool_disclosure(url: str, fn: Path):
    print("loading " + url)
    r = requests.get(url)
    if not r.ok:
        print("http status " + r.status_code)
        raise Exception(f"failed to load page {url}")
    soup = BeautifulSoup(r.text, "html.parser")
    df = extract_fool_stocks(soup)
    print(df)
    df.to_csv(fn, index=False)
    print("saved " + str(fn))


def main() -> None:
    name = f"Motley Fool Disclosure {date.today().isoformat()}.csv"
    fn = Path.home() / "Downloads" / name
    return download_fool_disclosure("https://www.fool.com/legal/fool-disclosure/", fn)


def print_changes() -> None:
    fn = Path.home() / "Downloads"
    paths = [s for s in fn.glob("Motley Fool Disclosure*.csv")]
    # load each csv into a df and index by date
    # df has cols rank and ticker and makes the ticker the index
    ds = {date.fromisoformat(str(f)[-14:-4]): pd.read_csv(f, usecols=["rank", "ticker"]) for f in paths}
    for k, v in ds.items():
        v.set_index("ticker", inplace=True)
        v.rename(columns={"rank": k}, inplace=True)
    # concat the df's into a single df using the date
    df = pd.concat(ds.values(), axis=1, keys=ds)
    # add an action col indicating the change between the first and last rank
    a = {e: "drop" for e in df[df.iloc[:, -1].isna()].index}
    b = {e: "new" for e in df[df.iloc[:, 0].isna()].index}
    c = {e: "up" for e in df[df.iloc[:, 0] > df.iloc[:, -1]].index}
    d = {e: "down" for e in df[df.iloc[:, 0] < df.iloc[:, -1]].index}
    m = {k: v for k, v in itertools.chain(a.items(), b.items(), c.items(), d.items())}
    df["action"] = pd.Series(m)
    df["action"] = df["action"].fillna(".")
    print(df[:20])


if __name__ == "__main__":
    # main()
    # print_changes()
    items = download_ft()
    pprint(items)
