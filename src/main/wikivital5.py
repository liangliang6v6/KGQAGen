import csv
import requests
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Iterable

API = "https://en.wikipedia.org/w/api.php"
ROOT_CAT = "Category:Wikipedia level-5 vital articles"          # :contentReference[oaicite:0]{index=0}
OUT_CSV  = Path("data/raw/seed.csv")

def api_get(params: Dict) -> Dict:
    params.update({"format": "json"})
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_category_members(cmtitle: str,
                         cmtype: str = "subcat|page",
                         cmnamespace: str = "*") -> Iterable[Dict]:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": cmtitle,
        "cmlimit": "500",
        "cmtype": cmtype,
        "cmnamespace": cmnamespace,
    }
    while True:
        data = api_get(params)
        yield from data["query"]["categorymembers"]
        if "continue" not in data:
            break
        params.update(data["continue"])

def all_topic_categories(root: str = ROOT_CAT) -> List[str]:
    subcats = []
    for cm in get_category_members(root, cmtype="subcat"):
        title = cm["title"]
        if title.endswith("by topic"):        # dive one more level
            for sub in get_category_members(title, cmtype="subcat"):
                subcats.append(sub["title"])
        else:
            subcats.append(title)
    return subcats

def talk_to_article(talk_title: str) -> str:
    assert talk_title.startswith("Talk:")
    return talk_title[len("Talk:"):]

def harvest_article_titles(cat: str) -> List[str]:
    titles = []
    for cm in get_category_members(cat,
                                   cmtype="page",
                                   cmnamespace="1"):  # 1 = Talk ns
        titles.append(talk_to_article(cm["title"]))
    return titles

def titles_to_qids(titles: List[str]) -> Dict[str, str]:
    qids = {}
    for i in range(0, len(titles), 50):
        batch = "|".join(titles[i:i+50])
        data = api_get({
            "action": "query",
            "prop": "pageprops",
            "titles": batch,
            "ppprop": "wikibase_item",
        })
        for page in data["query"]["pages"].values():
            qid = page.get("pageprops", {}).get("wikibase_item")
            if qid:
                qids[page["title"]] = qid
    return qids

def titles_to_qids(titles: List[str]) -> Dict[str, str]:
    qids = {}
    for i in range(0, len(titles), 50):
        batch = "|".join(titles[i:i+50])
        data = api_get({
            "action": "query",
            "prop": "pageprops",
            "titles": batch,
            "ppprop": "wikibase_item",
        })
        for page in data["query"]["pages"].values():
            qid = page.get("pageprops", {}).get("wikibase_item")
            if qid:
                qids[page["title"]] = qid
    return qids

def build_seed_list() -> List[Dict]:
    seed_rows = []
    for topic_cat in tqdm(all_topic_categories(), desc="Topics"):
        titles = harvest_article_titles(topic_cat)
        qids = titles_to_qids(titles)
        for title, qid in qids.items():
            seed_rows.append({
                "title": title,
                "qid":   qid,
                "url":   f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            })
    return seed_rows


def main():
    rows = build_seed_list()
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["title", "qid", "url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
