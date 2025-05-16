import json
import re
import requests
import time
from typing import List
from openai import OpenAI
import os
import dotenv
import logging
from datetime import datetime
from tqdm import tqdm

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    filename=f"output/log/cleaning_{TIMESTAMP}.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()  
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI(api_key=OPENAI_KEY)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "QuestionOnGraph/1.0 (your email)"
}

SPARQL_CHECK = """
You are given a SPARQL query over Wikidata that returned no results.
Wrong SPARQL:
{sparql}

Your task is to revise this query to retrieve correct results from Wikidata.
Revision Guidelines:
1. Use only the needed patterns—no OPTIONAL blocks or FILTERs unless essential.
2. End with one SERVICE wikibase:label clause for English labels.
3. The structure should be executable on https://query.wikidata.org.
Return only a single JSON object in the format below—no extra output or markdown:
{{
"correct_sparql": "<REVISED SPARQL QUERY HERE>"
}}
"""

"""
Example 1:
Wrong SPARQL:
{{
"SELECT ?personLabel WHERE {{ ?person wdt:P27 wd:Q29520 ; wdt:P19 wd:Q8686 ; wdt:P22 ?father . wd:Q12118887 wdt:P22 ?person . wd:Q2995574 wdt:P57 ?person . wd:Q96957285 wdt:P527 ?person . SERVICE wikibase:label {{ bd:serviceParam wikibase:language \\\"en\\\" . }} }}"
}}
Output:
{{
    "correct_sparql": "SELECT DISTINCT ?personLabel WHERE {{ ?person wdt:P27 wd:Q29520 ; wdt:P19 wd:Q8686 . wd:Q12118887 wdt:P22 ?person . wd:Q2995574 wdt:P57 ?person . wd:Q96957285 wdt:P527 ?person . SERVICE wikibase:label {{ bd:serviceParam wikibase:language \\\"en\\\" . }} }}"
}}
"""

# normalize answer by removing "(Qxxxxxx)" and lowercasing
def normalize(answers):
    return sorted(set(re.sub(r"\s*\(Q\d+\)$", "", a.strip()) for a in answers))

def run_sparql(query: str) -> List[str]:
    try:
        r = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query},
            headers=SPARQL_HEADERS,
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", {}).get("bindings", []):
            label_value = next((v["value"] for k, v in item.items() if k.endswith("Label")), None)
            if label_value:
                results.append(label_value)
        time.sleep(0.5)
        return results
    except (requests.exceptions.HTTPError, requests.exceptions.RequestException, ValueError) as e:
        logging.warning(f"[SPARQL] Failed to run query: {e}")
        return []


def clean(entry, max_results=10, timeout=3):
    original_answers = entry.get("answer", [])
    sparql = entry["sparql"]
    flag = 0
    def safe_run_sparql(query):
        try:
            results = run_sparql(query)
            if len(results) > max_results:
                logger.warning(f"[x] ID {entry.get('id')} — Too many SPARQL results ({len(results)}), skipping.")
                return None
            return results
        except requests.exceptions.Timeout:
            logger.warning(f"[x] ID {entry.get('id')} — SPARQL timed out after {timeout}s, skipping.")
            return None
        except Exception as e:
            logger.error(f"[x] ID {entry.get('id')} — SPARQL error: {e}")
            return None

    while flag < 2:
        results = safe_run_sparql(sparql)
        if results:
            break
        flag += 1
        logger.info(f"[x] ID {entry.get('id')} — Attempt {flag}: empty SPARQL result. Trying to revise...")
        sparql = check_sparql(sparql)
    
    if not results:
        logger.info(f"[!] ID {entry.get('id')} — No valid SPARQL after {flag} attempts. Skipping.")
        return None

    norm_answers = normalize(original_answers)
    norm_sparql = normalize(results)
    if set(norm_answers).issubset(set(norm_sparql)) and len(set(norm_answers))<10: # filter the super long answers
        merged = sorted(set(norm_answers) | set(norm_sparql))
        if set(merged) != set(norm_answers):
            logger.info(f"[+] ID {entry.get('id')} — Answer updated with SPARQL result")
        entry["answer"] = merged
        entry["sparql"] = sparql
        return entry
    else:
        logger.info(f"[!] ID {entry.get('id')} — Answer not in SPARQL. Skipping. Answer: {set(norm_answers)}, SPARQL: {set(norm_sparql)}, check: {set(norm_answers).issubset(set(norm_sparql))}")
        return None
    
def check_sparql(sparql:str):
    user_content = SPARQL_CHECK.format(
        sparql=sparql,
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert SPARQL engineer for Wikidata-style graphs. "},
            {"role": "user",   "content": user_content}
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    # print("raw output", text)

    if text.startswith("```json") or text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return parsed["correct_sparql"]
    except Exception as e:
        raise ValueError(f"[!] Failed to parse revised SPARQL: {e}\nCleaned text:\n{text}")


def get_last_id(output_path):
    if not os.path.exists(output_path):
        return 0  # Start fresh if file doesn't exist

    with open(output_path, "r", encoding="utf-8") as fout:
        lines = fout.readlines()
        for line in reversed(lines):
            if line.strip():
                try:
                    entry = json.loads(line)
                    return entry.get("id", 0) + 1
                except json.JSONDecodeError:
                    continue
    return 0

def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    
    count_id = get_last_id(output_path)
    print("start from ID:", count_id)
    
    with open(output_path, "a", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Processing lines"):
            if not line.strip():
                continue
            entry = json.loads(line)
            cleaned = clean(entry)
            if cleaned:
                cleaned["id"] = count_id
                fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                count_id += 1

if __name__ == "__main__":
    input_file = "data/processed/gen_data.json"
    output_file = "data/processed/data.jsonl"
    main(input_file, output_file)
