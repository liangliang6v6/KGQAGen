import os
import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import re
import requests
from openai import OpenAI
import dotenv
from typing import Set, Tuple, List, Dict
from tqdm import tqdm


dotenv.load_dotenv()  
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
MODEL_SPARQL = os.getenv("MODEL_SPARQL", "gpt-4o")

client = OpenAI(api_key=OPENAI_KEY)


TIMESTAMP     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_DIR      = Path("ours")
LOG_DIR       = BASE_DIR / "logs"
OUTPUT_DIR    = BASE_DIR / f"{TIMESTAMP}/dataset"
SUBGRAPH_DIR  = OUTPUT_DIR / "subgraph"
# SUBGRAPH_DIR_LABEL  = OUTPUT_DIR / "subgraph/label"
SEED_CSV      = Path("data/processed/test.csv")
DATASET_FILE  = OUTPUT_DIR / "dataset.json"

# Constants
DEPTH         = int(os.getenv("DEPTH", 3))
BRANCH_FACTOR = int(os.getenv("BRANCH_FACTOR", 8))


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_HEADERS  = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "QuestionOnGraph/1.0 (youremail@example.com)"
}

WIKIDATA_API   = "https://www.wikidata.org/w/api.php"

# ID with question
QUESTION_PROMPT = """
Given a set of RDF triples from Wikidata, decide if they are sufficient to form a clear, multi-hop question suitable for a KGQA dataset.

Triples:
    {triples}

Rules:
1. Difficulty: Prefer questions that:
   - Combine heterogeneous and robust predicates (e.g., 'award' with 'birthPlace', which are factual and unchanging).
   - Require inverse traversal, set comparison, numeric or temporal constraints.
   - Use date, location, or other factual details to meaningfully narrow down possible answers.
   - Are non-trivial for a simple LLM to answer without access to the triples.
2. Naturalness: Write questions a curious human might actually ask. *Avoid forced, awkward, or self-evident formulations.*
3. Clarity: The question must be unambiguous and specifically identify the intended answer(s).
4. Correctness: The SPARQL query must be directly executable, standard, and valid.
5. Output strictly valid JSON only. No explanations or extra text.

- If the triples are sufficient, output:
  {{
    "sufficient": true,
    "question": "<the generated question>",
    "answer": ["<answer1>", "..."],
    "sparql": "<correct SPARQL query>"
  }}
- If the triples are insufficient, output:
  {{ "sufficient": false }}

Example 1:
Triples:
[
  ["Johann Martin Schleyer (Q12712)", "nominated for (P1411)", "Nobel Peace Prize (Q35637)"],
  ["International Volap\u00fck Academy (Q3358168)", "founded by (P112)", "Johann Martin Schleyer (Q12712)"]
]

Output:
{{
  "sufficient": true,
  "question": "Who among the nominees for the Nobel Peace Prize was also the founder of International Volap\u00fck Academy?",
  "answer": ["Johann Martin Schleyer (Q12712)"],
  "sparql": "SELECT ?personLabel WHERE {{ ?person wdt:P1411 wd:Q35637 . wd:Q3358168 wdt:P112 ?person . SERVICE wikibase:label {{ bd:serviceParam wikibase:language \\"en\\" . }} }}"
}}
"""

SPARQL_PROMPT = """
You will be given a SPARQL query intended to search Wikidata. Revise and correct the query so that:
- It is syntactically valid and directly runnable on the Wikidata Query Service API.
- It uses proper Wikidata property and entity identifiers (wd:, wdt:, etc.).
- It produces human-readable output.
- Do not change the query logic or intent—only improve correctness and WDQS compatibility.
- Only output the corrected SPARQL query text, nothing else.

SPARQL Input:
{sparql}
Return only the JSON object with key "sparql" and the corrected query string as its value.
JSON only.
"""

# fliter nosiy realtions
filter_list = ["video", "image","audio", "inspired by", "Wikidata property example", ]
NOISE_PROPS = {
    "P18", "P94", "P41", "P154", "P109", "P373",   # images
    "P79", "P10", "P1651", "P11731", "P6456", # video
    "P51","P989","P443", "P7501",   # audio
    "P31",    # instance of
    "P941",    # inspired by
    "P737",    # influenced by
}

def setup_dirs_and_logging():
    for d in (LOG_DIR, OUTPUT_DIR, SUBGRAPH_DIR):
        d.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_FILE.touch(exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / f"run_{TIMESTAMP}.log",
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_seeds(single: bool) -> List[str]:
    with open(SEED_CSV, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        qids = [row["qid"].strip() for row in reader if row.get("qid")]
    return qids[:1] if single and qids else qids


_label_cache: Dict[str, str] = {}

def extract_id(token: str) -> str:
    m = re.match(r'^"(.*)"(?:\^\^<?[^>]+>?)?(?:@[a-z]+)?$', token)
    if m:
        return m.group(1)

    if token.startswith("<") and token.endswith(">"):
        uri = token[1:-1].rstrip("/")
        return uri.rsplit("/", 1)[-1]

    if re.fullmatch(r"[QP]\d+", token):
        return token
    return token

def get_label(token: str) -> str:
    key = extract_id(token)
    if not key:
        return token
    # Non-Wikidata IDs or literals → just echo back
    if not re.fullmatch(r"[QP]\d+", key):
        return key
    # Wikidata entity → look up label (with cache)
    if key in _label_cache:
        return _label_cache[key]

    params = {
        "action":    "wbgetentities",
        "ids":       key,
        "format":    "json",
        "props":     "labels",
        "languages": "en",
    }
    r = requests.get(WIKIDATA_API, params=params, headers=SPARQL_HEADERS)
    r.raise_for_status()
    ent = r.json().get("entities", {}).get(key, {})

    lbl = ent.get("labels", {}).get("en", {}).get("value") or key
    _label_cache[key] = lbl
    return f"{lbl} ({token})"

# def get_entity(wd_id: str, lang: str = "en") -> dict:
#     if wd_id in _entity_cache:
#         return _entity_cache[wd_id]

#     params = {
#         "action":    "wbgetentities",
#         "ids":       wd_id,
#         "format":    "json",
#         "props":     "labels|descriptions|aliases|claims",
#         "languages": lang
#     }
#     r = requests.get(WIKIDATA_API, params=params, headers=API_HEADERS)
#     r.raise_for_status()
#     ent = r.json()["entities"].get(wd_id, {})

#     # Text fields
#     label       = ent.get("labels", {}).get(lang, {}).get("value")
#     description = ent.get("descriptions", {}).get(lang, {}).get("value")
#     aliases     = [a["value"] for a in ent.get("aliases", {}).get(lang, [])]

#     # Normalize claims
#     claims_out = {}
#     for prop, snak_list in ent.get("claims", {}).items():
#         vals = []
#         for snak in snak_list:
#             if snak.get("snaktype") != "value":
#                 continue
#             dv = snak["mainsnak"].get("datavalue", {})
#             dt = snak["mainsnak"].get("datatype")
#             v  = dv.get("value")

#             if dt == "time":
#                 # "+YYYY-MM-DDThh:mm:ssZ" → "YYYY-MM-DD"
#                 iso = v["time"].lstrip("+").split("T")[0]
#                 vals.append(iso)
#             elif dt == "quantity":
#                 vals.append(float(v["amount"]))
#             elif dt == "monolingualtext":
#                 vals.append(v["text"])
#             elif dt == "string":
#                 vals.append(v)
#             elif dt == "wikibase-entityid":
#                 vals.append(v["id"])
#             elif dt == "globecoordinate":
#                 vals.append((v["latitude"], v["longitude"]))
#             else:
#                 # fallback to raw JSON
#                 vals.append(v)
#         if vals:
#             claims_out[prop] = vals

#     result = {
#         "id":          wd_id,
#         "label":       label,
#         "description": description,
#         "aliases":     aliases,
#         "claims":      claims_out
#     }
#     _entity_cache[wd_id] = result
#     return result

def run_sparql(query: str) -> List[dict]:
    r = requests.get(SPARQL_ENDPOINT, params={"query": query}, headers=SPARQL_HEADERS)
    r.raise_for_status()
    time.sleep(0.1)
    return r.json()["results"]["bindings"]

def expand_from_neighbors(
    neighbor_qids: Set[str],
    limit: int = BRANCH_FACTOR,
    max_retries: int = 3
) -> Tuple[List[Tuple[str, str, str]], Set[str]]:

    triples: List[Tuple[str, str, str]] = []
    next_frontier: Set[str] = set()

    def safe_sparql(q: str):
        for attempt in range(1, max_retries + 1):
            try:
                return run_sparql(q)
            except Exception as e:
                backoff = 2 ** attempt
                logging.warning(f"SPARQL failed (attempt {attempt}): {e}. retrying in {backoff}s")
                time.sleep(backoff)
        logging.error("SPARQL failed after retries, returning empty list")
        return []

    for q in neighbor_qids:
        q_out = f"""
        SELECT ?p ?o WHERE {{
          wd:{q} ?p ?o .
          FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT {limit}
        """
        for b in safe_sparql(q_out):
            p_uri = b["p"]["value"]
            p = p_uri.rsplit("/", 1)[-1]        # e.g. "P31"
            o_binding = b["o"]
            o = (o_binding["value"].rsplit("/", 1)[-1]
                 if o_binding["type"] == "uri"
                 else o_binding["value"])

            if p in NOISE_PROPS:
                continue

            triples.append((q, p, o))
            if o.startswith("Q"):
                next_frontier.add(o)

        q_in = f"""
        SELECT ?s ?p WHERE {{
          ?s ?p wd:{q} .
          FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT {limit}
        """
        for b in safe_sparql(q_in):
            p_uri = b["p"]["value"]
            p = p_uri.rsplit("/", 1)[-1]
            s_uri = b["s"]["value"]
            s = s_uri.rsplit("/", 1)[-1]

            if p in NOISE_PROPS:
                continue

            triples.append((s, p, q))
            if s.startswith("Q"):
                next_frontier.add(s)
    time.sleep(0.1)
    return triples, next_frontier

# def expand_from_neighbors(
#     neighbor_qids: Set[str],
#     limit: int = BRANCH_FACTOR
# ) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
#     # fliter nosiy realtions
#     filter_list = ["video", "image","audio", "inspired by", "Wikidata property example", ]

#     triples: List[Tuple[str, str, str]] = []
#     next_frontier: Set[str] = set()
#     for q in neighbor_qids:
#     # outgoing claims (safe)
#     q_out = f"""
#     SELECT ?p ?o WHERE {{
#       wd:{q} ?p ?o .
#       FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
#     }} LIMIT {limit}
#     """
#     for b in run_sparql(q_out):
#         p = b["p"]["value"].rsplit("/", 1)[-1]
#         o = b["o"]["value"].rsplit("/", 1)[-1] if b["o"]["type"] == "uri" else None
#         if get_label(p) not in filter_list:
#             triples.append((q, p, o))
#             if o and o.startswith("Q"):
#                 next_frontier.add(o)

#     # incoming claims (dangerous — wrap with try/except)
#     q_in = f"""
#     SELECT ?s ?p WHERE {{
#       ?s ?p wd:{q} .
#       FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
#     }} LIMIT {limit}
#     """
#     try:
#         for b in run_sparql(q_in):
#             s = b["s"]["value"].rsplit("/", 1)[-1]
#             p = b["p"]["value"].rsplit("/", 1)[-1]
#             if get_label(p) not in filter_list:
#                 triples.append((s, p, q))
#                 if s.startswith("Q"):
#                     next_frontier.add(s)
#     except Exception as e:
#         print(f"[Warning] Incoming expansion failed for {q}: {e}")
#         # just skip and continue


#     return triples, next_frontier


def gen_qa(triples: List[List[str]]) -> Dict:
    # logging.info(f"Input labelled triples to LLM:\n{json.dumps(triples, indent=2)}")
    user_content = QUESTION_PROMPT.format(
        triples=json.dumps(triples, separators=(", ", ": "))
    )
    # user_content = "Triples:\n" + json.dumps(triples, indent=2)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert at generating multi-hop Knowledge-Graph based questions."},
            {"role": "user",   "content": user_content}
        ]
    )
    text = resp.choices[0].message.content.strip()
    logging.info(f"Raw LLM response for triples:\n{text}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from LLM response: {e}\nResponse was:\n{text}")
        raise

def gen_sparql(raw_sparql: str):
    user_content = SPARQL_PROMPT.format(
        sparql=raw_sparql
    )
    resp = client.chat.completions.create(
        model=MODEL_SPARQL,
        messages=[
            {"role": "system", "content": "You are an expert SPARQL engineer for Wikidata-style graphs. "},
            {"role": "user",   "content": user_content}
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:sparql|json)?\\s*", "", raw)
    raw = re.sub(r"\\s*```$", "", raw).strip()

    logging.info("Raw SPARQL LLM response:\n%s", text)
    try:
        payload = json.loads(raw)
        sparql = payload["sparql"]
    except Exception as e:
        logging.error("Failed to parse revision JSON: %s\n%s", e, raw)
        raise

    return sparql

def main():
    setup_dirs_and_logging()
    seeds = load_seeds(single=False)

    id_counter = 1

    with open(DATASET_FILE, 'w', encoding='utf-8') as df:
        pass

    for seed in tqdm(seeds, desc="Processing seeds"):
        logging.info(f"Processing seed {seed}")
        frontier = {seed}
        visited: List[Tuple[str, str, str]] = []

        for depth in range(1, DEPTH + 1):
            new_triples, next_frontier = expand_from_neighbors(frontier)
            visited.extend(new_triples)
            visited = list({t for t in visited})  # remove duplicates

            # filter the null parts
            labeled = [[get_label(s), get_label(p), get_label(o)] for s, p, o in visited if o]
            
            logging.info(f"Processed triples: \n{labeled}")
            try:
                result = gen_qa(labeled)
                # use 2 stage
                # stage 1: question, with the triples involved 
                # stage 2: input the triples IDs then get back the SPARQL

                logging.info(f"LLM results: \n{result}")
            except Exception as e:
                logging.error(f"LLM call failed at hop {depth} for {seed}: {e}")
                break

            logging.info(f"Seed {seed} hop {depth} sufficient? {result.get('sufficient')}")

            if result.get("sufficient"):
                # subgraph
                subgraph_file = SUBGRAPH_DIR / f"{id_counter}_{seed}.json"
                with open(subgraph_file, 'w', encoding='utf-8') as sf:
                    json.dump(labeled, sf, indent=2)
                # subgraph_file1 = SUBGRAPH_DIR_LABEL / f"{id_counter}_{seed}.json"
                # with open(subgraph_file1, 'w', encoding='utf-8') as sf:
                #     json.dump(labeled, sf, indent=2)

                # dataset
                entry = {
                    "id": id_counter,
                    "seed": seed,
                    "question": result.get("question"),
                    "answer": result.get("answer"),
                    "sparql": revised_sparql
                }
                with open(DATASET_FILE, 'a', encoding='utf-8') as df:
                    df.write(json.dumps(entry) + '\n')
                

                id_counter += 1

                break

            frontier = next_frontier
            time.sleep(0.1)

if __name__ == "__main__":
    main()
