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
MODEL_SPARQL = os.getenv("MODEL_SPARQL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_KEY)


TIMESTAMP     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_DIR      = Path("output")
LOG_DIR       = BASE_DIR / "logs"
OUTPUT_DIR    = BASE_DIR / f"/dataset"
SUBGRAPH_DIR  = OUTPUT_DIR / "subgraph/full"
SUBGRAPH_DIR_PROOF  = OUTPUT_DIR / "subgraph/proof"
SEED_CSV      = Path("data/processed/vital.csv")
DATASET_FILE  = OUTPUT_DIR / "raw-data.json"

# Constants
DEPTH         = int(os.getenv("DEPTH", 10))
BRANCH_FACTOR = int(os.getenv("BRANCH_FACTOR", 15))


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_HEADERS  = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "QuestionOnGraph/1.0 (liangz.cs@outlook.com)"
}

WIKIDATA_API   = "https://www.wikidata.org/w/api.php"

# ID with question
QUESTION_PROMPT = """
You are given a small set of RDF triples from **Wikidata**.

Format: Each triple is a 3-item array:
  [ "<English label> (<Q-ID>)", "<predicate label> (<P-ID>)", "<English label> (<Q-ID>)" ]

Triples:
    {triples}

Your task is to decide whether this subgraph is **sufficient to support a challenging and non-trivial question** for a knowledge graph question answering (KGQA) benchmark.
Rules:
1. **Reasoning Depth**:
   - Prioritize questions that require **at least 2-hop deep** logical reasoning.
   - Selected candidate entities should be specific instead of big topic, don't include subclass, prefer the specific award instead of 
   - Use **factual constraints** (e.g., date, country, affiliation) only as needed to **disambiguate the correct answer** or narrow the search space meaningfully.
   - Avoid over-specifying. Use **just enough** constraints for a well-formed, specific answer.

2. **Entity Selection and Expansion**:
   - Each candidate must be a concrete, instance-level entity, such as Q7186 (Marie Curie), not an abstract or generic type like Q5 (human) or Q11424 (film).
   - Avoid generic classes such as “scientist”, “award”, “event”, “type of”, or any form of “subclass of” or “instance of”.
   - Prioritize entities that suggest meaningful expansion paths supporting deeper reasoning—e.g., related locations, affiliations, awarded recognitions, time periods, or collaborations.

3. Difficulty:
   - Incorporate **inverse relations**, **numeric or date constraints**, **comparative logic**, or **set membership conditions**.
   - Encourage use of **factual filters** (e.g., birth years, countries, time periods) to **narrow the candidate answer set**.
   - Design the question so that it is **difficult to solve using only general knowledge**.
   - The proof should contain all the knowledge to answer the question.

4. Naturalness: 
   - Use natural, fluent phrasing that a curious human might actually use.
   - Keep questions self-contained, concise, and free of redundancy, especially for obvious or common facts.
   - Avoid awkward structure, artificial phrasing, or referring to data/triples/lists.
   - Prohibited phrases include:
     - "listed here"
     - "from the given data"
     - "among these entities"
     - "according to the triples"
     - "which of the following"
   - Phrase questions as if the user has no access to any dataset.
5. Clarity: 
   - The question must be unambiguous and specifically identify the intended answer(s).
   - Avoid vague terms and ensure the expected answer is clearly implied by the question logic.

If NOT sufficient, return exactly with select candidates(3-5) in curret triples for next explore:
{{ 
    "sufficient": false,
    "candidate": [<QID>, ..., <QID>]
}}

If sufficient: return  
{{
  "sufficient": true,
  "question": "<natural-language question>",
  "answer": ["<answer-label (QID)>", "..."],
  "proof": [
    ["<label (QID)>", "<predicate label (PID)>", "<label (QID)>"],
    ...
  ]
}}

Output **strict JSON only**—no commentary.

Example 1:
Triples:
[
  ["Johann Martin Schleyer (Q12712)", "nominated for (P1411)", "Nobel Peace Prize (Q35637)"],
  ["International Volap\u00fck Academy (Q3358168)", "founded by (P112)", "Johann Martin Schleyer (Q12712)"],
  ["Johann Martin Schleyer (Q12712)","place of birth (P19)","Oberlauda Q885402)"]
]
Output:
{{
  "sufficient": true,
  "question": "Who among the nominees for the Nobel Peace Prize was also the founder of International Volap\u00fck Academy?",
  "answer": ["Johann Martin Schleyer (Q12712)"],
  "proof": [
  ["Johann Martin Schleyer (Q12712)", "nominated for (P1411)", "Nobel Peace Prize (Q35637)"],
  ["International Volap\u00fck Academy (Q3358168)", "founded by (P112)", "Johann Martin Schleyer (Q12712)"]
]
}}

Example 2:
Triples:
[
    ['Karakalpakstan (Q484245)', 'capital (P36)', 'Nukus (Q489898)'],
    ['Karakalpakstan (Q484245)', 'shares border with (P47)', 'Mangystau Region (Q238931)'], 
    ['Karakalpakstan (Q484245)', 'shares border with (P47)', 'Daşoguz Region (Q487393)'], 
    ['Karakalpakstan (Q484245)', 'official language (P37)', 'Karakalpak (Q33541)'], 
    ['Karakalpakstan (Q484245)', 'official language (P37)', 'Uzbek (Q9264)'], 
    ['Karakalpakstan (Q484245)', 'country (P17)', 'Uzbekistan (Q265)']
]
Output:
{{
  "sufficient": false,
  "candidate": [Q487393, Q33541, Q9264]
}}

Example 3:
Triples:
[
    ['Astronomy and Astrophysics (Q752075)', 'publisher (P123)', 'EDP Sciences (Q114404)'], 
    ['Astronomy and Astrophysics (Q752075)', 'editor (P98)', 'Thierry Forveille (Q46260676)'],
    ['Astronomy and Astrophysics (Q752075)', 'field of work (P101)', 'theoretical astronomy (Q3635271)'],
    ['Astronomy and Astrophysics (Q752075)', 'country (P17)', 'France (Q142)']
    ['Zeitschrift für Astrophysik (Q3575110)', 'followed by (P156)', 'Astronomy and Astrophysics (Q752075)'],
]
Output:
{{
  "sufficient": true,
  "question": "What astronomical journal, published by EDP Sciences and edited by Thierry Forveille, succeeded Zeitschrift für Astrophysik as its immediate follower?",
  "answer": ["Astronomy and Astrophysics (Q752075)"],
  "proof": [['Astronomy and Astrophysics (Q752075)', 'publisher (P123)', 'EDP Sciences (Q114404)'], ['Astronomy and Astrophysics (Q752075)', 'editor (P98)', 'Thierry Forveille (Q46260676)'], ['Zeitschrift für Astrophysik (Q3575110)', 'followed by (P156)', 'Astronomy and Astrophysics (Q752075)']]
}}
"""

SPARQL_PROMPT = """
You are given:
- Triples (Entity and predicate strings are "Label (ID)"): {triples}
- Question: "{question}"
- Answer list: {answer}

Write the **simplest valid SPARQL** for Wikidata that yields exactly the answers.

Rules for simplicity
1. Use only the needed patterns—no OPTIONAL blocks or FILTERs unless essential.
2. Extract Q-IDs (Qxxxx) and P-IDs (Pxxxx) from the triple strings and use:
     wd:Qxxxx   for entities
     wdt:Pxxxx  for direct predicates
3. End with one SERVICE wikibase:label clause for English labels.
4. Return a **single JSON object** using this format, and no extra explanation:
{{
"sparql": "<SPARQL query here>"
}}

Example 1:
Triples:
[
  ["Johann Martin Schleyer (Q12712)", "nominated for (P1411)", "Nobel Peace Prize (Q35637)"],
  ["International Volap\u00fck Academy (Q3358168)", "founded by (P112)", "Johann Martin Schleyer (Q12712)"]
]
Question:
"Who among the nominees for the Nobel Peace Prize was also the founder of International Volap\u00fck Academy?"
Answer:
["Johann Martin Schleyer (Q12712)"]

Output:
{{
    "sparql": "SELECT ?personLabel WHERE {{ ?person wdt:P1411 wd:Q35637 . wd:Q3358168 wdt:P112 ?person . SERVICE wikibase:label {{ bd:serviceParam wikibase:language \\"en\\" . }} }}"
}}
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
    # for d in (LOG_DIR, OUTPUT_DIR, SUBGRAPH_DIR, SUBGRAPH_DIR_PROOF):
    for d in (LOG_DIR, OUTPUT_DIR):
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
        return f"{_label_cache[key]} ({token})"

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
            # filter null and noise predicates
            if not p or not o or p in NOISE_PROPS: 
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
            # filter null and noise predicates
            if not p or not s or p in NOISE_PROPS:
                continue

            triples.append((s, p, q))
            if s.startswith("Q"):
                next_frontier.add(s)
    time.sleep(0.1)
    return triples, next_frontier

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

def gen_sparql(proof_triples: List[List[str]],
               question: str,
               answers: List[str]) -> str:
    user_content = SPARQL_PROMPT.format(
        triples=json.dumps(proof_triples, separators=(", ", ": ")),
        question=question,
        answer=json.dumps(answers, separators=(", ", ": "))
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert SPARQL engineer for Wikidata-style graphs. "},
            {"role": "user",   "content": user_content}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    text = resp.choices[0].message.content.strip()
    # try:
    #     return json.loads(text)["sparql"]
    # except (KeyError, json.JSONDecodeError) as e:
    #     logging.error(f"SPARQL JSON parse failed: {e}\n{text}")
    #     raise
    if text.startswith("```json") or text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return parsed
    except Exception as e:
        logging.error(f"[!] Failed to parse revised SPARQL: {e}\nCleaned text:\n{text}")
        raise 

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
            # print("before",new_triples)
            new_triples = [(get_label(s), get_label(p), get_label(o)) for s, p, o in new_triples]
            visited.extend(new_triples)
            visited = list({t for t in visited})  # remove duplicates
            # filter the null parts
            labeled = [[get_label(s), get_label(p), get_label(o)] for s, p, o in visited if o and s and p]
            
            logging.info(f"Processed triples: \n{labeled}")
            # step 1: generate q-a with proof triples
            try:
                result = gen_qa(labeled)
                logging.info(f"LLM results: \n{result}")
            except Exception as e:
                logging.error(f"LLM call failed at hop {depth} for {seed}: {e}")
                break

            logging.info(f"Seed {seed} hop {depth} sufficient? {result.get('sufficient')}")

            if result.get("sufficient"):
                # step 2: use the q-a with proof generate the SPARQL
                sparql_result = gen_sparql(
                    proof_triples=result["proof"],
                    question=result["question"],
                    answers=result["answer"]
                )
                # # subgraph
                # subgraph_file = SUBGRAPH_DIR / f"{id_counter}_{seed}.json"
                # with open(subgraph_file, 'w', encoding='utf-8') as sf:
                #     json.dump(labeled, sf, indent=2)
                # subgraph_file1 = SUBGRAPH_DIR_PROOF / f"{id_counter}_{seed}.json"
                # with open(subgraph_file1, 'w', encoding='utf-8') as sf:
                #     json.dump(result["proof"], sf, indent=2)

                # dataset
                entry = {
                    "id": id_counter,
                    "seed": seed,
                    "question": result.get("question"),
                    "answer": result.get("answer"),
                    "sparql": sparql_result.get("sparql"),
                    "proof": result.get("proof")
                }
                with open(DATASET_FILE, 'a', encoding='utf-8') as df:
                    df.write(json.dumps(entry) + '\n')
                
                id_counter += 1
                break

            frontier = next_frontier
            time.sleep(0.1)

if __name__ == "__main__":
    main()
