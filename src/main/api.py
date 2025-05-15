import requests
from typing import List, Tuple, Set, Dict

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "NeighborHopExpander/1.0 (you@example.com)"
}

_label_cache: Dict[str,str] = {}

def get_label(wd_id: str) -> str:
    if wd_id in _label_cache:
        return _label_cache[wd_id]
    params = {
        "action": "wbgetentities",
        "ids": wd_id,
        "format": "json",
        "props": "labels",
        "languages": "en"
    }
    r = requests.get(WIKIDATA_API, params=params, headers=HEADERS)
    r.raise_for_status()
    ent = r.json().get("entities", {}).get(wd_id, {})
    lbl = ent.get("labels", {}).get("en", {}).get("value", wd_id)
    _label_cache[wd_id] = lbl
    return lbl

def run_sparql(query: str) -> List[dict]:
    r = requests.get(SPARQL_ENDPOINT, params={"query": query}, headers=HEADERS)
    r.raise_for_status()
    return r.json()["results"]["bindings"]

def expand_from_neighbors(
    neighbor_qids: Set[str],
    limit: int = 5
) -> Tuple[List[Tuple[str,str,str]], Set[str]]:
    triples: List[Tuple[str,str,str]] = []
    next_frontier: Set[str] = set()

    for q in neighbor_qids:
        # outgoing direct claims
        q_out = f"""
        SELECT ?p ?o WHERE {{
          wd:{q} ?p ?o .
          FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT {limit}
        """
        for b in run_sparql(q_out):
            p_id = b["p"]["value"].rsplit("/", 1)[-1]
            o_id = b["o"]["value"].rsplit("/", 1)[-1] \
                   if b["o"]["type"] == "uri" else None
            triples.append((q, p_id, o_id))
            if o_id and o_id.startswith("Q"):
                next_frontier.add(o_id)

        # incoming direct claims
        q_in = f"""
        SELECT ?s ?p WHERE {{
          ?s ?p wd:{q} .
          FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT {limit}
        """
        for b in run_sparql(q_in):
            s_id = b["s"]["value"].rsplit("/", 1)[-1]
            p_id = b["p"]["value"].rsplit("/", 1)[-1]
            triples.append((s_id, p_id, q))
            if s_id.startswith("Q"):
                next_frontier.add(s_id)

    return triples, next_frontier

def map_id_triples_to_labels(
    id_triples: List[Tuple[str,str,str]]
) -> List[List[str]]:
    return [
        [ get_label(s), get_label(p), get_label(o) if o else "" ]
        for s,p,o in id_triples
    ]


if __name__ == "__main__":
    seed = "Q26876"
    hops = 2                        
    per_node_limit = 5              

    frontier = {seed}
    all_hop_results: Dict[int, List[Tuple[str,str,str]]] = {}

    for depth in range(1, hops+1):
        ids, next_frontier = expand_from_neighbors(frontier, limit=per_node_limit)
        all_hop_results[depth] = ids
        print(f"\n--- Hop {depth} (raw ID-triples: {len(ids)}) ---")
        for t in ids:
            print(t)
        frontier = next_frontier

    for depth, id_triples in all_hop_results.items():
        lbls = map_id_triples_to_labels(id_triples)
        print(f"\n--- Hop {depth} (label-triples) ---")
        for s,p,o in lbls:
            print(f"[{s!r}, {p!r}, {o!r}]")
