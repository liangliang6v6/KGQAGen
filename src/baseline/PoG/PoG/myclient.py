import requests
import typing as tp
import time
import itertools
from concurrent.futures import ThreadPoolExecutor


class SPARQLQueryClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def run_sparql(self, query: str) -> tp.List[str]:
        try:
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": "QoG-SPARQL-Client/1.0 (liangz.cs@outlook.com)"
            }
            response = requests.get(
                self.endpoint,
                params={"query": query},
                headers=headers,
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
            return [
                list(binding.values())[0]["value"]
                for binding in data["results"]["bindings"]
            ]
        except Exception as e:
            print(f"[SPARQL ERROR] {e}")
            return []

    def qid2label(self, qid: str) -> tp.List[str]:
        query = f"""
        SELECT ?label WHERE {{
            wd:{qid} rdfs:label ?label .
            FILTER (lang(?label) = "en")
        }} LIMIT 1
        """
        return self.run_sparql(query)

    def label2qid(self, label: str) -> tp.List[str]:
        query = f"""
        SELECT ?p WHERE {{
            ?p rdfs:label "{label}"@en .
            FILTER STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")
        }} LIMIT 1
        """
        results = self.run_sparql(query)
        return [p.split("/")[-1] for p in results]
        
    def get_all_relations_of_an_entity(self, qid: str) -> tp.Dict[str, tp.List[str]]:
        outgoing_query = f"""
        SELECT DISTINCT ?p WHERE {{
            wd:{qid} ?p ?o .
            FILTER STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")
        }}
        """
        incoming_query = f"""
        SELECT DISTINCT ?p WHERE {{
            ?s ?p wd:{qid} .
            FILTER STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")
        }}
        """

        tail_rels = self.run_sparql(outgoing_query)
        head_rels = self.run_sparql(incoming_query)

        cleaned_tail = [p.split("/")[-1] for p in tail_rels if p.startswith("http://www.wikidata.org/prop/direct/")]
        cleaned_head = [p.split("/")[-1] for p in head_rels if p.startswith("http://www.wikidata.org/prop/direct/")]

        return {"head": cleaned_head, "tail": cleaned_tail}

    def get_tail_entities_given_head_and_relation(self, head_qid: str, pid: str) -> tp.Dict[str, tp.List[str]]:
        query = f"""
        SELECT ?obj WHERE {{
            wd:{head_qid} wdt:{pid} ?obj .
            FILTER STRSTARTS(STR(?obj), "http://www.wikidata.org/entity/Q")
        }}
        """
        return {"head": [head_qid], "tail": self.run_sparql(query)}

    def get_tail_values_given_head_and_relation(self, head_qid: str, pid: str) -> tp.List[str]:
        query = f"""
        SELECT ?value WHERE {{
            wd:{head_qid} wdt:{pid} ?value .
            FILTER (!isIRI(?value))
        }}
        """
        return self.run_sparql(query)


class MultiServerWikidataQueryClient:
    def __init__(self, urls: tp.List[str]):
        self.clients = [SPARQLQueryClient(url) for url in urls]
        self.executor = ThreadPoolExecutor(max_workers=len(urls))

    def query_all(self, method: str, *args) -> tp.Union[tp.List[str], tp.Dict[str, tp.List[str]], str]:
        futures = [
            self.executor.submit(getattr(client, method), *args)
            for client in self.clients
        ]
        results = [f.result() for f in futures]

        is_dict_return = method in {
            "get_all_relations_of_an_entity",
            "get_tail_entities_given_head_and_relation",
        }

        if is_dict_return:
            merged = {"head": [], "tail": []}
            for r in results:
                if isinstance(r, dict):
                    merged["head"].extend(r.get("head", []))
                    merged["tail"].extend(r.get("tail", []))
            merged["head"] = list(set(merged["head"]))
            merged["tail"] = list(set(merged["tail"]))
            return merged if merged["head"] or merged["tail"] else "Not Found!"
        else:
            merged = set()
            for r in results:
                if isinstance(r, list):
                    merged.update(r)
            return list(merged) if merged else "Not Found!"
