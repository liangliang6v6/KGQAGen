"""
Property converter converts between Freebase and Wikidata properties.
"""
import logging
import requests
import time
import random
from bs4 import BeautifulSoup
from SPARQLWrapper import SPARQLWrapper, JSON

class PropertyConverter:
    """Convert Freebase property to Wikidata property and vice versa. The data are scraped from
    https://www.wikidata.org/wiki/Wikidata:WikiProject_Freebase/Mapping
    """
    def __init__(self, url="https://www.wikidata.org/wiki/Wikidata:WikiProject_Freebase/Mapping"):
        """Create a property converter

        Args:
            url (str, optional): The mapping page from Freebase properties to Wikidata properties.
                Defaults to "https://www.wikidata.org/wiki/Wikidata:WikiProject_Freebase/Mapping".
        """
        self.url = url
        self.freebase2wikidata = {}
        self.wikidata2freebase = {}
        self.update_mapping()

    # def get_wikidata_property(self, freebase_property):
    #     """Get the wikidata property from freebase property

    #     Args:
    #         freebase_property (str): The freebase property, e.g. /organization/organization/child

    #     Returns:
    #         str | None: wikidata property identifier, e.g. P355
    #             If the freebase property is not mapped, return None
    #     """
    #     return self.freebase2wikidata.get(freebase_property, None)
    def sparql_fallback(self, freebase_property: str):
        SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
        SPARQL_HEADERS = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "QuestionOnGraph/1.0 (liangz.cs@outlook.com)"
        }

        tokens = freebase_property.strip("/").split("/")
        filters = " || ".join([
            f'CONTAINS(LCASE(?propLabel), "{token.lower()}")'
            for token in tokens
        ])

        query = f"""
        SELECT ?prop ?propLabel WHERE {{
        ?prop a wikibase:Property ;
                rdfs:label ?propLabel .
        FILTER({filters})
        FILTER(LANG(?propLabel) = "en")
        }}
        LIMIT 1
        """

        params = {"query": query}
        max_retries = 5

        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(1.5, 3.0))
                response = requests.get(SPARQL_ENDPOINT, params=params, headers=SPARQL_HEADERS, timeout=30)

                if response.status_code == 429:
                    print(f"[SPARQL Error 429] ({attempt+1}/{max_retries}) {freebase_property}")
                    time.sleep((2 ** attempt) + random.uniform(0, 2))
                    continue

                response.raise_for_status()
                data = response.json()

                for result in data.get("results", {}).get("bindings", []):
                    prop_id = result["prop"]["value"].split("/")[-1]
                    print(f"[SPARQL Match] {freebase_property} → {prop_id}")
                    return prop_id

                return None

            except Exception as e:
                print(f"[SPARQL Error] ({attempt+1}/{max_retries}) for {freebase_property}: {e}")
                if attempt == max_retries - 1:
                    return None



    def get_wikidata_property(self, freebase_property):
        """
        Try scraped mapping first, then fall back to SPARQL label search.
        """
        pid = self.freebase2wikidata.get(freebase_property)
        if pid is not None:
            return pid

        print(f"[Fallback Triggered] Trying SPARQL for: {freebase_property}")
        return self.sparql_fallback(freebase_property)


    def get_freebase_property(self, wikidata_property):
        """Get the freebase property from wikidata property

        Args:
            wikidata_property (str): The wikidata property, e.g. P355

        Returns:
            str | None: freebase property identifier, e.g. /organization/organization/child
                If the wikidata property is not mapped, return None
        """
        return self.wikidata2freebase.get(wikidata_property, None)

    def update_mapping(self):
        """Update the mapping from freebase property to wikidata property
        by crawling https://www.wikidata.org/wiki/Wikidata:WikiProject_Freebase/Mapping
        """
        response = requests.get(self.url, timeout=60)
        html_doc = response.text
        soup = BeautifulSoup(html_doc, features="html.parser")
        freebase2wikidata = {}
        rows = soup.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            try:
                freebase_url = cols[0].a.attrs["href"]
            except AttributeError:
                continue
            except IndexError:
                continue
            if 'www.freebase.com' not in freebase_url:
                continue
            try:
                wikidata_url = cols[1].a.attrs["href"]
            except AttributeError:
                wikidata_url = None
            freebase2wikidata[freebase_url] = wikidata_url

        logging.info("# exsit mapping: %d", len(freebase2wikidata))
        logging.info("# not none mapping: %d", len([x for x in freebase2wikidata.values()
                                                    if x is not None]))
        logging.info("# wikidata properties mapped: %d", len(set(freebase2wikidata.values())))

        freebase2wikidata = {k.split('https://www.freebase.com')[-1]: v.split('/wiki/Property:')[-1]
                             if v is not None else v
                             for k, v in freebase2wikidata.items()}
        wikidata2freebase = {v: k for k, v in freebase2wikidata.items() if v is not None}
        self.freebase2wikidata = freebase2wikidata
        self.wikidata2freebase = wikidata2freebase
