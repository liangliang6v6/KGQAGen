"""
Entity converter converts between Freebase and Wikidata entity IDs.
"""
from SPARQLWrapper import JSON, SPARQLWrapper

class EntityConverter:
    """A converter to convert between freebase and wikidata entities.
    """
    def __init__(self, endpoint="http://localhost:1234/api/endpoint/sparql") -> None:
        sparql = SPARQLWrapper(endpoint)
        sparql.setReturnFormat(JSON)
        self.sparql = sparql
    
    def query_wikidata(self, query):
        """Query wikidata

        Args:
            query (str): SPARQL query

        Returns:
            dict: SPARQL query result
        """
        self.sparql.setQuery(query)
        try:
            ret = self.sparql.queryAndConvert()
            return ret
        except Exception as e:
            print(e)
            return None

    def get_freebase_id(self, entity, limit=1):
        """Query freebase id from wikidata id

        Args:
            entity (str): Wikidata id, e.g. Q42

        Returns:
            str | list[str]: Corresponding freebase id, e.g. /m/0bwd_0
                If limit is greater than 1, return a list of freebase ids.
        """
        query = f"""
            SELECT DISTINCT ?mid WHERE {{
                <http://www.wikidata.org/entity/{entity}> <http://www.wikidata.org/prop/direct/P646> ?mid
            }}
            LIMIT {limit}
            """
        response = self.query_wikidata(query)
        if response is None or "results" not in response:
            return None
 
        bindings = response["results"]["bindings"]
        if len(bindings) > 0:
            if limit == 1:
                mid = bindings[0]["mid"]["value"]
                return mid
            mids = [b["mid"]["value"] for b in bindings]
            return mids
        return None


    def get_wikidata_id(self, entity, limit=1):
        """Convert freebase id to wikidata id

        Args:
            entity (str): Freebase id, e.g. /m/0bwd_0

        Returns:
            str | list[str]: Corresponding wikidata id, e.g. Q42
                If limit is greater than 1, return a list of wikidata ids.
        """
        query = f"""
            SELECT DISTINCT ?qid WHERE {{
                ?qid wdt:P646 "{entity}".
            }}
            LIMIT {limit}
            """
        response = self.query_wikidata(query)
        if response is None or "results" not in response:
            return None
        bindings = response["results"]["bindings"]
        if len(bindings) > 0:
            if limit == 1:
                qid = bindings[0]["qid"]["value"]
                qid = qid.split("/")[-1]
                return qid
            qids = [b["qid"]["value"] for b in bindings]
            qids = [qid.split("/")[-1] for qid in qids]
            return qids
        return None
