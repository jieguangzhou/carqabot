from rdflib import Graph

PREFIX = """PREFIX p:<http://www.demo.com/predicate/>
PREFIX item:<http://www.demo.com/kg/item/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


class KG:
    __instance = None

    def __new__(cls, *args, **kwargs):

        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    def __init__(self, path):
        self.graph = Graph()
        self.graph.load(path, format='turtle')

    def get_object(self, subject, predicate):
        query = """
            SELECT ?object
            WHERE {
              <%(subject)s> <%(predicate)s> ?object
            }
            """ % {'subject': subject, 'predicate': predicate}
        self.query(query)

    def query(self, query):
        query = PREFIX + query
        print(query)
        result = self.graph.query(query)
        for r in result:
            print(str(r['object']))
