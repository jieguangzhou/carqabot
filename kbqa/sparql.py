from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON
from config import Path
from logging import getLogger
endpoint = 'http://localhost:3030/car/query'

PREFIX = """PREFIX p:<http://www.demo.com/predicate/>
PREFIX item:<http://www.demo.com/kg/item/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

logger = getLogger('sparql')
class KG1:
    __instance = None

    def __new__(cls, *args, **kwargs):

        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    def __init__(self):
        self.graph = Graph()
        self.graph.load(Path.kg, format='turtle')

    def get_object(self, subject, predicate):
        query = """
            SELECT ?object
            WHERE {
              <%(subject)s> <%(predicate)s> ?object
            }
            """ % {'subject': subject, 'predicate': predicate}
        return self.query(query)

    def query(self, query):
        query = PREFIX + query
        result = self.graph.query(query)
        objects = []
        for r in result:
            obj = str(r['object'])
            objects.append(obj)
        return objects


class KG2:
    __instance = None

    def __new__(cls, *args, **kwargs):

        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    def __init__(self):
        self.sparql = SPARQLWrapper(endpoint)

    def get_new_car(self, cat_train):
        query = """
        SELECT ?car ?time
        WHERE {
          ?car p:CarTrain <%(cat_train)s>;
               p:上市时间 ?time.   
        }order by DESC(?time)
        """ % {'cat_train': cat_train}
        results = []
        top_time = None
        for result in self.query(query):
            time = result["time"]["value"]
            top_time = top_time or time
            car = result["car"]["value"]
            if time == top_time:
                results.append({'car': car, 'time': time})
        return results

    def get_brand_train(self, subject, relation):
        query = """
            SELECT distinct ?train ?train_name
            WHERE {
              <%(subject)s> p:own ?train.
              ?train rdfs:label ?train_name.
              ?car p:CarTrain ?train;
                   <%(relation)s> ?object.
              }
            """ % {'subject': subject, 'relation': relation}
        objects = []
        for result in self.query(query):
            value = result["train_name"]["value"]
            train = result["train"]["value"]
            if check_value(value):
                objects.append({
                    'value': value,
                    'iri': train
                })
        return objects

    def get_train_object(self, train, relation):
        query = """
                SELECT ?car ?time ?object ?name
                WHERE {
                  ?car p:CarTrain <%(cat_train)s>;
                       p:上市时间 ?time;
                       <%(relation)s> ?object;
                       rdfs:label ?name.
                }order by DESC(?time)
                """ % {'cat_train': train, 'relation': relation}
        results = []
        top_time = None
        for result in self.query(query):
            time = result["time"]["value"]
            top_time = top_time or time
            car = result["car"]["value"]
            value = result["object"]["value"]
            name = result["name"]["value"]
            if time == top_time and check_value(value):
                results.append({
                    'car': car,
                    'time': time,
                    'value': value,
                    'name': name,
                })
        return results

    def get_object(self, subjects, predicate):
        objects = []
        if isinstance(subjects, str):
            subjects = [subjects]
        for subject in subjects:
            query = """
                SELECT ?object ?style
                WHERE {
                  <%(subject)s> <%(predicate)s> ?object;
                  p:style ?style.
                }
                """ % {'subject': subject, 'predicate': predicate}
            for result in self.query(query):
                value = result["object"]["value"]
                name = result["style"]["value"]
                if check_value(value):
                    objects.append({
                        'value': value,
                        'name': name,
                    })
        return objects


    def query(self, query):
        query = PREFIX + query
        # logger.debug(query)
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()["results"]["bindings"]
        return results


def check_value(value):
    return value != '-'


KG = KG2

if __name__ == '__main__':
    kg = KG()
    print(kg.get_new_car('http://www.demo.com/kg/item/train#s4658'))
