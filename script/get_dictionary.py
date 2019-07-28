from SPARQLWrapper import SPARQLWrapper, JSON
import os

endpoint = 'http://localhost:3030/car/query'
sparql = SPARQLWrapper(endpoint)

PREFIX = """PREFIX p:<http://www.demo.com/predicate/>
PREFIX item:<http://www.demo.com/kg/item/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

floder = '../data/dictionary'


def get_brand():
    save_path = '{}/Brand.txt'.format(floder)
    query = """
    SELECT ?subject ?name
    WHERE {
      ?subject rdf:type item:CBrand;
               rdfs:label ?name
    }
    """
    query = PREFIX + query
    print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    with open(save_path, 'w') as w_f:
        sparql_result = results["results"]["bindings"]
        print(len(sparql_result))
        for result in sparql_result:
            sub = result["subject"]["value"]
            name = result["name"]["value"]
            w_f.write('\t'.join([sub, name]) + '\n')


def get_train():
    save_path = '{}/Train.txt'.format(floder)
    query = """
    SELECT ?subject ?name
    WHERE {
      ?subject rdf:type item:CTrain;
               rdfs:label ?name
    }
    """
    query = PREFIX + query
    print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    with open(save_path, 'w') as w_f:
        sparql_result = results["results"]["bindings"]
        print(len(sparql_result))
        for result in sparql_result:
            sub = result["subject"]["value"]
            name = result["name"]["value"]
            w_f.write('\t'.join([sub, name]) + '\n')


def get_car():
    save_path = '{}/Car.txt'.format(floder)
    query = """
    SELECT ?subject ?train_name ?name
    WHERE {
      ?subject rdf:type item:CCar;
               rdfs:label ?name;
               p:CarTrain ?cartrain.
      ?cartrain rdfs:label ?train_name
               
    }
    """
    query = PREFIX + query
    print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    with open(save_path, 'w') as w_f:
        sparql_result = results["results"]["bindings"]
        print(len(sparql_result))
        for result in sparql_result:
            sub = result["subject"]["value"]
            train_name = result["train_name"]["value"]
            name = result["name"]["value"]
            w_f.write('\t'.join([sub, train_name, name]) + '\n')


def get_config():
    save_path = '{}/config.txt'.format(floder)
    query = """
    SELECT ?predicate ?name
    WHERE {
      ?predicate rdf:type item:config;
               rdfs:label ?name
    }
    """
    query = PREFIX + query
    print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    with open(save_path, 'w') as w_f:
        sparql_result = results["results"]["bindings"]
        print(len(sparql_result))
        for result in sparql_result:
            predicate = result["predicate"]["value"]
            name = result["name"]["value"]
            w_f.write('\t'.join([predicate, name]) + '\n')


get_brand()
get_train()
get_car()
get_config()
