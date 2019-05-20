from rdflib import Graph

graph = Graph()
graph.load('/home/guang/project/carqabot/data/kg/car.ttl', format='turtle')
print(len(graph))