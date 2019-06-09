from kbqa.method.commond import BaseQA
from logging import getLogger
from dm.status import DMStatus
import re

logger = getLogger('ComplexQA')


class ComplexQA(BaseQA):
    def __init__(self):
        super(ComplexQA, self).__init__()
        self.name = self.__class__.__name__


class PriceLessThenX(ComplexQA):
    def __init__(self):
        super(PriceLessThenX, self).__init__()
        self.re_price_find = re.compile('(\d+)[万wW]')
        self.re_less = re.compile('以下|低|小|少')
        self.re_more = re.compile('以上|高|大')

    def predict(self, text, entities):
        price_find = self.re_price_find.findall(text)
        if price_find:
            price = float(price_find[0]) * 10000
            if self.re_less.findall(text):
                price_type = 'less'
            elif self.re_more.findall(text):
                price_type = 'more'
            else:
                price_type = 'almost'

            if not entities:
                entities = [{}]

            entity_data = entities[0].get('entity_linking', [{}])[0]
            results = self.get_price_object(price, price_type, entity_data)
            entity = [i['car'] for i in results]
            if len(entity) == 1:
                entity = entity[0]
            result = {
                'module': 'kbqa',
                'type': 'Car',
                'data': results,
                'entity': entity,
                'relation': None
            }
        else:
            result = {}
        return result

    def get_price_object(self, price, price_type, entity_data=None):
        if price_type == 'less':
            filter = '?price <= {} && ?price > 0'.format(price)
        elif price_type == 'more':
            filter = '?price >= {}'.format(price)
        else:
            filter = '?price >= {} && ?price <= {}'.format(price * 0.9, price * 1.1)

        entity_class = entity_data.get('class')
        entity_iri = entity_data.get('iri')
        entity_query = ""
        logger.debug(entity_data)
        logger.debug(entity_iri)
        if entity_iri:
            if entity_class == 'Brand':
                entity_query = "?car p:BrandName <{}> .".format(entity_iri)
            elif entity_class == 'Train':
                entity_query = "?car p:CarTrain <{}> .".format(entity_iri)
        logger.debug(entity_query)
        query = """
            SELECT ?car ?price ?car_name
            WHERE {
              ?car <http://www.demo.com/predicate/厂商指导价(元)> ?price.
              ?car rdfs:label ?car_name.
              %(entity_query)s
              FILTER (%(filter)s)
            }
            LIMIT 5
            """ % {'filter': filter, 'entity_query': entity_query}
        results = []
        logger.debug(query)
        for result in self.kg.query(query):
            car = result["car"]["value"]
            price = result["price"]["value"]
            car_name = result["car_name"]["value"]
            results.append({
                'value': price,
                'car': car,
                'name': car_name,
            })
        return results
