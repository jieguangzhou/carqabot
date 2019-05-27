from logging import getLogger
from kbqa.sparql import KG

logger = getLogger('KBQA')


class BaseQA:
    def __init__(self):
        self.kg = KG()
        pass


    def predict(self, *args, **kwargs):
        return NotImplementedError



    def search_train_object(self, train, relation):
        logger.debug('search_train_object {} {}'.format(train, relation))
        results = self.kg.get_train_object(train, relation)
        return results

    def search_car_object(self, car, relation):
        logger.debug('search_car_object {} {}'.format(car, relation))
        results = self.kg.get_object(car, relation)
        return results

    def search_brand(self, brand, relation):
        logger.debug('search_brand {} {}'.format(brand, relation))
        results = self.kg.get_brand_train(brand, relation)
        return results





# class KBQA:
#     def __init__(self):
#         self.entity = Entity()
#         self.relation = Relation()
#         self.kg = KG()
#
#     def __call__(self, text, status: DMStatus = None, other_entity_iri=None):
#         return self.predict(text, status=status, other_entity_iri=None)
#
#     def predict(self, text, status: DMStatus = None, other_entity_iri=None):
#         entitys = self.entity.predict(text)
#         relation, relation_confidence = self.relation.predict(text)
#         results = []
#         entitys_linkings = []
#         for entity in entitys:
#             if 'entity_linking' not in entity:
#                 continue
#             last_iri_class = ''
#             for entitylinking in entity['entity_linking']:
#                 iri = entitylinking['iri']
#                 iri_class = entitylinking['class']
#                 if iri_class == 'Train':
#                     result_data = self.search_train_object(iri, relation)
#                 elif iri_class == 'Car':
#                     result_data = self.search_car_object(iri, relation)
#                 else:
#                     result_data = self.search_brand(iri, relation)
#                 if result_data:
#                     entitys_linkings.append(entitylinking)
#                     results.append({
#                         'module': 'kbqa',
#                         'type': iri_class,
#                         'data': result_data,
#                         'entity': iri,
#                         'relation': relation
#                     })
#                     if last_iri_class != iri_class:
#                         break
#         logger.debug('result: {}'.format(results))
#         status = {'entity': entitys_linkings, 'relation': relation}
#         return results, status
#
#     def search_train_object(self, train, relation):
#         logger.debug('search_train_object {} {}'.format(train, relation))
#         results = self.kg.get_train_object(train, relation)
#         return results
#
#     def search_car_object(self, car, relation):
#         logger.debug('search_car_object {} {}'.format(car, relation))
#         results = self.kg.get_object(car, relation)
#         return results
#
#     def search_brand(self, brand, relation):
#         logger.debug('search_brand {} {}'.format(brand, relation))
#         results = self.kg.get_brand_train(brand, relation)
#         return results
