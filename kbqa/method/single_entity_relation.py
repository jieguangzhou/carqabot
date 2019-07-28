from kbqa.sparql import KG
from kbqa.method.commond import BaseQA
from logging import getLogger
from dm.status import DMStatus

logger = getLogger('SingleEntityQA')


class SingleEntityQA(BaseQA):
    """
    回答普通KBQA，不使用上文
    如直接问 奥迪R8的最高车速是多少公里
    """
    def __init__(self):
        super(SingleEntityQA, self).__init__()

    def predict(self, entities, relation, status: DMStatus = None):
        entities = [entity for entity in entities if entity['entity_linking']]
        if len(entities) != 1:
            return None

        entity = entities[0]
        if 'entity_linking' not in entity:
            return None
        result = None
        for entitylinking in entity['entity_linking']:
            iri = entitylinking['iri']
            iri_class = entitylinking['class']
            if iri_class == 'Train':
                result_data = self.search_train_object(iri, relation)
            elif iri_class == 'Car':
                result_data = self.search_car_object(iri, relation)
            else:
                result_data = self.search_brand(iri, relation)
            if result_data:
                result = {
                    'module': 'kbqa',
                    'type': iri_class,
                    'data': result_data,
                    'entity': iri,
                    'relation': relation
                }
                break

        logger.debug('result: {}'.format(result))
        return result
