from kbqa.method.commond import BaseQA
from logging import getLogger
from dm.status import DMStatus

logger = getLogger('HistoryEntityQA')


class HistoryEntityQA(BaseQA):
    def __init__(self):
        super(HistoryEntityQA, self).__init__()

    def predict(self, entity_iri, status: DMStatus = None):
        relation = status.kb_status.last_relation
        logger.debug('history relation {}'.format(relation))
        result = None
        iri = entity_iri
        if 'train' in iri:
            iri_class = 'Train'
        elif 'car' in iri:
            iri_class = 'Car'
        else:
            iri_class = 'Brand'

        if iri_class == 'Train':
            result_data = self.search_train_object(iri, relation)
        elif iri_class == 'Car':
            result_data = self.search_car_object(iri, relation)
        else:
            result_data = self.search_brand(iri, relation)
        logger.debug(result_data)
        if result_data:
            result = {
                'module': 'kbqa',
                'type': iri_class,
                'data': result_data,
                'entity': iri,
                'relation': relation
            }

        logger.debug('result: {}'.format(result))
        return result
