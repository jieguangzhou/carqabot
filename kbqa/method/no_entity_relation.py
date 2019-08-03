from kbqa.method.commond import BaseQA
from logging import getLogger
from dm.status import DMStatus

logger = getLogger('NoEntityQA')


class NoEntityQA(BaseQA):
    """
    解决关系更换之后，回答历史实体的模块
    如上次问了  奥迪A7多少钱
    然后机器人回答了之后又接着问 百米加速是多少呢
    """
    def __init__(self):
        super(NoEntityQA, self).__init__()

    def predict(self, relation, status: DMStatus = None):
        result = None
        iri = status.kb_status.last_entity
        logger.debug(iri)
        if not iri:
            return result
        if isinstance(iri, list):
            iri_class = 'Car'
        else:
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
