from kbqa.method.commond import BaseQA
from logging import getLogger
from dm.status import DMStatus

logger = getLogger('HistoryEntityQA')


class HistoryEntityQA(BaseQA):
    """
    解决做选择时实体更换之后，回答历史关系的模块
    如上次问了  奔驰多少钱
    然后机器人回答了
        你是想问哪个车系的厂商指导价呢？
        1.奔驰E级AMG
        2.凌特
        3.奔驰G级
        4.奔驰C级(进口)
        5.奔驰CLS级
    然后回答 E级
    那么就会通过选择模块进行匹配得到 奔驰E级AMG，然后进行回答
    """
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
