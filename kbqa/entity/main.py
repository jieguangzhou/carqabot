from logging import getLogger
from kbqa.entity.ner import Ner
from kbqa.entity.entitylinking import EntityLinking
from kbqa.common.dictionary_match import Matcher

logger = getLogger('Entity')

class Entity:
    """
    命名实体识别与实体链接模块
    """
    def __init__(self):
        # 基于序列标注的命名实体识别
        self.ner = Ner()
        # 基于词典匹配的命名实体识别
        self.matcher = Matcher()
        self.entity_linking = EntityLinking()

    def predict(self, text):
        match_result = self.matcher.match(text)
        entitys = self.ner.predict(text)
        if not entitys and match_result:
            entitys = [i for i in match_result if 'Predicate' not in i['tag']]
            logger.info('use match entity {}'.format(entitys))
        self.entity_linking.linking(entitys, text)
        return entitys