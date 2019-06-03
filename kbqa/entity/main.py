from logging import getLogger
from kbqa.entity.ner import Ner
from kbqa.entity.entitylinking import EntityLinking

logger = getLogger('Entity')

class Entity:
    def __init__(self):
        self.ner = Ner()
        self.entity_linking = EntityLinking()

    def predict(self, text, match_result):
        entitys = self.ner.predict(text)
        if not entitys and match_result:
            entitys = [i for i in match_result if 'Predicate' not in i['tag']]
            logger.info('use match entity {}'.format(entitys))
        self.entity_linking.linking(entitys, text)
        return entitys