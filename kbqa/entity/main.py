from kbqa.entity.ner import Ner
from kbqa.entity.entitylinking import EntityLinking


class Entity:
    def __init__(self):
        self.ner = Ner()
        self.entity_linking = EntityLinking()

    def predict(self, text):
        entitys = self.ner.predict(text)
        self.entity_linking.linking(entitys, text)
        return entitys