from kbqa.entity.main import Entity
from kbqa.relation.main import Relation

class KBQA:
    def __init__(self, ner_model_path, relation_classifier_model_path):
        self.entity = Entity(ner_model_path)
        self.relation = Relation(relation_classifier_model_path)

    def __call__(self, text, history=None):
        return self.predict(text, history=history)

    def predict(self, text, history=None):
        entitys = self.entity.predict(text)
        relations = self.relation.predict(text)
        return entitys, relations