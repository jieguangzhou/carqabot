from kbqa.entity.main import Entity
from kbqa.relation.main import Relation
from kbqa.sparql import KG

class KBQA:
    def __init__(self, dictionary_dir, kg_path, ner_model_path, relation_classifier_model_path):
        self.entity = Entity(ner_model_path, dictionary_dir)
        self.relation = Relation(relation_classifier_model_path, dictionary_dir)
        self.kg = KG(kg_path)

    def __call__(self, text, history=None):
        return self.predict(text, history=history)

    def predict(self, text, history=None):
        entitys = self.entity.predict(text)
        relation, relation_confidence = self.relation.predict(text)
        for entity in entitys:
            if 'entitylinking' not in entity:
                continue
            entitylinkings = entity['entitylinking']
            for entitylinking in entitylinkings:
                subject = entitylinking['iri']
                self.kg.get_object(subject, relation)
        return entitys, (relation, relation_confidence)