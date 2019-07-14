from logging import getLogger

from kbqa.relation.relation_recognition import RelationClassifier
from kbqa.relation.relation_mapping import RelationMapping
from kbqa.relation.relation_match import RelationMatch

logger = getLogger('relation')

class Relation:
    """
    关系识别模块，主要有关系分类，关系匹配以及关将关系映射到iri
    """
    def __init__(self):
        self.relation_classifier = RelationClassifier()
        self.relation_match = RelationMatch()
        self.relation_mapping = RelationMapping()

    def predict(self, text):
        relation, confidence = self.relation_classifier.predict(text)
        if not relation:
            relation, confidence = self.relation_match.predict(text)
        iri_relation = self.relation_mapping(relation)
        return iri_relation, confidence
