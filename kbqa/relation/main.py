from kbqa.relation.relation_recognition import RelationClassifier
from kbqa.relation.relation_mapping import RelationMapping


class Relation:
    def __init__(self, relation_classifier_model_path, dictionary_dir):
        self.relation_classifier = RelationClassifier(relation_classifier_model_path)
        self.relation_mapping = RelationMapping(dictionary_dir)

    def predict(self, text):
        relation, confidence = self.relation_classifier.predict(text)
        iri_relation = self.relation_mapping(relation)
        return iri_relation, confidence
