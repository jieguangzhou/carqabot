from kbqa.relation.relation_recognition import RelationClassifier


class Relation:
    def __init__(self, relation_classifier_model_path):
        self.relation_classifier = RelationClassifier(relation_classifier_model_path)

    def predict(self, text):
        result = self.relation_classifier.predict(text)
        return result
