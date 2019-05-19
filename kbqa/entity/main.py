from kbqa.entity.ner import Ner

class Entity:
    def __init__(self, ner_model_path):
        self.ner = Ner(ner_model_path)

    def predict(self, text):
        entitys = self.ner.predict(text)
        return entitys