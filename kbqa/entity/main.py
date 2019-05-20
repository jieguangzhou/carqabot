from kbqa.entity.ner import Ner
from kbqa.entity.entitylinking import EntityLinking

class Entity:
    def __init__(self, ner_model_path, dictionary_dir):
        self.ner = Ner(ner_model_path)
        self.entity_linking = EntityLinking(dictionary_dir)

    def predict(self, text):
        entitys = self.ner.predict(text)
        self.entity_linking.match_style(entitys, text)
        return entitys