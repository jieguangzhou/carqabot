from kbqa.main import KBQA

class QA:
    def __init__(self, dictionary_dir, kg_path, ner_model_path, relation_classifier_model_path):
        self.kbqa = KBQA(dictionary_dir, kg_path, ner_model_path, relation_classifier_model_path)

    def predict(self, text, history=None):
        result = self.kbqa(text)
        return result