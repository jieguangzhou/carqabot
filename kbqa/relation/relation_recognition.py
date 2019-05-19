from model.text_classification import Predictor

class RelationClassifier:
    def __init__(self, model_path):
        self.predictor = Predictor(model_path)

    def predict(self, text):
        result = self.predictor.predict_text(text)
        return result