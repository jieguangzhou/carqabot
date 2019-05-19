from model.sequence_labeling import Predictor
class Ner:
    def __init__(self, model_path):
        self.predictor = Predictor(model_path)

    def predict(self, text):
        return self.predictor.predict_text(text)

