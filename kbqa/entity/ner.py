from model.sequence_labeling import Predictor
from config import Path
from logging import getLogger

logger = getLogger('Ner')
class Ner:
    def __init__(self):
        self.predictor = Predictor(Path.ner_model)

    def predict(self, text):
        ners = self.predictor.predict_text(text)
        logger.debug(ners)
        return ners

