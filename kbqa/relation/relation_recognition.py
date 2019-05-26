from model.text_classification import Predictor
from config import Path
from logging import getLogger
logger = getLogger('RelationClassifier')


class RelationClassifier:
    def __init__(self):
        self.predictor = Predictor(Path.relation_classifier_model)

    def predict(self, text):
        result = self.predictor.predict_text(text)
        logger.debug(result)
        return result
