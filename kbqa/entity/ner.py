from model.sequence_labeling import Predictor, debug_message
from config import Path
from logging import getLogger


logger = getLogger('Ner')
class Ner:
    """
    命名实体识别模块，使用bert做识别
    """
    def __init__(self):
        self.predictor = Predictor(Path.ner_model)

    def predict(self, text):
        ners = self.predictor.predict_text(text)
        logger.debug(ners)
        return ners

