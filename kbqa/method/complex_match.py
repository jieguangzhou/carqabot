from model.text_classification import Predictor
from config import Path
from logging import getLogger
import os
import pandas as pd

logger = getLogger('ComplexMatch')

# todo:去除实体词进行
class ComplexMatch:
    def __init__(self):
        self.data, self.questions = self.load_complex_qa()
        self.predictor = Predictor(Path.relation_match_model)

    def predict(self, text):
        results = []
        result = self.predictor.predict_texts(text_a_s=text, text_b_s=self.questions)
        for (qa_type, question), (label, confidence) in zip(self.data, result):
            if label == 'No':
                confidence = 1 - confidence
            results.append((qa_type, question, confidence))
        top_5_result = sorted(results, key=lambda x: x[2], reverse=True)[:5]
        for qa_type, question, confidence in top_5_result:
            logger.debug('{} {} {}'.format(qa_type, question, confidence))

        top_qa_type, question, top_confidence = top_5_result[0]
        if top_confidence < 0.97:
            top_qa_type = ''
        return top_qa_type, top_confidence

    def load_complex_qa(self):
        df = pd.read_excel(os.path.join(Path.data_path, 'complex_qa.xlsx'))
        data = []
        for _, row in df.iterrows():
            data.append((row['qa_type'], row['question']))
        questions = [i[1] for i in data]
        return data, questions
