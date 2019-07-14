from model.text_classification import Predictor
from config import Path
from logging import getLogger
import os
from kbqa.common.dictionary import Dictionary

logger = getLogger('RelationMatch')


class RelationMatch:
    """
    关系匹配，使用bert文本相似度方法进行
    """
    def __init__(self):
        self.predicates = self.load_predictes()
        self.predictor = Predictor(Path.relation_match_model)

    def predict(self, text):
        results = []
        result = self.predictor.predict_texts(text_a_s=text, text_b_s=self.predicates)
        for predicate, (label, confidence) in zip(self.predicates, result):
            if label == 'No':
                confidence = 1 - confidence
            results.append((predicate, confidence))
        top_5_result = sorted(results, key=lambda x: x[1], reverse=True)[:5]
        for label, confidence in top_5_result:
            logger.debug('{} {}'.format(label, confidence))

        top_label, top_confidence = top_5_result[0]
        if top_confidence < 0.97:
            top_label = ''
        return top_label, top_confidence

    def load_predictes(self):
        predicates = []
        config_path = os.path.join(Path.dictionary, 'config.txt')
        with open(config_path, 'r') as r_f:
            for line in r_f:
                line = line.rstrip('\n')
                if not line:
                    continue
                iri, *words = line.split('\t')
                for word in words:
                    predicates.append(word)
        return predicates
