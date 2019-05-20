from kbqa.common.dictionary import Dictionary
import os
import jieba
from difflib import SequenceMatcher


class EntityLinking:
    def __init__(self, dictionary_dir):
        self.dictionary_brand = Dictionary(os.path.join(dictionary_dir, 'Brand.txt'))
        self.dictionary_car = Dictionary(os.path.join(dictionary_dir, 'Car.txt'))
        self.dictionary_train = Dictionary(os.path.join(dictionary_dir, 'Train.txt'))
        self.style_data = self.get_style_data(os.path.join(dictionary_dir, 'Car.txt'))

    def linking(self, entitys, text):
        self.match_style(entitys, text)

    def match_style(self, entitys, text):
        text = jieba.lcut(text.lower())
        for entity in entitys:
            word = entity['word']
            iri_cars = self.dictionary_car.match(word)
            iri_scores = []
            max_score = 0
            for iri in iri_cars:
                style = self.style_data.get(iri, [])
                score = SequenceMatcher(None, text, style).ratio()
                iri_scores.append({'iri': iri, 'score': score})
                if score > max_score:
                    max_score = score
            iri_scores = [iri_score for iri_score in iri_scores if iri_score['score'] == max_score]
            if iri_scores:
                entity['entitylinking'] = iri_scores


    def get_style_data(self, path):
        style_data = {}
        with open(path, 'r') as r_f:
            for line in r_f:
                line = line.rstrip('\n')
                if not line:
                    continue
                iri, *_, style = line.split('\t')
                style_data[iri] = jieba.lcut(style.lower())
        return style_data