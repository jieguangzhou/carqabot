from kbqa.common.dictionary import Dictionary
import os
import jieba
from difflib import SequenceMatcher
from config import Path
from logging import getLogger

logger = getLogger('EntityLinking')


class EntityLinking:
    def __init__(self):
        dictionary_dir = Path.dictionary
        self.dictionary_brand = Dictionary(os.path.join(dictionary_dir, 'Brand.txt'))
        self.dictionary_train = Dictionary(os.path.join(dictionary_dir, 'Train.txt'))
        self.dictionary_car = Dictionary(os.path.join(dictionary_dir, 'Car.txt'))
        self.iri_features = self.get_iri_features(dictionary_dir)

    def linking(self, entities, text):
        self.match_entity(entities, text)

    def match_entity(self, entities, text):
        for entity in entities:
            word = entity['word']
            iris_brand = self.match_word(word, self.dictionary_brand)
            iris_train = self.match_word(word, self.dictionary_train)
            iris_car = self.match_word(word, self.dictionary_car)
            iris = iris_brand | iris_train | iris_car
            start = entity['start']
            end = entity['end']
            # replace_text = text[:start] + ' ' + text[end + 1:]
            replace_text_words = set([i for i in jieba.lcut(text, HMM=False) if i.strip()])
            entity_lingkings_scores = []
            for iri in iris:
                feature_data = self.iri_features[iri]
                score = self.jaccard(replace_text_words, feature_data['feature'])
                entity_lingkings_scores.append({
                    'score': score,
                    'iri': iri,
                    'rank': feature_data['rank'],
                    'class': feature_data['class'],
                    'texts': feature_data['texts'],
                })

            entity_lingkings_scores = sorted(entity_lingkings_scores, key=lambda x: (x['score'], -x['rank']),
                                             reverse=True)[:3]
            logger.debug(entity_lingkings_scores)
            entity['entity_linking'] = entity_lingkings_scores

    def get_style_data(self, path):
        style_data = {}
        with open(path, 'r') as r_f:
            for line in r_f:
                line = line.rstrip('\n')
                if not line:
                    continue
                iri, *_, style = line.split('\t')
                style_data[iri] = jieba.lcut(style.lower()), style
        return style_data

    def match_word(self, word, dictionary: Dictionary):
        iri_cars = dictionary.match(word)
        if not iri_cars:
            matchs = []
            max_score = 0
            for key in dictionary.dict.keys():
                score = SequenceMatcher(None, word, key).ratio()
                if score < 0.5:
                    continue
                match = {'score': score, 'word': key}
                if score > max_score:
                    max_score = score
                matchs.append(match)

            matchs = [iri_score for iri_score in matchs if iri_score['score'] == max_score]
            logger.warning(matchs)
            iri_cars = set()
            words = []
            for match in matchs:
                fuzzy_iri_cars = dictionary.match(match['word'])
                iri_cars.update(fuzzy_iri_cars)
                words.append(match['word'])
            logger.warning('fuzzy match {}'.format(words))
        return iri_cars

    def get_iri_features(self, dictionary_dir):
        features = {}
        for rank, file_name in enumerate(['Brand.txt', 'Train.txt', 'Car.txt']):
            class_name = file_name.split('.')[0]
            with open(os.path.join(dictionary_dir, file_name), 'r') as r_f:
                for line in r_f:
                    iri, *texts = line.rstrip('\n').split()
                    feature = []
                    for text in texts:
                        words = [i for i in jieba.lcut(text.lower(), HMM=False) if i.strip()]
                        feature.extend(words)
                    features[iri] = {
                        'feature': set(feature),
                        'rank': rank,
                        'class': class_name,
                        'texts': texts
                    }

        return features

    @staticmethod
    def jaccard(set1: set, set2: set):
        jiao = set1 & set2
        bing = set1 | set2
        return len(jiao) / len(bing)
