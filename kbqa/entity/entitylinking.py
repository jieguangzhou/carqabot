import os
from difflib import SequenceMatcher
from config import Path
from logging import getLogger

from kbqa.common.dictionary import Dictionary
from kbqa.common.tokenizer import Tokenizer

logger = getLogger('EntityLinking')


class EntityLinking:
    """
    实体链接模块，使用字典映射以及模糊匹配
    """

    def __init__(self):
        dictionary_dir = Path.dictionary
        self.dictionary_brand = Dictionary(os.path.join(dictionary_dir, 'Brand.txt'))
        self.dictionary_train = Dictionary(os.path.join(dictionary_dir, 'Train.txt'))
        self.dictionary_car = Dictionary(os.path.join(dictionary_dir, 'Car.txt'))
        self.tokenizer = self.init_tokenizer()
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
            logger.debug(iris)
            replace_text_words = set([i for i in self.tokenizer.lcut(text.lower()) if i.strip()])
            entity_lingkings_scores = []
            for iri in iris:
                feature_data = self.iri_features[iri]
                score = self.jaccard(feature_data['feature'], replace_text_words)
                if score <= 0.1:
                    continue
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
            iri_cars = set()
            words = []
            for match in matchs:
                fuzzy_iri_cars = dictionary.match(match['word'])
                iri_cars.update(fuzzy_iri_cars)
                words.append(match['word'])
            logger.debug('fuzzy match {}'.format(words))
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
                        words = [i for i in self.tokenizer.lcut(text.lower()) if i.strip()]
                        feature.extend(words)

                    features[iri] = {
                        'feature': set(feature),
                        'rank': rank,
                        'class': class_name,
                        'texts': texts,
                        'name': ' '.join(texts)
                    }
        # for n, (iri, data) in enumerate(features.items()):
        #     if n <= 20:
        #         print('{}\t{}'.format(iri, data['feature']))

        return features

    @staticmethod
    def jaccard(set1: set, set2: set):
        jiao = set()
        scores = []
        for word1 in set1:
            max_score = 0
            for word2 in set2:
                score = len(set(word1) & set(word2)) / max(len(set(word1)), len(set(word2)))
                if word1.startswith(word2) or word2.startswith(word1) or score > 0.5:
                    if score > max_score:
                        max_score = score
                    jiao.add(word1)
            scores.append(max_score)
        bing = set1 | set2
        weight = (sum(scores) / len(jiao)) if jiao else 0

        return weight * len(jiao) / len(bing)

    def init_tokenizer(self):
        tokenizer = Tokenizer()
        for dictionary in [self.dictionary_train, self.dictionary_brand]:

            for key in dictionary.dict.keys():
                tokenizer.suggest_freq(key, tune=True)
        return tokenizer
