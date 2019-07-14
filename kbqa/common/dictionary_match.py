from ahocorasick import Automaton
import os
from config import Path
import re
from collections import defaultdict
from kbqa.common.tokenizer import Tokenizer


class Matcher:
    """字典匹配"""
    def __init__(self):
        self.automaton = self.__create_automaton()
        self.mapping = defaultdict(set)
        self.tokenizer = self._init_tokenizer()

    def match(self, text):
        words = self.tokenizer.lcut(text.lower())
        index = 0
        result = []
        for word in words:
            if word not in self.mapping:
                continue
            word_len = len(word)
            start = index
            end = start + word_len - 1
            confidence = 1
            result.append({'start': start,
                           'end': end,
                           'word': word,
                           'confidence': confidence,
                           'tag': self.mapping[word]
                           })
            index += word_len
        return result
        # for word in words:

    def _init_tokenizer(self):
        tokenizer = Tokenizer()
        paths = [
            ('Brand', os.path.join(Path.dictionary, 'Brand.txt')),
            ('Car', os.path.join(Path.dictionary, 'Car.txt')),
            ('Train', os.path.join(Path.dictionary, 'Train.txt')),
            ('Predicate', os.path.join(Path.dictionary, 'config.txt'))
        ]
        for tag, path in paths:
            with open(path, 'r') as r_f:
                for line in r_f:
                    line = line.rstrip('\n')
                    _, *words = line.split('\t')
                    for word in words:
                        word = re.sub('\(.*?\)', '', word.lower())
                        tokenizer.suggest_freq(word, tune=True)
                        self.mapping[word].add(tag)
        return tokenizer

    def __create_automaton(self):
        paths = [
            ('Brand', os.path.join(Path.dictionary, 'Brand.txt')),
            ('Car', os.path.join(Path.dictionary, 'Car.txt')),
            ('Train', os.path.join(Path.dictionary, 'Train.txt')),
            ('Predicate', os.path.join(Path.dictionary, 'config.txt'))
        ]
        automaton = Automaton()
        for tag, path in paths:
            with open(path, 'r') as r_f:
                for line in r_f:
                    line = line.rstrip('\n')
                    _, *words = line.split('\t')
                    for word in words:
                        word = re.sub('\(.*?\)', '', word.lower())
                        _, tag_set = automaton.get(word, (word, set()))
                        tag_set.add(tag)
                        automaton.add_word(word, (word, tag_set))

        automaton.make_automaton()
        return automaton
