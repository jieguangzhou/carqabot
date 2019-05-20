from typing import List, Dict
from logging import getLogger
from collections import defaultdict

logger = getLogger('automaton')


class Dictionary:
    def __init__(self, dictionary_path):
        self.dict = defaultdict(set)
        self.load_data(dictionary_path)

    def match(self, word: str) -> List[Dict]:
        return self.dict.get(word.lower(), set())

    def load_data(self, dictionary_path):
        with open(dictionary_path, 'r') as r_f:
            for line in r_f:
                line = line.rstrip('\n')
                if not line:
                    continue
                iri, *words = line.split('\t')
                for word in words:
                    self.dict[word.lower()].add(iri)
