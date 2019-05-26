from kbqa.common.dictionary import Dictionary
import os
import jieba
from difflib import SequenceMatcher

from config import Path

class RelationMapping:
    def __init__(self):
        self.dictionary_car = Dictionary(os.path.join(Path.dictionary, 'config.txt'))

    def __call__(self, relation):
        iris = self.dictionary_car.match(relation)
        if iris:
            iri = list(iris)[0]
        else:
            iri = None
        return iri