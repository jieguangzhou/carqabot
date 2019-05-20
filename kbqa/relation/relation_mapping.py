from kbqa.common.dictionary import Dictionary
import os
import jieba
from difflib import SequenceMatcher


class RelationMapping:
    def __init__(self, dictionary_dir):
        self.dictionary_car = Dictionary(os.path.join(dictionary_dir, 'config.txt'))

    def __call__(self, relation):
        iris = self.dictionary_car.match(relation)
        if iris:
            iri = iris.pop()
        else:
            iri = None
        return iri