from kbqa.entity.main import Entity
from kbqa.relation.main import Relation
from kbqa.sparql import KG
from kbqa.method.single_entity_relation import SingleEntityQA
from kbqa.method.history_entity_relation import HistoryEntityQA
from kbqa.method.no_entity_relation import NoEntityQA
from kbqa.common.dictionary_match import Matcher
from dm.status import DMStatus
from logging import getLogger
from kbqa.method.complex_match import ComplexMatch
from kbqa.method.complex_qa import ComplexQA

logger = getLogger('KBQA')


class KBQA:
    def __init__(self):
        self.entity = Entity()
        self.relation = Relation()
        self.matcher = Matcher()
        self.qa_single_entity = SingleEntityQA()
        self.qa_history_entity = HistoryEntityQA()
        self.qa_no_entity = NoEntityQA()
        self.complex_match = ComplexMatch()
        self.complex_qa = self.init_complex_qa()

    def predict(self, text, status: DMStatus = None, other_entity_iri=None):
        logger.debug('{} {}'.format('other_entity_iri', other_entity_iri))
        match_result = self.matcher.match(text)
        entitys = self.entity.predict(text, match_result)
        top_qa_type, top_confidence = self.complex_match.predict(text)
        if top_qa_type:
            result = self.run_complex_qa(top_qa_type, text, entitys, status)
        else:
            result = self.run_simple_qa(text, status, entitys, other_entity_iri)

        if result:
            entity = result.get('entity')
            relation = result.get('relation')
        else:
            entity = None
            relation = None

        status = {'entity': entity, 'relation': relation}
        return result, status

    def run_complex_qa(self, qa_type, text, entitys, status: DMStatus = None):
        qa = self.complex_qa[qa_type]
        result = qa.predict(text, entitys)
        return result

    def run_simple_qa(self, text, status, entitys, other_entity_iri=None):
        relation, relation_confidence = self.relation.predict(text)
        if other_entity_iri:
            result = self.qa_history_entity.predict(other_entity_iri, status)
        else:
            if not entitys:
                result = self.qa_no_entity.predict(relation, status)
            else:
                result = self.qa_single_entity.predict(entitys, relation)
        return result

    def init_complex_qa(self):
        complex_qa = {}
        for subclass in ComplexQA.__subclasses__():
            complex_qa[subclass.__name__] = subclass()
        return complex_qa
