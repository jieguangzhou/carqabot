from kbqa.entity.main import Entity
from kbqa.relation.main import Relation
from kbqa.sparql import KG
from kbqa.method.single_entity_relation import SingleEntityQA
from kbqa.method.history_entity_relation import HistoryEntityQA
from kbqa.method.no_entity_relation import NoEntityQA
from dm.status import DMStatus
from logging import getLogger

logger = getLogger('KBQA')


class KBQA:
    def __init__(self):
        self.entity = Entity()
        self.relation = Relation()
        self.qa_single_entity = SingleEntityQA()
        self.qa_history_entity = HistoryEntityQA()
        self.qa_no_entity = NoEntityQA()

    def predict(self, text, status: DMStatus = None, other_entity_iri=None):
        logger.debug('{} {}'.format('other_entity_iri', other_entity_iri))
        if other_entity_iri:
            result = self.qa_history_entity.predict(other_entity_iri, status)
        else:
            relation, relation_confidence = self.relation.predict(text)
            entitys = self.entity.predict(text)
            if not entitys:
                result = self.qa_no_entity.predict(relation, status)
            else:
                result = self.qa_single_entity.predict(entitys, relation)

        if result:
            entity = result.get('entity')
            relation = result.get('relation')
        else:
            entity = None
            relation = None
        status = {'entity': entity, 'relation': relation}
        return result, status
