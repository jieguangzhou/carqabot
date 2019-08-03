from kbqa.main import KBQA
from dbqa.main import DBQA
from nlu.special import ChooseNLU
from dm.status import DMStatus


class NLU:
    def __init__(self):
        self.choose = ChooseNLU()
        self.kbqa = KBQA()
        self.dbqa = DBQA()

    def predict(self, text, status: DMStatus = None):
        iri = self.choose.predict(text, status.nlg_status.last_status)
        result, status = self.kbqa.predict(text, status, other_entity_iri=iri)
        if result is None:
            result = self.dbqa.predict(text)
            status = {}
        return result, status
