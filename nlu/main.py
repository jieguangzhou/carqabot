from kbqa.main import KBQA
from nlu.special import ChooseNLU
from dm.status import DMStatus


class NLU:
    def __init__(self):
        self.choose = ChooseNLU()
        self.kbqa = KBQA()

    def predict(self, text, status: DMStatus = None):
        iri = self.choose.predict(text, status.nlg_status.last_status)
        result, status = self.kbqa.predict(text, status, other_entity_iri=iri)
        return result, status
