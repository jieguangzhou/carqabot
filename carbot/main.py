from nlu.main import NLU
from dm.main import DM
from nlg.main import NLG


class Request:
    def __init__(self, id, text):
        self.id = id
        self.text = text


class CarBot:
    def __init__(self):
        self.nlu = NLU()
        self.dm = DM()
        self.nlg = NLG()

    def predict(self, request: Request):
        history_status = self.dm.get_history_status(request.id)
        nlu_result, nlu_status = self.nlu.predict(text=request.text, status=history_status)
        self.dm.add_nlu_status(request.id, nlu_status)
        policy = self.dm.get_policy(nlu_result, history_status)
        nlg_result = self.nlg.generate(policy)
        self.dm.add_nlg_status(request.id, nlg_result)
        return nlg_result.text
