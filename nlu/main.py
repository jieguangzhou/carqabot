from kbqa.main import KBQA


class NLU:
    def __init__(self):
        self.kbqa = KBQA()

    def predict(self, text, history=None):
        results, status = self.kbqa(text)
        return results, status
