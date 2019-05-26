DEFAULT_TEXT = '这个我还不清楚'


class NLGResult:
    def __init__(self, text: str = DEFAULT_TEXT, action: str = '', **kwargs):
        self.text = text
        self.action = action
        self.data = kwargs

    def __str__(self):
        return str(self.__dict__)


class BaseGenerator:
    def general(self, *args, **kwargs) -> NLGResult:
        raise NotImplementedError
