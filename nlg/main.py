from nlg.kbqa import KBQAGenerator
from nlg.common import NLGResult
from dm.dp import Policy


class NLG:
    def __init__(self):
        self.kbqa_generator = KBQAGenerator()

    def generate(self, policy: Policy) -> NLGResult:
        if policy.module == 'kbqa':
            nlg_result = self.kbqa_generator.general(policy)
        else:
            nlg_result = NLGResult()
        return nlg_result
