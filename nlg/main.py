from nlg.kbqa import KBQAGenerator
from nlg.sample import SampleGenerator
from nlg.common import NLGResult
from dm.dp import Policy, Action


class NLG:
    def __init__(self):
        self.kbqa_generator = KBQAGenerator()
        self.sample_generator = SampleGenerator()

    def generate(self, policy: Policy) -> NLGResult:
        if policy.module == 'kbqa':
            nlg_result = self.kbqa_generator.general(policy)
        elif policy.action == Action.guide:
            nlg_result = self.sample_generator.general(policy)
        else:
            nlg_result = NLGResult()
        return nlg_result
