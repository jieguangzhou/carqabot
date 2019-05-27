from nlg.common import BaseGenerator, NLGResult
from dm.dp import Policy, Action
from config import Path
import os
import random

class SampleGenerator(BaseGenerator):
    def __init__(self):
        sample_dir = Path.sample_question
        with open(os.path.join(sample_dir, 'base.txt'), 'r') as r_f:
            self.base_questions = [line.strip() for line in r_f]


    def general(self, policy):
        if policy.action == Action.guide:
            result = self.deal_base_sample(policy)
        else:
            result = NLGResult()
        return result

    def deal_base_sample(self, policy):
        questions = random.sample(self.base_questions, 3)
        text = '这个我还不清楚，你可以这样问\n' + '\n'.join(questions)
        result = NLGResult(text=text, action=policy.action)
        return result
