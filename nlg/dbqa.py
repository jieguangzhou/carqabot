from nlg.common import BaseGenerator, NLGResult



class DBQAGenerator(BaseGenerator):

    def general(self, policy):
        data = policy.data
        answer = data['answer']
        result = NLGResult(text=answer, action=policy.action)
        return result

