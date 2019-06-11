from nlg.common import BaseGenerator, NLGResult
from dm.dp import Policy, Action
import re

RE_PREDICATE_name = re.compile('predicate/([^\(\)]*)')
RE_PREDICATE_unit = re.compile('predicate/.*?\((.*?)\)')

class KBQAGenerator(BaseGenerator):

    def general(self, policy):
        if policy.action == Action.inform:
            result = self.deal_inform(policy)
        elif policy.action == Action.choose:
            result = self.deal_choose(policy)
        else:
            result = NLGResult()
        return result

    def deal_inform(self, policy):
        datas = policy.data['data']
        relation = policy.data.get('relation', '') or ''
        if relation:
            unit = RE_PREDICATE_unit.findall(relation)[0] if '(' in relation else ''
            relation = RE_PREDICATE_name.findall(relation)[0]
        else:
            unit = ''
        texts = []
        for data in datas:
            name = data.get('name', '')
            value = data.get('value')
            texts.append(' '.join([name, '的{}是'.format(relation), value + unit]))
        text = '\n'.join(texts)
        result = NLGResult(text=text, action=policy.action)
        return result

    def deal_choose(self, policy):
        datas = policy.data['data']
        relation = policy.data.get('relation', '') or ''
        if relation:
            relation = RE_PREDICATE_name.findall(relation)[0]
        texts = ['你是想问哪个车系的{}呢？'.format(relation)]
        option = []
        for n, data in enumerate(datas[:5]):
            value = data.get('value')
            iri = data.get('iri')
            if value:
                texts.append(str(n + 1) + '.' + value)
                option.append({'iri': iri, 'value': value, 'index': n + 1})
        text = '\n'.join(texts)
        result = NLGResult(text=text, action=policy.action, option=option)
        return result
