from logging import getLogger
import re

logger = getLogger('ChooseNLU')

class ChooseNLU:
    mapping = {
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
    }

    def __init__(self):
        pass

    def predict(self, text, nlg_status):
        if not nlg_status:
            return None
        options = nlg_status.data.get('option', [])
        iri = self.match_text(text, options) or self.match_index(text, options)
        logger.info(iri)
        return iri

    def match_text(self, text, options):
        iri = None
        for option in options:
            if text in option['value']:
                iri = option['iri']
                break
        return iri

    def match_index(self, text, options):
        text = text.strip()
        r1 = re.findall('第([1-5一二三四五])', text)
        r2 = re.findall('^[1-5一二三四五]$', text)
        r = r1 or r2
        iri = None
        if r:
            r_num = int(self.mapping.get(r[0], r[0]))
            iri = [option['iri'] for option in options if option['index'] == r_num]
            if iri:
                iri = iri[0]
        return iri
