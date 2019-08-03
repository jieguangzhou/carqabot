from model.mrc import Predictor
from config import Path
from dbqa.crawl import search_docs

from logging import getLogger

logger = getLogger('DBQA')


class DBQA:
    def __init__(self):
        self.mrc = Predictor(Path.mrc_model)

    def predict(self, text):
        question_docs = search_docs(text)
        answer = None
        pro = 0.0
        for question, doc in question_docs:
            answer, pro = self.mrc.predict_text_docs(text, [doc])
            if answer and pro >= 0.5 and answer not in {'。'}:
                logger.info('answer: {}, pro: {} \nquestion:{} doc: {}'.format(answer, pro, question, doc))
                break
        if answer is not None:
            result = {
                'answer': answer,
                'confidence': pro,
                'module': 'dbqa'
            }
        else:
            result = None
        return result


if __name__ == '__main__':
    dbqa = DBQA()
    print(dbqa.predict('兰博基尼和布加迪威龙哪个快?'))
