import random
import logging
import time
import logger_color
from pprint import pprint

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def test_qa():
    from carbot.main import CarBot, Request
    carbot = CarBot()

    text = '奔驰E级多少钱'
    request = Request(id='123', text=text)
    results = carbot.predict(request)
    print(results)


def chat():
    from carbot.main import CarBot, Request
    carbot = CarBot()
    id = random.randint(1, 99999999)
    while True:
        print('\n')
        time.sleep(0.05)
        text = input('user: ')
        request = Request(id=id, text=text)
        results = carbot.predict(request)
        time.sleep(0.05)
        print('bot: ')
        print(results)


def test_predicate_match():
    from model.text_classification import Predictor
    from config import Path
    import pandas as pd
    test_data_path = 'data/train/kbqa/predicate_match/test.xlsx'
    df = pd.read_excel(test_data_path)
    predictor = Predictor(Path.relation_match_model)
    trues = []
    pred = []
    from tqdm import tqdm
    for n, row in tqdm(df.iterrows(), total=len(df)):

        question = row['question']
        text = row['predicate']
        label, _ = predictor.predict_text(text_a=question, text_b=text)
        trues.append(row['label'])
        pred.append(label)
        if len(pred) >= 2000:
            break
    from sklearn.metrics import precision_recall_fscore_support
    result = precision_recall_fscore_support(trues, pred, average=None,  labels=['Yes', 'No'])
    print(result)

test_predicate_match()
