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


def test_matcher():
    from kbqa.common.dictionary_match import Matcher
    matcher = Matcher()
    print(matcher.match('奔驰E级 的座位数'))


def test_complex_qa():
    from kbqa.common.dictionary_match import Matcher
    matcher = Matcher()
    print(matcher.match('奔驰E级 的座位数'))

chat()
