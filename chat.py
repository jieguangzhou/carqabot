import random
import logging
import time
import logger_color
from pprint import pprint

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

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


chat()
