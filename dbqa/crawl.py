import requests
from urllib import parse
from bs4 import BeautifulSoup

from logging import getLogger

BASE_URL = 'https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&word={}'

headers = {"Accept": "text/html, application/xhtml+xml, image/jxr, */*",

           "Accept - Encoding": "gzip, deflate, br",

           "Accept - Language": "zh - CN",

           "Connection": "Keep - Alive",
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
           "referer": "baidu.com"}

logger = getLogger('crawl')


def search_docs(query):
    question_docs = []
    try:
        url = BASE_URL.format(parse.quote_plus(query))
        logger.debug(url)
        respone = requests.get(BASE_URL.format(query), headers=headers)
        bs = BeautifulSoup(respone.content.decode('gbk'), features="html.parser")

        inner = bs.select('div.list-inner')
        dls = inner[0].select('dl')
        for index, dl in enumerate(dls):
            dt = dl.select('dt')[0]
            question = dt.text.strip('\n')
            dd = dl.select('dd.answer')[0]
            answer = dd.text
            question_docs.append((question, answer))
    except:
        logger.error('网上检索失败')
    return question_docs


if __name__ == '__main__':
    for question, answer in search_docs('兰博基尼和布加迪威龙哪个快?'):
        print({'question':question, 'doc':answer})
