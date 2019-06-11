import random
import logging
import logger_color
from gevent import monkey

monkey.patch_all()
from gevent import pywsgi
from flask import request, Flask, jsonify
from carbot.main import CarBot, Request

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger('server')
app = Flask('CarBot')
app.config['JSON_AS_ASCII'] = False

carbot = CarBot()


@app.route('/chat')
def chat():
    user_id = request.args.get('user_id', random.randint(1, 99999999))
    question = request.args.get('question', '')
    bot_request = Request(id=user_id, text=question)
    try:
        result = carbot.predict(bot_request)
        success = 1
        message = ''
    except Exception as e:
        message = str(e)
        logger.error(message, exc_info=True)
        result = ''
        success = 0

    response = jsonify(result=result, success=success, id=user_id, question=question, message=message)
    return response


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 10000), app)
    server.serve_forever()
