from utils.common_defs import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.interfaces.nlp.interface import *

from flask import Flask, render_template, jsonify, request
import requests, json
import sys, gc

log = Log('apps')

def _cleanup():
    gc.collect()

HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

class SentimentApp():
    def __init__(self, *args, **kwargs):
        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "args: {}".format(args))

        if args:
            if len(args[0]) > 1:
                self.sentiment = DetectSentiment(lang=args[0][0], hypothesis_template=args[0][1])
            else:
                self.sentiment = DetectSentiment(lang=args[0][0])
        else:
            self.sentiment = DetectSentiment()

    def sentiment_app(self):
        app = Flask(__name__)

        @app.route('/', methods=['GET'])
        def home():
            return "Server for Sentiment Detector is up."

        @app.route('/predict', methods=['POST'])
        def pred():
            to_dict = request.get_json(force=True)
            label_pred, scores_pred, lang = self.sentiment.pred(to_dict['sequence'])
            return jsonify({"predicted_label": label_pred, "predicted_scores": scores_pred})

        return app

if __name__ == '__main__':
    if len(sys.argv) > 1:
        app = SentimentApp(sys.argv[1:])
    else:
        app = SentimentApp()

    app.sentiment_app().run(debug=False)

    _cleanup()

