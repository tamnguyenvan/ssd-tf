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

class EmotionApp():
    def __init__(self, *args, **kwargs):
        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "args: {}".format(args))

        self.emotion = DetectEmotion()

    def emotion_app(self):
        app = Flask(__name__)

        @app.route('/', methods=['GET'])
        def home():
            return "Server for Emotion Detector is up."

        @app.route('/predict', methods=['POST'])
        def pred():
            to_dict = request.get_json(force=True)
            label_pred, scores_pred = self.emotion.pred(to_dict['sequence'],
                                                        fast=to_dict['force'])
            return jsonify({"predicted_label": label_pred, "predicted_scores": scores_pred})

        return app

if __name__ == '__main__':
    app = EmotionApp()

    app.emotion_app().run(debug=False)

    _cleanup()

