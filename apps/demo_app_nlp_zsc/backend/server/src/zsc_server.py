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

class ZSCApp():
    def __init__(self, *args, **kwargs):
        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "args: {}".format(args))

        if args:
            if len(args[0]) > 1:
                self.zsc = ZeroShotClf(lang=args[0][0], hypothesis_template=args[0][1])
            else:
                self.zsc = ZeroShotClf(lang=args[0][0])
        else:
            self.zsc = ZeroShotClf()

    def zsc_app(self):
        app = Flask(__name__)

        @app.route('/', methods=['GET'])
        def home():
            return "Server for Zero-Shot Multi-label Classifier is up."

        @app.route('/get_languages_supported', methods=['GET'])
        def get_langs_supported():
            result = self.zsc.get_langs_supported()
            return jsonify({"languages_supported": result})

        @app.route('/set_language', methods=['POST'])
        def set_lang():
            to_dict = request.get_json(force=True)
            result = self.zsc.set_lang(to_dict['lang'])
            if result:
                return jsonify({"set_language": result})
            else:
                return "Language auto detection mode is set up."

        @app.route('/set_language_auto_detection_mode', methods=['GET'])
        def set_auto_detect_lang():
            _ = self.zsc.set_lang()
            return "Language auto detection mode is set up."

        @app.route('/predict', methods=['POST'])
        def pred():
            to_dict = request.get_json(force=True)
            label_pred, scores_pred, lang, _ = self.zsc.pred(to_dict['sequence'],
                                                             to_dict['candidate_labels'],
                                                             to_dict['multi_label'])
            return jsonify({"predicted_label": label_pred, "predicted_scores": scores_pred, "language": lang})

        @app.route('/predict_explain', methods=['POST'])
        def pred_explain():
            to_dict = request.get_json(force=True)
            _, scores_pred, _, attributions = self.zsc.pred(to_dict['sequence'],
                                                            to_dict['candidate_labels'],
                                                            to_dict['multi_label'],
                                                            explain=True)
            return jsonify({"predicted_scores": scores_pred, "attributions": attributions})

        @app.route('/label_explain', methods=['POST'])
        def label_explain():
            to_dict = request.get_json(force=True)
            attributions = self.zsc.label_explain(to_dict['sequence'], to_dict['label'])
            return jsonify({"attributions": attributions})

        return app

if __name__ == '__main__':
    if len(sys.argv) > 1:
        app = ZSCApp(sys.argv[1:])
    else:
        app = ZSCApp()

    app.zsc_app().run(debug=False)

    _cleanup()

