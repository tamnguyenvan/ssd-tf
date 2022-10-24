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

class ProfilerApp():
    def __init__(self, *args, **kwargs):
        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "args: {}".format(args))

        self.profiler = TextProfiler()

    def profiler_app(self):
        app = Flask(__name__)

        @app.route('/', methods=['GET'])
        def home():
            return "Server for Text Profiler is up."

        @app.route('/get_key_phrases', methods=['POST'])
        def get_key_phrases():
            to_dict = request.get_json(force=True)
            key_phrases = self.profiler.get_key_phrases(to_dict['sequence'])
            return jsonify({"key_phrases": key_phrases})

        @app.route('/get_lexical_features', methods=['POST'])
        def get_lexical_features():
            to_dict = request.get_json(force=True)
            features = self.profiler.get_lexical_features(to_dict['sequence'])
            return jsonify({"features": features})

        @app.route('/get_stylometric_features', methods=['POST'])
        def get_stylometric_features():
            to_dict = request.get_json(force=True)
            features = self.profiler.get_stylometric_features(to_dict['sequence'])
            return jsonify({"features": features})

        return app

if __name__ == '__main__':
    app = ProfilerApp()

    app.profiler_app().run(debug=False)

    _cleanup()

