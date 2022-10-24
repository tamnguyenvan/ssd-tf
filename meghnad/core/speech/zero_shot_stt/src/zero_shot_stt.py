#######################################################################################################################
# Zero-shot automatic speech recognition (speech to text).
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.speech.zero_shot_stt.cfg.config import ZeroShotSTTConfig

import sys, os, shutil, librosa, ffmpeg
import soundfile as sf
from fastpunct import FastPunct
#from rpunct import RestorePuncts

import torch
from transformers import pipeline
from pyannote.audio import Pipeline

log = Log()

@class_header(
description='''
Zero-shot multi-language automatic speech recognition (speech to text).''')
class ZeroShotSTT():
    def __init__(self, num_speakers_known:bool=False, *args, **kwargs):
        if 'mode' in kwargs:
            self.configs = ZeroShotSTTConfig(MeghnadConfig(), mode=kwargs['mode'])
        else:
            self.configs = ZeroShotSTTConfig(MeghnadConfig())

        if 'lang' in kwargs:
            self._load_model(kwargs['lang'], num_speakers_known=num_speakers_known)
        else:
            self._load_model(num_speakers_known=num_speakers_known)

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Extract text from speech (WAV file).''',
        arguments='''
        wav_file_path: path to the WAV file input to be recognized
        num_speakers [optional]: number of speakers''',
        returns='''
        extracted text''')
    def pred(self, wav_file_path:str, num_speakers:int=None) -> str:
        seq = ""
        default_sample_rate = self.configs.get_zsstt_settings('sample_rate')
        default_slice_duration = self.configs.get_zsstt_settings('stream_split_slice_len_in_secs')

        dir_to_dump_tmp_proc = self.configs.get_meghnad_configs('INT_PATH') + 'zero_shot_stt/'
        if os.path.exists(dir_to_dump_tmp_proc):
            shutil.rmtree(dir_to_dump_tmp_proc)
        os.mkdir(dir_to_dump_tmp_proc)

        sample_rate = librosa.get_samplerate(wav_file_path)
        #duration = librosa.get_duration(filename=wav_file_path, sr=sample_rate)

        stream = librosa.stream(wav_file_path,
                                block_length=default_slice_duration,
                                frame_length=sample_rate,
                                hop_length=sample_rate)

        for speech in stream:
            if len(speech.shape) > 1:
                speech = speech[:, 0] + speech[:, 1]

            if sample_rate != default_sample_rate:
                speech = librosa.resample(speech, sample_rate, default_sample_rate)

            file_path = dir_to_dump_tmp_proc + "tmp.wav"
            sf.write(file_path, speech, default_sample_rate, format='WAV')

            sliced_output = self.model(file_path, return_timestamps='word')

            if sliced_output['text']:
                if self.configs.diarization:
                    chunks = sliced_output['chunks']
                    if num_speakers:
                        speakers = self.diarization_model(file_path, num_speakers=num_speakers)
                    else:
                        speakers = self.diarization_model(file_path)

                    chunk_idx = 0
                    for turn, _, speaker in speakers.itertracks(yield_label=True):
                        seq_diarized = ""
                        while chunk_idx < len(chunks) and chunks[chunk_idx]['timestamp'][1] <= turn.end:
                            seq_diarized += chunks[chunk_idx]['text'] + ' '
                            chunk_idx += 1

                        seq_diarized = self._punct_case_corrector(seq_diarized.lower().strip())
                        if seq_diarized:
                            seq += "{}: ''{}'' from {:.3f}-{:.3f}".format(speaker, seq_diarized,
                                                                          turn.start, turn.end) +\
                                   self.configs.get_zsstt_settings('end_of_slice_marker')
                else:
                    seq += self._punct_case_corrector(sliced_output['text'].lower()) +\
                           self.configs.get_zsstt_settings('end_of_slice_marker')

        seq = seq.strip()

        shutil.rmtree(dir_to_dump_tmp_proc)

        return seq

    @method_header(
        description='''
                Convert between different audio file formats.''',
        arguments='''
                in_file_path: input file path
                out_file_path: output file path''')
    def convert(self, in_file_path:str, out_file_path:str):
        ffmpeg.input(in_file_path).output(out_file_path).run()

    # Load appropriate pipeline
    def _load_model(self, lang:str=None, num_speakers_known:bool=False):
        if lang:
            model_config = self.configs.get_zsstt_model(lang=str(lang))
        else:
            model_config = self.configs.get_zsstt_model()

        self.model = pipeline('automatic-speech-recognition',
                              model=model_config['ckpt'],
                              feature_extractor=model_config['ckpt'])

        self.punctuators = []
        if model_config['lang'] == 'en':
            self.punctuators.append(FastPunct(checkpoint_local_path=self.configs.get_meghnad_configs('FASTPUNCT_PATH')))
            #if torch.cuda.is_available():
            #    self.punctuators.append(RestorePuncts())

        if self.configs.diarization:
            model_config = self.configs.get_zsstt_diarization_model(num_speakers_known=num_speakers_known)
            self.diarization_model = Pipeline.from_pretrained(model_config['ckpt'],
                                                              cache_dir=self.configs.get_meghnad_configs('PYANNOTE_PATH'))

    def _punct_case_corrector(self, seq:str) -> str:
        if self.punctuators:
            seq = self.punctuators[0].punct(seq)
            #if torch.cuda.is_available():
            #    seq = self.punctuators[1].punctuate(seq)
        return seq

if __name__ == '__main__':
    pass

