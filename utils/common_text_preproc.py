#######################################################################################################################
# Common text preprocessing. Intended for use internally within Ixolerator.
#
#  Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log

import sys, string, nltk, re

log = Log()

@method_header(
    description='''
            Strip off punctuations.''',
    arguments='''
            sequence: text input
            punctuations [optional]: list of punctuations to be removed''',
    returns='''
            transformed text''')
def strip_punctuations(sequence:str, punctuations:[str]=[]) -> str:
    if not punctuations:
        punctuations = string.punctuation
    sequence = sequence.translate(str.maketrans('', '', punctuations))

    sequence = re.sub('\W', ' ', sequence)
    sequence = re.sub('\s+', ' ', sequence)

    return sequence

@method_header(
    description='''
            Strip off URLs.''',
    arguments='''
            sequence: text input''',
    returns='''
            transformed text''')
def strip_urls(sequence: str) -> str:
    sequence = re.sub(r'http\S+', '', sequence)
    sequence = re.sub(r'www\S+', '', sequence)

    return sequence

@method_header(
    description='''
            Strip off stop words.''',
    arguments='''
            sequence: text input
            stop_words [optional]: list of stop words to be removed''',
    returns='''
            transformed text''')
def strip_stop_words(sequence:str, stop_words:[str]=[]) -> str:
    if not stop_words:
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words = [word.lower() for word in stop_words]
    sequence = ' '.join([word for word in sequence.split() if word.lower() not in stop_words])
    return sequence

@method_header(
    description='''
            Strip off digits.''',
    arguments='''
            sequence: text input
            digits [optional]: list of digits to be removed''',
    returns='''
            transformed text''')
def strip_digits(sequence:str, digits:str=None) -> str:
    if not digits:
        digits = '0123456789'
    sequence = sequence.translate({ord(ch): None for ch in digits})
    return sequence

@method_header(
    description='''
            Clean abbreviations.''',
    arguments='''
            sequence: text input''',
    returns='''
            transformed text''')
def clean_abbreviations(sequence:str) -> str:
    sequence = re.sub(r"what's", "what is ", sequence)
    sequence = re.sub(r"What's", "What is ", sequence)

    sequence = re.sub(r"where's", "where is ", sequence)
    sequence = re.sub(r"Where's", "Where is ", sequence)

    sequence = re.sub(r"why's", "why is ", sequence)
    sequence = re.sub(r"Why's", "Why is ", sequence)

    sequence = re.sub(r"who's", "who is ", sequence)
    sequence = re.sub(r"Who's", "Who is ", sequence)

    sequence = re.sub(r"how's", "how is ", sequence)
    sequence = re.sub(r"How's", "How is ", sequence)

    sequence = re.sub(r"that's", "that is ", sequence)
    sequence = re.sub(r"That's", "That is ", sequence)

    sequence = re.sub(r"it's", "it is ", sequence)
    sequence = re.sub(r"It's", "It is ", sequence)

    sequence = re.sub(r"i'm", "i am ", sequence)
    sequence = re.sub(r"I'm", "I am ", sequence)

    sequence = re.sub(r"can't", "can not ", sequence)
    sequence = re.sub(r"Can't", "Can not ", sequence)

    sequence = re.sub(r"shan't", "shall not ", sequence)
    sequence = re.sub(r"Shan't", "Shall not ", sequence)

    sequence = re.sub(r"won't", "will not ", sequence)
    sequence = re.sub(r"Won't", "Will not ", sequence)

    sequence = re.sub(r"ain't", "am not ", sequence)
    sequence = re.sub(r"Ain't", "Am not ", sequence)

    sequence = re.sub(r"\'s", " ", sequence)
    sequence = re.sub(r"\'ve", " have ", sequence)
    sequence = re.sub(r"n't", " not ", sequence)
    sequence = re.sub(r"\'re", " are ", sequence)
    sequence = re.sub(r"\'d", " would ", sequence)
    sequence = re.sub(r"\'ll", " will ", sequence)
    sequence = re.sub(r"\'scuse", " excuse ", sequence)

    return sequence

@method_header(
    description='''
            Given a list of strings, find a subset that subsumes others.''',
    arguments='''
            sequences: lost of strings''',
    returns='''
            dictionary containing subsuming strings (with key same as its original index) and a list of 2 member tuples 
            containing original indices of the strings it subsumed and those strings''')
def get_subsuming_strings(sequences:[str]) -> dict:
    sequences = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)
    matches = {}

    for idx, elem in enumerate(sequences):
        match = {}
        match['subsuming_seq'] = elem[1]
        match['subsumed_seqs'] = []

        for x in sequences[idx+1: ]:
            if elem[1].lower().find(strip_punctuations(x[1].lower())) != -1:
                match['subsumed_seqs'].append(x)
                sequences.remove(x)

        if not match['subsumed_seqs']:
            del match['subsumed_seqs']

        matches[elem[0]] = match

    return matches

@method_header(
    description='''
            Identify names in a text file.''',
    arguments='''
            sequence: text from which the names have to be identified
            names: dictionary of names which has to be identified
            multi_word_name_sep [optional]: If multi-word names have a common seperator within them, then specify''',
    returns='''
            dictionary containing the names as keys and the regex span start, span end (both relative w.r.t each para), 
            text and para id as values''')
def identify_names(sequence:str, names:{str: [str]}, multi_word_name_sep:str=" ") -> {str: [(int, int, str, int)]}:
    pattern = _identify_names_prep(names, multi_word_name_sep)

    identified_names = {}
    paras = [para for para in sequence.splitlines() if len(para) > 1]

    for key in pattern.keys():
        identified_name_list = []

        for style in pattern[key]:

            for para_id, para in enumerate(paras):

                for match in re.finditer(style, para):
                    span = list(match.span())
                    span.append(match.group())
                    span.append(para_id)
                    identified_name_list.append(tuple(span))

        identified_names[key] = identified_name_list

    return identified_names

# Helper function to preprocess the names
def _identify_names_prep(dictionary:{str: [str]}, multi_word_name_sep:str) -> {str: [str]}:
    processed_dictionary = {}  # dictionary to store the pre-processed names

    for key in dictionary.keys():
        names = dictionary[key]

        temp = []     # temporary list to store the pre-processed names
        pattern = []  # list to store the final pre-processed names

        # loop to pre-process the names (except changing the letter case)
        for name in names:
            temp.append("".join(name.split(multi_word_name_sep)))   # join the seperated individual names without space
            temp.append("-".join(name.split(multi_word_name_sep)))  # join the seperated individual names with '-'
            temp.append(" ".join(name.split(multi_word_name_sep)))  # join the seperated individual names with space

            for word in name.split(multi_word_name_sep):
                temp.append(word)  # gets the split individual names

                final_name_list = list(set(names + temp))  # final list having the original name and pre-processed name

        # loop to have different combination of letter cases for each name
        for text in final_name_list:
            pattern.append(text)
            pattern.append(text.lower())
            pattern.append(text[0] + text[1:].lower())

        pattern = list(set(pattern))

        processed_dictionary[key] = pattern

    return processed_dictionary

if __name__ == '__main__':
    pass

