# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/25/2017'
__doc__ = """
Module uses Morfessor to segment words.
"""

try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc

import morfessor


# MODEL_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/model/model.bin"
# SAMPLE_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/toy50k/1.1580753.txt.iso"

def split_text(txt):
    """
    Simply split 'txt' on whitespace.
    Return generator of all continuous non-whitespace character
    sequences (e.g. words, punctuation, etc.).

    For best results 'txt' should be a unicode string.
    """
    return (token for token in txt.split())


def load_morfessor_model(filename):
    io = morfessor.MorfessorIO()
    return io.read_binary_model_file(filename)


def segment_token(model):
    """
    Returns a closure that uses the models segment methods to segment strings.
    """
    def segment(token):
        try:
            return model.segment(token)
        except KeyError:
            # if the token is new
            return model.viterbi_segment(token)


def should_flatten(flatten=True):
    """
    If True it returns toolz concat function.
    If False it returns the identity funcion.
    """
    if flatten:
        return tlz.concat
    else:
        return tlz.identity


@tlz.curry
def segment_text(model, txt, flatten=True):
    """
    Splits the text into tokens and then segments the tokens.

    Curried
    """
    segment_f = segment_token(model)
    return tlz.pipe(txt,
                    split_text,
                    tlzc.map(segment_f),
                    should_flatten(flatten),
                    list)


def segment_many(model, txts):
    return map(segment_text(model), txts)
