# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/25/2017'
__doc__ = """
Module uses Morfessor to segment words.
"""

import logging

import morfessor
try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc

from h_topic_model import textproc_utils as tpu


LOG = logging.getLogger(__name__)


def load_morfessor_model(filename):
    io = morfessor.MorfessorIO()
    return io.read_binary_model_file(filename)


def unpack_viterbi_segment(result):
    return tlz.pipe(result,
                    tlz.first,
                    lambda segs: [segs] if type(segs) in (unicode, str) else segs,
                    lambda segs: segs if type(segs) == list else list(segs))


def mk_segmenter(model):
    """
    Returns a closure that uses the models segment methods to segment strings.
    """
    def segment(token):
        try:
            return model.segment(token)
        except KeyError:
            LOG.debug("{}  -  token missing from segment model.")
            # if the token is new
            return tlz.pipe(token,
                            model.viterbi_segment,
                            unpack_viterbi_segment)

    return segment


# For compatablity.
mk_segment_token = mk_segmenter


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

    Curried.
    """
    return tlz.pipe(txt,
                    tpu.simple_split_txt,
                    tlzc.map(mk_segmenter(model)),
                    should_flatten(flatten),
                    # tlz.concat,
                    list)


def segment_many(model, txts, flatten=True):
    return tlz.pipe(txts,
                    tlzc.map(segment_text(model, flatten=flatten)),
                    should_flatten(flatten))
