# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '04/08/2017'
__doc__ = """
Module uses Morfessor to train a segmentation model.
"""

import locale
import logging
import math
import random

import morfessor
from morfessor import MorfessorIO
from morfessor import BaselineModel
from morfessor.exception import ArgumentException
import arrow
try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc


LOG = logging.getLogger(__name__)


# L = lambda : load_data("/Users/steven_c/projects/h_topic_model/data/toy50k/flatcat/wc.csv")
# H = lambda a: tlz.map(tlz.second, tlz.filter(tlz.first, a))


def load_data(trainfilespath, encoding=locale.getpreferredencoding(),
              separator=None, cseparator=r"\s+", lowercase=False):
    # TODO (sc) DOCS!
    io = MorfessorIO(encoding=encoding,
                     # compound_separator=args.cseparator,
                     compound_separator=cseparator,
                     atom_separator=separator,
                     lowercase=lowercase)
    return io.read_corpus_file(trainfilespath)


def mkmodel(data, trainmode='init+batch', forcesplit=['-'], corpusweight=1.0, skips=False,
            nosplit=None, freqthreshold=1, splitprob=None, algorithm="recursive",
            finish_threshold=0.005, maxepochs=None, dampening="ones",
            viterbismooth=0, viterbimaxlen=30, randseed=None):
    """
    splitprob - initialize new words by random splitting using the given
                split probability (default no splitting)
    algorithm - algorithm type ('recursive', 'viterbi'; default 'recursive')
    dampening - frequency dampening for training data ('none', 'log', or "
                'ones'; default 'ones')
    """
    # TODO (sc) DOCS!
    random.seed(randseed)
    ts = arrow.now()

    model = BaselineModel(forcesplit_list=forcesplit,
                          corpusweight=corpusweight,
                          use_skips=skips,
                          nosplit_re=nosplit)

    # Set frequency dampening function
    if dampening == 'none':
        dampfunc = None
    elif dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % dampening)

    # Set algorithm parameters
    if algorithm == 'viterbi':
        algparams = (viterbismooth, viterbimaxlen)
    else:
        algparams = ()

    # if trainmode == 'init+batch':
    c = model.load_data(data, freqthreshold, dampfunc, splitprob)
    e, c = model.train_batch(algorithm, algparams, finish_threshold, maxepochs)

    te = arrow.now()
    LOG.info("Epochs: {}".format(e))
    LOG.info("Final cost: {}".format(c))
    LOG.info("Training time: {}".format(te - ts))

    return model


def save_model(savefile, model):
    # TODO (sc) DOCS!
    io = MorfessorIO()
    io.write_binary_model_file(savefile, model)


def save_segmentations(savefile, model):
    # TODO (sc) DOCS!
    io = MorfessorIO()
    io.write_segmentation_file(savefile, model.get_segmentations())
