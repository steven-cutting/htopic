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


def load_data(trainfilespath, encoding=locale.getpreferredencoding(),
              separator=None, cseparator=r"\s+", lowercase=False):
    # TODO (sc) DOCS!
    io = MorfessorIO(encoding=encoding,
                     # compound_separator=args.cseparator,
                     compound_separator=cseparator,
                     atom_separator=separator,
                     lowercase=lowercase)
    return io.read_corpus_list_files([trainfilespath, ])
    # return io.read_corpus_file(trainfilespath)


def mk_frequency_dampening_func(dampening):
    """
    Return frequency dampening function based on dampening.
    """
    if dampening == 'none':
        return None
    elif dampening == 'log':
        return lambda x: int(round(math.log(x + 1, 2)))
    elif dampening == 'ones':
        return lambda x: 1
    else:
        raise ArgumentException("unknown dampening type {}".format(dampening))


def new_model(forcesplit=['-'], corpusweight=1.0, skips=False, nosplit=None, **kwargs):
    """
    forcesplit - force split on given atoms (default '-'). The list argument
                 is a string of characthers, use '' for no forced splits.
    corpusweight - Corpus weight parameter (default 1.0); sets the initial
                   value if other tuning options are used
    skips - use random skips for frequently seen compounds to speed up training
    nosplit - if the expression matches the two surrounding characters, do
              not allow splitting (default None)
    """
    return BaselineModel(forcesplit_list=forcesplit,
                         corpusweight=corpusweight,
                         use_skips=skips,
                         nosplit_re=nosplit)


@tlz.curry
def load_annots_to_model(filename, model, analysisseparator=',', encoding="utf-8",
                         cseparator=' ', annotationweight=None, **kwargs):
    """
    Mutates model!

    model - The morfessor model opbject to add annotations to.
    filename - Load annotated data for semi-supervised learning.
    encoding - File encoding, default 'utf-8'.
    analysisseparator - Separator for different analyses in an annotation file.
                        Use NONE for only allowing one analysis per line
    cseparator - Construction separator for test segmentation files.
    annotationweight - corpus weight parameter for annotated data (if unset,
                       the weight is set to balance the number of tokens in
                       annotated and unannotated data sets)
    """
    analysis_sep = (analysisseparator
                    if analysisseparator != 'NONE' else None)

    io = MorfessorIO(encoding=encoding,
                     compound_separator=cseparator)

    annotations = io.read_annotations_file(filename,
                                           analysis_sep=analysis_sep)
    # 364
    model.set_annotations(annotations, annotationweight)

    return model


@tlz.curry
def maybe_load_annots_to_model(filename, model, **kwargs):
    """
    Same as load_annots_to_model except that if 'filename' is None, it will simply
    return the model.
    """
    if filename is not None:
        return load_annots_to_model(filename, model, **kwargs)
    else:
        return model


@tlz.curry
def fit_model(data, model, dampening="ones", algorithm="recursive", viterbismooth=0,
              viterbimaxlen=30, freqthreshold=1, splitprob=None,
              finish_threshold=0.005, maxepochs=None, **kwargs):
    """
    data - Morfessor corpus object.
    dampening - frequency dampening for training data ('none', 'log', or "
                'ones'; default 'ones')
    algorithm - algorithm type ('recursive', 'viterbi'; default 'recursive')
    viterbismooth - additive smoothing parameter for Viterbi training and
                    segmentation (default 0)
    viterbimaxlen - maximum construction length in Viterbi training and
                    segmentation (default 30)
    freqthreshold - compound frequency threshold for batch training (default 1)
    splitprob - initialize new words by random splitting using the given
                split probability (default no splitting)
    finish_threshold - Stopping threshold. Training stops when the improvement
                       of the last iteration is smaller then finish_threshold *
                       #boundaries; (default 0.005)
    maxepochs - hard maximum of epochs in training
    """
    # Set frequency dampening function
    dampfunc = mk_frequency_dampening_func(dampening)

    # Set algorithm parameters
    if algorithm == 'viterbi':
        algparams = (viterbismooth, viterbimaxlen)
    else:
        algparams = ()

    c = model.load_data(data, freqthreshold, dampfunc, splitprob)
    epochs, cost = model.train_batch(algorithm, algparams, finish_threshold, maxepochs)

    LOG.info("Epochs: {}".format(epochs))
    LOG.info("Final cost: {}".format(cost))

    return model


def mkmodel(data, annotfile=None, **kwargs):
    """
    annotfile - File path for annotation file.
    """
    return tlz.pipe(new_model(**kwargs),
                    maybe_load_annots_to_model(annotfile, **kwargs),
                    fit_model(data, **kwargs))


def save_model(savefile, model):
    # TODO (sc) DOCS!
    io = MorfessorIO()
    io.write_binary_model_file(savefile, model)


def save_segmentations(savefile, model):
    # TODO (sc) DOCS!
    io = MorfessorIO()
    io.write_segmentation_file(savefile, model.get_segmentations())
