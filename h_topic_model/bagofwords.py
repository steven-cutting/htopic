# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/25/2017'
__doc__ = """
For creating bag of words representation of text for use with Gensim topic models.
"""

import itertools as itls

from gensim import corpora
try:
    import cytoolz as tlz
except ImportError:
    import toolz as tlz


STOPLIST = {u"", }


def token_in_stoplist(token):
    if token in STOPLIST:
        True
    else:
        False


def filter_tokens(tokens):
    return itls.ifilterfalse(token_in_stoplist, tokens)


def mk_dict(tokenSeqs):
    return corpora.Dictionary(tokenSeqs)


def filter_dict(dictionary, no_above=0.5, no_below=5, keep_n=100000):
    """
    !Operates in place!
    """
    dictionary.filter_extremes(no_below=no_below,
                               no_above=no_above,
                               keep_n=keep_n)
    dictionary.compactify()
    return dictionary


def mk_corpus(dictionary, tokenSeqs):
    return tlz.map(dictionary.doc2bow, tokenSeqs)


def save_dict(filename, dictionary):
    dictionary.save(filename)


def save_corpus(filename, corpus):
    """
    Save corpus to memory mapped format.
    """
    corpora.MmCorpus.serialize(filename, corpus)


def load_dict(filename):
    return corpora.Dictionary.load(filename)


def load_corpus(filename):
    """
    Load memory mapped corpus.
    """
    return corpora.MmCorpus(filename)
