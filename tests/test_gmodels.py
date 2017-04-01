# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/31/2017'


import toolz as tlz
from toolz import curried as tlzc
import pytest

from h_topic_model import bagofwords as bow
from h_topic_model import gmodels as gm


def non_empty_and_no_zeros(corpus):
    return tlz.pipe(corpus,
                    tlz.concat,
                    tlz.concat,
                    tlzc.filter(tlz.identity),  # Filter out zeros
                    set,
                    bool)


@pytest.mark.parametrize("tokens",
                         [[[u"foo", u"foo", u"bar", u"baz"],
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4']],
                          ])
def test__fit_tfidif(tokens):

    dictionary = bow.mk_dict(tokens)
    corpus = list(bow.mk_corpus(dictionary, tokens))

    tfidfModel = gm.fit_tfidf(corpus, dictionary)

    assert(non_empty_and_no_zeros(tfidfModel[corpus]))


@pytest.mark.parametrize("tokens",
                         [[[u"foo", u"foo", u"bar", u"baz"],
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4']],
                          ])
def test__fit_lsi(tokens):

    dictionary = bow.mk_dict(tokens)
    corpus = list(bow.mk_corpus(dictionary, tokens))

    lsiModel = gm.fit_lsi(corpus, dictionary)

    assert(non_empty_and_no_zeros(lsiModel[corpus]))


@pytest.mark.parametrize("tokens",
                         [[[u"foo", u"foo", u"bar", u"baz"],
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4']],
                          ])
def test__fit_lda(tokens):

    dictionary = bow.mk_dict(tokens)
    corpus = list(bow.mk_corpus(dictionary, tokens))

    numTopics = 5
    ldaModel = gm.fit_lda(corpus, dictionary, random_state=2017, num_topics=numTopics)
    ldaModelMulti = gm.fit_lda(corpus, dictionary, random_state=2017, num_topics=numTopics,
                               workers=3)

    assert(ldaModel.num_topics == numTopics)
    assert(ldaModelMulti.num_topics == numTopics)

    assert(non_empty_and_no_zeros(gm.to_model_vec_space(ldaModel, corpus)))
    assert(non_empty_and_no_zeros(gm.to_model_vec_space(ldaModelMulti, corpus)))
