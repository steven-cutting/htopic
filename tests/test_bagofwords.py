# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/31/2017'

from tempfile import NamedTemporaryFile

import toolz as tlz
from toolz import curried as tlzc
import pytest

from h_topic_model import bagofwords as bow


@pytest.mark.parametrize("string,expected,stoplist",
                         [(u"foo", True, {u"foo", u"bar", u"baz"}),
                          # Hebrew unicode code points.
                          (u'\u05d4\u05d5\u05d0', True,
                           {u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4'}),
                          (u"foo", False, {u"bar", u"baz"}),
                          # Hebrew unicode code points.
                          (u'\u05d4\u05d5\u05d0', False,
                           {u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4'}),
                          ])
def test__token_in_stoplist(string, expected, stoplist):
    assert(bow.token_in_stoplist(string, stoplist=stoplist) == expected)


@pytest.mark.parametrize("tokens,expected,stoplist",
                         [([u"foo", u"bar", u"baz",
                            u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4'],
                           [u"baz",
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4'],
                           {u"foo", u"bar",
                            u'\u05d4\u05d5\u05d0'}),
                          ])
def test__filter_tokens(tokens, expected, stoplist):
    assert(list(bow.filter_tokens(tokens, stoplist=stoplist)) == expected)


@pytest.mark.parametrize("tokens",
                         [([[u"foo", u"bar", u"baz"],
                            [u'\u05d4\u05d5\u05d0',
                             u'\u05e6\u05d9\u05dc\u05dd',
                             u'\u05e2\u05dc\u05d9\u05d5',
                             u'\u05db\u05ea\u05d1\u05d4']]),
                          ])
def test__mk_dict(tokens):
    assert(tlz.pipe(tokens,
                    bow.mk_dict,
                    lambda d: d.values(),
                    set) ==
           tlz.pipe(tokens,
                    tlz.concat,
                    set))


@pytest.mark.parametrize("tokens,elementType,uniqueCounts",
                         [([[u"foo", u"foo", u"bar", u"baz"],
                            [u'\u05d4\u05d5\u05d0',
                             u'\u05e6\u05d9\u05dc\u05dd',
                             u'\u05e6\u05d9\u05dc\u05dd',
                             u'\u05e2\u05dc\u05d9\u05d5',
                             u'\u05db\u05ea\u05d1\u05d4']],
                           int,
                           {1, 2}),
                          ])
def test__mk_corpus(tokens, elementType, uniqueCounts):

    tknCounts = tlz.pipe(tokens,
                         bow.mk_dict,
                         lambda d: bow.mk_corpus(d, tokens),
                         tlz.concat,
                         tlzc.map(tlz.second),
                         list)

    assert(tlz.pipe(tknCounts,
                    tlzc.map(type),
                    set,
                    tlz.first) ==
           elementType)

    assert(set(tknCounts) == uniqueCounts)


@pytest.mark.parametrize("tokens",
                         [[[u"foo", u"foo", u"bar", u"baz"],
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4']],
                          ])
def test__save_and_load_dict(tokens):

    dictionary = bow.mk_dict(tokens)

    with NamedTemporaryFile() as f:
        # Eventually add more checks.
        bow.save_dict(f.name, dictionary)
        ldict = bow.load_dict(f.name)

    assert(set(ldict.values()) ==
           tlz.pipe(tokens,
                    tlz.concat,
                    set))


@pytest.mark.parametrize("tokens",
                         [[[u"foo", u"foo", u"bar", u"baz"],
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4']],
                          ])
def test__save_and_load_corpus(tokens):

    dictionary = bow.mk_dict(tokens)
    corpus = bow.mk_corpus(dictionary, tokens)

    with NamedTemporaryFile() as f:
        # Eventually add more checks.
        bow.save_corpus(f.name, corpus)
        assert(bow.load_corpus(f.name))
