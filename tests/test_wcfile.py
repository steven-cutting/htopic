# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/24/2017'


from tempfile import NamedTemporaryFile

import pytest


from h_topic_model import wcfile as wc


@pytest.mark.parametrize("tokens,expected",
                         [([u"foo", u"bar", u"baz",
                            u"foo", u"bar", ],
                           set([(2, u"foo"), (2, u"bar"), (1, u"baz")])),
                          ])
def test__count_tokens(tokens, expected):
    assert(set(wc.count_tokens(tokens)) == expected)


@pytest.mark.parametrize("tokens,expected",
                         [([(2, u"foo"), (1, u"baz"), (2, u"bar")],
                           [(2, u"foo"), (2, u"bar"), (1, u"baz")],
                           ),
                          ])
def test__sort_token_counts(tokens, expected):
    assert(wc.sort_token_counts(tokens) == expected)


@pytest.mark.parametrize("row,expected,encoding",
                         [((2, u"foo"),
                           (2, u"foo".encode("utf-8")),
                           "utf-8"),
                          ((2, u"foo"),
                           (2, u"foo".encode("ascii")),
                           "ascii"),
                          ((2, u"הוא צילם עליו כתבה"),
                           (2, u"הוא צילם עליו כתבה".encode("utf-8")),
                           "utf-8"),
                          ])
def test__encode_token_in_row(row, expected, encoding):
    assert(wc.encode_token_in_row(row, encoding=encoding) == expected)


@pytest.mark.parametrize("tokens,expected",
                         [([(2, u"foo"), (1, u"baz"), (2, u"bar")],
                           "2\tfoo\r\n1\tbaz\r\n2\tbar\r\n"
                           ),
                          ])
def test__token_counts_to_cvs(tokens, expected):
    with NamedTemporaryFile() as f:
        wc.token_counts_to_cvs(f.name, tokens)
        assert(f.read() == expected)
