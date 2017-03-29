# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/24/2017'


from operator import eq

import toolz as tlz
import pytest


from h_topic_model import wcfile as wc


c_eq = tlz.curry(eq)


@pytest.mark.parametrize("string,expected",
                         [(u"foo bar baz", [u"foo", u"bar", u"baz"]),
                          (u"foo\n bar\tbaz\r\n", [u"foo", u"bar", u"baz"]),
                                                 # Hebrew unicode code points.
                          (u"הוא צילם עליו כתבה", [u'\u05d4\u05d5\u05d0',
                                                  u'\u05e6\u05d9\u05dc\u05dd',
                                                  u'\u05e2\u05dc\u05d9\u05d5',
                                                  u'\u05db\u05ea\u05d1\u05d4'])
                          ])
def test__simple_split_txt(string, expected):
    assert(tlz.pipe(string,
                    wc.simple_split_txt,
                    list,
                    c_eq(expected)))

    
@pytest.mark.parametrize("string,expected",
                         [([u"foo bar baz",
                            u"foo\n bar\tbaz\r\n",
                            u"הוא צילם עליו כתבה",
                            ],
                           [u"foo", u"bar", u"baz",
                            u"foo", u"bar", u"baz",
                            # Hebrew unicode code points.
                            u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4',
                            ])
                          ])
def test__simple_split_many(string, expected):
    assert(tlz.pipe(string,
                    wc.simple_split_many,
                    list,
                    c_eq(expected)))


@pytest.mark.parametrize("tokens,expected",
                         [([u"foo", u"bar", u"baz",
                            u"foo", u"bar",],
                           {(2, u"foo"), (2, u"bar"), (1, u"baz")}),
                          ])
def test__count_tokens(tokens, expected):
    assert(set(wc.count_tokens(tokens)) == expected)
