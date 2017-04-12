# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/31/2017'


import toolz as tlz
import pytest

from h_topic_model import segment as seg


class MockSegmentModel(object):
    def __init__(self, segDict={u"foo": [u"foo"], u"bar": [u"bar"], u"baz": [u"baz"],
                                u"foobarbaz": [u"foo", u"bar", u"baz"]}):
        self.segDict = segDict

    def segment(self, token):
        return self.segDict[token]

    def viterbi_segment(self, token):
        return [token, ]


@pytest.mark.parametrize("string,expected",
                         [(u"foo bar baz", [u"foo", u"bar", u"baz"]),
                          (u"foo\n bar\tbaz\r\n", [u"foo", u"bar", u"baz"]),
                          (u"הוא צילם עליו כתבה",
                          # Hebrew unicode code points.
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4'])
                          ])
def test__split_txt(string, expected):
    assert(tlz.pipe(string,
                    seg.split_text,
                    list) ==
           expected)


@pytest.mark.parametrize("string,expected",
                         [(u"foobarbaz", [u"foo", u"bar", u"baz"]),
                          (u"foo", [u"foo"]),
                          (u"הוא צילם עליו כתבה", [u"הוא צילם עליו כתבה"]),
                          ])
def test__mk_segment_token(string, expected):
    model = MockSegmentModel()
    segment = seg.mk_segment_token(model)
    assert(segment(string) == expected)


@pytest.mark.parametrize("string,expected",
                         [(u"foo bar baz", [u"foo", u"bar", u"baz"]),
                          (u"foo", [u"foo"]),
                          (u"הוא צילם עליו כתבה",
                           [u'\u05d4\u05d5\u05d0',
                            u'\u05e6\u05d9\u05dc\u05dd',
                            u'\u05e2\u05dc\u05d9\u05d5',
                            u'\u05db\u05ea\u05d1\u05d4']),
                          ])
def test__segment_text(string, expected):
    model = MockSegmentModel()
    assert(seg.segment_text(model, string) == expected)


@pytest.mark.parametrize("strings,expected,flatten",
                         [([u"foo bar baz", u"foo", u"הוא צילם עליו כתבה"],
                           [[[u"foo"], [u"bar"], [u"baz"]],
                            [[u"foo"], ],
                            [[u'\u05d4\u05d5\u05d0'],
                             [u'\u05e6\u05d9\u05dc\u05dd'],
                             [u'\u05e2\u05dc\u05d9\u05d5'],
                             [u'\u05db\u05ea\u05d1\u05d4']]],
                           False),
                          ([u"foo bar baz", u"foo"],
                           [u"foo", u"bar", u"baz",
                            u"foo"],
                           True),
                          ])
def test__segment_many(strings, expected, flatten):
    model = MockSegmentModel()
    assert(list(seg.segment_many(model, strings, flatten=flatten)) == expected)
