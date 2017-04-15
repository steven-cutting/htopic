# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/31/2017'


import pytest

from h_topic_model import segment as seg
from tests import t_utils as tu


@pytest.mark.parametrize("string,expected",
                         [(u"foobarbaz", [u"foo", u"bar", u"baz"]),
                          (u"foo", [u"foo"]),
                          (u"הוא צילם עליו כתבה", [u"הוא צילם עליו כתבה"]),
                          ])
def test__mk_segmenter(string, expected):
    model = tu.MockMorfessorSegmentModel()
    segment = seg.mk_segmenter(model)
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
    model = tu.MockMorfessorSegmentModel()
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
    model = tu.MockMorfessorSegmentModel()
    assert(list(seg.segment_many(model, strings, flatten=flatten)) == expected)
