# -*- coding: utf-8 -*-

__title__ = 'morphoff'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '04/13/2017'


EXSEGDICT = {u"foo": [u"foo"],
             u"bar": [u"bar"],
             u"baz": [u"baz"],
             u"foobarbaz": [u"foo", u"bar", u"baz"]}


class MockFlatCatSegmentModel(object):
    def __init__(self, segDict=EXSEGDICT):
        self.segDict = segDict

    def viterbi_segment(self, token):
        try:
            return (self.segDict[token], 111)
        except KeyError:
            return (token, 111)


class MockMorfessorSegmentModel(object):
    def __init__(self, segDict=EXSEGDICT):
        self.segDict = segDict

    def segment(self, token):
        return self.segDict[token]

    def viterbi_segment(self, token):
        return [token, ]
