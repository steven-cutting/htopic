# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '04/13/2017'
__doc__ = """
Module for general utilies for text processing.
"""

try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc
from text2math import text2tokens as t2t


# Source: https://en.wikipedia.org/wiki/Hebrew_punctuation
HPUNCTUATION = {u"\u05f3",  # ׳  U+05f3  HEBREW PUNCTUATION GERESH
                u"\u05f4",  # ״  U+05f4  HEBREW PUNCTUATION GERSHAYIM
                u"\u0027",  # '  U+0027  APOSTROPHE
                u"\u0022",  # "  U+0022  QUOTATION MARK
                u"\u05C3",  # ׃  HEBREW PUNCTUATION SOF PASUQ
                u"\u003A",  # :  COLON
                u"\u05C0",  # ׀  HEBREW PUNCTUATION PASEQ
                u"\u007C",  # |  VERTICAL LINE
                u"\u05be",  # ־  HEBREW PUNCTUATION MAQAF
                u"\u002d",  # -  HYPHEN-MINUS
                }


def remove_h_punct(txt, punctuation=HPUNCTUATION):
    """
    Replaces punctuation, common in Hebrew, with a space.
    Source for punctuation:
        https://en.wikipedia.org/wiki/Hebrew_punctuation
    """
    return tlz.reduce(lambda string, p: string.replace(p, u" "),
                      punctuation,
                      txt)


def simple_split_txt(txt):
    """
    Simply split 'txt' on whitespace.
    Return generator of all continuous non-whitespace character
    sequences (e.g. words, punctuation, etc.).

    For best results 'txt' should be a unicode string.
    """
    return tlz.pipe(txt,
                    t2t.punct_to_space,
                    t2t.drop_punct,
                    remove_h_punct,  # rm punctuation common in Hebrew
                    lambda t: t.split(),
                    tlzc.filter(tlz.identity))  # filter out falsy values


def simple_split_many(txts):
    """
    Creates a single sequence of all of the continuous non-whitespace
    character sequences (e.g. words, punctuation, etc.) from all of the strings
    in txts.

    For best results the strings in txts should be unicode strings.
    """
    return tlz.mapcat(simple_split_txt, txts)
