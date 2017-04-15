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
except ImportError:
    import toolz as tlz


punctuation = {u"\u05f3",  # ׳  U+05f3  HEBREW PUNCTUATION GERESH
               u"\u05f4",  # ״  U+05f4  HEBREW PUNCTUATION GERSHAYIM
               u"\u0027",  # '  U+0027  APOSTROPHE
               u"\u0022",  # "  U+0022  QUOTATION MARK
               }


def simple_split_txt(txt):
    """
    Simply split 'txt' on whitespace.
    Return generator of all continuous non-whitespace character
    sequences (e.g. words, punctuation, etc.).

    For best results 'txt' should be a unicode string.
    """
    return (token for token in txt.split())  # TODO (sc) Add punctuation removal
    # leave the loose punctuation in.
    # if not punctuation.__contains__(token))


def simple_split_many(txts):
    """
    Creates a single sequence of all of the continuous non-whitespace
    character sequences (e.g. words, punctuation, etc.) from all of the strings
    in txts.

    For best results the strings in txts should be unicode strings.
    """
    return tlz.mapcat(simple_split_txt, txts)
