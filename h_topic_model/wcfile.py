# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/23/2017'
__doc__ = """
Module for functions dedicated to creating a word count file from many
documents.
"""


import csv

try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc


def count_tokens(tokens):
    """
    Counts the number of occurances of the tokens.
    Returns the frequences as a generator of tuples in this format:
        [(count0, token0),
         (count1, token1),
         ...
         ]
    """
    return tlz.pipe(tokens,
                    tlz.frequencies,
                    lambda countdict: countdict.iteritems(),
                    tlzc.map(reversed),
                    tlzc.map(tuple),)


def sort_token_counts(seq, g2l=True):
    """
    Sort tokens by count size.

    g2l - Sort by greatest to least.
          Default True
    """
    return sorted(seq, key=tlz.first, reverse=True)


def encode_token_in_row(row, encoding='utf-8'):
    return (row[0], row[1].encode(encoding))


def token_counts_to_cvs(filename, seq, delimiter='\t', sort_f=tlz.identity, encoding="utf-8"):
    """
    Does not include a header.
    The tuples in 'seq' should have this format:
        (count, token)

    Delimiter defaults to tabs.
    sort_f - A function that is applied to seq before writting it to the file.
             Default is the identity function.
    """
    csv.register_dialect('tab-sep', delimiter=delimiter)
    with open(filename, 'wb') as out:
        csv_out = csv.writer(out, 'tab-sep')
        for row in sort_f(seq):
            tlz.pipe(row,
                     encode_token_in_row,
                     csv_out.writerow)
