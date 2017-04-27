# -*- coding: utf-8 -*-
"""
Module has functions that are useful for reading text that contians segmentation markings
and transforming that text into a format that can be used by Morfessor 2.0 and FlatCat
as labeled data.

1) Start by loading the files lines.
2) Feed them to get_seg_tokens
3) Then mk_comp_and_annots
4) Apply annot_file_format_many
5) Finally write to a file
"""
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '04/20/2017'


import re
import itertools as itls

import cytoolz as tlz
from cytoolz import curried as tlzc
from text2math import text2tokens as t2t

from h_topic_model import textproc_utils as tpu

c_starmap = tlz.curry(itls.starmap)


# SEGMENTED_SENTS_FILE = "/Users/steven_c/projects/h_topic_model/data/labled_data/ex_segments/ex_sentences_w_segmented_words.txt"
#
# ANNOTATIONS_FILE = "/Users/steven_c/projects/h_topic_model/data/labled_data/traindata.txt"
#
# def load_file_lines(filename):
#     with open(filename) as f:
#         return [r2t.adv_decode(txt) for txt in f.readlines()]

# LINES = load_file_lines(SEGMENTED_SENTS_FILE)
# len(LINES)


SEGMARKER = u"f60968a6d89e"
OGMARKER = u"~"

# ## Parsing out the tokens


def clean_txt(txt):
    """
    Removes most of the punctuation.
    """
    return tlz.pipe(txt,
                    t2t.drop_punct,
                    lambda t: re.subn(t2t.PUNCTUATIONPAT, u"", t)[0],
                    tpu.remove_h_punct(replace=u""))


@tlz.curry
def get_seg_tokens(lines, ogmarker=OGMARKER, segmarker=SEGMARKER):
    return tlz.pipe(lines,
                    tlzc.map(lambda token: token.replace(ogmarker, segmarker)),
                    tlzc.map(tpu.basic_split),
                    tlz.concat,
                    # Filter out tokens that are not segmented.
                    tlzc.filter(lambda t: segmarker in t),
                    tlzc.map(clean_txt),
                    tlzc.filter(tlz.identity))


# ## Putting together compounds and annotations

def replace_with_seg_marker(txt, replacement, segmarker=SEGMARKER):
    return txt.replace(segmarker, replacement)


@tlz.curry
def comp_from_annots(txt, segmarker=SEGMARKER):
    return replace_with_seg_marker(txt, u"", segmarker=segmarker)


@tlz.curry
def seg_marker_to_space(txt, segmarker=SEGMARKER, replacement=u" "):
    return replace_with_seg_marker(txt, replacement, segmarker=segmarker)


@tlz.curry
def group_annots_by_comp(seq, segmarker=SEGMARKER):
    return tlz.pipe(seq,
                    tlzc.groupby(comp_from_annots(segmarker=segmarker)),
                    lambda d: ((k, tuple(v)) for k, v in d.iteritems()))


def filter_empty_annots(seq):
    """
    Remove from seq those items that have zero annotation and
    remove the annotations that are the same as the commpound.
    """
    return tlz.pipe(seq,
                    c_starmap(lambda k, vs: (k, filter(lambda t: t != k, vs))),
                    tlzc.filter(tlz.second))  # filter out those that now have zero annotations.


def mk_comp_and_annots(seq, segmarker=SEGMARKER):
    """
    Accepts a sequence of tokens that have had their segmentations marked.

    Does not preserve order.
    Returns
        >>> list(mk_comp_and_annots(["token-0", "token-1", "to-ken-1"], segmarker="-"))
        [(u'token1', ('token-1', 'to-ken-1')), (u'token0', ('token-0',))]
    """
    return tlz.pipe(seq,
                    tlz.unique,
                    group_annots_by_comp(segmarker=segmarker),
                    filter_empty_annots)


def _to_annot_file_fromat(seq):
    """
    Takes a commpound and annotation data-structure. Returns compound and
    annotations in the format need for Morfessor 2.0 annotation file.

        >>> _to_annot_file_fromat(["compound", ["annotation-0", "annotation-1", \
                                                "annotation-n"]])
        'compound annotation-0, annotation-1, annotation-n'

    [Annotation file](http://morfessor.readthedocs.io/en/latest/filetypes.html#annotation-file)
    """
    return u" ".join([seq[0], ", ".join(seq[1])])


@tlz.curry
def to_annot_file_fromat(seq, segmarker=SEGMARKER):
    """
    Takes a commpound and annotation data-structure in the form of:
        [u"compound" [u"annotation-0", u"annotation-1", u"annotation-n"]]

    Returns compound and annotations in the format need for Morfessor 2.0
    annotation file.
        "compound annotation-0, annotation-1, ... , annotation-n"

        >>> to_annot_file_fromat(["compound", ["annotation-0", \
                                               "annotation-1", \
                                               "annotation-n"]], \
                                 segmarker="-")
        'compound annotation 0, annotation 1, annotation n'

    [Annotation file](http://morfessor.readthedocs.io/en/latest/filetypes.html#annotation-file)

    Also, replaces segmentation marker with a space.
    """
    return tlz.pipe(seq,
                    _to_annot_file_fromat,
                    seg_marker_to_space(segmarker=segmarker),
                    )


def annot_file_format_many(seq, segmarker=SEGMARKER):
    """
    ## Make Annotation Strings

    Takes many commpound and annotation data-structures. Returns the
    compounds and annotations in the format need for Morfessor 2.0
    annotation file.

        >>> seq = [["token0", ["token-0"]], \
                   ["token1", ["token-1", "to-ken-1"]], \
                   ]

        >>> list(annot_file_format_many(seq, segmarker="-"))
        ['token0 token 0', 'token1 token 1, to ken 1']

    ---

    [Annotation file](http://morfessor.readthedocs.io/en/latest/filetypes.html#annotation-file)

    An annotation file contains one compound and one or more annotations per
    compound on each line. The separators between the annotations (default ', ')
    and between the constructions (default ' ') are configurable.

    **Specification**

    ```
    <compound> <analysis1construction1>[ <analysis1constructionN>][,
    <analysis2construction1> [<analysis2constructionN>]*]*
    ```

    **Example**
    ```
    kahvikakku kahvi kakku, kahvi kak ku
    kahvikilon kahvi kilon
    kahvikoneemme kahvi konee mme, kah vi ko nee mme
    ```
    """
    return tlz.map(to_annot_file_fromat(segmarker=segmarker), seq)
