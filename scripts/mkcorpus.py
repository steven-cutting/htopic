#! /usr/bin/env python
# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/25/2017'


import itertools as itls

try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc

from h_topic_model import utils as u
from h_topic_model import segment as seg
from h_topic_model import bagofwords as bow

TEXT_DIR = "/Users/steven_c/projects/h_topic_model/data/toy50k/toy50k/"
MODEL_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/model/model.bin"

DICT_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/gensim/dict.dict"
CORPUS_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/gensim/corpus.mm"

TOTAl = 20000

model = seg.load_morfessor_model(MODEL_FILE)

filenames = itls.islice(u.spelunker_gen(TEXT_DIR), TOTAl)
txts = list(tlz.map(u.load_and_decode, filenames))

print "Number of texts {}.".format(len(txts))

tokensSeqs = tlz.pipe(seg.segment_many(model, txts),
                      tlzc.map(bow.filter_tokens),
                      tlzc.map(list),
                      list)

print "Number of token sets {}".format(len(tokensSeqs))

dictionary = bow.mk_dict(tokensSeqs)
dictionary = bow.filter_dict(dictionary)

bowCorpus = bow.mk_corpus(dictionary, tokensSeqs)

# --- save ---
bow.save_dict(DICT_FILE, dictionary)
bow.save_corpus(CORPUS_FILE, bowCorpus)
