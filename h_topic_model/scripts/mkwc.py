#! /usr/bin/env python
# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/24/2017'

import itertools as itls

try:
    import cytoolz as tlz
except ImportError:
    import toolz as tlz

from h_topic_model import wcfile as wc
from h_topic_model import utils as u


TEXT_DIR = "/Users/steven_c/projects/h_topic_model/data/toy50k/toy50k/"
COUNTS_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/counts/counts.csv"
TOTAl = 50000

filenames = itls.islice(u.spelunker_gen(TEXT_DIR), TOTAl)
txts = tlz.map(u.load_and_decode, filenames)

tokenFreqs = tlz.pipe(txts, wc.simple_split_many, wc.count_tokens)

wc.token_counts_to_cvs(COUNTS_FILE, tokenFreqs)  # dont sort it's not needed
#                                                # , sort_f=wc.sort_token_counts)
