#! /usr/bin/env python
# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/25/2017'

from h_topic_model import bagofwords as bow
from h_topic_model import gmodels as gm

DICT_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/gensim/dict.dict"
CORPUS_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/gensim/corpus.mm"

TFIDF_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/gensim/model.tfidf"
LDA_FILE = "/Users/steven_c/projects/h_topic_model/data/toy50k/gensim/model.lda"

dictionary = bow.load_dict(DICT_FILE)
bowCorpus = bow.load_corpus(CORPUS_FILE)

print "Fitting tfidf model."
tfidfModel = gm.fit_tfidf(bowCorpus)
print "Done fitting tfidf model."

tfidfCorpus = gm.to_model_vec_space(tfidfModel, bowCorpus)


print "Fitting lda model"
ldaModel = gm.fit_lda(tfidfCorpus, dictionary, workers=3)
print "Done fitting lda model."

print "Saving models"
gm.save_model(TFIDF_FILE, tfidfModel)
gm.save_model(LDA_FILE, ldaModel)
print "Done saving models"
