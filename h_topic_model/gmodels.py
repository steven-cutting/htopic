# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/25/2017'
__doc__ = """
For creating Gensim models.
"""

from gensim import models


def fit_tfidf(corpus):
    return models.TfidfModel(corpus)


def fit_lsi(corpus, dictionary, numTopics=200):
    return models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)


def fit_lda(corpus, dictionary=None, num_topics=100, workers=None,
            passes=5, alpha='symmetric', iterations=50, random_state=None):
    return models.ldamulticore.LdaMulticore(corpus=corpus,
                                            num_topics=num_topics,
                                            id2word=dictionary,
                                            workers=workers,
                                            passes=passes,
                                            alpha=alpha,
                                            random_state=random_state,
                                            iterations=iterations)


def to_model_vec_space(model, corpus):
    return model[corpus]


def save_model(filename, model):
    model.save(filename)


def load_tfidf(filename):
    return models.TfidfModel.load(filename)


def load_lsi(filename):
    return models.LsiModel.load(filename)

    
def load_lda(filename):
    return models.LdaMulticore.load(filename)
