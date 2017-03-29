# -*- coding: utf-8 -*-
from __future__ import print_function
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/28/2017'


import subprocess
import logging
import sys
import os
from os import path
from pprint import pprint

import click
from arrow import now 
try:
    import cytoolz as tlz
    from cytoolz import curried as tlzc
except ImportError:
    import toolz as tlz
    from toolz import curried as tlzc

from h_topic_model.scripts import s_utils as su
from h_topic_model import utils as u
from h_topic_model import wcfile as wc
from h_topic_model import segment as seg
from h_topic_model import bagofwords as bow
from h_topic_model import gmodels as gm


LOG = logging.getLogger(__name__)
SCRIPT_NAME = __name__.split(".")[-1]
LOGFMT = '%(levelname)s:%(asctime)s:%(name)s\tscript-name:{s}\t%(message)s'.format(s=SCRIPT_NAME)

CWD = os.getcwd()
D_WC_FILE = "wc.csv"
D_WC_FILE_P = path.join(CWD, D_WC_FILE)
D_MORF_MODEL_FILE = "morfessor_model.bin"
D_MORF_MODEL_FILE_P = path.join(CWD, D_MORF_MODEL_FILE)
D_G_DICT_FILE = "gensim_dict.dict"
D_G_DICT_FILE_P = path.join(CWD, D_G_DICT_FILE)
D_G_CORPUS_FILE = "gensim_corpus.mm"
D_G_CORPUS_FILE_P = path.join(CWD, D_G_CORPUS_FILE)
D_G_TFIDF_FILE = "gensim_model.tfidf"
D_G_LDA_FILE = "gensim_model.lda"
D_G_MODELS_P = CWD


@click.group()
@click.option("--loglvl", default="info", type=str,
              help="Logging level. Default info. Options: debug, info, warning, error, critical")
@click.option("--logfile", "-L", default=None, type=str,
              help="Write log messages to this file. Defaults to stdout.")
@click.option("--random_state", "-R", default=None, type=int,
              help="Random State. Allows for reproducible results.")  # TODO (sc) Improve help
@click.pass_context
def cli(ctx, loglvl, logfile, random_state):
    ctx.obj = {}
    ctx.obj["LOGFILE"] = logfile
    ctx.obj["STARTTIME"] = now()
    ctx.obj["RANDOMSTATE"] = random_state

    logging.basicConfig(format=LOGFMT,
                        level=su.str_to_log_level_code(loglvl),
                        stream=sys.stdout,
                        filename=logfile)
    logging.root.level = su.str_to_log_level_code(loglvl)
    logging.basicConfig


# TODO (steven_c) Add encoding options. Both for input and output.
@cli.command(help="""
    Creates a word count file from the text contained in the files in the input directory.
    Looks for files recursively.

    To provide reproducible results, provide the random_state parameter.
 """)
@click.argument("input_directory")
@click.option("--wcfile", "-W", type=str, default=D_WC_FILE_P,
              help="".join(["Write word counts to this file. Default: ",
                            D_WC_FILE_P]))
@click.option("--prob", "-P", default=1.0, type=float,
              help="Probability a file will be chosen. Create a random sample. 1.0 = 100%")
@click.pass_context
def mkwc(ctx, input_directory, wcfile, prob):

    LOG.info("Starting mkwc")

    filenames = tlz.random_sample(prob=prob,
                                  seq=u.spelunker_gen(input_directory),
                                  random_state=ctx.obj["RANDOMSTATE"])
    txts = tlz.map(u.load_and_decode, filenames)

    tokenFreqs = tlz.pipe(txts, wc.simple_split_many, wc.count_tokens)

    wc.token_counts_to_cvs(wcfile, tokenFreqs)  # dont sort it's not needed
    #                                           # , sort_f=wc.sort_token_counts)

    runtime = su.log_run_time(ctx.obj["STARTTIME"])
    LOG.info("Finished mkwc")

    
@cli.command(help="""
    Uses morfessor-train to construct a model for string segmentation
    using the provided training data in the form of a file of 'word' counts.

    The word count file should have only one word per line. The lines
    should be in the following format:

        count word

    To provide reproducible results, provide the random_state parameter.
""")
@click.option("--wcfile", "-W", type=str, default=D_WC_FILE_P,
              help="".join(["Token source for morfessor-train. Default: ",
                            D_WC_FILE_P]))
@click.option("--morf_model_file", "-M", type=str, default=D_MORF_MODEL_FILE_P,
              help="".join(["Save morfessor model to this file. Default: ",
                            D_MORF_MODEL_FILE_P]))
@click.option("--encoding", "-E", default="utf-8",
              help="Character encoding of the wcfile file.")
@click.pass_context
def morfessor(ctx, morf_model_file, wcfile, encoding):
    LOG.info("Starting morfessor")
    logfile = ctx.obj["LOGFILE"]
    randomState = ctx.obj["RANDOMSTATE"]
    subprocess.call(["morfessor-train",
                     "--encoding={}".format(encoding),
                     "--randseed={}".format(randomState) if randomState else "",
                     "--traindata-list",
                     "--logfile={}".format(logfile) if logfile else "",
                     "-s", morf_model_file,
                     wcfile])

    runtime = su.log_run_time(ctx.obj["STARTTIME"])
    LOG.info("Finished morfessor")


@cli.command(help="""
    Create Gensim style corpus for use with Gensim models (e.g. LDA, LSI, TfIdf).

    To provide reproducible results, provide the random_state parameter.
""")
@click.argument("input_directory")
@click.option("--morf_model_file", "-M", type=str, default=D_MORF_MODEL_FILE_P,
              help="".join(["File that contains the morfessor model to use. Default: ",
                            D_MORF_MODEL_FILE_P]))
@click.option("--dict_file", "-D", type=str, default=D_G_DICT_FILE_P,
              help="".join(["Save Gensim dictionary to this file. Default: ",
                            D_G_DICT_FILE_P]))
@click.option("--corpus_file", "-C", type=str, default=D_G_CORPUS_FILE_P,
              help="".join(["Save Gensim corpus to this file. Default: ",
                            D_G_CORPUS_FILE_P]))
@click.option("--prob", "-P", default=1.0, type=float,
              help="Probability a file will be chosen. Create a random sample. 1.0 = 100%")
@click.pass_context
def mkcorpus(ctx, input_directory, morf_model_file, dict_file, corpus_file, prob):
    LOG.info("Starting mkcorpus")

    # -- Loading Morfessor Model --
    LOG.info("Loading morfessor model: {}".format(morf_model_file))
    model = seg.load_morfessor_model(morf_model_file)

    # -- Finding and Loading input files --
    LOG.info("Loading input files from: {}".format(input_directory))
    filenames = tlz.random_sample(prob=prob,
                                  seq=u.spelunker_gen(input_directory),
                                  random_state=ctx.obj["RANDOMSTATE"])

    txts = list(tlz.map(u.load_and_decode, filenames))

    LOG.info("Number of texts {}.".format(len(txts)))

    # -- Tokenizing input texts --
    tokensSeqs = tlz.pipe(seg.segment_many(model, txts),
                        tlzc.map(bow.filter_tokens),
                        tlzc.map(list),
                        list)

    LOG.info("Number of token sets {}".format(len(tokensSeqs)))

    # -- Creating Dictionary and Corpus --
    LOG.info("Creating Gensim dictionary.")
    dictionary = bow.mk_dict(tokensSeqs)

    LOG.info("Number of unique tokens (pre filter): {}".format(len(dictionary.items())))

    dictionary = bow.filter_dict(dictionary)

    LOG.info("Number of unique tokens (post filter): {}".format(len(dictionary.items())))

    LOG.info("Creating Gensim corpus.")
    bowCorpus = bow.mk_corpus(dictionary, tokensSeqs)

    # -- Saving --
    LOG.info("Saving Gensim dictionary to: {}".format(dict_file))
    bow.save_dict(dict_file, dictionary)
    LOG.info("Saving Gensim corpus to: {}".format(corpus_file))
    bow.save_corpus(corpus_file, bowCorpus)

    runtime = su.log_run_time(ctx.obj["STARTTIME"])
    LOG.info("Finished mkcorpus")


@cli.command(help="""
Train Gensim TfIdf and LDA model.
""")
@click.option("--topic_model_dir", "-T", type=str, default=D_G_MODELS_P, 
              help="".join(["Directory to save the Gensim models to. Default: ",
                            D_G_MODELS_P]))
@click.option("--dict_file", "-D", type=str, default=D_G_DICT_FILE_P,
              help="".join(["Gensim dictionary file to use. Default: ",
                            D_G_DICT_FILE_P]))
@click.option("--corpus_file", "-C", type=str, default=D_G_CORPUS_FILE_P,
              help="".join(["Gensim corpus file to use. Default: ",
                            D_G_CORPUS_FILE_P]))
@click.pass_context
def mkmodel(ctx, topic_model_dir, dict_file, corpus_file):

    randomState = ctx.obj["RANDOMSTATE"]

    # -- Loading Dictionary and Corpus --
    LOG.info("Loading Gensim dictionary from: {}".format(dict_file))
    dictionary = bow.load_dict(dict_file)

    LOG.info("Loading Gensim corpus from: {}".format(corpus_file))
    bowCorpus = bow.load_corpus(corpus_file)

    # -- Fitting --
    LOG.info("Fitting Gensim TfIdf model.")
    tfidfModel = gm.fit_tfidf(bowCorpus)

    LOG.info("Fitting Gensim TfIdf model.")
    tfidfCorpus = gm.to_model_vec_space(tfidfModel, bowCorpus)

    LOG.info("Fitting Gensim LDA model.")
    ldaModel = gm.fit_lda(tfidfCorpus, dictionary, workers=3, random_state=randomState)

    # -- Saving --
    LOG.info("Saving Gensim models in: {}".format(topic_model_dir))

    LOG.info("Saving Gensim TfIdf model to: {}".format(D_G_TFIDF_FILE))
    gm.save_model(path.join(topic_model_dir, D_G_TFIDF_FILE), tfidfModel)

    LOG.info("Saving Gensim LDA model to: {}".format(D_G_LDA_FILE))
    gm.save_model(path.join(topic_model_dir, D_G_LDA_FILE), ldaModel)

    runtime = su.log_run_time(ctx.obj["STARTTIME"])
    LOG.info("Finished mkcorpus")


@cli.command(help="""
    Get Gensim LDA topics for input_file.
""")
@click.argument("input_file")
@click.option("--morf_model_file", "-M", type=str, default=D_MORF_MODEL_FILE_P,
              help="".join(["File that contains the morfessor model to use. Default: ",
                            D_MORF_MODEL_FILE_P]))
@click.option("--topic_model_dir", "-T", type=str, default=D_G_MODELS_P, 
              help="".join(["Directory where the Gensim models are. Default: ",
                            D_G_MODELS_P]))
@click.option("--dict_file", "-D", type=str, default=D_G_DICT_FILE_P,
              help="".join(["Gensim dictionary file to use. Default: ",
                            D_G_DICT_FILE_P]))
@click.option("--corpus_file", "-C", type=str, default=D_G_CORPUS_FILE_P,
              help="".join(["Gensim corpus file to use. Default: ",
                            D_G_CORPUS_FILE_P]))
@click.pass_context
def filetopics(ctx, input_file, morf_model_file, topic_model_dir, dict_file, corpus_file):

    morfModel = seg.load_morfessor_model(morf_model_file)
    dictionary = bow.load_dict(dict_file)
    tfidfModel = gm.load_tfidf(path.join(topic_model_dir, D_G_TFIDF_FILE))
    ldaModel = gm.load_lda(path.join(topic_model_dir, D_G_LDA_FILE))

    tlz.pipe(input_file,
             u.load_and_decode,
             lambda t: seg.segment_text(morfModel, t),
             dictionary.doc2bow,
             lambda t: tfidfModel[t],
             ldaModel.get_document_topics,
             tlz.curried.map(tlz.first),
             list,
             tlzc.map(unicode),
             lambda l: u", ".join(l),
             lambda t: u"\n\nTopics for input_file: {}\n\n".format(t),
             print,
             )


@cli.command(help="""
    Get Gensim LDA topic descriptions.
""")
@click.option("--topic_model_dir", "-T", type=str, default=D_G_MODELS_P, 
              help="".join(["Directory where the Gensim models are. Default: ",
                            D_G_MODELS_P]))
@click.pass_context
def gettopics(ctx, topic_model_dir):
    ldaModel = gm.load_lda(path.join(topic_model_dir, D_G_LDA_FILE))

    topics = [ldaModel.show_topic(i) for i in xrange(ldaModel.num_topics)]

    for i, topic in enumerate(topics):
        print(u'\n'.encode("utf-8"))
        print(u'--- TOPIC: {} ---'.format(i).encode("utf-8"))
        print(u'\n'.encode("utf-8"))
        # print(u'Probability\t\tWord')
        # print(u'-----------\t\t----')
        for prob, word in topic:
            print(u"\t{}".format(prob, word).encode("utf-8"))
