# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/23/2017'

# import flatcat

punctuation = [u"\u05f3",  # ׳  U+05f3  HEBREW PUNCTUATION GERESH
               u"\u05f4",  # ״  U+05f4  HEBREW PUNCTUATION GERSHAYIM
               u"\u0027",  # '  U+0027  APOSTROPHE
               u"\u0022",  # "  U+0022  QUOTATION MARK
               ]


# def load_flatcat_model(filename):
#     """
#     filename - a flatcat model in binary pickle.
#     """
#     io = flatcat.FlatcatIO()
#     return io.read_binary_model_file(filename)


# def segment_text(model, text):
#     return [model.viterbi_segment(w) for w in text.split()]
