# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/23/2017'
__doc__ = """
General utils that can be used thoughout the package.
"""

import os

from text2math import raw2text as r2t
from text2math.raw2text import verify_unicode, verify_bytestring


def spelunker_gen(rootdir):
    """
    Recursively find all files in 'rootdir' and
    return a generator of their paths.
    """
    for dirname, subdirlist, filelist in os.walk(rootdir):
        for fname in filelist:
            yield os.path.join(dirname, fname)


def load_and_decode(filename):
    """
    Reads files and handles any decoding issues.
    Returns text from file as Unicode.
    """
    with open(filename) as t:
        return r2t.adv_decode(t.read())


def ensure_bytestring(string, encoding='utf-8'):
    try:
        return verify_bytestring(string)
    except AssertionError:
        return string.encode(encoding)
