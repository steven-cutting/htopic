# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/24/2017'


import sys
from operator import eq

import toolz as tlz
import pytest


from h_topic_model import utils as u


if sys.version_info[0] < 3:
    _BYTESTRING = str
    _UNICODESTRING = unicode
else:
    _BYTESTRING = bytes
    _UNICODESTRING = str

c_eq = tlz.curry(eq)


@pytest.mark.parametrize("string,totype",
                         [(u"unicode", _BYTESTRING),
                          (b"byte", _BYTESTRING),
                          ])
def test__ensure_bytestring(string, totype):
    assert(tlz.pipe(string,
                    u.ensure_bytestring,
                    type,
                    c_eq(totype)))
