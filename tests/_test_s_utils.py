# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/24/2017'


import pytest


from h_topic_model.scripts import s_utils as su


k_error = pytest.mark.xfail(raises=KeyError,
                            reason="Test to ensure it does raise error on proper value.")


@pytest.mark.parametrize("logLvl,logCode",
                         [(u"debug", 10),
                          (u"info", 20),
                          (u"warning", 30),
                          (u"error", 40),
                          (u"critical", 50),
                          k_error((u"foobar", 0)),
                          ])
def test__str_to_log_level_obj(logLvl, logCode):
    assert(su.str_to_log_level_code(logLvl) == logCode)
