# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '03/23/2017'
__doc__ = """
General utils for scripts.
"""

import logging

from arrow import now

LOG = logging.getLogger(__name__)


def str_to_log_level_code(string):
    """
    Returns logging level obj based on 'string'.
    Options: debug, info, warning, error, critical
    """
    logdict = {"debug": logging.DEBUG,
               "info": logging.INFO,
               "warning": logging.WARNING,
               "error": logging.ERROR,
               "critical": logging.CRITICAL}
    try:
        return logdict[string]
    except KeyError as e:
        LOG.error("{} - Improper log level supplied. Raising KeyError.".format(e))

        basemsg = "{} - Is not a proper log level.".format(e)
        otheropts = "Did you mean: debug, info, warning, error, critical"
        raise KeyError("{b}  {o}".format(b=basemsg, o=otheropts))


def log_run_time(starttime):
    runtime = now() - starttime
    LOG.info("RunTime: {}".format(runtime))
    return runtime
