# -*- coding: utf-8 -*-

__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '04/14/2017'


from tempfile import NamedTemporaryFile
import csv
import itertools as itls

import pytest

from h_topic_model import segmodel as sm


def test__load_data():
    """
    Simply Check to see if it will run using simple data.
    """
    inputdata = [(i, n) for i, n in zip(xrange(10), itls.repeat(u"d", 10))]

    csv.register_dialect('tab-sep', delimiter="\t")
    with NamedTemporaryFile() as f:
        with open(f.name, "w+") as csvfile:
            csv_out = csv.writer(csvfile, 'tab-sep')
            for row in inputdata:
                csv_out.writerow(row)

        assert(list(sm.load_data(f.name)))


def test__mkmodel():
    """
    Simply Check to see if it will run using simple data.
    """
    inputdata = [(i, n) for i, n in zip(xrange(10), itls.repeat(u"d", 10))]

    csv.register_dialect('tab-sep', delimiter="\t")
    with NamedTemporaryFile() as f:
        with open(f.name, "w+") as csvfile:
            csv_out = csv.writer(csvfile, 'tab-sep')
            for row in inputdata:
                csv_out.writerow(row)

        # print list(sm.load_data(f.name))
        model = sm.mkmodel(list(sm.load_data(f.name)))
        print model
        assert(list(model.get_segmentations()))


def test__save_model_and_save_segmentations():
    """
    Simply Check to see if it will run using simple data.
    """
    inputdata = [(i, n) for i, n in zip(xrange(10), itls.repeat(u"d", 10))]

    csv.register_dialect('tab-sep', delimiter="\t")
    with NamedTemporaryFile() as f:
        with open(f.name, "w+") as csvfile:
            csv_out = csv.writer(csvfile, 'tab-sep')
            for row in inputdata:
                csv_out.writerow(row)

        model = sm.mkmodel(sm.load_data(f.name))

    with NamedTemporaryFile() as mdlfile:
        sm.save_model(mdlfile.name, model)

    with NamedTemporaryFile() as segfile:
        sm.save_segmentations(segfile.name, model)
