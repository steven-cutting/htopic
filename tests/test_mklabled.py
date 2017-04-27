# -*- coding: utf-8 -*-
__title__ = 'h_topic_model'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@gmail.com'
__created_on__ = '04/20/2017'

import pytest
import toolz as tlz

from h_topic_model import mklabled as ml


@pytest.mark.parametrize("tokens,expected",
                         [(u"token-0",
                           u"token0",),
                          (u"@%#$%^#$%,.::';/?!",
                           u"")
                          ])
def test__clean_txt(tokens, expected):
    assert(ml.clean_txt(tokens) ==
           expected)


@pytest.mark.parametrize("text,expected",
                         [([u"token&0 token&1", u"to&ken&1"],
                           [u'tokenf60968a6d89e0', u'tokenf60968a6d89e1',
                            u'tof60968a6d89ekenf60968a6d89e1']),
                          ])
def test__get_seg_tokens(text, expected):
    assert(list(ml.get_seg_tokens(text, ogmarker=u"&")) ==
           expected)


@pytest.mark.parametrize("tokens,expected",
                         [([u"token-0", u"token-1", u"to-ken-1"],
                           {(u'token0', (u'token-0', )),
                            (u'token1', (u'token-1', u'to-ken-1')),
                            })
                          ])
def test__group_annots_by_comp(tokens, expected):
    assert(set(ml.group_annots_by_comp(tokens, segmarker=u"-")) ==
           expected)


@pytest.mark.parametrize("seq,expected",
                         [([['token0', ['token0']],
                            ['token1', ['token-1', 'to-ken-1']],
                            ],
                           [('token1', ['token-1', 'to-ken-1']), ])
                          ])
def test__filter_empty_annots(seq, expected):
    assert(list(ml.filter_empty_annots(seq)) ==
           expected)


@pytest.mark.parametrize("seq,expected",
                         [(["token-0", ],
                           {('token0', ('token-0',))}),
                          (["token-0", "token-1", "to-ken-1"],
                           {('token0', ('token-0',)),
                            ('token1', ('token-1', 'to-ken-1'))}),
                          (['derp-derp', 'ferp-fe-rp'],
                           {('ferpferp', ('ferp-fe-rp',)), ('derpderp', ('derp-derp',))}),
                          ])
def test__mk_comp_and_annots(seq, expected):
    assert(set(ml.mk_comp_and_annots(seq, segmarker="-")) ==
           expected)


@pytest.mark.parametrize("seq,expected",
                         [([u"token0", [u"token-0", ]],
                           u"token0 token-0")
                          ])
def test___to_annot_file_fromat(seq, expected):
    assert(ml._to_annot_file_fromat(seq) ==
           expected)


@pytest.mark.parametrize("seq,expected",
                         [([[u"token0", [u"token-0"]],
                            [u"token1", [u"token-1", u"to-ken-1"]]],
                           [u"token0 token 0",
                            u"token1 token 1, to ken 1"]),
                          ])
def test__annot_file_format_many(seq, expected):
    assert(list(ml.annot_file_format_many(seq, segmarker=u"-")) ==
           expected)


@pytest.mark.parametrize("text,expected",
                         [([u"token&0 token&1", u"to&ken&1"],
                           {u"token0 token 0",
                            u"token1 token 1, to ken 1"}),
                          ])
def test__all(text, expected):
    assert(tlz.pipe(text,
                    ml.get_seg_tokens(ogmarker=u"&"),
                    ml.mk_comp_and_annots,
                    ml.annot_file_format_many,
                    set) ==
           expected)
