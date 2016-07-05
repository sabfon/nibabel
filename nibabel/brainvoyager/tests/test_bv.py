# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test main BV module."""

import os
from ...tmpdirs import InTemporaryDirectory
from ..bv import (readCString, parse_BV_header, pack_BV_header,
                               calc_BV_header_size)
from ..bv_vtc import VTC_HDR_DICT_PROTO
from ...testing import (assert_equal, data_path)


vtc_file = os.path.join(data_path, 'test.vtc')
vmp_file = os.path.join(data_path, 'test.vmp')


def test_readCString():
    # sample binary block
    binary = 'test.fmr\x00test.prt\x00'
    with InTemporaryDirectory():
        # create a tempfile
        path = 'test.header'
        fwrite = open(path, 'w')

        # write the binary block to it
        fwrite.write(binary)
        fwrite.close()
        del fwrite

        # open it again
        fread = open(path, 'r')

        # test readout of one string
        assert_equal([s for s in readCString(fread)], ['test.fmr'])

        # test new file position
        assert_equal(fread.tell(), 9)

        # manually rewind
        fread.seek(0)

        # test readout of two strings
        assert_equal([s for s in readCString(fread, 2, rewind=True)],
                     ['test.fmr', 'test.prt'])

        # test automatic rewind
        assert_equal(fread.tell(), 0)

        # test readout of two strings with trailing zeros
        assert_equal([s for s in readCString(fread, 2, strip=False)],
                     ['test.fmr\x00', 'test.prt\x00'])

        # test new file position
        assert_equal(fread.tell(), 18)

        # test readout of one string from given position
        fread.seek(0)
        assert_equal([s for s in readCString(fread, startPos=9)], ['test.prt'])

        del fread


def test_parse_BV_header():
    # open vtc test file
    fileobj = open(vtc_file, 'r')
    hdrDict = parse_BV_header(VTC_HDR_DICT_PROTO, fileobj)
    assert_equal(hdrDict['fmr'], 'test.fmr')
    assert_equal(hdrDict['XStart'], 120)
    assert_equal(hdrDict['TR'], 2000.0)


def test_pack_BV_header():
    # open vtc test file
    fileobj = open(vtc_file, 'r')
    hdrDict = parse_BV_header(VTC_HDR_DICT_PROTO, fileobj)
    binaryblock = pack_BV_header(VTC_HDR_DICT_PROTO, hdrDict)
    assert_equal(binaryblock, ''.join([
        '\x03\x00test.fmr\x00\x01\x00test.prt\x00\x00\x00\x02\x00\x05\x00\x03',
        '\x00x\x00\x96\x00x\x00\x96\x00x\x00\x96\x00\x01\x01\x00\x00\xfaD'
    ]))


def test_calc_BV_header_size():
    # open vtc test file
    fileobj = open(vtc_file, 'r')
    hdrDict = parse_BV_header(VTC_HDR_DICT_PROTO, fileobj)
    hdrSize = calc_BV_header_size(VTC_HDR_DICT_PROTO, hdrDict)
    assert_equal(hdrSize, 48)
