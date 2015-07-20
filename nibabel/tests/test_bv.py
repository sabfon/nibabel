# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test Analyze headers.

See test_wrapstruct.py for tests of the wrapped structarr-ness of the Analyze
header
"""

import os
import re
import logging
import pickle
import tempfile

import numpy as np

from ..externals.six import BytesIO, StringIO
from ..volumeutils import array_to_file
from ..spatialimages import (HeaderDataError, HeaderTypeError)
from ..bv import readCString, parse_BV_header, pack_BV_header, calc_BV_header_size
from ..bv_vtc import VtcHeader, VtcImage, _make_vtc_hdrDict
from ..bv_msk import MskHeader, MskImage
from ..bv_vmp import VmpHeader, VmpImage
from ..nifti1 import Nifti1Header
from ..loadsave import read_img_data
from ..import imageglobals
from ..casting import as_int

from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal)

from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path)

from .test_wrapstruct import _TestLabeledWrapStruct
from . import test_spatialimages as tsi

vtc_file = os.path.join(data_path, 'test.vtc')
vmp_file = os.path.join(data_path, 'test.vmp')

def test_readCString():
    # sample binary block
    binary = 'test.fmr\x00test.prt\x00'
    try:
        # create a tempfile
        file, path = tempfile.mkstemp()
        fwrite = open(path, 'w')

        # write the binary block to it
        fwrite.write(binary)
        fwrite.close()

        # open it again
        fread = open(path, 'r')

        # test readout of one string
        assert_equal([s for s in readCString(fread)], ['test.fmr'])

        # test new file position
        assert_equal(fread.tell(), 9)

        # manually rewind
        fread.seek(0)

        # test readout of two strings
        assert_equal([s for s in readCString(fread, 2, rewind=True)], ['test.fmr', 'test.prt'])

        # test automatic rewind
        assert_equal(fread.tell(), 0)

        # test readout of two strings with trailing zeros
        assert_equal([s for s in readCString(fread, 2, strip=False)], ['test.fmr\x00', 'test.prt\x00'])

        # test new file position
        assert_equal(fread.tell(), 18)

        # test readout of one string from given position
        fread.seek(0)
        assert_equal([s for s in readCString(fread, startPos=9)], ['test.prt'])
    except:
        os.remove(path)
        raise
    os.remove(path)

def test_parse_BV_header():
    # open vtc test file
    fileobj = open(vtc_file, 'r')
    hdrDict = _make_vtc_hdrDict()
    hdrDict = parse_BV_header(hdrDict, fileobj)
    assert_equal(hdrDict['fmr']['value'], 'test.fmr')
    assert_equal(hdrDict['XStart']['value'], 120)
    assert_equal(hdrDict['TR']['value'], 2000.0)

def test_pack_BV_header():
    # open vtc test file
    fileobj = open(vtc_file, 'r')
    hdrDict = _make_vtc_hdrDict()
    hdrDict = parse_BV_header(hdrDict, fileobj)
    binaryblock = pack_BV_header(hdrDict)
    print binaryblock
    assert_equal(binaryblock, '\x03\x00test.fmr\x00\x01\x00test.prt\x00\x00\x00\x02\x00\x05\x00\x03\x00x\x00\x96\x00x\x00\x96\x00x\x00\x96\x00\x01\x01\x00\x00\xfaD')

def test_calc_BV_header_size():
    # open vtc test file
    fileobj = open(vtc_file, 'r')
    hdrDict = _make_vtc_hdrDict()
    hdrDict = parse_BV_header(hdrDict, fileobj)
    hdrSize = calc_BV_header_size(hdrDict)
    assert_equal(hdrSize, 48)

def test_VtcImage():
    # load vtc image from filename
    vtc = VtcImage.from_filename(vtc_file)
