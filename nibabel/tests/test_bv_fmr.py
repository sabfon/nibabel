from os.path import join as pjoin
import numpy as np

from ..brainvoyager.bv import BvError
from ..brainvoyager.bv_fmr import BvFmrImage, FMR_HDR_DICT_PROTO, parse_BV_header
from ..testing import (assert_equal, data_path)
from ..externals import OrderedDict

fmr_file = pjoin(data_path, 'test.fmr')
fileobj = open(fmr_file, 'r')

def test_parse_BVFMR_header():
    parse_BV_header(FMR_HDR_DICT_PROTO, fileobj)
    #fmr = BvFmrImage.from_filename(fmr_file)

