"""Testing bv_msk module."""

from os.path import join as pjoin, dirname, basename
from glob import glob
from warnings import simplefilter
import shutil

import numpy as np
from numpy import array as npa

from .. import load as top_load

from ..openers import Opener
from ..fileholders import FileHolder
from ..volumeutils import array_from_file
from ..externals import OrderedDict

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from ..testing import catch_warn_reset, suppress_warnings

from .test_arrayproxy import check_mmap
from . import test_spatialimages as tsi


DATA_PATH = pjoin(dirname(__file__), 'data')
MSK = pjoin(DATA_PATH, 'test.msk')

EXAMPLE_IMAGES = [
    dict(
        fname = MSK,
        shape = (10, 10, 10),
        dtype = np.uint8,
        affine = np.array([[ -3.,   0.,   0., -21.],
       [  0.,   0.,  -3., -21.],
       [  0.,  -3.,   0., -21.],
       [  0.,   0.,   0.,   1.]]),
        zooms = (3.0, 3.0, 3.0),
        data_summary = dict(
            min = 0,
            max = 1,
            mean = 0.499),
        is_proxy = True)
]

MSK_EXAMPLE_HDRDICT = \
    OrderedDict([
        ('Resolution', 3),
        ('XStart', 120),
        ('XEnd', 150),
        ('YStart', 120),
        ('YEnd', 150),
        ('ZStart', 120),
        ('ZEnd', 150)
        ])
