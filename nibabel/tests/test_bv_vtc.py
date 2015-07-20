"""Testing bv_vtc module."""

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
VTC = pjoin(DATA_PATH, 'test.vtc')

EXAMPLE_IMAGES = [
    # Parameters come from load of Philips' conversion to NIfTI
    # Loaded image was ``phantom_EPI_asc_CLEAR_2_1.nii`` from
    # http://psydata.ovgu.de/philips_achieva_testfiles/conversion
    dict(
        fname = VTC,
        shape = (10, 10, 10, 5),
        dtype = np.float32,
        # We disagree with Philips about the right affine, for the moment, so
        # use our own affine as determined from a previous load in nibabel
        affine = np.array([[ -3.,   0.,   0., -21.],
       [  0.,   0.,  -3., -21.],
       [  0.,  -3.,   0., -21.],
       [  0.,   0.,   0.,   1.]]),
        zooms = (3.0, 3.0, 3.0),
        data_summary = dict(
            min = 0.009668911807239056,
            max = 199.93548583984375,
            mean = 100.19720458984375),
        is_proxy = True)
]

VTC_EXAMPLE_HDRDICT = \
    OrderedDict([
        ('version', {'default': 3, 'dt': '<h', 'value': 3}),
        ('fmr', {'default': '', 'dt': '<s', 'value': 'test.fmr'}),
        ('nPrt', {'default': 0, 'dt': '<h', 'value': 1}),
        ('prts', {
            'default': [OrderedDict([('filename', {'default': '', 'dt': '<s', 'value': 'test.prt'})])],
            'dt': 'multi', 'nField': 'nPrt',
            'value': [OrderedDict([('filename', {'default': '', 'dt': '<s', 'value': 'test.prt'})])]}),
        ('currentPrt', {'default': 0, 'dt': '<h', 'value': 0}),
        ('datatype', {'default': 2, 'dt': '<h', 'value': 2}),
        ('volumes', {'default': 0, 'dt': '<h', 'value': 5}),
        ('Resolution', {'default': 3, 'dt': '<h', 'value': 3}),
        ('XStart', {'default': 57, 'dt': '<h', 'value': 120}),
        ('XEnd', {'default': 231, 'dt': '<h', 'value': 150}),
        ('YStart', {'default': 52, 'dt': '<h', 'value': 120}),
        ('YEnd', {'default': 172, 'dt': '<h', 'value': 150}),
        ('ZStart', {'default': 59, 'dt': '<h', 'value': 120}),
        ('ZEnd', {'default': 197, 'dt': '<h', 'value': 150}),
        ('LRConvention', {'default': 1, 'dt': '<b', 'value': 1}),
        ('RefSpace', {'default': 3, 'dt': '<b', 'value': 1}),
        ('TR', {'default': 2000.0, 'dt': '<f', 'value': 2000.0})
        ])
