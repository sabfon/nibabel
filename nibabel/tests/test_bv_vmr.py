# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test BV module for VMR files."""

from os.path import join as pjoin
import numpy as np
from ..brainvoyager.bv import BvError
from ..brainvoyager.bv_vmr import BvVmrImage, BvVmrHeader
from ..testing import (assert_equal, data_path)
from ..externals import OrderedDict

vmr_file = pjoin(data_path, 'test.vmr')

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.vmr'),
        shape=(5, 4, 3),
        dtype=np.uint8,
        affine=np.array([[-3., 0, 0, -21.],
                         [0, 0, -3., -21.],
                         [0, -3., 0, -21.],
                         [0, 0, 0, 1.]]),
        zooms=(3., 3., 3.),
        # These values are from NeuroElf
        data_summary=dict(
            min=7,
            max=218,
            mean=120.3),
        is_proxy=True)
]

EXAMPLE_HDR = OrderedDict({
    'version': 4,
    'dimX': 3,
    'dimY': 4,
    'dimZ': 5,
    'offsetX': 0,
    'offsetY': 0,
    'offsetZ': 0,
    'framingCube': 256,
    'posInfosVerified': 0,
    'coordSysEntry': 1,
    'slice1CenterX': 127.5,
    'slice1CenterY': 0,
    'slice1CenterZ': 0,
    'sliceNCenterX': -127.5,
    'sliceNCenterY': 0,
    'SliceNCenterZ': 0,
    'sliceRowDirX': 0,
    'sliceRowDirY': 1,
    'sliceRowDirZ': 0,
    'sliceColDirX': 0,
    'sliceColDirY': 0,
    'sliceColDirZ': -1,
    'nrRowsSlice': 256,
    'nrColSlice': 256,
    'foVRowDir': 256,
    'foVColDir': 256,
    'sliceThick': 1,
    'gapThick': 0,
    'nrOfPastSpatTrans': 2,
    'pastST': [{
        'name': 'NoName',
        'type': 2,
        'sourceFile': '/home/test.vmr',
        'numTransVal': 16,
        'transfVal': [
                {'value': 1.0},
                {'value': 0.0},
                {'value': 0.0},
                {'value': -1.0},
                {'value': 0.0},
                {'value': 1.0},
                {'value': 0.0},
                {'value': 0.0},
                {'value': 0.0},
                {'value': 0.0},
                {'value': 1.0},
                {'value': -1.0},
                {'value': 0.0},
                {'value': 0.0},
                {'value': 0.0},
                {'value': 1.0}
        ],
    },
        {
        'name': 'NoName',
                'type': 2,
                'sourceFile': '/home/test_TRF.vmr',
                'numTransVal': 16,
                'transfVal': [
                    {'value': 1.0},
                    {'value': 0.0},
                    {'value': 0.0},
                    {'value': 1.0},
                    {'value': 0.0},
                    {'value': 1.0},
                    {'value': 0.0},
                    {'value': 1.0},
                    {'value': 0.0},
                    {'value': 0.0},
                    {'value': 1.0},
                    {'value': 0.0},
                    {'value': 0.0},
                    {'value': 0.0},
                    {'value': 0.0},
                    {'value': 1.0}
                ]
    }],
    'lrConvention': 1,
    'referenceSpace': 0,
    'voxResX': 1,
    'voxResY': 1,
    'voxResZ': 1,
    'flagVoxResolution': 0,
    'flagTalSpace': 0,
    'minIntensity': 0,
    'meanIntensity': 127,
    'maxIntensity': 255,
})


def compareValues(header, testHeader):
    for key in header:
        if (type(header[key]) is list):
            num = len(testHeader[key])
            for i in range(0, num):
                compareValues(header[key][i], testHeader[key][i])
        assert_equal(header[key], testHeader[key])


def test_parse_BVVMR_header():
    vmr = BvVmrImage.from_filename(vmr_file)
    compareValues(vmr.header._hdrDict, EXAMPLE_HDR)


def test_wrong_input():
    vmr = BvVmrHeader()
    try:
        vmr.set_zooms(('a', 2.0, 3))
    except BvError:
        print ("Wrong type parameter for set_zoom: 2.0, 3")
        pass

    try:
        vmr.set_zooms((2.0, 3.0))
    except BvError:
        print ("Wrong number of input parameter for set_zoom")
        pass