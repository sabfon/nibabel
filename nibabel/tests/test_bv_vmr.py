import os
from os.path import join as pjoin
import numpy as np
import ast
from .. brainvoyager.bv_vmr import *
from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path)


vmr_file = os.path.join(data_path, 'test.vmr')
fileobj = open(vmr_file, 'r')
test_file = ast.literal_eval(open(os.path.join(data_path, 'test_vmr_header.txt')).read()) #data obtained from NeuroElf

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


def compareValues(header, testHeader):
    for key in header:
        if (type(header[key]) is list):
            num = len(testHeader[key])
            for i in range(0,num):
                compareValues(header[key][i], testHeader[key][i] )

        assert_equal(header[key], testHeader[key])

def test_parse_VMR_header():
    vmr = BvVmrHeader()
    header = vmr.from_fileobj(fileobj)
    header = header._hdrDict

    compareValues(header, test_file)

def test_wrong_input():
    vmr = VmrHeader()
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
