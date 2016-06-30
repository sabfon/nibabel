import os
from os.path import join as pjoin
import numpy as np
from .. brainvoyager.bv_vtc import *
from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path)

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.vtc'),
        shape=(10, 10, 10, 5),
        dtype=np.float32,
        affine=np.array([[-3., 0, 0, -21.],
                         [0, 0, -3., -21.],
                         [0, -3., 0, -21.],
                         [0, 0, 0, 1.]]),
        zooms=(3., 3., 3.),
        # These values are from NeuroElf
        data_summary=dict(
            min=0.0096689118,
            max=199.93549,
            mean=100.19728),
        is_proxy=True)
]
