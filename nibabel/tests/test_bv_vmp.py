import os
from os.path import join as pjoin
import numpy as np
from .. brainvoyager.bv_vmp import *
from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path)

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.vmp'),
        shape=(1, 10, 10, 10),
        dtype=np.float32,
        affine=np.array([[-3., 0, 0, -21.],
                         [0, 0, -3., -21.],
                         [0, -3., 0, -21.],
                         [0, 0, 0, 1.]]),
        zooms=(3., 3., 3.),
        # These values are from NeuroElf
        data_summary=dict(
            min=0.0033484352752566338,
            max=7.996956825256348,
            mean=3.9617851),
        is_proxy=True)
]
