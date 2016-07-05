# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test BV module for MSK files."""

from os.path import join as pjoin
import numpy as np
from ...testing import (data_path)

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'test.msk'),
        shape=(10, 10, 10),
        dtype=np.uint8,
        affine=np.array([[-3., 0, 0, -21.],
                         [0, 0, -3., -21.],
                         [0, -3., 0, -21.],
                         [0, 0, 0, 1.]]),
        zooms=(3., 3., 3.),
        # These values are from NeuroElf
        data_summary=dict(
            min=0,
            max=1,
            mean=0.499),
        is_proxy=True)
]
