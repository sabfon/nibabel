# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test BV module for VTC files."""

from os.path import join as pjoin
import numpy as np
from .. brainvoyager.bv_vtc import BvVtcHeader
from ..testing import (data_path)
from numpy.testing import (assert_array_equal)

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


def test_get_base_affine():
    hdr = BvVtcHeader()
    hdr.set_data_shape((3, 5, 7, 9))
    hdr.set_zooms((3, 3, 3, 3))
    assert_array_equal(hdr.get_base_affine(),
                       np.asarray([[-3.,  0.,  0.,  195.],
                                   [0.,  0.,  -3., 183.],
                                   [0.,  -3.,  0., 207.],
                                   [0.,  0.,  0.,  1.]]))
