# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) VTC files.

for documentation on the file format see:
http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html

Author: Thomas Emmerling
"""

import numpy as np
from .bv import BvError, BvFileHeader, BvFileImage
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report

VTC_HDR_DICT_PROTO = (
    ('version', 'h', 3),
    ('fmr', 'z', b''),
    ('nPrt', 'h', 0),
    ('prts', (('filename', 'z', b''),), 'nPrt'),
    ('currentPrt', 'h', 0),
    ('datatype', 'h', 2),
    ('volumes', 'h', 0),
    ('Resolution', 'h', 3),
    ('XStart', 'h', 57),
    ('XEnd', 'h', 231),
    ('YStart', 'h', 52),
    ('YEnd', 'h', 172),
    ('ZStart', 'h', 59),
    ('ZEnd', 'h', 197),
    ('LRConvention', 'b', 1),
    ('RefSpace', 'b', 3),
    ('TR', 'f', 2000.0)
    )


class VtcHeader(BvFileHeader):

    """
    Header for Brainvoyager (BV) VTC files.

    For documentation on the file format see:
    http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html
    """

    # format defaults
    allowed_dtypes = [2]
    default_dtype = 2
    hdr_dict_proto = VTC_HDR_DICT_PROTO

    def get_data_shape(self):
        """Get shape of data."""
        hdr = self._hdrDict
        # calculate dimensions
        z = (hdr['ZEnd'] -
             hdr['ZStart']) / hdr['Resolution']
        y = (hdr['YEnd'] -
             hdr['YStart']) / hdr['Resolution']
        x = (hdr['XEnd'] -
             hdr['XStart']) / hdr['Resolution']
        t = hdr['volumes']
        return tuple(int(d) for d in [z, y, x, t])

    def set_data_shape(self, shape=None, zyx=None, t=None):
        """Set shape of data.

        To conform with nibabel standards this implements shape.
        However, to fill the VtcHeader with sensible information
        use the zyxt parameter instead.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        zyxt: 3x2 nested list [[1,2],[3,4],[5,6]]
           array storing borders of data
        t: int
           number of volumes
        """
        if (shape is None) and (zyx is None) and (t is None):
            raise BvError('Shape, zyx, or t needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from standard coordinates.
            if len(shape) != 4:
                raise BvError('Shape for VTC files must be 4 dimensional!')
            self._hdrDict['XEnd'] = \
                57 + (shape[2] * self._hdrDict['Resolution'])
            self._hdrDict['YEnd'] = \
                52 + (shape[1] * self._hdrDict['Resolution'])
            self._hdrDict['ZEnd'] = \
                59 + (shape[0] * self._hdrDict['Resolution'])
            self._hdrDict['volumes'] = shape[3]
            return
        self._hdrDict['XStart'] = zyx[2][0]
        self._hdrDict['XEnd'] = zyx[2][1]
        self._hdrDict['YStart'] = zyx[1][0]
        self._hdrDict['YEnd'] = zyx[1][1]
        self._hdrDict['ZStart'] = zyx[0][0]
        self._hdrDict['ZEnd'] = zyx[0][1]
        if t is not None:
            self._hdrDict['volumes'] = t

    def get_xflip(self):
        """Get xflip for data."""
        xflip = int(self._hdrDict['LRConvention'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')

    def set_xflip(self, xflip):
        """Set xflip for data."""
        if xflip is True:
            self._hdrDict['LRConvention'] = 1
        elif xflip is False:
            self._hdrDict['LRConvention'] = 2
        else:
            self._hdrDict['LRConvention'] = 0

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class."""
        return (klass._chk_fileversion,
                klass._chk_sizeof_hdr,
                klass._chk_datatype,
                )

    ''' Check functions in format expected by BatteryRunner class '''

    @classmethod
    def _chk_fileversion(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['version'] == 3:
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'only fileversion 3 is supported at the moment!'
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep

    @classmethod
    def _chk_sizeof_hdr(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr.template_dtype.itemsize == len(hdr.binaryblock):
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'size of binaryblock should be ' + \
            str(hdr.template_dtype.itemsize)
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep

    @classmethod
    def _chk_datatype(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dtype = klass._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 40
            rep.problem_msg = 'data code %d not recognized' % code
        else:
            if dtype.itemsize == 0:
                rep.problem_level = 40
                rep.problem_msg = 'data code %d not supported' % code
            else:
                return hdr, rep
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep


class VtcImage(BvFileImage):

    """Class for BrainVoyager VTC images."""

    # Set the class of the corresponding header
    header_class = VtcHeader

    # Set the label ('image') and the extension ('.vtc') for a VTC file
    files_types = (('image', '.vtc'),)

load = VtcImage.load
save = VtcImage.instance_to_filename
