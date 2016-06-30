# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) MSK files.

for documentation on the file format see:
http://www.brainvoyager.com/ubb/Forum8/HTML/000087.html

Author: Thomas Emmerling
"""

from .bv import BvError, BvFileHeader, BvFileImage

MSK_HDR_DICT_PROTO = (
    ('Resolution', 'h', 3),
    ('XStart', 'h', 57),
    ('XEnd', 'h', 231),
    ('YStart', 'h', 52),
    ('YEnd', 'h', 172),
    ('ZStart', 'h', 59),
    ('ZEnd', 'h', 197),
    )



class BvMskHeader(BvFileHeader):
    """Class for BrainVoyager MSK header."""

    # format defaults
    allowed_dtypes = [3]
    default_dtype = 3
    hdr_dict_proto = MSK_HDR_DICT_PROTO

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

        return tuple(int(d) for d in [z, y, x])

    def set_data_shape(self, shape=None, zyx=None):
        """Set shape of data.

        To conform with nibabel standards this implements shape.
        However, to fill the VtcHeader with sensible information use
        the zyxt parameter instead.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list [[XStart,XEnd],[YStart,YEnd],[ZStart,ZEnd]]
           array storing borders of data
        """
        if (shape is None) and (zyx is None):
            raise BvError('Shape or zyx needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from standard coordinates.
            if len(shape) != 3:
                raise BvError('Shape for MSK files must be 3 dimensional!')
            self._hdrDict['XEnd'] = self._hdrDict['XStart'] + \
                (shape[2] * self._hdrDict['Resolution'])
            self._hdrDict['YEnd'] = self._hdrDict['YStart'] + \
                (shape[1] * self._hdrDict['Resolution'])
            self._hdrDict['ZEnd'] = self._hdrDict['ZStart'] + \
                (shape[0] * self._hdrDict['Resolution'])
            return
        self._hdrDict['XStart'] = zyx[0][0]
        self._hdrDict['XEnd'] = zyx[0][1]
        self._hdrDict['YStart'] = zyx[1][0]
        self._hdrDict['YEnd'] = zyx[1][1]
        self._hdrDict['ZStart'] = zyx[2][0]
        self._hdrDict['ZEnd'] = zyx[2][1]

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class."""
        return ()


class BvMskImage(BvFileImage):
    """Class for BrainVoyager MSK masks.

    MSK files are technically binary images
    """

    # Set the class of the corresponding header
    header_class = BvMskHeader

    # Set the label ('image') and the extension ('.msk') for a MSK file
    files_types = (('image', '.msk'),)
    valid_exts = ('.msk',)

load = BvMskImage.load
save = BvMskImage.instance_to_filename
