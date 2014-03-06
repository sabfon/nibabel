# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Reading / writing functions for Brainvoyager (BV) MSK files

for documentation on the file format see:
http://www.brainvoyager.com/ubb/Forum8/HTML/000087.html

Author: Thomas Emmerling
'''

import numpy as np
from .bv import BvError,BvFileHeader,BvFileImage

# template for MSK header
msk_header_dtd = \
    [
        ('Resolution', 'i2'),
        ('XStart', 'i2'),
        ('XEnd', 'i2'),
        ('YStart', 'i2'),
        ('YEnd', 'i2'),
        ('ZStart', 'i2'),
        ('ZEnd', 'i2')
    ]

class MskHeader(BvFileHeader):
    '''Class for BrainVoyager MSK header
    '''
    #copy of module-level template definition
    template = msk_header_dtd

    # format defaults
    allowed_dtypes = [3]
    default_dtype = 3

    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._structarr
        # calculate dimensions
        z = (hdr['ZEnd'] - hdr['ZStart']) / hdr['Resolution']
        y = (hdr['YEnd'] - hdr['YStart']) / hdr['Resolution']
        x = (hdr['XEnd'] - hdr['XStart']) / hdr['Resolution']

        return tuple(int(d) for d in [z,y,x])

    def set_data_shape(self, shape=None, zyx=None):
        ''' Set shape of data
        To conform with nibabel standards this implements shape.
        However, to fill the VtcHeader with sensible information use the zyxt parameter instead.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list [[XStart,XEnd],[YStart,YEnd],[ZStart,ZEnd]]
           array storing borders of data

        '''
        if (shape is None) and (zyx is None):
            raise BvError('Shape or zyx needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape. Dimensions will start from standard coordinates.
            if len(shape) != 3:
                raise BvError('Shape for MSK files must be 3 dimensional!')
            self._structarr['XEnd'] = 57 + (shape[2] * self._structarr['Resolution'])
            self._structarr['YEnd'] = 52 + (shape[1] * self._structarr['Resolution'])
            self._structarr['ZEnd'] = 59 + (shape[0] * self._structarr['Resolution'])
            return
        self._structarr['XStart'] = zyx[0][0]
        self._structarr['XEnd'] = zyx[0][1]
        self._structarr['YStart'] = zyx[1][0]
        self._structarr['YEnd'] = zyx[1][1]
        self._structarr['ZStart'] = zyx[2][0]
        self._structarr['ZEnd'] = zyx[2][1]

    def update_template_dtype(self,binaryblock=None, item=None, value=None):
        dt = np.dtype(self.template)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt

        return self.template

    @classmethod
    def default_structarr(klass):
        ''' Return header data for empty header (filled with standard values from BV documentation)
        '''
        dt = np.dtype(klass.template)
        hdr = np.zeros((), dtype=dt)

        hdr['Resolution'] = 3
        hdr['XStart'] = 57
        hdr['XEnd'] = 231
        hdr['YStart'] = 52
        hdr['YEnd'] = 172
        hdr['ZStart'] = 59
        hdr['ZEnd'] = 197

        return hdr

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return ()

class MskImage(BvFileImage):
    ''' Class for BrainVoyager MSK masks
    MSK files are technically binary images
    '''
    # Set the class of the corresponding header
    header_class = MskHeader

    # Set the label ('image') and the extension ('.msk') for a MSK file
    files_types = (('image', '.msk'),)

load = MskImage.load
save = MskImage.instance_to_filename