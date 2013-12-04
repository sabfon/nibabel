# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Header reading / writing functions for Brainvoyager (BV) file formats

Author: Thomas Emmerling
'''

import numpy as np
from .bv import BvError,BvFileHeader,BvFileImage, data_type_codes

# List of fields to expect in VMP header
vmp_header_dtype = [
    ('version', 'i2'),
    ('type', 'i2'),
    ('volumes', 'i2'),
    ('relResolution', 'i2'),
    ('XStart', 'i2'),
    ('XEnd', 'i2'),
    ('YStart', 'i2'),
    ('YEnd', 'i2'),
    ('ZStart', 'i2'),
    ('ZEnd', 'i2'),
    ('LRConvention', 'i1'),
    ('RefSpace', 'i1'),
    ('TR', 'f4'),
    ]

class MskHeader(BvFileHeader):
    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._structarr
        # calculate dimensions
        z = (hdr['ZEnd'] - hdr['ZStart']) / hdr['relResolution']
        y = (hdr['YEnd'] - hdr['YStart']) / hdr['relResolution']
        x = (hdr['XEnd'] - hdr['XStart']) / hdr['relResolution']

        return tuple(int(d) for d in [z,y,x])

    def update_template_dtype(self,binaryblock=None, item=None, value=None):

        msk_header_dtd = \
            [
                ('relResolution', 'i2'),
                ('XStart', 'i2'),
                ('XEnd', 'i2'),
                ('YStart', 'i2'),
                ('YEnd', 'i2'),
                ('ZStart', 'i2'),
                ('ZEnd', 'i2')
            ]

        dt = np.dtype(msk_header_dtd)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt

        return dt

    @classmethod
    def default_structarr(klass):
        ''' Return header data for empty header
        '''

        msk_header_dtd = \
            [
                ('relResolution', 'i2'),
                ('XStart', 'i2'),
                ('XEnd', 'i2'),
                ('YStart', 'i2'),
                ('YEnd', 'i2'),
                ('ZStart', 'i2'),
                ('ZEnd', 'i2')
            ]

        dt = np.dtype(msk_header_dtd)
        hdr = np.zeros((), dtype=dt)

        hdr['relResolution'] = 3
        hdr['XStart'] = 57
        hdr['XEnd'] = 231
        hdr['YStart'] = 52
        hdr['YEnd'] = 172
        hdr['ZStart'] = 59
        hdr['ZEnd'] = 197

        return hdr

    def get_data_dtype(self):
        ''' Get numpy dtype for data

        For examples see ``set_data_dtype``
        '''
        return np.uint8

    def set_data_dtype(self, datatype):
        ''' Set numpy dtype for data from code or dtype or type
        '''
        raise BvError('Mask files can only be np.uint8!')

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return ()

class MskImage(BvFileImage):
    # Set the class of the corresponding header
    header_class = MskHeader

    # Set the label ('image') and the extension ('.msk') for a MSK file
    files_types = (('image', '.msk'),)

load = MskImage.load
save = MskImage.instance_to_filename