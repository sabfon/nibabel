# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) VMP files.
for documentation on the file format see:
http://support.brainvoyager.com/installation-introduction/23-file-formats/377-users-guide-23-the-format-of-nr-vmp-files.html
Author: Thomas Emmerling
"""

from .bv import (BvError, BvFileHeader, BvFileImage, _proto2default,
                 update_BV_header)
from ..spatialimages import HeaderDataError

VMP_HDR_DICT_PROTO = (
    ('MagicNumber', 'I', 2712847316),
    ('VersionNumber', 'h', 6),
    ('DocumentType', 'h', 1),
    ('NrOfSubMaps', 'i', 1),
    ('NrOfTimePoints', 'i', 0),
    ('NrOfComponentParams', 'i', 0),
    ('ShowParamsRangeFrom', 'i', 0),
    ('ShowParamsRangeTo', 'i', 0),
    ('UseForFingerprintParamsRangeFrom', 'i', 0),
    ('UseForFingerprintParamsRangeTo', 'i', 0),
    ('XStart', 'i', 57),
    ('XEnd', 'i', 231),
    ('YStart', 'i', 52),
    ('YEnd', 'i', 172),
    ('ZStart', 'i', 59),
    ('ZEnd', 'i', 197),
    ('Resolution', 'i', 3),
    ('DimX', 'i', 256),
    ('DimY', 'i', 256),
    ('DimZ', 'i', 256),
    ('NameOfVTCFile', 'z', b'<none>'),
    ('NameOfProtocolFile', 'z', b''),
    ('NameOfVOIFile', 'z', b''),
    ('Maps', (
        ('TypeOfMap', 'i', 1),
        ('MapThreshold', 'f', 1.6500),
        ('UpperThreshold', 'f', 8),
        ('MapName', 'z', b'New Map'),
        ('PosMinR', 'B', 255),
        ('PosMinG', 'B', 0),
        ('PosMinB', 'B', 0),
        ('PosMaxR', 'B', 255),
        ('PosMaxG', 'B', 255),
        ('PosMaxB', 'B', 0),
        ('NegMinR', 'B', 255),
        ('NegMinG', 'B', 0),
        ('NegMinB', 'B', 255),
        ('NegMaxR', 'B', 0),
        ('NegMaxG', 'B', 0),
        ('NegMaxB', 'B', 255),
        ('UseVMPColor', 'B', 0),
        ('LUTFileName', 'z', b'<default>'),
        ('TransparentColorFactor', 'f', 1.0),
        ('NrOfLags', 'i', (0, 'TypeOfMap', 3)),
        ('DisplayMinLag', 'i', (0, 'TypeOfMap', 3)),
        ('DisplayMaxLag', 'i', (0, 'TypeOfMap', 3)),
        ('ShowCorrelationOrLag', 'i', (0, 'TypeOfMap', 3)),
        ('ClusterSizeThreshold', 'i', 50),
        ('EnableClusterSizeThreshold', 'B', 0),
        ('ShowValuesAboveUpperThreshold', 'i', 1),
        ('DF1', 'i', 249),
        ('DF2', 'i', 0),
        ('ShowPosNegValues', 'B', 3),
        ('NrOfUsedVoxels', 'i', 45555),
        ('SizeOfFDRTable', 'i', 0),
        ('FDRTableInfo', (
            ('q', 'f', 0),
            ('critStandard', 'f', 0),
            ('critConservative', 'f', 0),
        ), 'SizeOfFDRTable'),
        ('UseFDRTableIndex', 'i', 0),
    ), 'NrOfSubMaps'),
    ('ComponentTimePoints', (
        ('Timepoints', (('Timepoint', 'f', 0),), 'NrOfTimePoints'),
    ), 'NrOfSubMaps'),
    ('ComponentParams', (
        ('ParamName', 'z', b''),
        ('ParamValues', (('Value', 'f', 0),), 'NrOfSubMaps')
    ), 'NrOfComponentParams')
    )


class BvVmpHeader(BvFileHeader):
    ''' Class for BrainVoyager NR-VMP header
    '''

    # format defaults
    allowed_dtypes = [2]
    default_dtype = 2
    hdr_dict_proto = VMP_HDR_DICT_PROTO

    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._hdrDict
        # calculate dimensions
        z = (hdr['ZEnd'] - hdr['ZStart']) / hdr['Resolution']
        y = (hdr['YEnd'] - hdr['YStart']) / hdr['Resolution']
        x = (hdr['XEnd'] - hdr['XStart']) / hdr['Resolution']
        n = hdr['NrOfSubMaps']
        return tuple(int(d) for d in [n, z, y, x])

    def set_data_shape(self, shape=None, zyx=None, n=None):
        ''' Set shape of data
        To conform with nibabel standards this implements shape.
        However, to fill the BvVtcHeader with sensible information use the
        zyxn parameter instead.
        Parameters
        ----------
        shape: sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list [[1,2],[3,4],[5,6]]
           array storing borders of data
        n: int
           number of submaps
        '''
        hdrDict_old = self._hdrDict.copy()
        if (shape is None) and (zyx is None) and (n is None):
            raise HeaderDataError('Shape, zyx, or n needs to be specified!')

        if ((n is not None) and (n < 1)) or \
           ((shape is not None) and (shape[0] < 1)):
            raise HeaderDataError('NR-VMP files need at least one sub-map!')

        nc = self._hdrDict['NrOfSubMaps']
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from default coordinates.
            if len(shape) != 4:
                raise HeaderDataError(
                    'Shape for VMP files must be 4 dimensional (NZYX)!')
            self._hdrDict['XEnd'] = self._hdrDict['XStart'] + \
                (shape[3] * self._hdrDict['Resolution'])
            self._hdrDict['YEnd'] = self._hdrDict['YStart'] + \
                (shape[2] * self._hdrDict['Resolution'])
            self._hdrDict['ZEnd'] = self._hdrDict['ZStart'] + \
                (shape[1] * self._hdrDict['Resolution'])
            # if shape[0] > nc:
            #     for m in range(shape[0] - nc):
            #         self._hdrDict['Maps']\
            #             .append(_proto2default(self.hdr_dict_proto[23][1]))
            # elif shape[0] < nc:
            #     for m in range(nc - shape[0]):
            #         self._hdrDict['Maps'].pop()
            self._hdrDict['NrOfSubMaps'] = int(shape[0])
            self._hdrDict = update_BV_header(self.hdr_dict_proto,
                                             hdrDict_old, self._hdrDict)
            return
        if zyx is not None:
            self._hdrDict['XStart'] = zyx[2][0]
            self._hdrDict['XEnd'] = zyx[2][1]
            self._hdrDict['YStart'] = zyx[1][0]
            self._hdrDict['YEnd'] = zyx[1][1]
            self._hdrDict['ZStart'] = zyx[0][0]
            self._hdrDict['ZEnd'] = zyx[0][1]
        if n is not None:
            # if n > nc:
            #     for m in range(n - nc):
            #         self._hdrDict['Maps']\
            #             .append(_proto2default(self.hdr_dict_proto[23][1]))
            # elif n < nc:
            #     for m in range(nc - n):
            #         self._hdrDict['Maps'].pop()
            self._hdrDict['NrOfSubMaps'] = int(n)
            self._hdrDict = update_BV_header(self.hdr_dict_proto,
                                             hdrDict_old, self._hdrDict)

    def get_framing_cube(self):
        ''' Get the dimensions of the framing cube that constitutes
        the coordinate system boundaries for the bounding box.
        '''
        hdr = self._hdrDict
        return hdr['DimZ'], hdr['DimY'], hdr['DimX']

    def set_framing_cube(self, fc):
        ''' Set the dimensions of the framing cube that constitutes
        the coordinate system boundaries for the bounding box.
        For VMP files this puts the values also into the header.
        '''
        self._hdrDict['DimZ'] = fc[0]
        self._hdrDict['DimY'] = fc[1]
        self._hdrDict['DimX'] = fc[2]
        self._framing_cube = fc


class BvVmpImage(BvFileImage):
    # Set the class of the corresponding header
    header_class = BvVmpHeader

    # Set the label ('image') and the extension ('.vmp') for a VMP file
    files_types = (('image', '.vmp'),)
    valid_exts = ('.vmp',)

load = BvVmpImage.load
save = BvVmpImage.instance_to_filename