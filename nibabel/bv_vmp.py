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
from .spatialimages import HeaderDataError, HeaderTypeError
from .batteryrunners import Report

class VmpHeader(BvFileHeader):
    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._structarr
        # calculate dimensions
        z = (hdr['ZEnd'] - hdr['ZStart']) / hdr['Resolution']
        y = (hdr['YEnd'] - hdr['YStart']) / hdr['Resolution']
        x = (hdr['XEnd'] - hdr['XStart']) / hdr['Resolution']
        n = hdr['NrOfSubMaps']
        return tuple(int(d) for d in [n,z,y,x])

    def get_data_dtype(self):
        ''' Get numpy dtype for data

        For examples see ``set_data_dtype``
        '''
        dtype = self._data_type_codes.dtype[2]
        return dtype.newbyteorder(self.endianness)

    def set_data_dtype(self, datatype):
        ''' Set numpy dtype for data from code or dtype or type
        '''
        raise BvError('Cannot set different dtype for VMP files!')


    def update_template_dtype(self,binaryblock=None, item=None, value=None):
        if binaryblock is None:
            binaryblock = self.binaryblock

        # check for file version
        if int(np.fromstring(binaryblock[4:6],np.uint16)) != 6:
            raise BvError('Only NR-VMP files with file version 6 are supported!')

        ### OK, start with pre-parsing the binaryblock and include the stop byte ('\x00') in all strings
        point = 76

        # find the number of sub-maps/component maps
        nSubMaps = int(np.fromstring(binaryblock[8:12],np.uint32))
        if nSubMaps < 1:
            raise BvError('NR-VMP files need at least one sub-map!')

        # find the number of time points
        nTimePoints = int(np.fromstring(binaryblock[12:16],np.uint32))

        # find the number of component parameters
        nComponentParams = int(np.fromstring(binaryblock[16:20],np.uint32))

        # find length of vtc filename
        vtcl = binaryblock.find('\x00', point) - (point - 1)
        vtclt = 'S' + str(vtcl)
        point += vtcl

        # find length of prt filename
        prtl = binaryblock.find('\x00', point) - (point - 1)
        prtlt = 'S' + str(prtl)
        point += prtl

        # find length of voi filename
        voil = binaryblock.find('\x00', point) - (point - 1)
        voilt = 'S' + str(voil)
        point += voil

        # assemble first part of header dtype
        vmp_header_dtd = \
            [
                ('MagicNumber', 'i4'),
                ('VersionNumber', 'i2'),
                ('DocumentType', 'i2'),
                ('NrOfSubMaps', 'i4'),
                ('NrOfTimePoints', 'i4'),
                ('NrOfComponentParams', 'i4'),
                ('ShowParamsRangeFrom', 'i4'),
                ('ShowParamsRangeTo', 'i4'),
                ('UseForFingerprintParamsRangeFrom', 'i4'),
                ('UseForFingerprintParamsRangeTo', 'i4'),
                ('XStart', 'i4'),
                ('XEnd', 'i4'),
                ('YStart', 'i4'),
                ('YEnd', 'i4'),
                ('ZStart', 'i4'),
                ('ZEnd', 'i4'),
                ('Resolution', 'i4'),
                ('DimX', 'i4'),
                ('DimY', 'i4'),
                ('DimZ', 'i4'),
                ('NameOfVTCFile', vtclt),
                ('NameOfProtocolFile', prtlt),
                ('NameOfVOIRFile', voilt)
            ]

        ## start to pre-parse through loop over NrOfSubMaps
        submaps = []

        for submap in range(nSubMaps):
            maptemp = []

            # find type of map - certain fields are only stored of MapType == 3
            #mType = int(np.fromstring(binaryblock[point:point+4],np.uint32))
            mType = 2
            if mType == 3:
                mStore = 1
            else:
                mStore = 0

            point += 12
            # find length of map name
            mapl = binaryblock.find('\x00', point) - (point - 1)
            maplt = 'S' + str(mapl)
            point += (mapl + 13)

            # find length of LUT filename
            lutl = binaryblock.find('\x00', point) - (point - 1)
            lutlt = 'S' + str(lutl)
            if mType == 3:
                point += (lutl + 42)
            else:
                point += (lutl + 26)

            # find size of FRD table
            fdrl = int(np.fromstring(binaryblock[point:point+4],np.uint32))
            point += (8 + (3*4*fdrl))

            # fill FDR table according to size of FDR table
            #fdrtablerows = []
            #for row in range(fdrl):
            #    fdrtablerows.append(
            #        (
            #            'row' + str(row),
            #            [
            #                ('q', 'f4'),
            #                ('critStandard', 'f4'),
            #                ('critConservative', 'f4')
            #            ]
            #        )
            #    )
            #if fdrl == 0:
            #    fdrtablerows = 'V0'

            vmp_header_dtd.append(
                (
                    'map' + str(submap+1),[
                        ('TypeOfMap', 'i4'),
                        ('MapThreshold', 'i4'),
                        ('UpperThreshold', 'i4'),
                        ('MapName', maplt),
                        ('PosMin', [
                            ('R', 'b'),
                            ('G', 'b'),
                            ('B', 'b')
                        ]),
                        ('PosMax', [
                            ('R', 'b'),
                            ('G', 'b'),
                            ('B', 'b')
                        ]),
                        ('NegMin', [
                            ('R', 'b'),
                            ('G', 'b'),
                            ('B', 'b')
                        ]),
                        ('NegMax', [
                            ('R', 'b'),
                            ('G', 'b'),
                            ('B', 'b')
                        ]),
                        ('UseVMPColor', 'b'),
                        ('LUTFileName', lutlt),
                        ('TransparentColorFactor', 'f4'),
                        ('NrOfLags', 'i4', (mStore,)),
                        ('DisplayMinLag', 'i4', (mStore,)),
                        ('DisplayMaxLag', 'i4', (mStore,)),
                        ('ShowCorrelationOrLag', 'i4', (mStore,)),
                        ('ClusterSizeThreshold', 'i4'),
                        ('EnableClusterSizeThreshold', 'b'),
                        ('ShowValuesAboveUpperThreshold', 'i4'),
                        ('DF1', 'i4'),
                        ('DF2', 'i4'),
                        ('ShowPosNegValues', 'b'),
                        ('NrOfUsedVoxels', 'i4'),
                        ('SizeOfFDRTable', 'i4'),
                        ('FDRTableInfo', [
                            ('q', 'f4'),
                            ('critStandard', 'f4'),
                            ('critConservative', 'f4')
                        ], (fdrl,)),
                        ('UseFDRTableIndex', 'i4')
                    ]
                )
            )

        ## append loop for time course values
        vmp_header_dtd.append(
            ('timepoint', 'f4', (nTimePoints,))
        )

        ## start to pre-parse through loop over NrOfComponentsParams
        componentparams = []
        for componentparam in range(nComponentParams):
            cpnl = binaryblock.find('\x00', point) - (point - 1)
            cpnlt = 'S' + str(cpnl)
            point += (cpnl + (nSubMaps*4))

            componentparams.append(
                (
                    ('componentParamName' + str(componentparam), cpnlt),
                    ('componentParam' + str(componentparam), 'f4', (nSubMaps,))
                )
            )

        # append component parameters
        if nComponentParams != 0:
            vmp_header_dtd.append(componentparams)

        if item is not None:
            vmp_header_dtd = [(x[0], x[1]) if x[0] != item else (item, 'S'+str(len(value)+1)) for x in vmp_header_dtd]
        
        dt = np.dtype(vmp_header_dtd)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt

        return dt

    @classmethod
    def default_structarr(klass, endianness=None):
        ''' Return header data for empty header with given endianness
        '''

        vtc_header_dtd = \
            [
                ('version', 'i2'),
                ('fmr', 'S0'),
                ('nPrt', 'i2'),
                ('prts', 'S0'),
                ('currentPrt', 'i2'),
                ('datatype', 'i2'),
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
                ('TR', 'f4')
            ]

        dt = np.dtype(vtc_header_dtd)
        hdr = np.zeros((), dtype=dt)

        hdr['version'] = 3
        hdr['fmr'] = ''
        hdr['nPrt'] = 0
        hdr['prts'] = ''
        hdr['currentPrt'] = 0
        hdr['datatype'] = 2
        hdr['volumes'] = 0
        hdr['relResolution'] = 3
        hdr['XStart'] = 57
        hdr['XEnd'] = 231
        hdr['YStart'] = 52
        hdr['YEnd'] = 172
        hdr['ZStart'] = 59
        hdr['ZEnd'] = 197
        hdr['LRConvention'] = 1
        hdr['RefSpace'] = 3
        hdr['TR'] = 2000

        return hdr

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return ()

class VmpImage(BvFileImage):
    # Set the class of the corresponding header
    header_class = VmpHeader

    # Set the label ('image') and the extension ('.vmp') for a VMP file
    files_types = (('image', '.vmp'),)

load = VmpImage.load
save = VmpImage.instance_to_filename