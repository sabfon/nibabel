# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Reading / writing functions for Brainvoyager (BV) VMP files

for documentation on the file format see:
http://support.brainvoyager.com/installation-introduction/23-file-formats/377-users-guide-23-the-format-of-nr-vmp-files.html

Author: Thomas Emmerling
'''

import numpy as np
from .bv import BvError,BvFileHeader,BvFileImage
from .spatialimages import HeaderDataError, HeaderTypeError
from .batteryrunners import Report

def _make_vmp_header_dtd(vtclt,prtlt,voilt):
    ''' Helper for creating a VMP header dtype with given parameters
    '''
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
    return vmp_header_dtd

def _make_vmp_submap_header_dtd(submap,maplt,lutlt,mStore,fdrl):
    ''' Helper for creating a VMP submap header dtype with given parameters

    Parameters
    ----------
    submap: int
        number of the current submap (zero-based)
    maplt: string (dtype)
        length of the MapName string (e.g. 'S12')
    lutlt: string (dtype)
        length of the LUT name string (e.g. 'S10')
    mStore: int
        1: if TypeOfMap == 3
        0: if TypeOfMap != 3
    fdrl: int
        size of FDR table
    '''
    vmp_submap_header_dtd = \
        (
            'map' + str(submap+1),[
                ('TypeOfMap', 'i4'),
                ('MapThreshold', 'f4'),
                ('UpperThreshold', 'f4'),
                ('MapName', maplt),
                ('PosMin', [
                    ('R', 'u1'),
                    ('G', 'u1'),
                    ('B', 'u1')
                ]),
                ('PosMax', [
                    ('R', 'u1'),
                    ('G', 'u1'),
                    ('B', 'u1')
                ]),
                ('NegMin', [
                    ('R', 'u1'),
                    ('G', 'u1'),
                    ('B', 'u1')
                ]),
                ('NegMax', [
                    ('R', 'u1'),
                    ('G', 'u1'),
                    ('B', 'u1')
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
    return vmp_submap_header_dtd

def _fill_default_vmp_submap_header(hdr, submap):
    ''' Helper for filling a default submap for a VMP header
    '''
    hdr['map' + str(submap+1)]['TypeOfMap'] = 1
    hdr['map' + str(submap+1)]['MapThreshold'] = 1.6500
    hdr['map' + str(submap+1)]['UpperThreshold'] = 8
    hdr['map' + str(submap+1)]['MapName'] = 'New Map'
    hdr['map' + str(submap+1)]['PosMin']['R'] = 255
    hdr['map' + str(submap+1)]['PosMin']['G'] = 0
    hdr['map' + str(submap+1)]['PosMin']['B'] = 0
    hdr['map' + str(submap+1)]['PosMax']['R'] = 255
    hdr['map' + str(submap+1)]['PosMax']['G'] = 255
    hdr['map' + str(submap+1)]['PosMax']['B'] = 0
    hdr['map' + str(submap+1)]['NegMin']['R'] = 255
    hdr['map' + str(submap+1)]['NegMin']['G'] = 0
    hdr['map' + str(submap+1)]['NegMin']['B'] = 255
    hdr['map' + str(submap+1)]['NegMax']['R'] = 0
    hdr['map' + str(submap+1)]['NegMax']['G'] = 0
    hdr['map' + str(submap+1)]['NegMax']['B'] = 255
    hdr['map' + str(submap+1)]['UseVMPColor'] = 0
    hdr['map' + str(submap+1)]['LUTFileName'] = '<default>'
    hdr['map' + str(submap+1)]['TransparentColorFactor'] = 1.0
    hdr['map' + str(submap+1)]['ClusterSizeThreshold'] = 50
    hdr['map' + str(submap+1)]['EnableClusterSizeThreshold'] = 0
    hdr['map' + str(submap+1)]['ShowValuesAboveUpperThreshold'] = 1
    hdr['map' + str(submap+1)]['DF1'] = 249
    hdr['map' + str(submap+1)]['DF2'] = 0
    hdr['map' + str(submap+1)]['ShowPosNegValues'] = 3
    hdr['map' + str(submap+1)]['NrOfUsedVoxels'] = 45555
    hdr['map' + str(submap+1)]['SizeOfFDRTable'] = 0
    hdr['map' + str(submap+1)]['UseFDRTableIndex'] = 0
    return hdr

class VmpHeader(BvFileHeader):
    ''' Class for BrainVoyager NR-VMP header
    '''

    # format defaults
    allowed_dtypes = [2]

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

    def set_data_shape(self, shape=None, zyx=None, n=None):
        ''' Set shape of data
        To conform with nibabel standards this implements shape.
        However, to fill the VtcHeader with sensible information use the zyxn parameter instead.

        Parameters
        ----------
        shape: sequence
           sequence of integers specifying data array shape
        zyx: 3x2 nested list [[1,2],[3,4],[5,6]]
           array storing borders of data
        n: int
           number of submaps

        '''
        if (shape is None) and (zyx is None) and (n is None):
            raise BvError('Shape, zyx, or n needs to be specified!')

        nc = self._structarr['NrOfSubMaps']
        if shape is not None:
            # Use zyx and t parameters instead of shape. Dimensions will start from default coordinates.
            if len(shape) != 4:
                raise BvError('Shape for VMP files must be 4 dimensional (NZYX)!')
            self._structarr['XEnd'] = 57 + (shape[3] * self._structarr['Resolution'])
            self._structarr['YEnd'] = 52 + (shape[2] * self._structarr['Resolution'])
            self._structarr['ZEnd'] = 59 + (shape[1] * self._structarr['Resolution'])
            if shape[0] > nc:
                self._add_submap(shape[0] - nc)
            elif shape[0] < nc:
                self._rem_submap(nc - shape[0])
            self._structarr['NrOfSubMaps'] = shape[0]
            return
        self._structarr['XStart'] = zyx[2][0]
        self._structarr['XEnd'] = zyx[2][1]
        self._structarr['YStart'] = zyx[1][0]
        self._structarr['YEnd'] = zyx[1][1]
        self._structarr['ZStart'] = zyx[0][0]
        self._structarr['ZEnd'] = zyx[0][1]
        if n is not None:
            if n > nc:
                self._add_submap(n - nc)
            elif n < nc:
                self._rem_submap(nc - n)
            self._structarr['NrOfSubMaps'] = n

    def get_framing_cube(self):
        ''' Get the dimensions of the framing cube that constitutes the coordinate system boundaries for the bounding box
        '''
        hdr = self._structarr
        return hdr['DimZ'], hdr['DimY'], hdr['DimX']

    def set_framing_cube(self, fc):
        ''' Set the dimensions of the framing cube that constitutes the coordinate system boundaries for the bounding box
        For VMP files this puts the values also into the header
        '''
        self._structarr['DimZ'] = fc[0]
        self._structarr['DimY'] = fc[1]
        self._structarr['DimX'] = fc[2]
        self._framing_cube = fc

    def update_template_dtype(self,binaryblock=None, item=None, value=None):
        ''' (Re-)Parse the binaryblock to update the header dtype

        Parameters
        ----------
        binaryblock: structarr.tostring() instance
            binaryblock of the header to parse
        item: string
            string field in header to change
        value: string
            length is used to set new dtype for field

        Returns
        -------
        newTemplate: list
            new list of dtype fields for header
        '''
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
        newTemplate = _make_vmp_header_dtd(vtclt,prtlt,voilt)

        ## start to pre-parse through loop over NrOfSubMaps
        for submap in range(nSubMaps):
            # find type of map - certain fields are only stored of MapType == 3
            mType = int(np.fromstring(binaryblock[point:point+4],np.uint32))
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

            # find size of FDR table
            fdrl = int(np.fromstring(binaryblock[point:point+4],np.uint32))
            point += (8 + (3*4*fdrl))

            # append the new submap to the header template
            newTemplate.append(_make_vmp_submap_header_dtd(submap,maplt,lutlt,mStore,fdrl))

        ## append loop for time course values
        newTemplate.append(
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
            newTemplate.append(componentparams)

        # maybe a dtype field should be changed
        if item is not None:
            # is it in the first level?
            if type(item) == str:
                field1 = [x for x in enumerate(newTemplate) if (x[1][0] == item)]
                newTemplate[field1[0][0]] = (field1[0][1][0], 'S'+str(len(value)+1))

            # it is on second level
            else:
                field1 = [x for x in enumerate(newTemplate) if (x[1][0] == item[0])]
                field2 = [x for x in enumerate(newTemplate[field1[0][0]][1]) if (x[1][0] == item[1])]
                newTemplate[field1[0][0]][1][field2[0][0]] = (field2[0][1][0], 'S'+str(len(value)+1))
        
        dt = np.dtype(newTemplate)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt

        return newTemplate

    def _add_submap(self,n=1):
        ''' Add a submap to the VMP header

        Parameters
        ----------
        n: number of submaps to add
        '''
        # gather some data about the old header
        if n == 0:
            return
        if n < 0:
            raise BvError('Please use _rem_submap for removing submaps!')
        oldkeys = self.keys()
        oldhdr = self._structarr
        newTemplate = self.update_template_dtype()
        mapn = self._structarr['NrOfSubMaps']
        lastmapind = [ind for ind,field in enumerate(newTemplate) if 'map' in field[0]][-1]
        if lastmapind is None:
            raise BvError('No Maps defined in VMP header!')

        # insert the new submaps into the header dtype template
        for newmap in range(n):
            newTemplate.insert(lastmapind+1+newmap,_make_vmp_submap_header_dtd(mapn+newmap,'S8','S10',0,0))
        dt = np.dtype(newTemplate)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt
        hdr = np.zeros((), dtype=dt)

        # copy the current values into the new header
        for key in oldkeys:
            hdr[key] = oldhdr[key]

        # fill the new submaps with default data
        for newmap in range(n):
            hdr = _fill_default_vmp_submap_header(hdr,mapn+newmap)

        # save the new number of submaps to header
        hdr['NrOfSubMaps'] = mapn+n
        self._structarr = hdr
        self.update_template_dtype()

    def _rem_submap(self,n=1):
        ''' Remove a submap to the VMP header

        Parameters
        ----------
        n: number of submaps to remove
        '''
        if n == 0:
            return
        if n < 0:
            raise BvError('Please use _add_submap for adding submaps!')
        if n >= self._structarr['NrOfSubMaps']:
            raise BvError('NR-VMP files need at least one sub-map!')
        # gather some data about the old header
        oldhdr = self._structarr
        newTemplate = self.update_template_dtype()
        mapn = self._structarr['NrOfSubMaps']
        mapind = [ind for ind,field in enumerate(newTemplate) if 'map' in field[0]][-n]
        if mapind is None:
            raise BvError('No Maps defined in VMP header!')

        # remove submaps from the header dtype template
        if n > 1:
            for map in range(n):
                newTemplate.pop(mapind[map])
        else:
            newTemplate.pop(mapind)
        dt = np.dtype(newTemplate)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt
        hdr = np.zeros((), dtype=dt)

        # copy the current values into the new header
        for key in [i for i in hdr.dtype.fields]:
            hdr[key] = oldhdr[key]

        # save the new number of submaps to header
        hdr['NrOfSubMaps'] = mapn-n
        self._structarr = hdr
        self.update_template_dtype()

    @classmethod
    def default_structarr(klass, endianness=None):
        ''' Return header data for empty header with given endianness
        (filled with standard values from the BV documentation)
        '''

        newTemplate = _make_vmp_header_dtd('S1','S1','S1')
        newTemplate.append(_make_vmp_submap_header_dtd(0,'S8','S10',0,0))

        dt = np.dtype(newTemplate)
        hdr = np.zeros((), dtype=dt)

        hdr['MagicNumber'] = -1582119980
        hdr['VersionNumber'] = 6
        hdr['DocumentType'] = 1
        hdr['NrOfSubMaps'] = 1
        hdr['NrOfTimePoints'] = 0
        hdr['NrOfComponentParams'] = 0
        hdr['ShowParamsRangeFrom'] = 0
        hdr['ShowParamsRangeTo'] = 0
        hdr['UseForFingerprintParamsRangeFrom'] = 0
        hdr['UseForFingerprintParamsRangeTo'] = 0
        hdr['XStart'] = 57
        hdr['XEnd'] = 231
        hdr['YStart'] = 52
        hdr['YEnd'] = 172
        hdr['ZStart'] = 59
        hdr['ZEnd'] = 197
        hdr['Resolution'] = 3
        hdr['DimX'] = 256
        hdr['DimY'] = 256
        hdr['DimZ'] = 256
        hdr['NameOfVTCFile'] = ''
        hdr['NameOfProtocolFile'] = '' 
        hdr['NameOfVOIRFile'] = ''

        hdr = _fill_default_vmp_submap_header(hdr, 0)

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
