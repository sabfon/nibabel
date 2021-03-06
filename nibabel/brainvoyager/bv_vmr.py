# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) VMR files.
for documentation on the file format see:
http://support.brainvoyager.com/automation-aamp-development/23-file-formats/385-developer-guide-26-the-format-of-vmr-files.html
Author: Sabrina Fontanella and Thomas Emmerling
"""

from .bv import (BvError, BvFileHeader, BvFileImage, parse_BV_header,
                 pack_BV_header, calc_BV_header_size, combineST, parseST)
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report
import numpy as np


VMR_PRHDR_DICT_PROTO = (
    ('version', 'h', 4),
    ('dimX', 'h', 256),
    ('dimY', 'h', 256),
    ('dimZ', 'h', 256)
)

VMR_PSHDR_DICT_PROTO = (
    ('offsetX', 'h', 0),
    ('offsetY', 'h', 0),
    ('offsetZ', 'h', 0),
    ('framingCube', 'h', 256),
    ('posInfosVerified', 'i', 0),
    ('coordSysEntry', 'i', 1),
    ('slice1CenterX', 'f', 127.5),
    ('slice1CenterY', 'f', 0),
    ('slice1CenterZ', 'f', 0),
    ('sliceNCenterX', 'f', -127.5),
    ('sliceNCenterY', 'f', 0),
    ('SliceNCenterZ', 'f', 0),
    ('sliceRowDirX', 'f', 0),
    ('sliceRowDirY', 'f', 1),
    ('sliceRowDirZ', 'f', 0),
    ('sliceColDirX', 'f', 0),
    ('sliceColDirY', 'f', 0),
    ('sliceColDirZ', 'f', -1),
    ('nrRowsSlice', 'i', 256),
    ('nrColSlice', 'i', 256),
    ('foVRowDir', 'f', 256),
    ('foVColDir', 'f', 256),
    ('sliceThick', 'f', 1),
    ('gapThick', 'f', 0),
    ('nrOfPastSpatTrans', 'i', 0),
    ('pastST', (
        ('name', 'z', b''),
        ('type', 'i', b''),
        ('sourceFile', 'z', b''),
        ('numTransVal', 'i', b''),
        ('transfVal', (('value', 'f', b''),), 'numTransVal')
    ), 'nrOfPastSpatTrans'),
    ('lrConvention', 'B', 1),
    ('referenceSpace', 'B', 0),
    ('voxResX', 'f', 1),
    ('voxResY', 'f', 1),
    ('voxResZ', 'f', 1),
    ('flagVoxResolution', 'B', 0),
    ('flagTalSpace', 'B', 0),
    ('minIntensity', 'i', 0),
    ('meanIntensity', 'i', 127),
    ('maxIntensity', 'i', 255)
)


def computeOffsetPostHDR(hdrDict, fileobj):
    currentSeek = fileobj.tell()
    return currentSeek + (hdrDict['dimX']*hdrDict['dimY']*hdrDict['dimZ'])


def concatePrePos(preDict, posDict):
    temp = preDict.copy()
    temp.update(posDict)
    return temp


class BvVmrHeader(BvFileHeader):
    """Class for BrainVoyager VMR header."""

    # format defaults
    default_endianness = '<'
    allowed_dtypes = [3]
    default_dtype = 3
    hdr_dict_proto = VMR_PRHDR_DICT_PROTO + VMR_PSHDR_DICT_PROTO

    def get_data_shape(self):
        hdr = self._hdrDict
        # calculate dimensions
        z = hdr['dimZ']
        y = hdr['dimY']
        x = hdr['dimX']
        return tuple(int(d) for d in [z, y, x])

    def set_data_shape(self, shape=None, zyx=None):
        if (shape is None) and (zyx is None):
            raise HeaderDataError('Shape or zyx needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from standard coordinates.
            if len(shape) != 3:
                raise HeaderDataError(
                    'Shape for VMR files must be 3 dimensional!')
            self._hdrDict['dimX'] = shape[2]
            self._hdrDict['dimY'] = shape[1]
            self._hdrDict['dimZ'] = shape[0]
            return
        self._hdrDict['dimX'] = zyx[2][1] - zyx[2][0]
        self._hdrDict['dimY'] = zyx[1][1] - zyx[1][0]
        self._hDict['dimZ'] = zyx[0][1] - zyx[0][0]

    def set_data_offset(self, offset):
        """Set offset into data file to read data.
        The offset is always 8 for VMR files.
        """
        self._data_offset = 8

    def get_data_offset(self):
        """Return offset into data file to read data.
        The offset is always 8 for VMR files.
        """
        return 8

    def set_xflip(self, xflip):
        if xflip is True:
            self._hdrDict['lrConvention'] = 1
        elif xflip is False:
            self._hdrDict['lrConvention'] = 2
        else:
            self._hdrDict['lrConvention'] = 0

    def get_xflip(self):
        xflip = int(self._hdrDict['lrConvention'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')

    def get_base_affine(self):
        """Get affine from VMR header fields.

        Internal storage of the image is ZYXT, where (in patient coordiante/
        real world orientations):
        Z := axis increasing from right to left (R to L)
        Y := axis increasing from superior to inferior (S to I)
        X := axis increasing from anterior to posterior (A to P)
        T := volumes (if present in file format)
        """
        zooms = self.get_zooms()
        if not self.get_xflip():
            # make the BV internal Z axis neurological (left-is-left);
            # not default in BV files!
            zooms[0] *= -1

        # compute the rotation
        rot = np.zeros((3, 3))
        # make the flipped BV Z axis the new R axis
        rot[:, 0] = [-zooms[0], 0, 0]
        # make the flipped BV X axis the new A axis
        rot[:, 1] = [0, 0, -zooms[2]]
        # make the flipped BV Y axis the new S axis
        rot[:, 2] = [0, -zooms[1], 0]

        # compute the translation
        fcc = np.array(self.get_framing_cube())/2  # center of framing cube
        bbc = np.array(self.get_bbox_center())  # center of bounding box
        tra = np.dot((bbc-fcc), rot)

        # assemble
        M = np.eye(4, 4)
        M[0:3, 0:3] = rot
        M[0:3, 3] = tra.T

        # look for additional transformations in pastST
        if self._hdrDict['pastST']:
            STarray = []
            for st in range(len(self._hdrDict['pastSt'])):
                STarray.append(parseST(self._hdrDict['pastST'][st]))
        return M

    @classmethod
    def _get_checks(klass):
        return (klass._chk_fileversion,
                klass._chk_sizeof_hdr,
                klass._chk_datatype,
                )

    @classmethod
    def from_fileobj(klass, fileobj, endianness=default_endianness,
                     check=True):
        hdrDictPre = parse_BV_header(VMR_PRHDR_DICT_PROTO, fileobj)
        # calculate new seek for the post data header
        newSeek = computeOffsetPostHDR(hdrDictPre, fileobj)
        fileobj.seek(newSeek)
        hdrDictPos = parse_BV_header(VMR_PSHDR_DICT_PROTO, fileobj)
        hdrDict = concatePrePos(hdrDictPre, hdrDictPos)
        # The offset is always 8 for VMR files.
        offset = 8
        return klass(hdrDict, endianness, check, offset)

    def get_bbox_center(self):
        """ Get the center coordinate of the bounding box. """
        hdr = self._hdrDict

        x = hdr['offsetX'] + (hdr['dimX']/2)
        y = hdr['offsetY'] + (hdr['dimY']/2)
        z = hdr['offsetZ'] + (hdr['dimZ']/2)
        return z, y, x


    def get_zooms(self):
        return (self._hdrDict['voxResZ'], self._hdrDict['voxResY'],
                self._hdrDict['voxResX'])

    def set_zooms(self, zooms):
        # check if the input type is correct
        if all(isinstance(i, float) for i in zooms) is False:
            raise BvError('Zooms for VMR files must be float values!')
        if len(zooms) != 3:
            raise BvError('Zooms for VMR files must be 3 values (ZYX)!')
        self._hdrDict['voxResZ'] = float(zooms[0])
        self._hdrDict['voxResY'] = float(zooms[1])
        self._hdrDict['voxResX'] = float(zooms[2])

    def write_to(self, fileobj):
        binaryblock = pack_BV_header(self.hdr_dict_proto, self._hdrDict)
        # calculate size of preDataHeader
        sizePrH = calc_BV_header_size(VMR_PRHDR_DICT_PROTO, self._hdrDict)
        # write the preHeader
        fileobj.write(binaryblock[0:sizePrH])
        fileobj.seek(computeOffsetPostHDR(self._hdrDict, fileobj))
        fileobj.write(binaryblock[sizePrH:])

    ''' Check functions in format expected by BatteryRunner class '''

    @classmethod
    def _chk_fileversion(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['version'] == 4:
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'only fileversion 4 is supported at the moment!'
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


class BvVmrImage(BvFileImage):
    """Class for BrainVoyager VMR images."""

    # Set the class of the corresponding header
    header_class = BvVmrHeader

    # Set the label ('image') and the extension ('.vtc') for a VMR file
    files_types = (('image', '.vmr'),)
    valid_exts = ('.vmr',)

load = BvVmrImage.load
save = BvVmrImage.instance_to_filename