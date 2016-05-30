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

Author: Sabrina Fontanella
"""

from .bv import BvError, BvFileHeader, BvFileImage, parse_BV_header, pack_BV_header, calc_BV_header_size
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report


VMR_PRHDR_DICT_PROTO = (
    ('version', 'h', 4),
    ('dimX', 'h', 256),
    ('dimY', 'h', 256),
    ('dimZ', 'h', 256)
)

VMR_PSHDR_DICT_PROTO = (
    ('xOffset', 'h', 0),
    ('yOffset', 'h', 0),
    ('zOffsed', 'h', 0),
    ('framingCube', 'h', 256),
    ('posInfosVerified','i',0),
    ('CoordSysEntry','i', 1),
    ('slice1CenterX','f', 127.5),
    ('slice1CenterY','f', 0),
    ('slice1CenterZ', 'f',0),
    ('sliceNCenterX','f', -127.5),
    ('sliceNCenterY','f', 0),
    ('SliceNCenterZ','f', 0),
    ('sliceRowDirX', 'f', 0),
    ('sliceRowDirY', 'f',1),
    ('sliceRowDirZ', 'f',0),
    ('sliceColDirX', 'f',0),
    ('sliceColDirY', 'f',0),
    ('sliceColDirZ','f', -1),
    ('nrRowsSlice','i', 256),
    ('nrColSlice','i', 256),
    ('foVRowDir','f', 256),
    ('foVColDir','f', 256),
    ('SliceThick','f', 1),
    ('GapThick','f', 0),
    ('nrOfPastSpatTrans','i', 0),
    ('pastST', (('name', 'z', b''),('type', 'i', b''),('sourceFile','z',b''),
                ('numTransVal','i',b''),('transfVal', (('value', 'f', b''),), 'numTransVal')),
     'nrOfPastSpatTrans'),
    ('LRConvention','b', 1),
    ('referenceSpace','b', 0),
    ('voxResX','f', 1),
    ('voxResY','f', 1),
    ('voxResZ','f', 1),
    ('flagVoxResolution','b',0),
    ('flagTalSpace','b', 0),
    ('minIntensity','i', 0),
    ('meanIntensity','i', 127),
    ('maxIntensity','i', 255)
)




def computeOffsetPostHDR(hdrDict, fileobj):
    currentSeek = fileobj.tell()
    return currentSeek + (hdrDict['dimX']*hdrDict['dimY']*hdrDict['dimZ'])

def concatePrePos(preDict, posDict):
    temp = preDict.copy()
    temp.update(posDict)
    return temp



class VmrHeader(BvFileHeader):
    default_endianness = '<'
    hdr_dict_proto = VMR_PRHDR_DICT_PROTO + VMR_PSHDR_DICT_PROTO


    def get_data_shape(self):

     hdr = self._hdrDict
     # calculate dimensions
     z =  hdr['dimZ']
     y =  hdr['dimY']
     x =  hdr['dimX']
     return tuple(int(d) for d in [z, y, x])




    def set_data_shape(self, shape=None, zyx=None):

        if (shape is None) and (zyx is None):
            raise BvError('Shape or zyx needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape.
            # Dimensions will start from standard coordinates.
            if len(shape) != 3:
                raise BvError('Shape for VMR files must be 3 dimensional!')
            self._hdrDict['dimX'] = shape[2]
            self._hdrDict['dimY'] = shape[1]
            self._hdrDict['dimZ'] = shape[0]
            return
        self._hdrDict['dimX'] = zyx[2][1] - zyx[2][0]
        self._hdrDict['dimY'] = zyx[1][1] - zyx[1][0]
        self._hDict['dimZ'] =  zyx[0][1] - zyx[0][0]



    def set_xflip(self, xflip):
        if xflip is True:
            self._hdrDict['LRConvention'] = 1
        elif xflip is False:
            self._hdrDict['LRConvention'] = 2
        else:
            self._hdrDict['LRConvention'] = 0


    def get_xflip(self):
        xflip = int(self._hdrDict['LRConvention'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')


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
        newSeek = computeOffsetPostHDR(hdrDictPre,fileobj) #calculate new seek for the post data header
        fileobj.seek(newSeek)
        hdrDictPos = parse_BV_header(VMR_PSHDR_DICT_PROTO, fileobj)
        hdrDict = concatePrePos(hdrDictPre, hdrDictPos)
        offset = fileobj.tell()
        return klass(hdrDict, endianness, check, offset)




    def get_bbox_center(self):
        """Get the center coordinate of the bounding box.
           Not required for VMR files
        """
        return 0,0,0

    def get_zooms(self):
        return (self._hdrDict['voxResX'], self._hdrDict['voxResY'], self._hdrDict['voxResZ'])



    def set_zooms(self, zooms):

       #check if the input type is correct
        if all(isinstance(i, float) for i in zooms) == False:
            raise BvError('Zooms for VMR files must be float values!')

        if len(zooms) != 3:
            raise BvError('Zooms for VMR files must be 3 values!')

        self._hdrDict['voxResX'] = float(zooms[0])
        self._hdrDict['voxResY'] = float(zooms[1])
        self._hdrDict['voxResZ'] = float(zooms[2])


    def write_to(self, fileobj):

        binaryblock = pack_BV_header(self.hdr_dict_proto, self._hdrDict)
        sizePrH = calc_BV_header_size(VMR_PRHDR_DICT_PROTO, self._hdrDict) #calculate size of preDataHeader
        fileobj.write(binaryblock[0:sizePrH]) #write the preHeader

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







"""Class for BrainVoyager VMR images."""
class VMRImage(BvFileImage):

    # Set the class of the corresponding header
    header_class = VmrHeader

    # Set the label ('image') and the extension ('.vtc') for a VMR file
    files_types = (('image', '.vmr'),)


load = VMRImage.load
save = VMRImage.instance_to_filename
