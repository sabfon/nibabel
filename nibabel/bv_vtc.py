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
from .spatialimages import HeaderDataError
from .batteryrunners import Report
from .externals import OrderedDict


def _make_vtc_hdrDict():
    """Helper for creating a VTC header OrderedDict with default parameters.

    Create an OrderedDict that contains keys with the header fields.
    The value of each field is a dict containing the DataType (dt), a default
    value (default) and the actual value (value).

    b := signed char (1 byte)
    h := signed short integer (2 bytes)
    s := string (variable bytes)
    f := float (4 bytes)
    """
    VTC_HEADER_PROTO = OrderedDict([
        ('version', {'dt': '<h', 'default': 3}),
        ('fmr', {'dt': '<s', 'default': ''}),
        ('nPrt', {'dt': '<h', 'default': 0}),
        ('prts',
            {'dt': 'multi', 'nField': 'nPrt', 'default': [
                OrderedDict([
                    ('filename', {'dt': '<s', 'default': '', 'value': ''})
                ])
            ]}
         ),
        ('currentPrt', {'dt': '<h', 'default': 0}),
        ('datatype', {'dt': '<h', 'default': 2}),
        ('volumes', {'dt': '<h', 'default': 0}),
        ('Resolution', {'dt': '<h', 'default': 3}),
        ('XStart', {'dt': '<h', 'default': 57}),
        ('XEnd', {'dt': '<h', 'default': 231}),
        ('YStart', {'dt': '<h', 'default': 52}),
        ('YEnd', {'dt': '<h', 'default': 172}),
        ('ZStart', {'dt': '<h', 'default': 59}),
        ('ZEnd', {'dt': '<h', 'default': 197}),
        ('LRConvention', {'dt': '<b', 'default': 1}),
        ('RefSpace', {'dt': '<b', 'default': 3}),
        ('TR', {'dt': '<f', 'default': 2000.0})
    ])

    for k in VTC_HEADER_PROTO.keys():
        VTC_HEADER_PROTO[k]['value'] = VTC_HEADER_PROTO[k]['default']
    return VTC_HEADER_PROTO

class VtcHeader(BvFileHeader):

    # format defaults
    allowed_dtypes = [2]
    default_dtype = 2

    def get_data_shape(self):
        """Get shape of data."""
        hdr = self._hdrDict
        # calculate dimensions
        z = (hdr['ZEnd']['value'] -
             hdr['ZStart']['value']) / hdr['Resolution']['value']
        y = (hdr['YEnd']['value'] -
             hdr['YStart']['value']) / hdr['Resolution']['value']
        x = (hdr['XEnd']['value'] -
             hdr['XStart']['value']) / hdr['Resolution']['value']
        t = hdr['volumes']['value']
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
            self._hdrDict['XEnd']['value'] = \
                57 + (shape[2] * self._hdrDict['Resolution']['value'])
            self._hdrDict['YEnd']['value'] = \
                52 + (shape[1] * self._hdrDict['Resolution']['value'])
            self._hdrDict['ZEnd']['value'] = \
                59 + (shape[0] * self._hdrDict['Resolution']['value'])
            self._hdrDict['volumes']['value'] = shape[3]
            return
        self._hdrDict['XStart']['value'] = zyx[2][0]
        self._hdrDict['XEnd']['value'] = zyx[2][1]
        self._hdrDict['YStart']['value'] = zyx[1][0]
        self._hdrDict['YEnd']['value'] = zyx[1][1]
        self._hdrDict['ZStart']['value'] = zyx[0][0]
        self._hdrDict['ZEnd']['value'] = zyx[0][1]
        if t is not None:
            self._hdrDict['volumes']['value'] = t

    def get_xflip(self):
        """Get xflip for data."""
        xflip = int(self._hdrDict['LRConvention']['value'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')

    def set_xflip(self, xflip):
        """Set xflip for data."""
        if xflip is True:
            self._hdrDict['LRConvention']['value'] = 1
        elif xflip is False:
            self._hdrDict['LRConvention']['value'] = 2
        else:
            self._hdrDict['LRConvention']['value'] = 0

    def update_template_dtype(self,binaryblock=None, item=None, value=None):
        if binaryblock is None:
            binaryblock = self.binaryblock

        # find length of fmr filename (start search after the first 2 bytes)
        # include the stop byte ('\x00') in the string
        fmrl = binaryblock.find('\x00', 2) - 1
        fmrlt = 'S' + str(fmrl)

        # find number of linked PRTs
        nPrt = int(np.fromstring(binaryblock[2+fmrl:2+fmrl+2], np.uint16))

        # find length of name(s) of linked PRT(s)
        if nPrt == 0:
            prts = [('prt1', 'S1')]
        else:
            prts = []
            point = 2 + fmrl + 3
            for prt in range(nPrt):
                prtl = binaryblock.find('\x00', point) - (point-2)
                prts.append(('prt' + str(prt+1), 'S' + str(prtl)))
                point += prtl

        # deep copy the template
        newTemplate = _make_vtc_header_dtd(fmrlt, prts)

        # handle the items that should be changed
        if item is not None:
            newTemplate = [(x[0], x[1]) if x[0] != item else (item, 'S'+str(len(value)+1)) for x in newTemplate]

        dt = np.dtype(newTemplate)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt

        return newTemplate

    @classmethod
    def _init_hdrDict(klass):
        """Return header data for empty header with given endianness."""
        hdrDict = _make_vtc_hdrDict()
        return hdrDict

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
        if hdr['version']['value'] == 3:
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
        code = int(hdr['datatype']['value'])
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
    # Set the class of the corresponding header
    header_class = VtcHeader

    # Set the label ('image') and the extension ('.vtc') for a VTC file
    files_types = (('image', '.vtc'),)

load = VtcImage.load
save = VtcImage.instance_to_filename
