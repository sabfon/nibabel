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

class VtcHeader(BvFileHeader):
    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._structarr
        # calculate dimensions
        z = (hdr['ZEnd'] - hdr['ZStart']) / hdr['relResolution']
        y = (hdr['YEnd'] - hdr['YStart']) / hdr['relResolution']
        x = (hdr['XEnd'] - hdr['XStart']) / hdr['relResolution']
        t = hdr['volumes']
        return tuple(int(d) for d in [z,y,x,t])

    def update_template_dtype(self,binaryblock=None, item=None, value=None):
        if binaryblock is None:
            binaryblock = self.binaryblock

        # find length of fmr filename (start search after the first 2 bytes)
        # include the stop byte ('\x00') in the string
        fmrl = binaryblock.find('\x00', 2) - 1
        fmrlt = 'S' + str(fmrl)

        # find number of linked PRTs
        nPrt = int(np.fromstring(binaryblock[2+fmrl:2+fmrl+2],np.uint16))

        # find length of name(s) of linked PRT(s)
        if nPrt == 0:
            prts = [('prt1', 'S0')]
        else:
            prts = []
            point = 2 + fmrl + 3
            for prt in range(nPrt):
                prtl = binaryblock.find('\x00', point) - (point-2)
                prts.append(('prt' + str(prt+1), 'S' + str(prtl)))
                point += prtl

        if len(prts)==1:
            prts = prts[0]
        else:
            prts = ('prts', prts)

        vtc_header_dtd = \
            [
                ('version', 'i2'),
                ('fmr', fmrlt),
                ('nPrt', 'i2'),
                prts,
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
                ('TR', 'f4'),
            ]

        if item is not None:
            vtc_header_dtd = [(x[0], x[1]) if x[0] != item else (item, 'S'+str(len(value)+1)) for x in vtc_header_dtd]
        
        dt = np.dtype(vtc_header_dtd)
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
        rep.problem_msg = 'size of binaryblock should be ' + str(hdr.template_dtype.itemsize)
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
    # Set the class of the corresponding header
    header_class = VtcHeader

    # Set the label ('image') and the extension ('.vtc') for a VTC file
    files_types = (('image', '.vtc'),)

load = VtcImage.load
save = VtcImage.instance_to_filename