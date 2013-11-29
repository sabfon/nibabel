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
from py3k import asbytes

from nibabel.volumeutils import allopen, array_to_file, array_from_file, Recoder
from nibabel.spatialimages import HeaderDataError, HeaderTypeError, ImageFileError, SpatialImage, Header
from nibabel.fileholders import FileHolder,  copy_file_map
from nibabel.arrayproxy import ArrayProxy
from nibabel.volumeutils import (shape_zoom_affine, apply_read_scaling, seek_tell, make_dt_codes,
                                 pretty_mapping, endian_codes, native_code, swapped_code)
from nibabel.arraywriters import make_array_writer, WriterError, get_slope_inter
from .wrapstruct import LabeledWrapStruct
from . import imageglobals as imageglobals
from .batteryrunners import Report, BatteryRunner

# List of fields to expect in filtered VTC header
vtc_header_dtd = [
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

# List of fields to expect in MSK header
msk_header_dtype = [
    ('relResolution', 'i2'),
    ('XStart', 'i2'),
    ('XEnd', 'i2'),
    ('YStart', 'i2'),
    ('YEnd', 'i2'),
    ('ZStart', 'i2'),
    ('ZEnd', 'i2')
    ]

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

_dtdefs = ( # code, conversion function, equivalent dtype, aliases
    (1, 'short int', np.dtype(np.uint16).newbyteorder('<')),
    (2, 'float', np.dtype(np.float32).newbyteorder('<')))

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)

class BvError(Exception):
    """Exception for BV format related problems.

    To be raised whenever there is a problem with a BV fileformat.
    """
    pass

class BvFileHeader(object):
    """Class to hold information from a BV file header.
    """

    default_x_flip = True

    # Copies of module-level definitions
    _data_type_codes = data_type_codes

    # data scaling capabilities
    has_data_slope = False
    has_data_intercept = False

    def __init__(self,
                 binaryblock=None,
                 endianness='<',
                 check=True):
        ''' Initialize header from binary data block

        Parameters
        ----------
        binaryblock : {None, string} optional
            binary block to set into header.  By default, None, in
            which case we insert the default empty header block
        endianness : {None, '<','>', other endian code} string, optional
            endianness of the binaryblock.  If None, guess endianness
            from the data.
        check : bool, optional
            Whether to check content of header in initialization.
            Default is True.
        template_dtype: numpy dtype
            full (pre-parsed) template_dtype for header
        '''

        template_dtype = self.template_dtype(binaryblock)
        if binaryblock is None:
            self._structarr = self.__class__.default_structarr(endianness)
            return
        # check size
        if len(binaryblock) != template_dtype.itemsize:
            raise BvError('Binary block is wrong size')
        wstr = np.ndarray(shape=(),
                         dtype=template_dtype,
                         buffer=binaryblock)
        if endianness is None:
            raise BvError('endianness has to be defined')
        else:
            endianness = endian_codes[endianness]
        if endianness != native_code:
            dt = template_dtype.newbyteorder(endianness)
            wstr = np.ndarray(shape=(),
                             dtype=dt,
                             buffer=binaryblock)
        self._structarr = wstr.copy()
        self.set_data_offset(self.template_dtype.itemsize)
        if check:
            self.check_fix()
        return

    @property
    def template_dtype(self,binaryblock=None):
        raise NotImplementedError

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        ''' Return read structure with given or guessed endiancode

        Parameters
        ----------
        fileobj : file-like object
           Needs to implement ``read`` method
        endianness : None or endian code, optional
           Code specifying endianness of read data

        Returns
        -------
        wstr : WrapStruct object
           WrapStruct object initialized from data in fileobj
        '''
        raw_str = fileobj.read(5000)
        return klass(raw_str, endianness, check)

    @property
    def binaryblock(self):
        ''' binary block of data as string

        Returns
        -------
        binaryblock : string
            string giving binary data block

        Examples
        --------
        >>> # Make default empty structure
        >>> wstr = WrapStruct()
        >>> len(wstr.binaryblock)
        2
        '''
        return self._structarr.tostring()

    def write_to(self, fileobj):
        ''' Write structure to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method

        Returns
        -------
        None

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> from io import BytesIO
        >>> str_io = BytesIO()
        >>> wstr.write_to(str_io)
        >>> wstr.binaryblock == str_io.getvalue()
        True
        '''
        fileobj.write(self.binaryblock)

    @property
    def endianness(self):
        ''' endian code of binary data

        The endianness code gives the current byte order
        interpretation of the binary data.

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> code = wstr.endianness
        >>> code == native_code
        True

        Notes
        -----
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization, or occasionally byteswapping the
        data - but this is done via the as_byteswapped method
        '''
        if self._structarr.dtype.isnative:
            return native_code
        return swapped_code

    def copy(self):
        ''' Return copy of structure

        >>> wstr = WrapStruct()
        >>> wstr['integer'] = 3
        >>> wstr2 = wstr.copy()
        >>> wstr2 is wstr
        False
        >>> wstr2['integer']
        array(3, dtype=int16)
        '''
        return self.__class__(
                self.binaryblock,
                self.endianness, check=False)

    def __eq__(self, other):
        ''' equality between two structures defined by binaryblock

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr2 = WrapStruct()
        >>> wstr == wstr2
        True
        >>> wstr3 = WrapStruct(endianness=swapped_code)
        >>> wstr == wstr3
        True
        '''
        this_end = self.endianness
        this_bb = self.binaryblock
        try:
            other_end = other.endianness
            other_bb = other.binaryblock
        except AttributeError:
            return False
        if this_end == other_end:
            return this_bb == other_bb
        other_bb = other._structarr.byteswap().tostring()
        return this_bb == other_bb

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, item):
        ''' Return values from structure data

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr['integer'] == 0
        True
        '''
        return self._structarr[item]

    def __setitem__(self, item, value):
        ''' Set values in structured data

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr['integer'] = 3
        >>> wstr['integer']
        array(3, dtype=int16)
        '''
        self._structarr[item] = value

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        ''' Return keys from structured data'''
        return list(self.template_dtype.names)

    def values(self):
        ''' Return values from structured data'''
        data = self._structarr
        return [data[key] for key in self.template_dtype.names]

    def items(self):
        ''' Return items from structured data'''
        return zip(self.keys(), self.values())

    def get(self, k, d=None):
        ''' Return value for the key k if present or d otherwise'''
        return (k in self.keys()) and self._structarr[k] or d

    def check_fix(self, logger=None, error_level=None):
        ''' Check structured data with checks '''
        if logger is None:
            logger = imageglobals.logger
        if error_level is None:
            error_level = imageglobals.error_level
        battrun = BatteryRunner(self.__class__._get_checks())
        self, reports = battrun.check_fix(self)
        for report in reports:
            report.log_raise(logger, error_level)

    @classmethod
    def diagnose_binaryblock(klass, binaryblock, endianness=None):
        ''' Run checks over binary data, return string '''
        wstr = klass(binaryblock, endianness=endianness, check=False)
        battrun = BatteryRunner(klass._get_checks())
        reports = battrun.check_only(wstr)
        return '\n'.join([report.message
                          for report in reports if report.message])

    @classmethod
    def guessed_endian(self, mapping):
        ''' Guess intended endianness from mapping-like ``mapping``

        Parameters
        ----------
        wstr : mapping-like
            Something implementing a mapping.  We will guess the endianness from
            looking at the field values

        Returns
        -------
        endianness : {'<', '>'}
           Guessed endianness of binary data in ``wstr``
        '''
        raise NotImplementedError

    @property
    def structarr(self):
        ''' Structured data, with data fields

        Examples
        --------
        >>> wstr1 = WrapStruct() # with default data
        >>> an_int = wstr1.structarr['integer']
        >>> wstr1.structarr = None
        Traceback (most recent call last):
           ...
        AttributeError: can't set attribute
        '''
        return self._structarr

    def as_byteswapped(self, endianness=None):
        ''' return new byteswapped object with given ``endianness``

        Guaranteed to make a copy even if endianness is the same as
        the current endianness.

        Parameters
        ----------
        endianness : None or string, optional
           endian code to which to swap.  None means swap from current
           endianness, and is the default

        Returns
        -------
        wstr : ``WrapStruct``
           ``WrapStruct`` object with given endianness

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr.endianness == native_code
        True
        >>> bs_wstr = wstr.as_byteswapped()
        >>> bs_wstr.endianness == swapped_code
        True
        >>> bs_wstr = wstr.as_byteswapped(swapped_code)
        >>> bs_wstr.endianness == swapped_code
        True
        >>> bs_wstr is wstr
        False
        >>> bs_wstr == wstr
        True

        If you write to the resulting byteswapped data, it does not
        change the original.

        >>> bs_wstr['integer'] = 3
        >>> bs_wstr == wstr
        False

        If you swap to the same endianness, it returns a copy

        >>> nbs_wstr = wstr.as_byteswapped(native_code)
        >>> nbs_wstr.endianness == native_code
        True
        >>> nbs_wstr is wstr
        False
        '''
        current = self.endianness
        if endianness is None:
            if current == native_code:
                endianness = swapped_code
            else:
                endianness = native_code
        else:
            endianness = endian_codes[endianness]
        if endianness == current:
            return self.copy()
        wstr_data = self._structarr.byteswap()
        return self.__class__(wstr_data.tostring(),
                              endianness,
                              check=False)

    def get_value_label(self, fieldname):
        ''' Returns label for coded field

        A coded field is an int field containing codes that stand for
        discrete values that also have string labels.

        Parameters
        ----------
        fieldname : str
           name of header field to get label for

        Returns
        -------
        label : str
           label for code value in header field `fieldname`

        Raises
        ------
        ValueError : if field is not coded

        Examples
        --------
        >>> from nibabel.volumeutils import Recoder
        >>> recoder = Recoder(((1, 'one'), (2, 'two')), ('code', 'label'))
        >>> class C(LabeledWrapStruct):
        ...     template_dtype = np.dtype([('datatype', 'i2')])
        ...     _field_recoders = dict(datatype = recoder)
        >>> hdr  = C()
        >>> hdr.get_value_label('datatype')
        '<unknown code 0>'
        >>> hdr['datatype'] = 2
        >>> hdr.get_value_label('datatype')
        'two'
        '''
        if not fieldname in self._field_recoders:
            raise ValueError('%s not a coded field' % fieldname)
        code = int(self._structarr[fieldname])
        try:
            return self._field_recoders[fieldname].label[code]
        except KeyError:
            return '<unknown code {0}>'.format(code)

    def __str__(self):
        ''' Return string representation for printing '''
        summary = "%s object, endian='%s'" % (self.__class__,
                                              self.endianness)
        def _getter(obj, key):
            try:
                return obj.get_value_label(key)
            except ValueError:
                return obj[key]

        return '\n'.join(
            [summary,
             pretty_mapping(self, _getter)])

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        raise NotImplementedError

    @classmethod
    def default_structarr(klass, endianness=None):
        raise NotImplementedError

    @classmethod
    def from_header(klass, header=None, check=True):
        ''' Class method to create header from another header

        Parameters
        ----------
        header : ``Header`` instance or mapping
           a header of this class, or another class of header for
           conversion to this type
        check : {True, False}
           whether to check header for integrity

        Returns
        -------
        hdr : header instance
           fresh header instance of our own class
        '''
        # own type, return copy
        if type(header) == klass:
            obj = header.copy()
            if check:
                obj.check_fix()
            return obj
        # not own type, make fresh header instance
        obj = klass(check=check)
        if header is None:
            return obj
        try: # check if there is a specific conversion routine
            mapping = header.as_bv_map()
        except AttributeError:
            # most basic conversion
            obj.set_data_dtype(header.get_data_dtype())
            obj.set_data_shape(header.get_data_shape())
            obj.set_zooms(header.get_zooms())
            return obj
        # header is convertible from a field mapping
        for key, value in mapping.items():
            try:
                obj[key] = value
            except (ValueError, KeyError):
                # the presence of the mapping certifies the fields as
                # being of the same meaning as for BV types
                pass
        # set any fields etc that are specific to this format (overriden by
        # sub-classes)
        obj._set_format_specifics()
        # Check for unsupported datatypes
        orig_code = header.get_data_dtype()
        try:
            obj.set_data_dtype(orig_code)
        except HeaderDataError:
            raise HeaderDataError('Input header %s has datatype %s but '
                                  'output header %s does not support it'
                                  % (header.__class__,
                                     header.get_value_label('datatype'),
                                     klass))
        if check:
            obj.check_fix()
        return obj

    def _set_format_specifics(self):
        ''' Utility routine to set format specific header stuff
        '''
        pass

    def raw_data_from_fileobj(self, fileobj):
        ''' Read unscaled data array from `fileobj`

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           unscaled data array
        '''
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset)

    def data_from_fileobj(self, fileobj):
        ''' Read scaled data array from `fileobj`

        Use this routine to get the scaled image data from an image file
        `fileobj`, given a header `self`.  "Scaled" means, with any header
        scaling factors applied to the raw data in the file.  Use
        `raw_data_from_fileobj` to get the raw data.

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           scaled data array

        Notes
        -----
        We use the header to get any scale or intercept values to apply to the
        data.  BV files don't have scale factors or intercepts, but
        this routine also works with formats based on Analyze, that do have
        scaling, such as SPM analyze formats and NIfTI.
        '''
        # read unscaled data
        data = self.raw_data_from_fileobj(fileobj)
        # get scalings from header.  Value of None means not present in header
        slope, inter = self.get_slope_inter()
        slope = 1.0 if slope is None else slope
        inter = 0.0 if inter is None else inter
        # Upcast as necessary for big slopes, intercepts
        return apply_read_scaling(data, slope, inter)

    def data_to_fileobj(self, data, fileobj):
        ''' Write `data` to `fileobj`, maybe modifying `self`

        In writing the data, we match the header to the written data, by
        setting the header scaling factors.  Thus we modify `self` in
        the process of writing the data.

        Parameters
        ----------
        data : array-like
           data to write; should match header defined shape
        fileobj : file-like object
           Object with file interface, implementing ``write`` and
           ``seek``

        Examples
        --------
        >>> from nibabel.analyze import AnalyzeHeader
        >>> hdr = AnalyzeHeader()
        >>> hdr.set_data_shape((1, 2, 3))
        >>> hdr.set_data_dtype(np.float64)
        >>> from io import BytesIO
        >>> str_io = BytesIO()
        >>> data = np.arange(6).reshape(1,2,3)
        >>> hdr.data_to_fileobj(data, str_io)
        >>> data.astype(np.float64).tostring('F') == str_io.getvalue()
        True
        '''
        data = np.asanyarray(data)
        shape = self.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        out_dtype = self.get_data_dtype()
        try:
            arr_writer = make_array_writer(data,
                                           out_dtype,
                                           self.has_data_slope,
                                           self.has_data_intercept)
        except WriterError as e:
            raise HeaderTypeError(str(e))
        seek_tell(fileobj, self.get_data_offset())
        arr_writer.to_fileobj(fileobj)
        self.set_slope_inter(*get_slope_inter(arr_writer))

    def get_data_dtype(self):
        ''' Get numpy dtype for data

        For examples see ``set_data_dtype``
        '''
        code = int(self._structarr['datatype'])
        dtype = self._data_type_codes.dtype[code]
        return dtype.newbyteorder(self.endianness)

    def set_data_dtype(self, datatype):
        ''' Set numpy dtype for data from code or dtype or type
        '''
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise HeaderDataError(
                'data dtype "%s" not recognized' % datatype)
        dtype = self._data_type_codes.dtype[code]
        # test for void, being careful of user-defined types
        if dtype.type is np.void and not dtype.fields:
            raise HeaderDataError(
                'data dtype "%s" known but not supported' % datatype)
        self._structarr['datatype'] = code
        self._structarr['bitpix'] = dtype.itemsize * 8

    def get_data_shape(self):
        ''' Get shape of data
        '''
        raise NotImplementedError

    def set_data_shape(self, shape):
        ''' Set shape of data
        '''
        raise NotImplementedError

    def get_base_affine(self):
        ''' Get affine from basic (shared) header fields

        Note that we get the translations from the center of the
        image.

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3, 2, 1))
        >>> hdr.default_x_flip
        True
        >>> hdr.get_base_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        '''
        hdr = self._structarr
        dims = hdr['dim']
        ndim = dims[0]
        return shape_zoom_affine(hdr['dim'][1:ndim+1],
                                 hdr['pixdim'][1:ndim+1],
                                 self.default_x_flip)

    get_best_affine = get_base_affine

    get_default_affine = get_base_affine

    def get_zooms(self):
        ''' Get zooms from header

        Returns
        -------
        z : tuple
           tuple of header zoom values

        Examples
        --------
        >>> hdr = AnalyzeHeader()
        >>> hdr.get_zooms()
        (1.0,)
        >>> hdr.set_data_shape((1,2))
        >>> hdr.get_zooms()
        (1.0, 1.0)
        >>> hdr.set_zooms((3, 4))
        >>> hdr.get_zooms()
        (3.0, 4.0)
        '''
        hdr = self._structarr
        dims = hdr['dim']
        ndim = dims[0]
        if ndim == 0:
            return (1.0,)
        pixdims = hdr['pixdim']
        return tuple(pixdims[1:ndim+1])

    def set_zooms(self, zooms):
        ''' Set zooms into header fields

        See docstring for ``get_zooms`` for examples
        '''
        hdr = self._structarr
        dims = hdr['dim']
        ndim = dims[0]
        zooms = np.asarray(zooms)
        if len(zooms) != ndim:
            raise HeaderDataError('Expecting %d zoom values for ndim %d'
                                  % (ndim, ndim))
        if np.any(zooms < 0):
            raise HeaderDataError('zooms must be positive')
        pixdims = hdr['pixdim']
        pixdims[1:ndim+1] = zooms[:]

    def as_analyze_map(self):
        raise NotImplementedError

    def set_data_offset(self, offset):
        """ Set offset into data file to read data
        """
        self._data_offset = offset

    def get_data_offset(self):
        ''' Return offset into data file to read data
        '''
        return self._data_offset

    def get_slope_inter(self):
        ''' Get scalefactor and intercept

        These are not implemented for BV files
        '''
        return None, None

    def set_slope_inter(self, slope, inter=None):
        ''' Set slope and / or intercept into header

        Set slope and intercept for image data, such that, if the image
        data is ``arr``, then the scaled image data will be ``(arr *
        slope) + inter``

        In this case, for Analyze images, we can't store the slope or the
        intercept, so this method only checks that `slope` is None or 1.0, and
        that `inter` is None or 0.

        Parameters
        ----------
        slope : None or float
            If float, value must be 1.0 or we raise a ``HeaderTypeError``
        inter : None or float, optional
            If float, value must be 0.0 or we raise a ``HeaderTypeError``
        '''
        if (slope is None or slope == 1.0) and (inter is None or inter == 0):
            return
        raise HeaderTypeError('Cannot set slope != 1 or intercept != 0 '
                              'for BV headers')
    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return (klass._chk_sizeof_hdr,
                klass._chk_datatype,
                klass._chk_bitpix,
                klass._chk_pixdims)

    ''' Check functions in format expected by BatteryRunner class '''

    @classmethod
    def _chk_sizeof_hdr(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['sizeof_hdr'] == klass.sizeof_hdr:
            return hdr, rep
        rep.problem_level = 30
        rep.problem_msg = 'sizeof_hdr should be ' + str(klass.sizeof_hdr)
        if fix:
            hdr['sizeof_hdr'] = klass.sizeof_hdr
            rep.fix_msg = 'set sizeof_hdr to ' + str(klass.sizeof_hdr)
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

    @classmethod
    def _chk_bitpix(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dt = klass._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 10
            rep.problem_msg = 'no valid datatype to fix bitpix'
            if fix:
                rep.fix_msg = 'no way to fix bitpix'
            return hdr, rep
        bitpix = dt.itemsize * 8
        if bitpix == hdr['bitpix']:
            return hdr, rep
        rep.problem_level = 10
        rep.problem_msg = 'bitpix does not match datatype'
        if fix:
            hdr['bitpix'] = bitpix # inplace modification
            rep.fix_msg = 'setting bitpix to match datatype'
        return hdr, rep

    @staticmethod
    def _chk_pixdims(hdr, fix=False):
        rep = Report(HeaderDataError)
        pixdims = hdr['pixdim']
        spat_dims = pixdims[1:4]
        if not np.any(spat_dims <= 0):
            return hdr, rep
        neg_dims = spat_dims < 0
        zero_dims = spat_dims == 0
        pmsgs = []
        fmsgs = []
        if np.any(zero_dims):
            level = 30
            pmsgs.append('pixdim[1,2,3] should be non-zero')
            if fix:
                spat_dims[zero_dims] = 1
                fmsgs.append('setting 0 dims to 1')
        if np.any(neg_dims):
            level = 35
            pmsgs.append('pixdim[1,2,3] should be positive')
            if fix:
                spat_dims = np.abs(spat_dims)
                fmsgs.append('setting to abs of pixdim values')
        rep.problem_level = level
        rep.problem_msg = ' and '.join(pmsgs)
        if fix:
            pixdims[1:4] = spat_dims
            rep.fix_msg = ' and '.join(fmsgs)
        return hdr, rep

    def data_to_fileobj(self, data, fileobj):
        ''' Write image data to file in fortran order '''
        dtype = self.get_data_dtype()
        fileobj.write(data.astype(dtype).tostring(order='C'))

    def data_from_fileobj(self, fileobj):
        ''' Read data in fortran order '''
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        data_size = int(np.prod(shape) * dtype.itemsize)
        data_bytes = fileobj.read(data_size)
        return np.ndarray(shape, dtype, data_bytes, order='C')
    


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

    def template_dtype(self,binaryblock=None):
        if binaryblock is None:
            binaryblock = self.binaryblock

        # find length of fmr filename (start search after the first 2 bytes)
        fmrl = 'S' + str(binaryblock.find('\x00', 2) - 2)

        # find number of linked PRTs
        nPrt = int(binaryblock[2+fmrl+1])

        # find length of name(s) of linked PRT(s)
        if nPrt == 0:
            prts = 'S0'
        prts = []
        point = 2 + fmrl + 2
        for prt in range(nPrt):
            prtl = binaryblock.find('\x00', point) - point
            prts.append(('prt' + str(prt), 'S' + str(prtl)))
            point += prtl

        vtc_header_dtd = [
            ('version', 'i2'),
            ('fmr', fmrl),
            ('nPrt', 'i2'),
            ('prts', prts),
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

        return np.dtype(vtc_header_dtd)

    @classmethod
    def default_structarr(klass, endianness=None):
        ''' Return header data for empty header with given endianness
        '''
        hdr_data = super(VtcHeader, klass).default_structarr(endianness)
        hdr_data['sizeof_hdr'] = klass.sizeof_hdr
        hdr_data['dim'] = 1
        hdr_data['dim'][0] = 0
        hdr_data['pixdim'] = 1
        hdr_data['datatype'] = 16 # float32
        hdr_data['bitpix'] = 32
        return hdr_data

class VtcImage(SpatialImage):
    # Set the class of the corresponding header
    header_class = VtcHeader

    # Set the label ('image') and the extension ('.vtc') for a VTC file
    files_types = (('image', '.vtc'),)

    # BV files are not compressed...
    _compressed_exts = ()

    # use the standard ArrayProxy
    ImageArrayProxy = ArrayProxy

    def get_header(self):
        ''' Return header
        '''
        return self._header

    def get_data_dtype(self):
        return self._header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._header.set_data_dtype(dtype)
    
    @classmethod
    def from_file_map(klass, file_map):
        '''Load image from `file_map`

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        '''
        vtcf = file_map['image'].get_prepare_fileobj('rb')
        header = klass.header_class.from_fileobj(vtcf)
        hdr_copy = header.copy()
        data = klass.ImageArrayProxy(vtcf, hdr_copy)
        img = klass(data, affine=None, header, file_map=file_map)
        img._load_cache = {'header': hdr_copy,
                           'affine': None,
                           'file_map': copy_file_map(file_map)}
        return img

    def _write_header(self, header_file, header, slope, inter):
        ''' Utility routine to write header

        Parameters
        ----------
        header_file : file-like
           file-like object implementing ``write``, open for writing
        header : header object
        slope : None or float
           slope for data scaling
        inter : None or float
           intercept for data scaling
        '''
        header.set_slope_inter(slope, inter)
        header.write_to(header_file)

    def _write_data(self, vtcfile, data, header):
        ''' Utility routine to write VTC image

        Parameters
        ----------
        vtcfile : file-like
           file-like object implementing ``seek`` or ``tell``, and
           ``write``
        data : array-like
           array to write
        header : analyze-type header object
           header
        '''
        shape = header.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        offset = header.get_data_offset()
        out_dtype = header.get_data_dtype()
        array_to_file(data, vtcfile, out_dtype, offset)

    def to_file_map(self, file_map=None):
        ''' Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        '''
        if file_map is None:
            file_map = self.file_map
        data = self.get_data()
        self.update_header()
        hdr = self.get_header()
        out_dtype = self.get_data_dtype()
        arr_writer = make_array_writer(data,
                                       out_dtype,
                                       hdr.has_data_slope,
                                       hdr.has_data_intercept)
        vtcf = file_map['image'].get_prepare_fileobj('wb')
        slope, inter = get_slope_inter(arr_writer)
        self._write_header(vtcf, hdr, slope, inter)
        # Write image
        shape = hdr.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        seek_tell(vtcf, hdr.get_data_offset())
        arr_writer.to_fileobj(vtcf)
        vtcf.close_if_mine()
        self._header = hdr
        self.file_map = file_map

load = VtcImage.load
save = VtcImage.instance_to_filename