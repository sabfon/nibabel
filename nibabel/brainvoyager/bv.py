# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) file formats.

please look at the support site of BrainInnovation for further informations
about the file formats: http://support.brainvoyager.com/

This file implements basic functionality for BV file formats. Look into bv_*.py
files for implementations of the different file formats.

Author: Thomas Emmerling
"""

import numpy as np

from ..volumeutils import array_to_file, array_from_file
from ..spatialimages import Header, HeaderDataError, SpatialImage
from ..fileholders import copy_file_map
from ..arrayproxy import CArrayProxy
from ..volumeutils import make_dt_codes
from struct import pack, unpack, calcsize
from ..externals import OrderedDict

_dtdefs = (  # code, conversion function, equivalent dtype, aliases
    (1, 'int16', np.uint16),
    (2, 'float32', np.float32),
    (3, 'uint8', np.uint8))

# Make full code alias bank, including dtype column
data_type_codes = make_dt_codes(_dtdefs)

def _proto2default(proto):
    """Helper for creating a VTC header OrderedDict with default parameters.

    Create an OrderedDict that contains keys with the header fields, and
    default values.

    See :func:`parse_BV_header` for description of `proto` format.
    """
    default_hdr = OrderedDict()
    for name, format, def_or_name in proto:
        default_hdr[name] = {}
        if isinstance(format, tuple):
            value = _proto2default(format)
            default_hdr[name] = [value]
        else:
            default_hdr[name] = def_or_name
    return default_hdr


def readCString(f, nStrings=1, bufsize=1000, startPos=None, strip=True,
                rewind=False):
    """Read a zero-terminated string from a file object.

    Read and return a zero-terminated string from a file object.

    Parameters
    ----------
    f : fileobj
       File object to use
    nStrings: int, optional
       Number of strings to search (and return). Default is 1.
    bufsize: int, optional
       Define the buffer size that should be searched for the string.
       Default is 1000 bytes.
    startPos: int, optional
       Define the start file position from which to search. If None then start
       where the file object currently points to. Default is None.
    strip : bool, optional
       Whether to strip the trailing zero from the returned string.
       Default is True.
    rewind: bool, optional
       Whether the fileobj f should be returned to the initial position after
       reading. Default is False.

    Returns
    -------
    str_list : generator of string(s)
    """
    currentPos = f.tell()
    if strip:
        suffix = ''
    else:
        suffix = '\x00'
    if startPos is not None:
        f.seek(startPos)
    data = f.read(bufsize)
    lines = data.split('\x00')
    str_list = []
    if rewind:
        f.seek(currentPos)
    else:
        offset = 0
        for s in range(nStrings):
            offset += len(lines[s])+1
        f.seek(currentPos+offset)
    for s in range(nStrings):
        str_list.append(lines[s] + suffix)
    return str_list


def parse_BV_header(hdr_dict_proto, fileobj):
    """Parse the header of a BV file format.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the ``prts`` field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes below.
    fileobj : fileobj
       File object to use. Make sure that the current position is at the
       beginning of the header (e.g. at 0).

    Notes
    -----
    The description of `hdr_dict_proto` below is notated according to
    https://docs.python.org/3/reference/introduction.html#notation

        hdr_dict_proto ::= ((element_proto))*
        element_proto ::= '(' name ',' pack_format ',' default ')'  |
                          '(' name ',' hdr_dict_proto ',' n_fields_name ')'
        pack_format ::= 'b' | 'h' | 'f' | 'z'
        name ::= str
        n_fields_name ::= str
        default ::= int | float | bytes

    The pack_format codes have meaning::

        b := signed char (1 byte)
        h := signed short integer (2 bytes)
        f := float (4 bytes)
        z := zero-terminated string (variable bytes)
    """
    hdr_dict = OrderedDict()
    for name, format, def_or_name in hdr_dict_proto:
        # handle zero-terminated strings
        if format == 'z':
            value = readCString(fileobj)[0]
        # handle array fields
        elif isinstance(format, tuple):
            value = []
            n_values = hdr_dict[def_or_name]
            for i in range(n_values):
                value.append(parse_BV_header(format, fileobj))
        else:  # pack string format
            bytes = fileobj.read(calcsize(format))
            value = unpack('<' + format, bytes)[0]
        hdr_dict[name] = value
    return hdr_dict


def pack_BV_header(hdr_dict_proto, hdr_dict):
    """Pack the header of a BV file format into a byte string.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the ``prts`` field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdrDict: OrderedDict
       hdrDict that contains the fields and values to for the respective
       BV file format.

    Returns
    -------
    binaryblock : bytes
        Binary representation of header ready for writing to file.
    """
    binary_parts = []
    for name, format, def_or_name in hdr_dict_proto:
        value = hdr_dict[name]
        # handle zero-terminated strings
        if format == 'z':
            part = value + b'\x00'
        # handle array fields
        elif isinstance(format, tuple):
            # check the length of the array to expect
            n_values = hdr_dict[def_or_name]
            sub_parts = []
            for i in range(n_values):
                sub_parts.append(pack_BV_header(format, value[i]))
            part = b''.join(sub_parts)
        else:
            part = pack('<' + format, value)
        binary_parts.append(part)
    return b''.join(binary_parts)


def calc_BV_header_size(hdr_dict_proto, hdr_dict):
    """Calculate the binary size of a hdrDict for a BV file format header.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the prts field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdrDict: OrderedDict
       hdrDict that contains the fields and values to for the respective
       BV file format.

    Returns
    -------
    hdr_size : int
        Size of header when packed into bytes ready for writing to file.
    """
    hdr_size = 0
    for name, format, def_or_name in hdr_dict_proto:
        value = hdr_dict[name]
        # handle zero-terminated strings
        if format == 'z':
            hdr_size += len(value) + 1
        # handle array fields
        elif isinstance(format, tuple):
            # check the length of the array to expect
            n_values = hdr_dict[def_or_name]
            for i in range(n_values):
                # recursively iterate through the fields of all items
                # in the array
                hdr_size += calc_BV_header_size(format, value[i])
        else:
            hdr_size += calcsize(format)
    return hdr_size


def _proto2default(proto):
    """Helper for creating a VTC header OrderedDict with default parameters.

    Create an OrderedDict that contains keys with the header fields, and
    default values.

    See :func:`parse_BV_header` for description of `proto` format.
    """
    default_hdr = OrderedDict()
    for name, format, def_or_name in proto:
        default_hdr[name] = {}
        if isinstance(format, tuple):
            value = _proto2default(format)
            default_hdr[name] = [value]
        else:
            if isinstance(def_or_name, tuple):
                default_hdr[name] = def_or_name[0]
            else:
                default_hdr[name] = def_or_name
    return default_hdr


def readCString(f, nStrings=1, bufsize=1000, startPos=None, strip=True,
                rewind=False):
    """Read a zero-terminated string from a file object.

    Read and return a zero-terminated string from a file object.

    Parameters
    ----------
    f : fileobj
       File object to use
    nStrings: int, optional
       Number of strings to search (and return). Default is 1.
    bufsize: int, optional
       Define the buffer size that should be searched for the string.
       Default is 1000 bytes.
    startPos: int, optional
       Define the start file position from which to search. If None then start
       where the file object currently points to. Default is None.
    strip : bool, optional
       Whether to strip the trailing zero from the returned string.
       Default is True.
    rewind: bool, optional
       Whether the fileobj f should be returned to the initial position after
       reading. Default is False.

    Returns
    -------
    str_list : generator of string(s)
    """
    currentPos = f.tell()
    if strip:
        suffix = ''
    else:
        suffix = '\x00'
    if startPos is not None:
        f.seek(startPos)
    data = f.read(bufsize)
    lines = data.split('\x00')
    str_list = []
    if rewind:
        f.seek(currentPos)
    else:
        offset = 0
        for s in range(nStrings):
            offset += len(lines[s])+1
        f.seek(currentPos+offset)
    for s in range(nStrings):
        str_list.append(lines[s] + suffix)
    return str_list


def parse_BV_header(hdr_dict_proto, fileobj):
    """Parse the header of a BV file format.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the ``prts`` field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes below.
    fileobj : fileobj
       File object to use. Make sure that the current position is at the
       beginning of the header (e.g. at 0).

    Notes
    -----
    The description of `hdr_dict_proto` below is notated according to
    https://docs.python.org/3/reference/introduction.html#notation

        hdr_dict_proto ::= ((element_proto))*
        element_proto ::= '(' name ',' pack_format ',' default ')'  |
                          '(' name ',' pack_format ',' '(' default ','
                            c_fields_name ',' c_fields_value ')' ')'  |
                          '(' name ',' hdr_dict_proto ',' n_fields_name ')'
        pack_format ::= 'b' | 'h' | 'f' | 'z'
        name ::= str
        n_fields_name ::= str
        c_fields_name ::= str
        c_fields_value ::= int | float | bytes
        default ::= int | float | bytes

    The pack_format codes have meaning::

        b := signed char (1 byte)
        h := signed short integer (2 bytes)
        f := float (4 bytes)
        z := zero-terminated string (variable bytes)

    The c_fields_name and c_fields_value parameters are used for header fields
    that are only written depending on the value of another header field (e.g.
    'NrOfLags' in the VMP file header).
    """
    hdr_dict = OrderedDict()
    for name, format, def_or_name in hdr_dict_proto:
        # handle zero-terminated strings
        if format == 'z':
            value = readCString(fileobj)[0]
        # handle array fields
        elif isinstance(format, tuple):
            value = []
            n_values = hdr_dict[def_or_name]
            for i in range(n_values):
                value.append(parse_BV_header(format, fileobj))
        else:  # pack string format
            bytes = fileobj.read(calcsize(format))
            value = unpack('<' + format, bytes)[0]
        hdr_dict[name] = value
    return hdr_dict


def pack_BV_header(hdr_dict_proto, hdr_dict):
    """Pack the header of a BV file format into a byte string.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the ``prts`` field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdrDict: OrderedDict
       hdrDict that contains the fields and values to for the respective
       BV file format.

    Returns
    -------
    binaryblock : bytes
        Binary representation of header ready for writing to file.
    """
    binary_parts = []
    for name, format, def_or_name in hdr_dict_proto:
        value = hdr_dict[name]
        # handle zero-terminated strings
        if format == 'z':
            part = value + b'\x00'
        # handle array fields
        elif isinstance(format, tuple):
            # check the length of the array to expect
            n_values = hdr_dict[def_or_name]
            sub_parts = []
            for i in range(n_values):
                sub_parts.append(pack_BV_header(format, value[i]))
            part = b''.join(sub_parts)
        else:
            part = pack('<' + format, value)
        binary_parts.append(part)
    return b''.join(binary_parts)


def calc_BV_header_size(hdr_dict_proto, hdr_dict):
    """Calculate the binary size of a hdrDict for a BV file format header.

    This function can be (and is) called recursively to iterate through nested
    fields (e.g. the prts field of the VTC header).

    Parameters
    ----------
    hdr_dict_proto: tuple
        tuple of format described in Notes of :func:`parse_BV_header`
    hdrDict: OrderedDict
       hdrDict that contains the fields and values to for the respective
       BV file format.

    Returns
    -------
    hdr_size : int
        Size of header when packed into bytes ready for writing to file.
    """
    hdr_size = 0
    for name, format, def_or_name in hdr_dict_proto:
        value = hdr_dict[name]
        # handle zero-terminated strings
        if format == 'z':
            hdr_size += len(value) + 1
        # handle array fields
        elif isinstance(format, tuple):
            # check the length of the array to expect
            n_values = hdr_dict[def_or_name]
            for i in range(n_values):
                # recursively iterate through the fields of all items
                # in the array
                hdr_size += calc_BV_header_size(format, value[i])
        else:
            hdr_size += calcsize(format)
    return hdr_size


class BvError(Exception):

    """Exception for BV format related problems.

    To be raised whenever there is a problem with a BV fileformat.
    """

    pass

class BvFileHeader(Header):

    """Class to hold information from a BV file header."""

    # Copies of module-level definitions
    _data_type_codes = data_type_codes
    _field_recoders = {'datatype': data_type_codes}

    # format defaults
    # BV files are radiological (left-is-right) by default
    # (VTC files have a flag for that, however)
    default_x_flip = True
    default_endianness = '<'  # BV files are always little-endian
    allowed_dtypes = [1, 2, 3]
    default_dtype = 2
    data_layout = 'C'
    hdr_dict_proto = ()

    def __init__(self,
                 hdrDict=None,
                 endianness=default_endianness,
                 check=True,
                 offset=None):
        """Initialize header from binary data block.

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
        offset : int, optional
            offset of the actual data into to binary file (in bytes)
        """
        if endianness != self.default_endianness:
            raise BvError('BV files are always little-endian')
        self.endianness = self.default_endianness
        if hdrDict is None:
            hdrDict = _proto2default(self.hdr_dict_proto)
        self._hdrDict = hdrDict
        if offset is None:
            self.set_data_offset(calc_BV_header_size(
                self.hdr_dict_proto, self._hdrDict))
        self._framing_cube = self._guess_framing_cube()
        if check:
            self.check_fix()
        return

    @classmethod
    def from_fileobj(klass, fileobj, endianness=default_endianness,
                     check=True):
        """Return read structure with given or guessed endiancode.

        Parameters
        ----------
        fileobj : file-like object
           Needs to implement ``read`` method
        endianness : None or endian code, optional
           Code specifying endianness of read data

        Returns
        -------
        header : BvFileHeader object
           BvFileHeader object initialized from data in fileobj
        """
        hdrDict = parse_BV_header(klass.hdr_dict_proto, fileobj)
        offset = fileobj.tell()
        return klass(hdrDict, endianness, check, offset)

    @classmethod
    def from_header(klass, header=None, check=False):
        """Class method to create header from another header.

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
        """
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
        try:  # check if there is a specific conversion routine
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

    def copy(self):
        """Copy object to independent representation.

        The copy should not be affected by any changes to the original
        object.
        """
        return self.__class__(self._hdrDict)

    def _set_format_specifics(self):
        """Utility routine to set format specific header stuff."""
        pass

    def data_from_fileobj(self, fileobj):
        """Read data array from `fileobj`.

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           data array
        """
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset,
                               order=self.data_layout)

    def get_data_dtype(self):
        """Get numpy dtype for data.

        For examples see ``set_data_dtype``
        """
        if 'datatype' in self._hdrDict:
            code = self._hdrDict['datatype']
        else:
            code = self.default_dtype
        dtype = self._data_type_codes.dtype[code]
        return dtype.newbyteorder(self.endianness)

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code or dtype or type."""
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise HeaderDataError(
                'data dtype "%s" not recognized' % datatype)
        if code not in self.allowed_dtypes:
            raise HeaderDataError(
                'data dtype "%s" not supported' % datatype)
        dtype = self._data_type_codes.dtype[code]
        if 'datatype' in self._hdrDict.keys():
            self._hdrDict['datatype'] = code
            return
        if dtype.newbyteorder(self.endianness) != self.get_data_dtype():
            raise HeaderDataError(
                'File format does not support setting of header!')

    def get_xflip(self):
        """Get xflip for data."""
        return self.default_x_flip

    def set_xflip(self, xflip):
        """Set xflip for data."""
        if xflip is True:
            return
        else:
            raise BvError('cannot change Left-right convention!')

    def get_data_shape(self):
        """Get shape of data."""
        raise NotImplementedError

    def set_data_shape(self, shape):
        """Set shape of data."""
        raise NotImplementedError

    def get_base_affine(self):
        """Get affine from basic (shared) header fields.

        Note that we get the translations from the center of the
        (guessed) framing cube of the referenced VMR (anatomical) file.

        Internal storage of the image is ZYXT, where (in RAS orientations)
        Z := axis increasing from right to left (R to L)
        Y := axis increasing from superior to inferior (S to L)
        X := axis increasing from anterior to posterior (A to P)
        T := volumes (if present in file format)

        Examples
        --------
        >>> hdr = BvFileHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3, 3, 3))
        >>> hdr.get_base_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
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

        return M

    get_best_affine = get_base_affine

    get_default_affine = get_base_affine

    get_affine = get_base_affine

    def _guess_framing_cube(self):
        """Guess the dimensions of the framing cube.

        Guess the dimensions of the framing cube that constitutes the
        coordinate system boundaries for the bounding box.

        For most BV file formats this need to be guessed from
        XEnd, YEnd, and ZEnd in the header.
        """
        # then start guessing...
        hdr = self._hdrDict
        # get the ends of the bounding box (highest values in each dimension)
        x = hdr['XEnd']
        y = hdr['YEnd']
        z = hdr['ZEnd']

        # compare with possible framing cubes
        for fc in [256, 384, 512, 768, 1024]:
            if any([d > fc for d in (x, y, z)]):
                continue
            else:
                return fc, fc, fc

    def get_framing_cube(self):
        """Get the dimensions of the framing cube.

        Get the dimensions of the framing cube that constitutes the
        coordinate system boundaries for the bounding box.
        For most BV file formats this need to be guessed from
        XEnd, YEnd, and ZEnd in the header.
        """
        return self._framing_cube

    def set_framing_cube(self, fc):
        """Set the dimensions of the framing cube.

        Set the dimensions of the framing cube that constitutes the
        coordinate system boundaries for the bounding box
        For most BV file formats this need to be guessed from
        XEnd, YEnd, and ZEnd in the header.
        Use this if you know about the framing cube for the BV file.
        """
        self._framing_cube = fc

    def get_bbox_center(self):
        """Get the center coordinate of the bounding box.

        Get the center coordinate of the bounding box with respect to the
        framing cube.
        """
        hdr = self._hdrDict
        x = hdr['XStart'] + \
            ((hdr['XEnd'] - hdr['XStart'])/2)
        y = hdr['YStart'] + \
            ((hdr['YEnd'] - hdr['YStart'])/2)
        z = hdr['ZStart'] + \
            ((hdr['ZEnd'] - hdr['ZStart'])/2)
        return z, y, x

    def get_zooms(self):
        shape = self.get_data_shape()
        return tuple(float(self._hdrDict['Resolution'])
                     for d in shape[0:3])

    def set_zooms(self, zooms):
        if type(zooms) == int:
            self._hdrDict['Resolution'] = zooms
        else:
            if any([zooms[i] != zooms[i+1] for i in range(len(zooms)-1)]):
                raise BvError('Zooms for all dimensions must be equal!')
            else:
                self._hdrDict['Resolution'] = int(zooms[0])

    def as_analyze_map(self):
        raise NotImplementedError

    def set_data_offset(self, offset):
        """Set offset into data file to read data."""
        self._data_offset = offset

    def get_data_offset(self):
        """Return offset into data file to read data."""
        return self._data_offset

    def get_slope_inter(self):
        """BV formats do not do scaling."""
        return None, None

    def check_fix(self):
        """Do some checks for the BV header."""
        pass

    def write_to(self, fileobj):
        """Write header to fileobj.

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method

        Returns
        -------
        None
        """
        binaryblock = pack_BV_header(self.hdr_dict_proto, self._hdrDict)
        fileobj.write(binaryblock)

class BvFileImage(SpatialImage):

    """Class to hold information from a BV image file."""

    # Set the class of the corresponding header
    header_class = BvFileHeader

    # Set the label ('image') and the extension ('.bv') for a (dummy) BV file
    files_types = (('image', '.bv'),)

    # BV files are not compressed...
    _compressed_exts = ()

    # use the row-major CArrayProxy
    ImageArrayProxy = CArrayProxy

    def update_header(self):
        """Harmonize header with image data and affine.

        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = SpatialImage(data, affine)
        >>> hdr = img.get_header()
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_header()
        >>> hdr.get_data_shape() == (2, 3, 4)
        True
        >>> hdr.get_zooms()
        (1.0, 2.0, 3.0)
        """
        hdr = self._header
        shape = self._dataobj.shape
        # We need to update the header if the data shape has changed.  It's a
        # bit difficult to change the data shape using the standard API, but
        # maybe it happened
        if hdr.get_data_shape() != shape:
            hdr.set_data_shape(shape)

    @classmethod
    def from_file_map(klass, file_map):
        """Load image from `file_map`.

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        bvf = file_map['image'].get_prepare_fileobj('rb')
        header = klass.header_class.from_fileobj(bvf)
        affine = header.get_affine()
        hdr_copy = header.copy()
        # use row-major memory presentation!
        data = klass.ImageArrayProxy(bvf, hdr_copy)
        img = klass(data, affine, header, file_map)
        img._load_cache = {'header': hdr_copy,
                           'affine': None,
                           'file_map': copy_file_map(file_map)}
        return img

    def _write_header(self, header_file, header):
        """Utility routine to write BV header.

        Parameters
        ----------
        header_file : file-like
           file-like object implementing ``write``, open for writing
        header : header object
        slope : None or float
           slope for data scaling
        inter : None or float
           intercept for data scaling
        """
        header.write_to(header_file)

    def _write_data(self, bvfile, data, header):
        """Utility routine to write BV image.

        Parameters
        ----------
        vtcfile : file-like
           file-like object implementing ``seek`` or ``tell``, and
           ``write``
        data : array-like
           array to write
        header : analyze-type header object
           header
        """
        shape = header.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' %
                                  ', '.join(str(s) for s in shape))
        offset = header.get_data_offset()
        out_dtype = header.get_data_dtype()
        array_to_file(data, bvfile, out_dtype, offset, order='C')

    def to_file_map(self, file_map=None):
        """Write image to `file_map` or contained ``self.file_map``.

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        if file_map is None:
            file_map = self.file_map
        data = self.get_data()
        self.update_header()
        hdr = self.get_header()

        with file_map['image'].get_prepare_fileobj('wb') as bvf:
            self._write_header(bvf, hdr)
            self._write_data(bvf, data, hdr)
        self._header = hdr
        self.file_map = file_map
