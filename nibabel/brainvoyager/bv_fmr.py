# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager(BV) FMR files.
for documentation on the file format see:
http://support.brainvoyager.com/automation-aamp-development/23-file-formats/383-developer-guide-26-the-format-of-fmr-files.html
"""

from .bv import (BvError, BvFileHeader, BvFileImage, parse_notBin_BV_header,
                 pack_BV_header, calc_BV_header_size)
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report
from ..externals import OrderedDict

FMR_HDR_DICT_PROTO = (
    ('FileVersion','h',7),
    ('NrOfVolumes','h',0),
    ('NrOfSlices','h',30),
    ('NrOfSkippedVolumes','h',0),
    ('Prefix','z',b''),
    ('DataStorageFormat','h',2),
    ('DataType','h',1),
    ('TR','h',2000),
    ('InterSliceTime','h',6.6667),
    ('TimeResolutionVerified','h',0),
    ('TE','h',30),
    ('SliceAcquisitionOrder','h',0),
    ('SliceAcquisitionOrderVerified','h',0),
    ('ResolutionX','h',10),
    ('ResolutionY','h',10),
    ('LoadAMRFile','z',""),
    ('ShowAMRFile','h',0),
    ('ImageIndex','h',0),
    ('LayoutNColumns','h',6),
    ('LayoutNRows','h',5),
    ('LayoutZoomLevel','h',1),
    ('SegmentSize','h',10),
    ('SegmentOffset','h',0),
    ('NrOfLinkedProtocols','h',1),
    ('ProtocolFile','z',b''),
    ('InplaneResolutionX','h',3),
    ('InplaneResolutionY','h',3),
    ('SliceThickness','h',3),
    ('SliceGap','h',1),
    ('VoxelResolutionVerified','h',0),
    ('PosInfosVerified','h',0),
    ('CoordinateSystem','h',1),
    ('Slice1CenterX','h',0),
    ('Slice1CenterY','h',0),
    ('Slice1CenterZ','h',-58),
    ('SliceNCenterX','h',0),
    ('SliceNCenterY','h',0),
    ('SliceNCenterZ','h',58),
    ('RowDirX','h',1),
    ('RowDirY','h',0),
    ('RowDirZ','h',0),
    ('ColDirX','h',0),
    ('ColDirY','h',1),
    ('ColDirZ','h',0),
    ('NRows','h',10),
    ('NCols','h',10),
    ('FoVRows','h',192),
    ('FoVCols','h',192),
    ('SliceThicknessFromImgHead','h',3),
    ('GapThickness','h',1),
    ('NrOfPastSpatialTransformations','h',0),
    ('LeftRightConvention','z',b''),
    ('FirstDataSourceFile','z',b''),
    ('MultibandSequence','h',0),
    ('SliceTimingTableSize','h',0),
    ('AcqusitionTime','z', b'')
)



class BvFmrHeader(BvFileHeader):
    """Class for BrainVoyager FMR header."""
    default_endianness = '<'
    hdr_dict_proto = FMR_HDR_DICT_PROTO


    def from_fileobj(self, fileobj, endianness=default_endianness,
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
        hdrDict = parse_notBin_BV_header(self.hdr_dict_proto, fileobj)
        offset = fileobj.tell()
        return self(hdrDict, endianness, check, offset)


class BvFmrImage(BvFileImage):
    """Class for BrainVoyager FMR images."""

    # Set the class of the corresponding header
    header_class = BvFmrHeader

    # Set the label ('image') and the extension ('.fmr') for a FMR file
    files_types = (('image', '.fmr'),)
    valid_exts = ('.fmr',)

load = BvFmrImage.load
save = BvFmrImage.instance_to_filename
