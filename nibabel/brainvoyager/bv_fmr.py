# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Reading / writing functions for Brainvoyager (BV) FMR files.
for documentation on the file format see:
http://support.brainvoyager.com/automation-aamp-development/23-file-formats/383-developer-guide-26-the-format-of-fmr-files.html
"""

from .bv import (BvError, BvFileHeader, BvFileImage, parse_BV_header,
                 pack_BV_header, calc_BV_header_size)
from ..spatialimages import HeaderDataError
from ..batteryrunners import Report
from ..externals import OrderedDict

FMR_HDR_DICT_PROTO = (
    ('FileVersion','h',6),
    ('NrOfVolumes','h',5),
    ('NrOfSlices','h',5),
    ('NrOfSkippedVolumes','h',0),
    ('Prefix','h',b''),
    ('DataStorageFormat','h',2),
    ('DataType','h',1),
    ('TR','h',2000),
    ('InterSliceTime','h',6.666667e+01),
    ('TimeResolutionVerified','h',0),
    ('TE','h',30),
    ('SliceAcquisitionOrder','h',0),
    ('SliceAcquisitionOrderVerified','h',0),
    ('ResolutionX','h',10),
    ('ResolutionY','h',10),
    ('LoadAMRFile','h',""),
    ('ShowAMRFile','h',0),
    ('ImageIndex','h',0),
    ('LayoutNColumns','h',6),
    ('LayoutNRows','h',5),
    ('LayoutZoomLevel','h',1),
    ('SegmentSize','h',10),
    ('SegmentOffset','h',0),
    ('NrOfLinkedProtocols','h',1),
    ('ProtocolFile','z',b''),
    ('InplaneResolutionX','h',3.000000),
    ('InplaneResolutionY','h',3.000000),
    ('SliceThickness','h',3.000000),
    ('SliceGap','h',1.000000),
    ('VoxelResolutionVerified','h',0),
    ('PosInfosVerified','h',0),
    ('CoordinateSystem','h',1),
    ('Slice1CenterX','h',0.00000),
    ('Slice1CenterY','h',0.00000),
    ('Slice1CenterZ','h',-58.00000),
    ('SliceNCenterX','h',0.00000),
    ('SliceNCenterY','h',0.00000),
    ('SliceNCenterZ','h',58.00000),
    ('RowDirX','h',1.000000),
    ('RowDirY','h',0.000000),
    ('RowDirZ','h',0.000000),
    ('ColDirX','h',0.000000),
    ('ColDirY','h',1.000000),
    ('ColDirZ','h',0.000000),
    ('NRows','h',10),
    ('NCols','h',10),
    ('FoVRows','h',192),
    ('FoVCols','h',192),
    ('SliceThickness','h',3.000000),
    ('GapThickness','h',1.000000),
    ('NrOfPastSpatialTransformations','h',0),
    ('LeftRightConvention','z',b''),
    ('FirstDataSourceFile','z',b'')





)

def parse_BV_header(hdr_dict_proto, fileobj, parent_hdr_dict=None):
    hdr_dict = OrderedDict()
    for name, format, def_or_name in hdr_dict_proto:
        line = fileobj.readline()
        while (line=='\r\n' or line.find(':')==-1): #skip all the blank and section title lines
           line = fileobj.readline()
        if isinstance(format, tuple):
            raise "Non implemented yet!"
        else:
            line =  (line.rstrip('\n')).rstrip('\r') #delete \n and \r from line
            token = line.split(":")
            token[1] = token[1].replace(' ', '') #delete all the spaces
            value = token[1]
        hdr_dict[name] = value



class BvFmrHeader(BvFileHeader):
    """Class for BrainVoyager FMR header."""
    default_endianness = '<'



class BvFmrImage(BvFileImage):
    """Class for BrainVoyager FMR images."""

    # Set the class of the corresponding header
    header_class = BvFmrHeader

    # Set the label ('image') and the extension ('.fmr') for a FMR file
    files_types = (('image', '.fmr'),)
    valid_exts = ('.fmr',)

load = BvFmrImage.load
save = BvFmrImage.instance_to_filename
