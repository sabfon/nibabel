from os.path import join as pjoin
import numpy as np

from nibabel.brainvoyager.bv import BvError, parse_notBin_BV_header
from nibabel.brainvoyager.bv_fmr import BvFmrImage, FMR_HDR_DICT_PROTO
from nibabel.testing import (assert_equal, data_path)
from nibabel.externals import OrderedDict

fmr_file = pjoin(data_path, 'test.fmr')
fileobj = open(fmr_file, 'r')



EXAMPLE_HDR = OrderedDict({
    'FileVersion': 7,
    'NrOfVolumes': 5,
    'NrOfSlices': 5,
    'NrOfSkippedVolumes': 0,
    'Prefix': 'test',
    'DataStorageFormat': 2,
    'DataType': 1,
    'TR': 2000,
    'InterSliceTime': 400,
    'TimeResolutionVerified': 1,
    'TE': 30,
    'SliceAcquisitionOrder': 0,
    'SliceAcquisitionOrderVerified': 0,
    'ResolutionX': 10,
    'ResolutionY': 10,
    'LoadAMRFile': '<none>',
    'ShowAMRFile': 1,
    'ImageIndex': 0,
    'LayoutNColumns': 6,
    'LayoutNRows': 5,
    'LayoutZoomLevel': 1,
    'SegmentSize': 10,
    'SegmentOffset': 0,
    'NrOfLinkedProtocols': 1,
    'ProtocolFile': '<none>',
    'InplaneResolutionX': 3,
    'InplaneResolutionY': 3,
    'SliceThickness': 3,
    'SliceGap': 1,
    'VoxelResolutionVerified': 0,
    'PosInfosVerified': 0,
    'CoordinateSystem': 1,
    'Slice1CenterX': 0,
    'Slice1CenterY': 0,
    'Slice1CenterZ': -58,
    'SliceNCenterX': 0,
    'SliceNCenterY': 0,
    'SliceNCenterZ': 58,
    'RowDirX': 1,
    'RowDirY': 0,
    'RowDirZ': 0,
    'ColDirX': 0,
    'ColDirY': 1,
    'ColDirZ': 0,
    'NRows': 10,
    'NCols': 10,
    'FoVRows': 192,
    'FoVCols': 192,
    'SliceThicknessFromImgHead': 3,
    'GapThickness': 1,
    'NrOfPastSpatialTransformations': 0,
    'LeftRightConvention': 'Radiological',
    'FirstDataSourceFile': 'test.stc',
    'MultibandSequence': 0,
    'SliceTimingTableSize': 0,
    'AcqusitionTime': 'NA',


})

def compareValues(header, testHeader):
    for key in header:
        if (type(header[key]) is list):
            num = len(testHeader[key])
            for i in range(0, num):
                compareValues(header[key][i], testHeader[key][i])
        assert_equal(header[key], testHeader[key])



def test_parse_BVFMR_header():
    hdr_dict = parse_notBin_BV_header(FMR_HDR_DICT_PROTO, fileobj)
    compareValues(hdr_dict, EXAMPLE_HDR)


