import os
from .. brainvoyager.bv_vmr import *
from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path)


vmr_file = os.path.join(data_path, 'test.vmr')
fileobj = open(vmr_file, 'r')
test_file = eval(open(os.path.join(data_path, 'check_vmr.txt')).read()) #data obtained from NeuroElf
#vmr_empty = os.path.join(data_path, 'test_toWrite.vmr')
#fileobjW = open(vmr_empty, 'w')


def compareValues(header, testHeader):
    for key in header:
        if (type(header[key]) is list):
            num = len(testHeader[key])
            for i in range(0,num):
                compareValues(header[key][i], testHeader[key][i] )

        assert_equal(header[key], testHeader[key])



def test_parse_VMR_header():
    vmr = VmrHeader()
    header = vmr.from_fileobj(fileobj)
    header = header._hdrDict

    compareValues(header, test_file)


def test_VMRImage():
    vmr = VMRImage.from_filename(vmr_file)
    print(vmr)

def test_right_input():
    vmr = VmrHeader()
    vmr.set_zooms((0.1,0.1,0.1))
    a = [1.0, 1.0, 1.0]
    vmr.set_zooms(a)
    vmr.set_zooms((float(1),float(1),float(1)))


def test_wrong_input():
    vmr = VmrHeader()
    try:
        vmr.set_zooms(('a', 2.0, 3))
    except BvError:
        print ("Wrong type parameter for set_zoom: 2.0, 3")
        pass

    try:
        vmr.set_zooms((2.0, 3.0))
    except BvError:
        print ("Wrong number of input parameter for set_zoom")
        pass

def test_write_to():
    fileobj = fileobj = open(vmr_file, 'r')
    vmrHead = VmrHeader()
    hdrDict = (vmrHead.from_fileobj(fileobj))._hdrDict
    binaryblock = pack_BV_header(VMR_PRHDR_DICT_PROTO+VMR_PSHDR_DICT_PROTO, hdrDict)
    #vmrHead.write_to(fileobjW)









