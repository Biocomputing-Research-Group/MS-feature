import zlib
import struct
from base64 import b64decode
from xml.dom import minidom
import numpy as np
from tqdm import tqdm

def decode(path):
    def DecodeFloatArray(code_str):
        code = b64decode(code_str)
        bytes = zlib.decompress(code)
        # bytes = code
        return np.array(struct.unpack("%df" % (len(bytes) / 4), bytes), dtype=np.float32)

    def DecodeDoubleArray(code_str):
        code = b64decode(code_str)
        bytes = zlib.decompress(code)
        # bytes = code
        return np.array(struct.unpack("%dd" % (len(bytes) / 8), bytes), dtype=np.float64)
    print('reading file...')
    data = minidom.parse(path)
    print('parsing file...')
    spectrums = data.getElementsByTagName('spectrum')

    listMz = []
    listIn = []
    listRt = []

    for spectrum in tqdm(spectrums):
        binaries = spectrum.getElementsByTagName('binary')
        mzArrayBinaryStr = binaries[0].firstChild.data
        intenArrayBinaryStr = binaries[1].firstChild.data
        mzArr = DecodeDoubleArray(mzArrayBinaryStr)
        intenArr = DecodeDoubleArray(intenArrayBinaryStr)
        retentionTime = float(spectrum.getElementsByTagName('scan')[0].getElementsByTagName('cvParam')[0].getAttribute('value'))

        listMz.append(mzArr)
        listIn.append(intenArr)
        listRt.append(retentionTime)

    return listMz, listIn, listRt

'''

import zlib
import struct
from base64 import b64decode
from xml.dom import minidom
import numpy as np
from tqdm import tqdm

class spectrumData:
    mzArr = []
    intenArr = []
    retentionTime = 0

    def __init__(self, mz, inten, rt):
        self.mzArr = mz
        self.intenArr = inten
        self.retentionTime = rt

    def __str__(self) -> object:
        return "mzArr:" + str(self.mzArr) + "\nintenArr:" + str(self.intenArr) + "\nRT:" + str(self.retentionTime)

def decode(path):
    def DecodeFloatArray(code_str):
        code = b64decode(code_str)
        bytes = zlib.decompress(code)
        # bytes = code
        return np.array(struct.unpack("%df" % (len(bytes) / 4), bytes), dtype=np.float32)

    def DecodeDoubleArray(code_str):
        code = b64decode(code_str)
        bytes = zlib.decompress(code)
        # bytes = code
        return np.array(struct.unpack("%dd" % (len(bytes) / 8), bytes), dtype=np.float64)
    print('reading file...')
    data = minidom.parse(path)
    print('parsing file...')
    spectrums = data.getElementsByTagName('spectrum')
    ret = []
    for spectrum in tqdm(spectrums):
        binaries = spectrum.getElementsByTagName('binary')
        mzArrayBinaryStr = binaries[0].firstChild.data
        intenArrayBinaryStr = binaries[1].firstChild.data
        mzArr = DecodeDoubleArray(mzArrayBinaryStr)
        intenArr = DecodeDoubleArray(intenArrayBinaryStr)
        retentionTime = float(spectrum.getElementsByTagName('scan')[0].getElementsByTagName('cvParam')[0].getAttribute('value'))

        spectrumObj = spectrumData(mzArr, intenArr, retentionTime)
        ret.append(spectrumObj)
    return ret

'''