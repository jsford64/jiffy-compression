import numpy as np
from dataclasses import dataclass,field
from collections import OrderedDict
import pyfastpfor as p4
import zstandard as zstd
import os
from io import IOBase,BytesIO,BufferedRandom, BufferedReader, BufferedWriter
from os import path
import sys

'''
    Copyright 2023 Jeff S. Ford and Jordan S. Ford

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION = np.uint16(VERSION_MAJOR * 256 + VERSION_MINOR)
MAGIC = b'JFFY'

############################### ByteStream Class ###############################

class ByteStream(BufferedRandom):
    '''
    ByteStream(buffer=b'')

    ByteStream() is a subclass of the python io.BufferedRandom class that
    allows Jiffy to write compressed scan data to a file or bytes object (b'')
    when encoding, or read compressed frames from files or bytes objects
    when decoding.

    Constructing a ByteStream() object
    ----------------------------------

    There are three different methods of constructing a ByteStream():

    ByteStream(buffer = b'') :  Default. ByteStream() with no parameters
                                will create and use an empty io.BytesIO
                                object io.BytesIO(b'').

                                ByteStream(<bytes object>) will append to
                                or read from an existing bytes object.

                                ByteStream(<str object>) will
                                attempt to open a file with the
                                path/filename string provided.
                                If the file does not exist, it will be created.
                                If the file exists, it will be opened for
                                binary r/w (wb+) with random access, and all
                                existing data will be lost.

                                ByteStream(<file object>) will append to
                                or read from an existing file object. No data
                                will be lost or overwitten. The file should
                                be opened for binary r/w with random access.

    ByteStream Attributes
    ---------------------

    buffer: io.BytesIO object   This is the bytesIO or file object containing
                                the compressed scan data. 
    '''


    def __init__(self, buffer=b''):

        if isinstance(buffer,bytes):
            # convert to BytesIO
            buffer = BytesIO(buffer)

        elif isinstance(buffer,(BufferedReader,BufferedWriter,BufferedRandom,BytesIO,ByteStream)):
            buffer = buffer

        elif isinstance(buffer,str):
            # If the file exists, open it for binary read with random access
            # If the file does not exist, create it and open it for r/w with random access
            if path.exists(buffer):
                buffer = open(buffer,'rb+')
            else:
                buffer = open(buffer,'wb+')
        else:
            print(type(buffer))
            raise TypeError

        super().__init__(buffer)


    def getBuffer(self, start=0, length = -1):
        '''
        Non-destructively return a copy of the buffer data as a bytes object,
        beginning at the byte offset 'start', with a total length given by
        'length'.
        '''
        self.flush()
        pos = self.tell()
        self.seek(start)
        buff = self.read(length)
        self.seek(pos)
        return buff

    @property
    def size(self):
        '''Get the size in bytes of the entire encoded buffer/file without
        using system calls or re-reading the whole file.
        '''
        pos = self.tell()
        self.seek(0, 2)
        len = self.tell()
        self.seek(pos)
        return len

    def rewind(self):
        '''
        Non-destructively set the current r/w position to 0.
        Existing data is preserved.
        '''
        self.flush()
        self.seek(0)

    def clear(self):
        '''
        Clear the buffer. All existing buffer data is lost. Does not change codec state.
        '''
        self.seek(0)
        self.truncate()

    def writeField(self, val):
        '''
        Write a numpy dtype to the buffer.
        Returns the number of bytes written.
        '''
        return self.write(val.tobytes())

    def readField(self, dtype):
        '''
        Read a numpy dtype from the buffer.
        '''
        return np.frombuffer(self.read(int(dtype().itemsize)),dtype=dtype)[0]
    
    def updateField(self,val,offset):
        '''
        Update a numpy scalar value already in the buffer at byte offset, 
        and preserve the r/w position.
        '''
        pos = self.tell()
        self.seek(offset)
        self.writeField(val)
        self.seek(pos)

    def peekField(self, dtype):
        '''
        Peek at the next n bytes without advancing the r/w position.
        '''
        pos = self.tell()
        data = self.read(dtype.itemsize)
        self.seek(pos)
        return data
    
    def eof(self):
        '''
        Returns True if the current r/w position is at the end of the buffer.
        '''
        return self.tell() == self.size

############################### CodecState Class ###############################

@dataclass
class CodecState:
    ''' Holds current codec state values that are read/written by various chained codecs.
    '''
    # any of these parameters can be modified on instantiation of CodecState().
    scanShape : tuple = None
    precision : np.uint8 = 1
    # Is this frame is an I or P scan? P == 0, I == 1
    # first scan of a sequence after instantiation is always an i-scan (intra-compressed)
    iScan:bool = True
    # Dynamically keep track of min size uint required for dynRangeBits
    # through the codec stages
    pipeDtype:np.dtype = np.dtype(np.uint32)
    # Record the original dtype of the scan data
    origDtype:np.dtype = np.dtype(np.uint32)
    # keeps track of dynamic range through the codec stages
    dynRangeBits:np.uint8 = 0
    bitmask:np.array = field(default=None, init=False)

    @property
    def dynRangeMask(self):
        # return an integer mask of all valid bits for the current dyn range
        return (1<<self.dynRangeBits)-1

    def setDynRange(self,bits):
        # update the current dynamic range and set the dtype to the min required uint
        self.dynRangeBits = bits

        # NOTE(JORDAN): We probably want to figure this out and use smaller types when we can!
        self.pipeDtype = np.int32
        return

        # Here's the thing to figure out...
        if bits < 8:
            self.pipeDtype = np.int8
        elif bits < 16:
            self.pipeDtype = np.int16
        elif bits < 32:
            self.pipeDtype = np.int32


############################### NumpyCodec Classes ###############################

@dataclass
class NumpyCodec:
    '''
    Base class for a codec that transforms Numpy arrays in some way,
    producing another Numpy array. For example, a quantizer might quantize
    a uint32 array on encode, producing a uint16 array. The decode would reverse
    the change in dtype and quantization.

    codecState can be changed from the default by passing the new CodecState on
    instantiation of the NumpyCodec.

    the encode() and decode() methods must be overridden by the subclass
    '''
    codecState:CodecState

    def encode (nparray):
        raise NotImplementedError

    def decode (nparray):
        raise NotImplementedError

    def reset(self):
        # reset the iScan flag or other state info
        pass


############################### Pipe Class ###############################


@dataclass
class Pipe:
    '''
    Pipe([list of NumpyCodec()])

    Implements a codec pipeline by sequentially chaining I/O between a list of
    codecs provided on instatiation.
    '''
    codecs:list = field(default_factory=list)

    def __getitem__(self, i):
        return self.codecs[i]

    def encode(self,arr):
        '''
        Chain out of codec n to in of codec n+1
        '''
        out = arr
        taplen = 0

        for i,c in enumerate(self.codecs):
            # Loop through each codec in the pipe
            out = c.encode(out)

        return out

    def decode(self, *args):
        '''
        Chain out of codec n to in of codec n-1, starting at the end of the
        codec list.
        First codec is a byte reader so there is no decodedArr input.

        To decode, just run the codecs in reverse order
        '''
        reversedCodecs = self.codecs[::-1]

        # first codec may or may not have initial values
        if len(args):
            decodedArr = reversedCodecs[0].decode(*args)
        else:
            decodedArr = reversedCodecs[0].decode()

        for c in reversedCodecs[1:]:
            # Run remaining codec decodes
            # After the first codec, start passing the output to the next codec
            decodedArr = c.decode(decodedArr)

        return decodedArr


############################### Numpy Codec Subclasses ###############################

@dataclass
class Quantize(NumpyCodec):
    '''
    Encode:
      - determine the dynamic range of the input array and set the codecState accordingly
      - If input array is not unsigned integer, round and convert it to the
        smallest uint dtype required.
      - Quantize to codecState.precision
      - Keep track of dtype and dynamic range in CodecState.
    Returns a quantized and rounded numpy array with the smallest uint dtype
    that accomodates the quantized dynamic range.

    Modifies dynamic range and dtype in codecState.

    Returns:
      Quantized scan with smallest possible dtype and same shape as input

    Decode:
      - Reverse the encode tranformations (and I/O) above
    '''

    def encode(self,arr):

        # determine the dynamic range of this scan
        self.codecState.origDtype = arr.dtype
        max = arr.max()

        # Determine dynamic range of output, quantize to precision, round, and
        # convert to the smallest uint dtype that will accomodate the quantized dynamic range
        scale = 1.0/self.codecState.precision
        dynRangeBits = np.ceil(np.log2(max * scale)).astype(np.uint8)
        self.codecState.setDynRange( dynRangeBits )

        return (arr * scale).round(0).astype(self.codecState.pipeDtype)

    def decode(self,arr):
        # Reverse the encode() process above.
        # Multiply the input array arr * precision and convert dtype to codecState.pipeDtype
        precision = self.codecState.precision
        if precision == 1:
            return arr.astype(self.codecState.origDtype)
        return (arr * precision).astype(self.codecState.origDtype)


@dataclass
class SelectIorPscan(NumpyCodec):
    '''
    Determine if the scan should be an I (intra-coded) or P (predicted from previous scan) scan.
    Use 4 evenly spaced lines as a test heuristic to make the I or P decision.
    '''
    # Not in class instantiation init parameters
    lines:object = field(default=None,init=False)           # Y indices of horiz. lines to test
    last:np.array = field(default=np.array([]), init=False) # Lines used in the previous call to encode()

    test_pipe:Pipe = field(default=None, init=False)
    test_byteStream:ByteStream = field(default=None, init=False)    # temp compressed output stream

    def __post_init__(self):
        height = self.codecState.scanShape[0]
        # 4 was experimentally determined to be the min number of lines
        # with high correlation to the optimal I/P selection outcome
        nLines = np.uint8(4)
        # Space the lines out vertically over the scan
        stride = (height/(nLines+1)).round().astype(np.uint8)
        self.lines = np.linspace(stride,height-1,nLines).round().astype(np.uint8)

        # create a throwaway pipe to test bitmask compression with
        self.test_byteStream = ByteStream()
        self.test_pipe = Pipe( [Delta(self.codecState),
                                Zigzag(self.codecState),
                                P4(self.codecState),
                                Bytes2Follow(self.codecState, self.test_byteStream, decodeDtype=np.uint32) ]
                             )

    def encode (self, arr):
        '''
        Determine if we should encode an intra-scan (iScan) or inter-scan (pScan).
        Put the answer in self.codecState.iScan and return the input array.
        '''
        if self.codecState.iScan:
            # force an encode an I scan if codecState.iScan is already set
            self.last = arr[self.lines]
            return arr

        # Otherwise, adaptive mode
        # Determine whether to make this an I or P scan by doing a test
        # compressions on a few test lines in the scan.

        a = arr[self.lines]
        l = self.last
        self.last = a

        mask = (a != 0)
        l = l[mask]
        a = a[mask]

        # We don't need to keep the test data around,
        # so clear the buffer for speed.
        self.test_byteStream.clear()
        # get the test size of I scan encode
        dI = self.test_pipe.encode(a)

        self.test_byteStream.clear()
        # get the test size of P scan encode
        dP = self.test_pipe.encode(a-l)

        # If dP is larger, use an I scan.
        # If dI is larger, use a P scan.
        self.codecState.iScan = dP > dI

        return arr

    def decode(self,arr):
        '''
        Nothing to do, just pass through
        '''
        return arr


@dataclass
class Flatten(NumpyCodec):
    '''
    This is just a simple numpy flatten/reshape in the form of a codec
    spurely for the sake of being explicit about the encode/decode operations
    being performed in the codec pipeline.
    '''
    def encode(self,arr):
        '''
        same as nparray.flatten().
        '''
        return arr.flatten()

    def decode(self,arr):
        '''
        Same as nparray.reshape() to original scan shape.
        '''
        return arr.reshape(self.codecState.scanShape)


@dataclass
class Delta(NumpyCodec):
    '''
    Horizontal Delta compression/decompression using numpy.diff() on encode,
    and numpy.cumsum() on decode.
    '''

    def encode(self,arr):
        '''
        Horizontal diff. Convert to smaller dtype if possible.
        '''
        self.codecState.setDynRange(self.codecState.dynRangeBits+1)
        return np.diff(arr, prepend=0, axis=0).astype(self.codecState.pipeDtype)

    def decode(self,arr):
        '''
        Horizontal cumulative sum to restore original data.
        Convert to original pre-encoded dtype if needed.
        '''
        dynRangeMask = self.codecState.dynRangeMask
        self.codecState.setDynRange(self.codecState.dynRangeBits-1)
        dtype = self.codecState.pipeDtype
        return np.cumsum(arr, axis=0, dtype=dtype)

@dataclass
class Zigzag(NumpyCodec):
    '''
    Zigzag encoder class to encode/decode signed arrays <--> unsigned arrays.
    '''
    def encode(self,arr):
        '''
        Convert negative numbers to odd positive values,
        convert positive numbers to even positive values.
        Convert to larger dtype if necessary.
        Example:
            In: -3 -2 -1  0  1  2  3
            Out: 5  3  1  0  2  4  6
        '''
        self.codecState.setDynRange(self.codecState.dynRangeBits+1)
        dtype = self.codecState.pipeDtype

        a = arr.astype(dtype) * 2
        a[a < 0] += 1

        return np.abs(a)

    def decode(self,arr):
        '''
        Restore negative numbers and original dtype.
        Example:
            In:   5  3  1  0  2  4  6
            Out: -3 -2 -1  0  1  2  3
        '''
        dtype = self.codecState.pipeDtype

        a = arr.astype(dtype)
        odds = a & 1 == 1
        a[odds] = a[odds] * -1
        a //= 2

        self.codecState.setDynRange(self.codecState.dynRangeBits-1)
        return a.astype(self.codecState.pipeDtype)


@dataclass
class Pack(NumpyCodec):
    '''
    Pack/unpack bitmask bits (bool) to bytes using numpy packbits()/unpackbits().
    '''

    def encode(self,bitmask):
        '''
        numpy packbits()
        '''
        return np.packbits(bitmask, axis=None, bitorder='little')

    def decode(self,arr):
        '''
        numpy unpackbits()
        '''
        bitmask = np.unpackbits(arr, axis=None, count=None, bitorder='little').astype(bool)
        return bitmask



@dataclass
class Bytes2FollowHeader:
    '''
    Encode:
      Write a 32 bit header word to the byteStream
      MS 5 bits: dynamic range of encoded data to follow (# of significant bits)
      LS 27 bits: number of bytes to follow

      Returns: number of bytes written to byteStream

    Decode:
      Returns:
          tuple ( number of significant bits in encoded data,
                  number of bytes in the encoded data segment
                )
    '''
    byteStream:ByteStream

    def encode(self, origDtype, dynRangeBits, length):
        nBytes =  self.byteStream.writeField( np.frombuffer(origDtype.char.encode(), dtype=np.uint8) )
        hdr = (dynRangeBits & 0x1f)<<27 | length & 0x7ffffff
        nBytes += self.byteStream.writeField( np.uint32(hdr) )
 
        return nBytes

    def decode(self):
        origDtype = np.dtype(self.byteStream.read(1))
        hdr = np.frombuffer( self.byteStream.read(4),dtype=np.uint32 )[0]
        dynRangeBits = (hdr>>27) & 0x1f
        length = hdr & 0x7ffffff
        return origDtype, dynRangeBits, length


@dataclass
class Bytes2Follow:
    '''
    Encode: Write a 'bytes to follow' header and the bytesIn buffer to byteStream.
            Returns: total number of bytes written to byteStream.

    Decode: Extract the dynamic range (# of significant bits of decoded buffer),
            length, and the encoded buffer bytes from stream.
            Returns: the buffer bytes as bytes if isBitmask == True
                     the buffer bytes cast as decodeDtype if isBitMask == False
    '''
    codecState:CodecState
    byteStream:ByteStream
    decodeDtype:np.dtype = None

    # Is this compressed buffer a bitmask?
    isBitMask:bool = False

    # not included in class instantiation parameters
    header:Bytes2FollowHeader = field(default=None, init=False)

    def __post_init__(self):
        self.header = Bytes2FollowHeader(self.byteStream)

    def encode(self,bytesIn):
        if self.isBitMask:
            nBytes = self.header.encode(np.bool_().dtype, 0, len(bytesIn))
        else:
            nBytes = self.header.encode(self.codecState.origDtype,
                                        self.codecState.dynRangeBits,
                                        len(bytesIn))
        nBytes += self.byteStream.write(bytesIn)
        return  nBytes # bytes written

    def decode(self):
        origDtype, dynRangeBits, length = self.header.decode()
        if not self.isBitMask:
            self.codecState.origDtype = origDtype
            self.codecState.setDynRange(dynRangeBits)

        buff = self.byteStream.read(length)
        if self.decodeDtype is None:
            return buff
        else:
           return np.frombuffer(buff,dtype=self.decodeDtype)


@dataclass
class MaskedScanDelta(NumpyCodec):
    '''
    Encode:
      Create bitmask from 0 values in scan.
      keep only non-zero values from scan according to bitmask.
      If iScan:
          done.
          Return: non-zero scan and bitmask in a tuple.
      Else:
          Calc delta scan and delta bitmask with scan and bitmask from previous scan.
          Return: delta scan and selta bitmask in a tuple

    Decode:
      Reverse the process above
      Return: reconstructed scan with 2D shape
    '''
    maskval:np.int32 = 0

    # Not in init parameters
    _lastscan:np.array = field(default=None, init=False)
    _lastbitmask:np.array = field(default=None, init=False)

    def encode(self, currentscan):

        bitmask =   currentscan != self.maskval

        if self.codecState.iScan:
            # I scan
            residualScan = currentscan[bitmask]
            residualBitmask = bitmask
        else:
            # P scan
            residualScan = (currentscan  - self._lastscan)[bitmask]
            residualBitmask = np.logical_xor(bitmask, self._lastbitmask)

        self._lastscan = currentscan
        self._lastbitmask = bitmask

        return residualScan,residualBitmask

    def decode(self, residualBitmask, residualScan):

        reconscan = np.full(residualBitmask.shape, self.maskval, dtype=residualScan.dtype)

        if self.codecState.iScan:
            # I scan
            reconBitmask = residualBitmask
            reconscan[reconBitmask] = residualScan
        else:
            # P scan
            reconBitmask = np.logical_xor(residualBitmask, self._lastbitmask)
            reconscan[reconBitmask] = residualScan + self._lastscan[reconBitmask]

        self._lastscan = reconscan
        self._lastbitmask = reconBitmask

        return reconscan


@dataclass
class P4:
    '''
    Compress/Decompress flat contiguous nparray of np.uint32s <-> byte string
    using the PyFastPFor algorithm.
    '''
    codecState:CodecState = CodecState()
    p4name:str = 'simdfastpfor256'

    def __post_init__(self):
        self._codec = p4.getCodec(self.p4name)

    def encode(self,arr,level=None):
        assert (arr.ndim == 1)

        aIn = arr.astype(np.uint32)

        aInSz = aIn.size
        aOut = np.empty(aInSz*2,dtype=np.uint32)
        aOut[0] = aInSz

        encSz = self._codec.encodeArray( aIn, aInSz, aOut[1:], len(aOut))
        return aOut[:encSz+1].tobytes()

    def decode(self,bytesIn):
        arrayIn = np.frombuffer(bytesIn,dtype=np.uint32)
        aOutSz = arrayIn[0]
        aOut = np.empty(aOutSz,dtype=np.uint32)

        decSz = self._codec.decodeArray(arrayIn[1:],len(arrayIn)-1,aOut,aOutSz)
        return aOut[:decSz].astype(np.uint32)


@dataclass
class Zstd:
    '''
    Encode/decode using the zstandard compression codec.
    '''
    codecState:CodecState = CodecState()
    default:int = 6
    levels:tuple = (0,9)

    def encode(self, arrayIn, level = None):
        if level is None:
            level = self.default
        return zstd.compress(arrayIn.tobytes(),level=level)

    def decode(self, bytesIn):
        dtype = np.uint8

        buff = zstd.decompress(bytesIn)
        arr = np.frombuffer(buff,dtype=dtype)

        return arr



# modes for Jiffy().encode()
ADAPTIVE = 0
ISCAN = 1

@dataclass
class Scan:
    '''
    Scan(shape, precision=1, bytestream=b'')

    Compress or decompress a sequence of scans of the same scan type (range, intensity, etc.).
    A scan codec is required per scan type, because the codec holds state information and previous scan data.

    shape:      a tuple indicating the shape of the scan to be
                encoded/decoded [width,height]

    precision:  On encode, scans will be rounded and truncated (quantized) to
                the nearest integer by default (precision=1). Increasing
                'precision'will make the quantization coarser, rounding  and
                truncating to the nearest integer multiple of 'precision',
                producing a corresponding increase in compression ratio.
                Valid precision values are positive, non-zero integers.
                Range scans are expected to be pre-scaled to integer units of
                millimeters such that a value of 1 == 1mm.

    byteStream: The byte buffer to encode into, can be specified as:
                      - a byte string (b'')
                      - an open, binary, random access file descriptor,
                        opened with 'wb+' or 'rb+'
                      - an io.BytesIO() or io.BufferedRandom() object,
                      - a filename that Scan will open in 'wb+' (rewrite) mode,
                      - a jiffy.ByteStream() object.
                Scan will append encoded data to any data already in byteStream,
                except in the case where byteStream is a filename to open.
                Scan defaults to using an empty bytes object (b'').

    Attributes:
    ----------------

    isIscan:    True if the encoded or decoded scan is an I scan, False if it is a P scan.

    nBytes:     The number of bytes written/read to byteStream on encode/decode.

    precision:  The quantization precision used to encode the scan.

    shape:      The shape of the scan.

    byteStream: The byte buffer to encode into.

    Methods:
    ----------------

    encode(scan, mode=ADAPTIVE)

        scan:   The scan to encode. Must be a numpy array of the same shape
                as the Scan object.

        mode:   The scan mode to use. Valid values are:
                    - ADAPTIVE:  The scan mode will be selected automatically
                                    based on the scan data.
                    - ISCAN:     The scan will be encoded as an I scan.

        Returns:    The ByteStream object containing the compressed scan.

    decode()

        Returns:    The decoded scan as a numpy array.

    '''
    
    shape:np.array
    byteStream:ByteStream = b''
    precision:np.uint8 = 1

    def __post_init__(self):

        self.codecState = CodecState(self.shape,self.precision)
        self.nBytes = 0     # number of bytes written to byteStream

        # On Scan.encode():
        #       pipe0.encode() runs first, followed by
        #       bitmaskPipe.encode(), which writes the compressed bitmask,
        #       then pipe1.encode(), which writes the compressed scan.

        # On Scan.decode():
        #       bitmaskPipe.decode() runs first, to decode the compressed
        #       bitmask, followed by pipe1.decode(), then pipe0.decode(),
        #       with the order of codec execution reversed in each pipe.

        self.pipe0 = Pipe (
                            [ Quantize(self.codecState),
                              # SelectIorPscan().encode() sets codecState.iScan
                              SelectIorPscan(self.codecState),
                              Flatten(self.codecState),
                              MaskedScanDelta(self.codecState)]
                          )

        # On encode, the compressed residualBitmask is written to byteStream first
        # On decode, the compressed residualBitmask is extracted from byteStream first

        self.bitmaskPipe = Pipe (
                                  [ Pack(self.codecState),
                                    Zstd(self.codecState),
                                    Bytes2Follow(self.codecState, self.byteStream, isBitMask=True)]
                                )

        # On encode, pipe0 MaskedScanDelta().encode() output is compressed and
        # written to byteStream after bitmaskPipe write of compressed residualBitmask.

        # On decode, input to pipe0 MaskedScanDelta().decode() is extracted
        # from byteStream after bitmaskPipe extraction of compressed residualBitmask.

        self.pipe1 = Pipe (
                            [ Delta(self.codecState),
                              Zigzag(self.codecState),
                              P4(self.codecState),
                              Bytes2Follow(self.codecState, self.byteStream, decodeDtype=np.uint32) ]
                          )

    @property
    def isIscan(self):
        '''
        Returns True if the encoded scan is an i-scan (intra-coded scan)
        or False if the encoded scan is a p-scan (temporospatial or p-coded scan).
        '''
        return self.codecState.iScan


    def encode(self, arr, mode=ADAPTIVE):
        '''
        Encode a LiDAR scan with Jiffy compression, and append it to the
        byteStream.

        arr:    2D numpy array to encode. The dtype of arr can be
                float, int, or uint of any number of bytes, but
                input values are expected to all be positive.
        mode:   Automatically select I or P scan (mode = ADAPTIVE),
                or force an I scan (mode=ISCAN).

                Returns the number of bytes written to the compressed ByteStream buffer.

                To get the compressed scan mode after calling encode(), use Scan.isIscan.
                To get the number of bytes written to byteStream, use Scan.nBytes.

        '''
        self.codecState.iScan = bool(mode)

        # Write a uint32 to byteStream indicating the scan type and encoded scan length
        # We don't know the encoded scan length yet, so we'll write a placeholder
        # value of 0, and update it later.

        # First, save the current byteStream offset
        scanOffset = self.byteStream.tell()
        nBytes = self.byteStream.writeField(np.uint32(0))

        # encode the scan, write the compressed bitmask first, then the compressed scan
        residualScan, residualBitmask = self.pipe0.encode(arr)
        nBytes += self.bitmaskPipe.encode(residualBitmask)
        nBytes += self.pipe1.encode(residualScan)
        self.nBytes = np.uint32(nBytes)
    
        # Update the scan length in the byteStream

        if self.isIscan:
            # If the mode is an IScan, the scan length is negative
            nBytes = -nBytes

        self.byteStream.updateField(np.uint32(nBytes), scanOffset)
        return self.nBytes, self.byteStream
    

    def decode(self):
        '''
        Decode each scan from the byteStream.
        '''
        # Read the scan length from the byteStream
        nBytes = self.byteStream.readField(dtype=np.int32)

        self.codecState.iScan = nBytes < 0
        nBytes = np.abs(nBytes)    

        # decode a scan
        self.nBytes = nBytes
        residualBitmask = self.bitmaskPipe.decode()
        residualScan = self.pipe1.decode()
        buff = self.pipe0.decode(residualBitmask,residualScan)
        return buff


@dataclass
class Stream:
    '''

        Stream( scansPerFrame:np.uint8 = 1, framesPerGroup:int = 10, byteStream:ByteStream = b'', 
                precision:np.uint8 = 1, header:bool = True )

    )
        Jiffy Stream codec class. Compresses/decompresses a LiDAR stream, a sequence of LiDAR scans of multiple scan types.
        A user should typically use either StreamReader() or StreamWriter, rather than Stream(),
        to avoid ambiguities regarding file handling.

        Stream() constructor arguments:
        -------------------------------

        scansPerFrame:  
            The number of scans, each with a unique scan type (range, intensity, 
            etc.), returned by the LiDAR in one sweep. The collection of scans returned 
            in one sweep is called a frame. 

            Ignored on decode, since the number of scans per frame is encoded in the stream.
            Must be > 0. Default is 1.

        framesPerGroup:

            A group of frames is a logical grouping of frames that are encoded 
            together, with the first frame in each group forced to be encoded as an 
            intra-coded frame (all scans in the first frame of every group are 
            I scans). Default is 10. 

            Ignored on decode, since the number of frames per group is encoded in the stream.

            If framesPerGroup == 0, no groups are encoded. All frames will be coded as 
            temporal (I) or spatial (P) scans if mode=ADAPTIVE, or as intra-coded scans 
            if mode=ISCAN.

            If framesPerGroup > 0, a group is encoded. All scans in the first frame 
            of each group will be intra-coded, so if framesPerGroup == 1, all scans will
            be intra-coded.
            All other frames in the group will be coded as temporal (I) or spatial (P) scans 
            if mode=ADAPTIVE, or as intra-coded scans if mode=ISCAN.

            framesPerGroup must be a positive integer.

            Groups of frames are used to improve random access and editing, and may
            potentially aid in error recovery when decoding in future versions.

        byteStream:
            A Jiffy ByteStream object containing a compressed stream for decode, or
            an empty ByteStream object to hold the encoded stream.

        precision:  
            A scalar precision value to apply to all scan types, or a list of 
            precision values, one for each scan type. Precision is used to quantize 
            the scan values to the nearest integer multiple of precision.

            On encode, scans will be rounded and truncated (quantized) to
            the nearest integer by default (precision=1). Increasing
            'precision'will make the quantization coarser, rounding  and
            truncating to the nearest integer multiple of 'precision',
            producing a corresponding increase in compression ratio.
            Valid precision values are positive, non-zero integers.
            Range scans are expected to be pre-scaled to integer units of
            millimeters such that a value of 1 == 1mm.

            On decode, precision is ignored, since the precision values are encoded
            in the stream.

        header:     
            If True, a header will be written to the byteStream on the first call to
            encode, and extracted from the byteStream on the first call to decode.
            If header is False, the header will not be written on encode, and the user
            is required to read the header (if it is present) before calling decode.
            The header contains information about the stream, including the number of
            scans per frame, the number of frames per group, and the precision values
            used to encode the stream. Default is True.

        Attributes
        ----------

        byteStream:
            The byteStream object used to encode/decode the stream.

        framePrecisions:
            A list of np.uint16 precision values used to encode/decode the stream, one for each
            scan type.

        scansPerFrame:
            The number of scans per frame used to encode/decode the stream.

        framesPerGroup:
            The number of frames per group used to encode/decode the stream.

        frameModes:
            The actual scan modes used to encode/decode the frame. frameModes is a list of
            scan modes, one for each scan in the frame. frameModes is only valid
            after the first call to encode or decode, and updates on each subsequent call.

        frameBytes:
            The number of bytes used to encode/decode the frame. frameBytes is a list of
            number of bytes, one for each scan in the frame. frameBytes is only valid
            after the first call to encode, and it updates on each subsequent call to encode.
            Ignored on decode.

        frameCount:
            The number of frames encoded/decoded. frameCount is only valid after the first
            call to encode or decode, and it updates on each subsequent call to encode or decode.
            if framesPerGroup > 0, frameCount is the number of scans encoded/decoded in the current group.

        Encoded stream format:

            All multi-byte fields are encoded as little endian.

            Stream:
            -------
            if header == True:
                Header                          Optional header, only if header == True

            if framesPerGroup == 0:
                [ iFrame ] + [ aFrame ] * n     no groups, first frame is an I scan, 
                                                all others are adaptive, n = number of frames encoded in stream - 1
            elif framesPerGroup == 1:
                [ iFrame ] * n                  all frames are I scans, n = number of frames encoded in stream

            else:
                [ Group ] * m                   groups of frames, first frame in group is an I scan,
                                                all others in the group are adaptive, m = number of groups encoded


            Header (optional):
            ------------------
            
                Field Name      Type,       Size (bytes)    Description
                ---------------------------------------------------------
                "JFFY"          ASCII,          4           File 'magic number' identifier (jiffy.JIFFY)
                version         uint8,          1           Version of the Jiffy file format (jiffy.VERSION)
                shape           uint32,         8           shape of scans, (rows, cols)
                scansPerFrame   uint8,          1           Number of scans in a frame, min=1
                framesPerGroup  uint32,         4           Number of frames in a group, 
                                                            0:          no groups, first frame is an I scan, 
                                                                        all others are adaptive

                                                            1:          all frames are I scans

                                                            2 or more:  first frame in group is an I scan, 
                                                                        all others in the group are adaptive

                framePrecisions uint16,     scansPerFrame   list of precision values, one for each scan type
        '''
    scansPerFrame:np.uint8 = 1
    framesPerGroup:int = 10
    byteStream:ByteStream = b''
    precision:np.uint16 = 1
    header:bool = True

    frameModes:list[bool] = field(init=False)
    framePrecisions:list[np.uint8] = field(init=False)
    frameBytes:list[np.uint32] = field(init=False)
    frameCount:np.uint64 = field(init=False)
    _encodedScans:list[Scan] = field(default_factory=list, init=False)

    def __post_init__(self):
        '''
        Initialize the Stream object.
        '''
        if  self.framesPerGroup < 0:
            raise ValueError('framesPerGroup must be a positive integer')

        if  self.scansPerFrame < 1:
            raise ValueError('scansPerFrame must be > 0')
        
        self.scansPerFrame = np.uint8(self.scansPerFrame)
        self.framesPerGroup = np.uint32(self.framesPerGroup)

        # make precision a list if it is not already
        if  not isinstance(self.precision, list):
            self.framePrecisions = [self.precision] * self.scansPerFrame
        else:
            self.framePrecisions = self.precision

        del self.precision

        assert all([p < np.iinfo(np.uint16).max for p in self.framePrecisions])
        

        self.frameCount = 0
        self._scanCodecs = []
        self._encodedScans = []
        self.frameModes = []


    def writeHeader(self):
        '''
        Write the Jiffy header to the byteStream.
        '''
        self.byteStream.write(b'JFFY')
        self.byteStream.writeField(VERSION)
        self.byteStream.writeField(np.uint32(self.shape[0]))
        self.byteStream.writeField(np.uint32(self.shape[1]))
        self.byteStream.writeField(np.uint8(self.scansPerFrame))
        self.byteStream.writeField(np.uint32(self.framesPerGroup))
        self.byteStream.write(np.array(self.framePrecisions, dtype=np.uint16).tobytes())
        self.rewind()
        self.readHeader()

    def readHeader(self):
        '''
        Read the Jiffy header from the byteStream and return a dict with the header contents.
        '''
        self.magic = self.byteStream.read(4)
        if  self.magic != MAGIC:
            raise ValueError('ByteStream is not a Jiffy compressed file.')
        self.version = self.byteStream.readField(dtype=np.uint16)
        self.shape = (self.byteStream.readField(dtype=np.uint32), self.byteStream.readField(dtype=np.uint32))
        self.scansPerFrame = self.byteStream.readField(dtype=np.uint8)
        self.framesPerGroup = self.byteStream.readField(dtype=np.uint32)
        self.framePrecisions = np.frombuffer(self.byteStream.read(np.uint16().itemsize*self.scansPerFrame), dtype=np.uint16)
        self.header = False

        return self.stats()
    
    def stats(self):
        '''
        Return a dictionary of stream statistics.
        '''
        return {'magic':self.magic,
                'version':self.version,
                'shape':self.shape, 
                'scansPerFrame':self.scansPerFrame, 
                'framesPerGroup':self.framesPerGroup, 
                'framePrecisions':self.framePrecisions}

    def _clearScanBuffers(self):
        '''
        Clear the encoded Scan buffers, one for each scantype.
        '''
        [codec.byteStream.clear() for codec in self._scanCodecs]
       

    def encode(self, frame):
        ''' 
        frame:  A frame is a list of 2D numpy arrays (scans) produced from one sweep of the lidar,
                with one array for each scan type (range, intensity, etc.) in the frame.
                The number of scans in a frame must match the number of scan types in the stream,
                and remain constant for the entire stream.
                
                If self.scansPerFrame == 0, frame can be a list of scans or
                a single scan (2D numpy array, not a list).

                The dtype of each scan can be float, int, or uint of any number of bytes, but
                input values are expected to all be positive.

                The shape of each scan must match the shape of the first scan encoded in the stream.
                '''

        # make frame a list if it's not already
        if  not isinstance(frame, (list,tuple)):
            frame = [frame]
        
        # check that frame is a list of 2D numpy arrays
        # and that all arrays are 2D
        if not np.all([isinstance(arr,np.ndarray) for arr in frame]):
            raise TypeError('Frame must be a list of numpy arrays.')
        if not np.all([arr.ndim == 2 for arr in frame]):
            raise TypeError('Frame must be a list of 2D numpy arrays.')
        
        if len(self._scanCodecs) == 0:
            # first frame, initialize the stream
            # This has to be done on the first call, because the shape of 
            # the frame is not known until encode is called.
            self.shape = frame[0].shape
            self._scanCodecs = [Scan(self.shape, self.byteStream, precision=p) for p in self.framePrecisions]
            self.frameCount = 0

        # check that the number of scans in the frame matches the number of scan types in the stream
        if len(frame) != self.scansPerFrame:
            raise ValueError('The number of scans per frame must remain constant for the entire stream.')

        # check that the shape of each scan in the frame is constant
        if not np.all([scan.shape == self.shape for scan in frame]):
            raise ValueError('The scan shape must remain constant for the entire stream.')
        
        if self.header:
            self.writeHeader()

        mode = ISCAN    # default to intra-coded scan

        if self.frameCount != 0:
            # this is not the first frame in the stream or group
            mode = ADAPTIVE
        
        # encode each scan in the frame and write it to the byteStream
        self.frameBytes = [codec.encode(scan, mode) for codec,scan in zip(self._scanCodecs, frame)]
        self.frameModes = [codec.isIscan for codec in self._scanCodecs]

        self.frameCount += 1

        if self.frameCount == self.framesPerGroup:
            # this is the last frame in a group
            self.frameCount = 0

        return self.byteStream.getBuffer()
        

 
    def decode(self):
        '''
        Decode a frame (list of scans) from the byteStream.
        This is a generator function that yields one frame of scans at a time.
        '''
        # Read the header if this is the first frame and readHeader has not been called.
        if self.header:
            self.readHeader()

        if len(self._scanCodecs) == 0:
            # first frame, initialize the stream
            # This has to be done on the first call, because the
            # header must be read before the relevant parameters are known.
            self._scanCodecs = [Scan(self.shape, self.byteStream, p) for p in self.framePrecisions]

        # Read a frame of encoded scans from the byteStream
        while not self.byteStream.eof():
            # decode the scans in a frame
            frame = [scan.decode() for scan in self._scanCodecs]
            self.frameModes = [scan.isIscan for scan in self._scanCodecs]
            self.frameBytes = [scan.nBytes for scan in self._scanCodecs]

            self.frameCount += 1
            if self.frameCount == self.framesPerGroup:
                # this is the last frame in a group
                self.frameCount = 0
        
            yield frame
       

    def close(self):
        '''
        Close the byteStream.
        Do not call this function if you intend to continue using the byteStream.
        Use rewind() instead.
        '''
        self.byteStream.close()

    def rewind(self):
        '''
        Rewind the byteStream to the beginning of the stream.
        Use this when you want to decode an encoded stream you just generated.
    
        Stream state is reset to:
            header = True
            frameCount = 0
        '''
        self.byteStream.rewind()
        self.header = True
        self.frameCount = 0
        self._encodedScans = []

class StreamReader(Stream):
    '''
        StreamReader( byteStream:ByteStream, *args, **kwargs )

        Jiffy StreamReader subclass of Stream(). Decompresses a LiDAR stream, a sequence of LiDAR scans of multiple scan types.
        
        StreamReader() constructor arguments:
        -------------------------------
        byteStream:
            A previously encoded ByteStream object containing a compressed stream,
            a byte string (b''), a file name, or a file opened in mode 'rb+'.
            
            On decode, Stream will extract encoded data from byteStream, producing 
            decoded frames.
        *args:
            Remaining arguments are passed to the Stream constructor.
        **kwargs:
            Remaining keyword arguments are passed to the Stream constructor.
    '''
    def __init__(self, byteStream, *args, **kwargs):
        if isinstance(byteStream, str): 
            byteStream = ByteStream(byteStream)
        super().__init__(byteStream=byteStream, *args, **kwargs)

class StreamWriter(Stream):
    '''
        StreamWriter( byteStream:ByteStream, *args, **kwargs )

        Jiffy StreamWriter subclass of Stream(). Compresses a LiDAR stream, a sequence of LiDAR scans of multiple scan types.
        
        StreamWriter() constructor arguments:
        -------------------------------
        byteStream:
            A byte string (b''), a file name, or a file opened in mode 'wb+'.
            
            StreamWriter will append encoded data to any data already in byteStream,
            except in the case where byteStream is a filename to open. If byteStream is a
            filename, it will be opened in wb+ mode, and any existing data will be lost.
            If an open file is used, it must be opened in wb+ mode. 
            
            byteStream defaults to using an empty bytes object (b'').
        *args:
            Remaining arguments are passed to the Stream constructor.
        **kwargs:
            Remaining keyword arguments are passed to the Stream constructor.
    '''
    def __init__(self, byteStream, *args, **kwargs):
        
        if isinstance(byteStream, str): 
            if os.path.exists(byteStream):
                os.remove(byteStream)
            byteStream = ByteStream(byteStream)
        elif isinstance(byteStream, bytes):
            byteStream = ByteStream(byteStream)
        elif isinstance(byteStream, IOBase):
            if byteStream.mode != 'wb+':
                raise ValueError('StreamWriter requires a file opened in wb+ mode.')
            byteStream = ByteStream(byteStream)
        elif isinstance(byteStream, ByteStream):
            pass
        else:
            raise TypeError('byteStream must be a ByteStream, a file name, a file opened in wb+ mode, or a byte string (b\'\').')


        super().__init__(byteStream=byteStream, *args, **kwargs)
