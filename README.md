# jiffy-compression
Jiffy- a fast, lossless SIMD compression codec for LiDAR streams

## Dependencies

Jiffy was written and tested using Python 10.3.

### Python Modules

    PyFastPfor:

        ARM:    git@github.com:jsford64/PyFastPFor-OSX-ARM64.git
        Intel:  git@github.com:searchivarius/PyFastPFor.git

    Zstandard:

        pip install zstandard

## Examples

Look at test.py for an example of how to compress and decompress a LiDAR sequence to a file or byte string.

## Jiffy Constants

    VERSION = np.uint16(1.0 * 256)  Current Jiffy version.
    MAGIC = b'JFFY'                 'Magic Number' for Jiffy encoded data/files.
    ADAPTIVE = 0                    Encode scan as I or P according to adaptive scan algorithm.
    ISCAN    = 1                    Encode scan as I only.

## Jiffy Stream Codec Class

    Stream( scansPerFrame:np.uint8 = 1, framesPerGroup:int = 10, byteStream:ByteStream = b'', 
            precision:np.uint8 = 1, header:bool = True )


    Compresses/decompresses a LiDAR stream, a sequence of LiDAR scans of multiple scan types.
    
### Stream() Constructor Arguments

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
        A previously encoded ByteStream object containing a compressed stream,
        a byte string (b''), a file name, or an open file.
        
        On encode, Stream will append encoded data to any data already in byteStream,
        except in the case where byteStream is a filename to open. If byteStream is a
        filename, it will be opened in wb+ mode, and any existing data will be lost.
        If an open file is used, it must be opened in rb+ or wb+ modes. 
        
        On decode, Stream will extract encoded data from byteStream, producing 
        decoded frames.
        byteStream defaults to using an empty bytes object (b'').

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

### Stream Attributes

    byteStream:
        The byteStream object used to encode/decode the stream.

    framePrecisions:
        A list of np.uint8 precision values used to encode/decode the stream, one for each
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

### Stream Methods

    writeHeader():
        Write the Jiffy header to the byteStream.

    readHeader():
        Read the Jiffy header from the byteStream and return a dict with the header contents.

    stats():
        Return a dictionary of header contents.

    encode(frame):
        frame:  A frame is a list of 2D numpy arrays (scans) produced from one sweep of the lidar,
                with one array for each scan type (range, intensity, etc.) in the frame.
                The number of scans in a frame must match the number of scan types in the encoded stream,
                and remain constant for the entire stream.
                
                If self.scansPerFrame == 0, frame can be a list of scans or
                a single scan (2D numpy array, not a list).

                The dtype of each scan can be float, int, or uint of any number of bytes, but
                input values are expected to all be positive.

                The shape of each scan must match the shape of the first scan encoded in the stream.

    decode():
        Decode a frame (list of scans) from the byteStream.
        This is a generator function that yields one frame of scans at a time.

    close():
        Close the byteStream.
        Do not call this function if you intend to continue using the byteStream.
        Use rewind() instead.

    rewind():
        Rewind the byteStream to the beginning of the stream.
        Use this when you want to decode an encoded stream you just generated.
    
        Stream state is reset to:
            header = True
            frameCount = 0


### Encoded Stream Format

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
    -------------------------------------------------------------------------------------------
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

    framePrecisions uint8,      scansPerFrame   list of precision values, one for each scan type
