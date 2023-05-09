#!/usr/bin/env python3
import jiffy as jf
import numpy as np
import matplotlib.pyplot as plt
from jf_pipecodecs import JiffyAdaptiveMaskP4 as OGjiffy
from jf_pipecodecs import SensorInfo

def fake_scan(shape, precision=1, out_of_range_percent=0.2, dtype=np.float32):
    scan = np.zeros(shape, dtype=np.float32)
    h,w = shape
    scan[:,0] = 0.1
    for j in range(1, w):
        # ramp horizontally from ~0.1 to ~width (w), with small positive
        # random offsets, keeping the resulting precision consistent with the
        # precision parameter.
        scan[:,j] = scan[:,j-1] + precision * np.random.randint(0, 4, (h,))

    # return a run length of zeros to insert for a line, with mean = out_of_range_percent relative to scan width
    run = lambda : np.uint32(np.abs(np.random.normal(out_of_range_percent*w, w//16)))

    # Insert a random length run() of zeros in the middle of the scan
    
    for i in range(h):
        dx = run()
        start = w//2 - dx//2
        end = start + dx
        scan[i,start:end] = 0.0

    return scan


shape = (128,1024)
precision = 1 # mm; or a list of integer precisions, one per scantype

framesPerGroup = 2
frameDtypes = [np.float32, np.int32, np.uint16, np.uint8]
scansPerFrame = len(frameDtypes)
nGroups = 2
fname = "test.jiffy"

# Create a new stream for encoding, must be opened in binary r/w mode ('wb+')
with open(fname, 'wb+') as f:

    stream = jf.Stream(scansPerFrame, framesPerGroup, byteStream=f, precision=precision)

    origFrames = []
    encodedModes = []

    # do a short encode
    for g in range(nGroups):
        for f in range(framesPerGroup):
            # Create a fake frame and save it for later comparison
            frame = [fake_scan(shape, precision=precision, dtype=d) for d in frameDtypes]
            origFrames.append(frame)
            # Encode the frame      
            stream.encode(frame)
            encodedModes.append(stream.frameModes)

# Close the stream
stream.close()

# Reopen (or rewind) the stream for decoding. 

# Stream must be opened in binary r/w mode ('rb+'),
# or you can rewind the encoded stream to the
# beginning with stream.rewind().

with open(fname, 'rb+') as f:
    stream = jf.Stream(byteStream=f)

    # Read the header- this is optional, decode() will do it for you if you don't,
    # but we are checking the header fields here.
    hdr = stream.readHeader()

    # check the header
    assert hdr['magic'] == jf.MAGIC
    assert hdr['version'] == jf.VERSION
    assert hdr['shape'] == shape
    assert hdr['scansPerFrame'] == scansPerFrame
    assert hdr['framesPerGroup'] == framesPerGroup
    assert hdr['framePrecisions'].all() == precision

    # make a frameAllClose() lambda for comparing decoded frames to the originals
    frameAllClose = lambda x,y : all([np.allclose(a,b) for a,b in zip(x,y)])
    # make a generic match lambda for comparing decoded frame modes and dtypes match the originals
    match = lambda x,y : all([a==b for a,b in zip(x,y)])

    # do a short decode
    for frame,ogFrame in zip(stream.decode(), origFrames):
        # ensure the frame is the same as the original
        assert frameAllClose(frame, ogFrame)
        # ensure the frame modes are the same as the original
        assert match(stream.frameModes, encodedModes.pop(0))
        # ensure the frame dtypes are the same as the original
        assert match(stream.frameDtypes, frameDtypes)

print('\nSuccess!')


