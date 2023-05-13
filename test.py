#!/usr/bin/env python3
import jiffy as jf
import numpy as np
import matplotlib.pyplot as plt

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

def fake_scan(shape, precision=1, out_of_range_percent=0.2, dtype=np.float32):
    scan = np.zeros(shape, dtype=dtype)
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
precision = 2 # mm; or a list of integer precisions, one per scantype

framesPerGroup = 2
frameDtypes = [np.float32, np.int32, np.uint16, np.uint8]
scansPerFrame = len(frameDtypes)
nGroups = 2
fname = "test.jiffy"

# Create a new stream for encoding, must be opened in binary r/w mode ('wb+')
stream = jf.StreamWriter(fname, scansPerFrame, framesPerGroup, precision=precision)

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

stream_rdr = jf.StreamReader(fname)

# Read the header- this is optional, decode() will do it for you if you don't,
# but we are checking the header fields here.
hdr = stream_rdr.readHeader()

# check the header
assert hdr['magic'] == jf.MAGIC
assert hdr['version'] == jf.VERSION
assert hdr['shape'] == shape
assert hdr['scansPerFrame'] == scansPerFrame
assert hdr['framesPerGroup'] == framesPerGroup

for hdr_prec in hdr['framePrecisions']:
    assert hdr_prec == precision

# make a frameAllClose() lambda for comparing decoded frames to the originals
frameAllClose = lambda x,y,eps : all([np.allclose(a,b, atol=eps) for a,b in zip(x,y)])
# make a generic match lambda for comparing decoded frame modes and dtypes match the originals
match = lambda x,y : all([a==b for a,b in zip(x,y)])

# do a short decode
for frame, ogFrame in zip(stream_rdr.decode(), origFrames):
    # ensure the frame is the same as the original
    assert frameAllClose(frame, ogFrame, precision/2.0)
    # ensure the frame modes are the same as the original
    assert match(stream_rdr.frameModes, encodedModes.pop(0))
    # ensure the frame dtypes are the same as the original
    for s, scan in enumerate(frame):
        assert np.issubdtype(scan.dtype, frameDtypes[s])

print('Success!')
