#!/usr/bin/env python3
import sys
import path
sys.path.append(str(path.Path(__file__).abspath().parent)+'/examples')
from encode_to_file import get_frames

import jiffy-lidar as jf
import numpy as np

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


# make a frameAllClose() lambda for comparing decoded frames to the originals
frameAllClose = lambda x,y,precision : all([np.allclose(a,b,atol=p/2.0) for a,b,p in zip(x,y,precision)])
# make a generic match lambda for comparing decoded frame modes and dtypes match the originals
match = lambda x,y : all([a==b for a,b in zip(x,y)])


if __name__ == '__main__':
 
    # Make a StreamWriter()
    shape = (128,1024)      # sensor scan shape
    scansPerFrame = 7       # 7 scans per frame, determined by the sensor
                            # Frames contain first- and second-return range, signal, and reflectivity plus ambient infrared.
    framesPerGroup = 10     # First frame of each group is intra-coded, all others are adaptively I or P coded
    precision = [5, 5, 1, 1, 1, 1, 1] # Use 5 millimeter precision for ranges. Don't quantize anything else.

    dataset_path = str(path.Path(__file__).abspath().parent+'/data/os0-128.npz')

    # Open the jiffy encoding stream, initializing the encoded byteStream with an empty bytes string b'', not a file.
    stream = jf.StreamWriter(b'', scansPerFrame, framesPerGroup, precision=precision)

    encodedModes = []  # save the I or P frame modes chosen by the encoder for later comparison
    encodedDtypes = [scan.dtype for scan in next(get_frames(dataset_path))] # save the dtypes given to the encoder for later comparison

    for frame in get_frames(dataset_path):
        # Encode the latest frame.
        stream.encode(frame)
        # Save the frame modes chosen by the encoder for later comparison
        encodedModes.append(stream.frameModes)
        encodedDtypes.append(encodedDtypes)

    # Rewind the stream to the beginning in preparation for decoding.
    stream.rewind()

    # Read the header- this is optional, decode() will do it for you if you don't,
    # but we are checking the header fields here.
    hdr = stream.readHeader()

    # check the header
    assert hdr['magic'] == jf.MAGIC
    assert hdr['version'] == jf.VERSION
    assert hdr['shape'] == shape
    assert hdr['scansPerFrame'] == scansPerFrame
    assert hdr['framesPerGroup'] == framesPerGroup

    # do a decode and check the results
    for frame,ogFrame in zip(stream.decode(), get_frames(dataset_path)):
        # ensure the frame is the same as the original to within the specified precision/2
        assert frameAllClose(frame, ogFrame, precision)
        # ensure the frame modes are the same as the original
        assert match(stream.frameModes, encodedModes.pop(0))
        # ensure the frame dtypes are the same as the original
        assert match([scan.dtype for scan in frame], encodedDtypes)


    print('\nSuccess!')
