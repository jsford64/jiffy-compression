#!/usr/bin/env python3

import os
import sys
import path
sys.path.append(path.Path(__file__).abspath().parent)
import jiffy as jf
import numpy as np
import matplotlib.pyplot as plt
from utils import DatasetReader

def get_frames(dataset_path, count=-1):
    rdr = DatasetReader(dataset_path)

    range_scans = rdr.getRange(count)
    range2_scans = rdr.getRange2(count)
    signal_scans = rdr.getSignal(count)
    signal2_scans = rdr.getSignal2(count)
    reflect_scans = rdr.getReflectivity(count)
    reflect2_scans = rdr.getReflectivity2(count)
    nearir_scans = rdr.getNearIR(count)
    
    return zip(range_scans, range2_scans, signal_scans, signal2_scans, reflect_scans, reflect2_scans, nearir_scans)

def decode_from_file(jiffy_file):
    with open(jiffy_file, 'rb+') as f:
        stream = jf.Stream(byteStream=f)

        header = stream.readHeader()
        print("Header")
        print("--------------------------------------")
        print(f"  magic:          {header['magic']}")
        print(f"  version:        {header['version']}")
        print(f"  shape:          {header['shape']}")
        print(f"  scansPerFrame:  {header['scansPerFrame']}")
        print(f"  framesPerGroup: {header['framesPerGroup']}")
        print(f"  scanPrecisions: {header['framePrecisions']}")

        frames = stream.decode()
        for frame in frames:
            for s in range(scansPerFrame):
                print(s)
                plt.imshow(frame[s])
                plt.show()
        stream.close()


if __name__=='__main__':

    # Parse command line arguments.
    if( len(sys.argv) != 2 ):
        print("usage: python3 decode_from_file.py <path-to-dataset.jiffy>")
        exit()
    jiffy_file = sys.argv[1]

    # Use jiffy to decode the scans in a .jiffy file.
    decode_from_file(jiffy_file) 
    
