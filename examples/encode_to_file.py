#!/usr/bin/env python3

import os
import sys
import path
sys.path.append(path.Path(__file__).abspath().parent.parent)
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

def encode_to_file(fname, dataset_path):

    scans_per_frame = 7               # Frames contain first- and second-return range, signal, and reflectivity plus ambient infrared.
    precision = [5, 5, 1, 1, 1, 1, 1] # Use 5 millimeter precision for ranges. Don't quantize anything else.
    frames_per_group = 10             # Force a keyframe every 10 frames.

    # Open the jiffy encoding stream.    
    stream = jf.StreamWriter(fname, scans_per_frame, frames_per_group, precision=precision)

    original_size_bytes = 0
    for fnum, frame in enumerate(get_frames(dataset_path)):
        # Encode the latest frame.
        stream.encode(frame[:scans_per_frame])

        # Add the raw data size of this frame in bytes to the total.
        for scan in frame[:scans_per_frame]:
            original_size_bytes += scan.size * scan.itemsize

    # Count the encoded stream size in bytes.
    encoded_size_bytes = stream.byteStream.size

    # Flush the jiffy stream to file and close the file.
    stream.close()

    return original_size_bytes, encoded_size_bytes


if __name__=='__main__':

    # Parse command line arguments.
    if( len(sys.argv) != 2 ):
        print("usage: python3 encode_to_file.py <path-to-dataset.npz>")
        exit()
    dataset_path = sys.argv[1]
    dataset_name = os.path.basename(dataset_path).split('.')[0]

    # Use jiffy to encode the scans into a .jiffy file.
    orig_sz, encoded_sz = encode_to_file(dataset_name+'.jiffy', dataset_path) 
    print("{:0.3f} MB -> {:0.3f} MB ({:0.2f}x)".format(orig_sz/1e6, encoded_sz/1e6, orig_sz/encoded_sz))
    
