#!/usr/bin/env python3

import os
import sys
import path
import jiffyCodec as jf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataset_reader import DatasetReader

def decode_from_file(jiffy_file):
    stream = jf.StreamReader(byteStream=jiffy_file)

    header = stream.readHeader()
    print("Header")
    print("--------------------------------------")
    print(f"  magic:          {header['magic']}")
    print(f"  version:        {header['version']//256}.{header['version']%256}")
    print(f"  shape:          {header['shape']}")
    print(f"  scansPerFrame:  {header['scansPerFrame']}")
    print(f"  framesPerGroup: {header['framesPerGroup']}")
    print(f"  scanPrecisions: {header['framePrecisions']}")

    for frame in stream.decode():
        yield frame
    stream.close()

class ScanVisParams:
    def __init__(self, title, cmap, gamma):
        self.title = title
        self.cmap = cmap
        self.gamma = gamma

if __name__=='__main__':

    # Parse command line arguments.
    if( len(sys.argv) != 2 ):
        print("usage: python3 decode_from_file.py <path-to-dataset.jiffy>")
        exit()

    jiffy_file = sys.argv[1]

    scanvis = [None]*7
    scanvis[0] = ScanVisParams(title='Range',         cmap=cm.magma,     gamma=2.2)
    scanvis[1] = ScanVisParams(title='Range2',        cmap=cm.magma,     gamma=2.2)
    scanvis[2] = ScanVisParams(title='Signal',        cmap=cm.gist_heat, gamma=2.2)
    scanvis[3] = ScanVisParams(title='Signal2',       cmap=cm.gist_heat, gamma=2.2)
    scanvis[4] = ScanVisParams(title='Reflectivity',  cmap=cm.jet,       gamma=2.2)
    scanvis[5] = ScanVisParams(title='Reflectivity2', cmap=cm.jet,       gamma=2.2)
    scanvis[6] = ScanVisParams(title='NearIR',        cmap=cm.bone,      gamma=8)

    # Use jiffy to decode the scans in a .jiffy file.
    for frame_num, frame in enumerate(decode_from_file(jiffy_file)):
        plt.subplots(len(frame), 1, sharex=True)
        for scan_num, scan in enumerate(frame):

            plt.subplot(len(frame), 1, scan_num+1)

            sv = scanvis[scan_num]

            plt.ylabel(sv.title, rotation=0)

            scan = scan ** (1.0/sv.gamma)

            plt.imshow(scan, cmap=sv.cmap)

        plt.suptitle(f"Decoded Frame {frame_num}")
        plt.show()
    
