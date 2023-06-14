import os
import os.path as osp
import numpy as np
import dataclasses

@dataclasses.dataclass
class SensorInfo:
    precision : float = 0.0
    minRange  : float = 0.0
    maxRange  : float = 0.0
    scanShape : tuple = (0, 0)
    beam_altitude_angles : list = dataclasses.field(default_factory=list,repr=False)
    beam_azimuth_angles  : list = dataclasses.field(default_factory=list,repr=False)

class DatasetReader:
    def __init__(self, path):
        self.dataset  = np.load(path)
        self.path     = path

        # Dataset Information
        self.numScans = len(self.timestamps)
        self.elapsedTime = self.stopTime-self.startTime

        # Scan Information
        self.scanRows = len(self.dataset['beam_altitude_angles'])
        self.scanCols = len(self.dataset['xyz/00000']) // self.scanRows
        self.scanShape = (self.scanRows, self.scanCols)

        # Sensor Information
        self.sensorInfo = SensorInfo()
        self.sensorInfo.precision = 1
        self.sensorInfo.scanShape = (self.scanRows, self.scanCols)
        self.sensorInfo.minRange = self.dataset['range_limits'][0]
        self.sensorInfo.maxRange = self.dataset['range_limits'][1]
        self.sensorInfo.beam_altitude_angles = self.beamAltitudeAngles
        self.sensorInfo.beam_azimuth_angles  = self.beamAzimuthAngles

    @property
    def name(self):
        return osp.splitext(osp.basename(self.path))[0]

    @property
    def timestamps(self):
        return self.dataset['timestamp']

    @property
    def scanInterval(self):
        return np.median(np.diff(self.timestamps))

    @property
    def startTime(self):
        return self.timestamps[0]

    @property
    def stopTime(self):
        return self.timestamps[-1] + self.scanInterval

    @property
    def beamAltitudeAngles(self):
        return np.array(np.pi/180.0 * self.dataset['beam_altitude_angles'])

    @property
    def beamAzimuthAngles(self):
        '''This is hardcoded for now.'''
        start = 0; stop = 2*np.pi
        return np.linspace(start, stop, num=self.scanCols, endpoint=True)

    def _get(self, scanType, count=-1, reshape=True):
        if count < 0:
            count = self.numScans
        for i in range(0, count):
            try:
                scan = self.dataset[f'{scanType}/{i:05d}']
            except:
                print(f'WARNING: {scanType} not included in {self.name}!')
                return []

            # Typically you want to reshape the linear scan array from the dataset into a 2d scan.
            # Sometimes you don't want to do that because the scan is really a list of xyz points.
            if reshape:
                yield scan.reshape(self.scanShape)
            else:
                yield scan

    def getXYZ(self, count=-1):
        return self._get('xyz', count, reshape=False)

    def getXYZ2(self, count=-1):
        for scan in self._get('xyz2', count, reshape=False):
            yield scan / 1000.0

    def getRange(self, count=-1):
        for xyz in self.getXYZ(count):
            scan = np.linalg.norm(xyz, axis=1).reshape(self.scanShape)
            #scan[scan < self.sensorInfo.minRange] = 0.0
            #scan[scan > self.sensorInfo.maxRange] = 0.0
            yield 1000.0 * scan

    def getRange2(self, count=-1):
        for xyz in self.getXYZ2(count):
            scan = np.linalg.norm(xyz, axis=1).reshape(self.scanShape)
            #scan[scan < self.sensorInfo.minRange] = 0.0
            #scan[scan > self.sensorInfo.maxRange] = 0.0
            yield 1000.0*scan

    def getSignal(self, count=-1):
        return self._get('signal', count)

    def getSignal2(self, count=-1):
        return self._get('signal2', count)

    def getReflectivity(self, count=-1):
        return self._get('reflectivity', count)

    def getReflectivity2(self, count=-1):
        return self._get('reflectivity2', count)

    def getNearIR(self, count=-1):
        return self._get('nearIR', count)

    def getTimestamp(self, count=-1):
        if count < 0:
            count = self.scanCount
        for i in range(count):
            yield self.timestamps[i]

    def getScans(self, scanType, count=-1):
        scanType = scanType.lower()

        if scanType == 'range' or scanType == 'ranges':
            return self.getRange(count)
        if scanType == 'range2' or scanType == 'ranges2':
            return self.getRange2(count)
        if scanType == 'signal':
            return self.getSignal(count)
        if scanType == 'signal2':
            return self.getSignal2(count)
        if scanType == 'reflectivity':
            return self.getReflectivity(count)
        if scanType == 'reflectivity2':
            return self.getReflectivity2(count)
        if scanType == 'nearir':
            return self.getNearIR(count)
        raise Exception(f"DatasetReader.get received request for unrecognized scantype '{scanType}'.")
