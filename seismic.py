''' 
A class that deals with seismic or high-rate GPS data (not finished)

Written by Z. Duputel, April 2013.
'''

# Externals
import os
import copy
import shutil
import numpy  as np
import pyproj as pp
import matplotlib.pyplot as plt


# Personals
from WaveMod    import sac
from .SourceInv import SourceInv

class seismic(SourceInv):
    
    def __init__(self,name,dtype='seismic',utmzone=None,ellps='WGS84'):
        '''
        Args:
            * name      : Name of the dataset.
            * dtype     : data type (optional, default='seismic')
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''
        
        super(self.__class__,self).__init__(name,utmzone,ellps) 

        # Initialize the data set 
        self.dtype = 'seismic'
        
        # Initialize Waveform Engine
        self.WaveformEngine = None

        # Initialize some things
        self.station = []
        self.sac     = []
        self.lat     = []
        self.lon     = []
        self.x       = []
        self.y       = []
    
        # All done
        return
    
    def readStatXY(self,station_file):
        '''
        Read station file and populate the Xr attribute (station coordinates)
        Args:
            * station_file:   station filename including station coordinates
        file format:
        STNAME  X_COORD Y_COORD
        '''
        
        # Assert if station file exists
        assert os.path.exists(station_file), 'Cannot read %s (no such file)'%(station_file)
        
        # Read the file and fill-up Xr
        self.station = []; 
	for l in open(station_file):
            if l.strip()[0]=='#':
                continue
            items = l.strip().split()
            self.stat.append(items[0].strip())
            self.x.append(float(items[1]))
            self.y.append(float(items[2]))
        self.x = np.array(self.x,dtype='float64')
        self.y = np.array(self.y,dtype='float64')
        
        # All done
        return    

    def buildGFs(self,dir_name,src_loc,out_type,strike,dip,rake,M0,stf_type,r_time,cleanup=True):
        '''
        Build Green's functions for a particular source location
        Args:
            * src_loc:  Point source coordinates (ndarray)
        '''
        
        # Go in dir_name
        cwd = os.getcwd()
        if cleanup and os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        else:
            os.mkdir(dir_name)
        os.chdir(dir_name)

        # Waveform simulation
        self.WaveformEngine.Xs = copy.deepcopy(src_loc)
        self.WaveformEngine.synthSDR(out_type,strike,dip,rake,M0,stf_type,r_time)
        
        # Go back
        os.chdir(cwd)

        # All done
        return
