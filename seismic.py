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
#xfrom WaveMod    import sac
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
        self.sta_name = []
        self.sac      = []
        self.sta_lat  = np.array([],dtype='float64')
        self.sta_lon  = np.array([],dtype='float64')
        self.sta_x    = np.array([],dtype='float64')
        self.sta_y    = np.array([],dtype='float64')
    
        # All done
        return
    
    def readStat(self,station_file,file_format='LL'):
        '''
        Read station file and populate the Xr attribute (station coordinates)
        Args:
            * station_file: station filename including station coordinates
            * file_format:  station file format (default= 'LL')
        file format:
        STNAME  X_COORD Y_COORD (if file_format='XY')
        STNAME  LON LAT (if file_format='LL')
        '''
        
        # Assert if station file exists
        assert os.path.exists(station_file), 'Cannot read %s (no such file)'%(station_file)

        # Assert file format
        assert file_format=='LL' or file_format=='XY', 'file_format can be either LL or XY'
        
        # Read the file 
        X = []; Y = []
	for l in open(station_file):
            if l.strip()[0]=='#':
                continue
            items = l.strip().split()
            self.sta_name.append(items[0].strip())
            X.append(float(items[1]))
            Y.append(float(items[2]))

        # Lat-Lon / UTM conversions
        if file_format=='LL':            
            self.sta_lon = np.append(self.sta_lon,X)
            self.sta_lat = np.append(self.sta_lat,Y)
            self.x, self.y = self.ll2xy(self.sta_lon,self.sta_lat)
        else:
            self.sta_x = np.append(self.sta_x,X)
            self.sta_y = np.append(self.sta_y,Y)
            self.lon, self.lat = self.ll2xy(self.sta_x,self.sta_y)

        # All done
        return    

    def writeModelFile(self,Vp,Vs,Rho,H):
        '''
        Create model file from input Vp, Vs and thickness (H)
        '''
        
        # Check Waveform Engine
        assert self.WaveformEngine != None, 'WaveformEngine must be assigned'

        # Create file
        self.WaveformEngine.writeModelFile(Vp,Vs,Rho,H)

        # All done
        return

    def readStatLL(self,station_file):
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
	for l in open(station_file):
            if l.strip()[0]=='#':
                continue
            items = l.strip().split()
            self.stat.append(items[0].strip())
            self.lat.append(float(items[1]))
            self.lon.append(float(items[2]))
        self.lat = np.array(self.x,dtype='float64')
        self.lon = np.array(self.y,dtype='float64')
        self.x,self.y = self.xy2ll(self.lon,self.lat)
        
        # All done
        return    

    def calcSynthetics(self,dir_name,strike,dip,rake,M0,r_time,stf_type='triangle',
                 out_type='D',src_loc=None,cleanup=True):
        '''
        Build Green's functions for a particular source location
        Args:
            * dir_name: Name of the directory where synthetics will be created
            * strike:   Fault strike
            * dip:      Fault dip
            * rake:     Fault rake
            * M0:       Seismic moment
            * r_time:   Rise time
            * stf_type: 
            * src_loc:  Point source coordinates (ndarray)
        '''
        
        # Check Waveform Engine
        assert self.WaveformEngine != None, 'WaveformEngine must be assigned'
        if src_loc == None:
            assert self.WaveformEngine.Xs != None, 'Source location must be assigned'
        else:
            self.WaveformEngine.Xs = copy.deepcopy(src_loc)

        if src_loc == None:
            assert self.WaveformEngine.Xr != None, 'Recever locations must be assigned'
        else:
            self.WaveformEngine.Xr = copy.deepcopy(sta_loc)

        # Go in dir_name
        cwd = os.getcwd()
        if cleanup and os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        else:
            os.mkdir(dir_name)
        os.chdir(dir_name)

        # Waveform simulation
        self.WaveformEngine.synthSDR(out_type,strike,dip,rake,M0,stf_type,r_time)
        
        # Go back
        os.chdir(cwd)

        # All done
        return
