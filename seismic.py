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
        self.dtype = dtype
        
        # Initialize Waveform Engine
        self.waveform_engine = None

        # Initialize some things
        self.sta_name = []
        self.lat  = np.array([],dtype='float64')
        self.lon  = np.array([],dtype='float64')
        self.x    = np.array([],dtype='float64')
        self.y    = np.array([],dtype='float64')
    
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
        X = []
        Y = []
        for l in open(station_file):
            if (l.strip()[0]=='#'):
                continue
            items = l.strip().split()
            self.sta_name.append(items[0].strip())
            X.append(float(items[1]))
            Y.append(float(items[2]))

        # Lat-Lon / UTM conversions
        if file_format=='LL':            
            self.lon = np.append(self.lon,X)
            self.lat = np.append(self.lat,Y)
            self.x, self.y = self.ll2xy(self.lon,self.lat)
        else:
            self.x = np.append(self.x,X)
            self.y = np.append(self.y,Y)
            self.lon, self.lat = self.ll2xy(self.x,self.y)

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
            if (l.strip()[0]=='#'):
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
        
    def initWaveInt(self,waveform_engine):
        '''
        Initialize Bob Hermann's wavenumber integration engine
        '''
        
        # Assign reference to waveform_engine
        self.waveform_engine = copy.deepcopy(waveform_engine)

        # Assign receiver location
        self.waveform_engine.setXr(self.sta_name,self.x,self.y)

        # All done
        return

    def calcSynthetics(self,dir_name,strike,dip,rake,M0,rise_time,stf_type='triangle',
                       rfile_name=None,out_type='D',src_loc=None,cleanup=True):
        '''
        Build Green's functions for a particular source location
        Args:
            * dir_name:  Name of the directory where synthetics will be created
            * strike:    Fault strike (in deg)
            * dip:       Fault dip (in deg)
            * rake:      Fault rake (in deg)
            * M0:        Seismic moment
            * rise_time: Rise time (in sec)
            * stf_type: 
            * src_loc:  Point source coordinates (ndarray)
            * rfile_name: pulse file name if stf_type='rfile'
        '''
        
        # Check Waveform Engine
        assert self.waveform_engine != None, 'waveform_engine must be assigned'
        if src_loc == None:
            assert self.waveform_engine.Xs != None, 'Source location must be assigned'
        else:
            self.waveform_engine.Xs = copy.deepcopy(src_loc)

        # Assign receiver locations
        assert self.waveform_engine.Xr != None, 'Recever locations must be assigned'

        # Go in dir_name
        cwd = os.getcwd()
        if cleanup and os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        else:
            os.mkdir(dir_name)
        os.chdir(dir_name)

        # Waveform simulation        
        self.waveform_engine.synthSDR(out_type,strike,dip,rake,M0,stf_type,rise_time,rfile_name)
        
        # Go back
        os.chdir(cwd)
        
        if cleanup:
            shutil.rmtree(dir_name)

        # All done
        return
    

