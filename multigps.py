''' 
A class that deals with multiple gps rates.

Written by R. Jolivet, May 2014.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import os
import copy
import sys

# Personals
from .gpsrates import gpsrates
from .gpstimeseries import gpstimeseries
if (sys.version_info[0]==2) and ('EDKS_HOME' in os.environ.keys()):
    from calcGreenFunctions_EDKS_subRectangles import *

class multigps(gpsrates):

    def __init__(self, name, gpsobjects=None, utmzone='10', ellps='WGS84', obs_per_station=2):
        '''
        Args:
            * name              : Name of the dataset.
            * gpsobjects        : A list of GPS rates objects.
            * utmzone           : UTM zone. (optional, default is 10 (Western US))
            * ellps             : ellipsoid (optional, default='WGS84')
            * obs_per_station   : Number of observations per stations.
        '''

        # Base class init
        super(multigps,self).__init__(name,utmzone,ellps,verbose=False) 
        
        # Set things
        self.dtype = 'multigps'
        self.obs_per_station = obs_per_station
 
        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize Multiple GPS array {}".format(self.name))

        # Initialize things
        self.vel_enu = None
        self.err_enu = None
        self.rot_enu = None
        self.synth = None

        # Set objects
        if gpsobjects is not None:
            self.setgpsrates(gpsobjects)

        # All done
        return
    
    def setgpsrates(self, gpsobjects):
        '''
        Takes list of gpsrates and build a multi gps function.
        Args:
            * gpsobjects    : List of gpsrates objects.
        '''

        # Get the list
        self.gpsobjects = gpsobjects

        # Loop over the stations to get the number of stations 
        ns = 0
        for gps in gpsobjects:
            ns += gps.station.shape[0]

        # Get the factor
        self.factor = gpsobjects[0].factor

        # Create the arrays
        self.station = np.zeros(ns).astype('|S4')
        self.lon = np.zeros(ns)
        self.lat = np.zeros(ns)
        self.x = np.zeros(ns)
        self.y = np.zeros(ns)
        self.vel_enu = np.zeros((ns,3))
        self.err_enu = np.zeros((ns,3))

        # obs per stations
        obs_per_station = []

        # Loop over the gps objects to feed self
        ns = 0
        for gps in gpsobjects:
            
            # Assert
            assert gps.factor==self.factor, 'GPS object have a different factor: Object {}'.format(gps.name)
            assert gps.utmzone==self.utmzone, 'UTM zone is not compatible: Object {}'.format(gps.utmzone)

            # Set starting and ending points
            st = ns
            ed = ns + gps.station.shape[0]

            # Get stations
            self.station[st:ed] = gps.station

            # Feed in 
            self.lon[st:ed] = gps.lon
            self.lat[st:ed] = gps.lat
            self.x[st:ed] = gps.x
            self.y[st:ed] = gps.y
            self.vel_enu[st:ed,:] = gps.vel_enu
            self.err_enu[st:ed,:] = gps.err_enu

            # Force number of observation per station
            gps.obs_per_station = self.obs_per_station

            # Update ns
            ns += gps.station.shape[0]

        # All done
        return

    def getNumberOfTransformParameters(self, transformation):
        '''
        Returns the number of transform parameters for the given transformation.
        Args:
            * transformation: List [main_transformation, [transfo_subnet1, transfo_subnet2, ....] ]
                 Each can be 'strain', 'full', 'strainnorotation', 'strainnotranslation', 'strainonly'
        '''

        # Separate 
        mainTrans = transformation[0]
        subTrans = transformation[1]

        # Assert
        assert type(mainTrans) is str, 'First Item of transformation list needs to be string'
        assert type (subTrans) is list, 'Second Item of transformation list needs to be a list'

        # Nhere
        nMain = super(multigps,self).getNumberOfTransformParameters(transformation[0])
        
        # Each subnetwork
        nSub = 0
        for trans, gps in zip(subTrans, self.gpsobjects):
            nSub += gps.getNumberOfTransformParameters(trans)

        # Sum
        Npo = nSub + nMain

        # All done
        return Npo

    def getTransformEstimator(self, transformation):
        '''
        Returns the estimator for the transform.
        Args:
            * transformation : List [main_transformation, [transfo_subnet1, transfo_subnet2, ....] ]
                    Each item can be 'strain', 'full', 'strainnorotation', 'strainnotranslation', 'strainonly'
        '''

        # Separate
        mainTrans = transformation[0]
        subTrans = transformation[1]

        # Assert
        assert type(mainTrans) is str, 'First Item of transformation list needs to be string'
        assert type (subTrans) is list, 'Second Item of transformation list needs to be a list'

        # Need the number of columns and lines
        nc = self.getNumberOfTransformParameters(transformation)
        ns = self.station.shape[0]*self.obs_per_station

        # Create the big holder
        orb = np.zeros((ns,nc))

        # Get the main transforms
        Morb = super(multigps,self).getTransformEstimator(mainTrans)
        cst = Morb.shape[1]

        # Put it where it should be
        orb[:, :cst] = Morb
        
        # Loop over the subnetworks
        lst_east = 0
        lst_north = self.station.shape[0]
        for trans, gps in zip(subTrans, self.gpsobjects):
            # Get the transform
            Sorb = gps.getTransformEstimator(trans)
            # Set the indexes right
            ced = cst + Sorb.shape[1]
            led_east = lst_east + gps.station.shape[0]
            led_north = lst_north + gps.station.shape[0]
            # Put it where it should be 
            orb[lst_east:led_east, cst:ced] = Sorb[:gps.station.shape[0],:]
            orb[lst_north:led_north, cst:ced] = Sorb[gps.station.shape[0]:,:]
            # Update column
            cst += Sorb.shape[1]
            # update lines
            lst_east += gps.station.shape[0]
            lst_north += gps.station.shape[0]

        # All done
        return orb

    def computeTransformation(self, fault):
        '''
        Computes the transformation that is stored with a particular fault.
        Stores it in transformation.
        '''

        # Get the transformation 
        transformation = fault.poly[self.name]

        # Separate
        mainTrans = transformation[0]
        subTrans = transformation[1]

        # Get the solution
        Tvec = fault.polysol[self.name]

        # Assert
        assert type(mainTrans) is str, 'First Item of transformation list needs to be string'
        assert type (subTrans) is list, 'Second Item of transformation list needs to be a list'

        # Compute the main transformation 
        super(multigps, self).computeTransformation(fault)
        st = super(multigps, self).getNumberOfTransformParameters(mainTrans)

        # Loop over the transformations
        sst = 0
        for trans, gps in zip(subTrans, self.gpsobjects):
            
            # Put the transform in the fault
            fault.poly[gps.name] = trans

            # Put the solution in the fault
            ed = st + gps.getNumberOfTransformParameters(trans)
            fault.polysol[self.name] = Tvec[st:ed]

            # Do the transform 
            gps.computeTransformation(fault)

            # Add it up
            sed = sst + gps.station.shape[0]
            self.transformation[sst:sed,:] += gps.transformation

            # Remove unwanted garbage in the fault
            del fault.poly[gps.name]
            del fault.polysol[gps.name]

        # All done
        return

#EOF
