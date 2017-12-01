'''
A class that deals with transformations

Written by R. Jolivet, Dec 2017
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys
import os

#class transformation
class transformation(SourceInv):

    # ----------------------------------------------------------------------
    # Initialize class 
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        super(transformation,self).__init__(name,
                                            utmzone = utmzone,
                                            ellps = ellps, 
                                            lon0 = lon0, 
                                            lat0 = lat0, 
                                            verbose = verbose)
        # Initialize the class
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing transformation {}".format(self.name))

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Build the Green's functions for the transformations
    def buildGFs(self, datas, transformations, verbose=True):
        '''
        Builds the design matrix for the datasets given. 

        The GFs are stored in a dictionary. 
        Each entry of the dictionary is named after the corresponding dataset. 
        Each of these entry is a dictionary that contains the different cases of 
        transformations.

        Args:   
            * datas             : List of datasets (gps, insar, optical, ...)
            * transformations   : List of transformation types

        Kwargs:
            * verbose           : Talk to me

        Returns:
            * None

            Transformation types can be:
                    
                 For InSAR, Optical, GPS:
                       1 -> estimate a constant offset
                       3 -> estimate z = ax + by + c
                       4 -> estimate z = axy + bx + cy + d
                       'strain'              -> Estimates a strain tensor 
             
                 For GPS only:
                       'full'                -> Estimates a rotation, 
                                                translation and scaling
                                                (Helmert transform).
                       'translation'         -> Estimates a translation
                       'rotation'            -> Estimates a rotation

        '''

        # Pre compute Normalizing factors
        for data in datas:
            self.computeTransformNormFactor(data)

        # Iterate over the data
        for data, transformation in zip(datas, transformations):
            
            # Check something
            assert data.dtype in ('insar', 'gps', 'opticorr', 'tsunami'), \
                    'Unknown data type {}'.format(data.dtype)

            # Get the transformation estimator for this guy
            T = data.getTransformEstimator(transformation, computeNormFact=False)

        # All done 
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Compute the Normalizing factors
    def computeTransformNormFactor(self, data):
        '''
        Computes quantities needed to build the transformation object for 
        a dataset

        Args:
            * data          : instance of a data class
        '''

        # Calculate
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)
        normX = np.abs(data.x - x0).max()
        normY = np.abs(data.y - y0).max()
        base_max = np.max([np.abs(base_x).max(), np.abs(base_y).max()])

        # Set in place
        data.TransformNormalizingFactor['x'] = normX
        data.TransformNormalizingFactor['y'] = normY
        data.TransformNormalizingFactor['ref'] = [x0, y0]
        data.TransformNormalizingFactor['base'] = base_max

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Assemble the data vector
    def assembled(self, datas, verbose=True):
        '''
        Assembles a data vector for inversion using the list datas
        Assembled vector is stored in self.dassembled

        Args:
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Check 
        if type(datas) is not list:
            datas = [datas]

        if verbose:
            # print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling d vector")

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Assemble the Green's functions
    def assembleGFs(self, datas, verbose=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        Assembled Greens' functions are stored in self.Gassembled

        Special case: If 'strain' is in self.transformations, this parameter will 
        be placed as first and will be common to all data sets (i.e. there is
        only one strain tensor for a region, although there can be multiple 
        translation, rotations, etc for individual networks)

        Args:   
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Check 
        if type(datas) is not list:
            datas = [datas]

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print("Assembling G for transformation {}".format(self.name))

        # All done
        return


#EOF
