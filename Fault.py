'''
A parent Fault class

Written by R. Jolivet, Z. Duputel and B. Riel, March 2014
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from . import triangularDisp as tdisp
from scipy.linalg import block_diag
import copy
import sys
import os

# Personals
from .SourceInv import SourceInv


class Fault(SourceInv):
    
    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(Fault,self).__init__(name, utmzone, ellps)

        # Initialize the fault
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initializing fault {}".format(self.name))

        # Specify the type of patch
        self.patchType = None

        # Set the reference point in the x,y domain (not implemented)
        self.xref = 0.0
        self.yref = 0.0

        # Allocate fault trace attributes
        self.xf   = None # original non-regularly spaced coordinates (UTM)
        self.yf   = None
        self.xi   = None # regularly spaced coordinates (UTM)
        self.yi   = None
        self.loni = None # regularly spaced coordinates (geographical)
        self.lati = None
        
        # Allocate depth attributes
        self.top = None             # Depth of the top of the fault
        self.depth = None           # Depth of the bottom of the fault
        
        # Allocate patches
        self.patch     = None
        self.slip      = None
        self.totalslip = None
        self.Cm        = None
        
        # Create a dictionnary for the polysol
        self.polysol = {}
        
        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}
        
        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None
        
        # Adjacency map for the patches
        self.adjacencyMap = None
        
        # All done
        return
        
    def duplicateFault(self):
        '''
        Returns a copy of the fault.
        '''
        
        return copy.deepcopy(self)

    def initializeslip(self, n=None):
        '''
        Re-initializes the fault slip array to zero values.
        Args:
            * n     : Number of slip values. If None, it'll take the number of patches.
        '''

        if n is None:
           n = len(self.patch)

        self.slip = np.zeros((n,3))

        # All done
        return

    def addfaults(self, filename):
        '''
        Add some other faults to plot with the modeled one.

        Args:
            * filename      : Name of the fault file (GMT lon lat format).
        '''

        # Allocate a list 
        self.addfaults = []

        # Read the file
        fin = open(filename, 'r')
        A = fin.readline()
        tmpflt=[]
        while len(A.split()) > 0:
            if A.split()[0] is '>':
                if len(tmpflt) > 0:
                    self.addfaults.append(np.array(tmpflt))
                tmpflt = []
            else:
                lon = float(A.split()[0])
                lat = float(A.split()[1])
                tmpflt.append([lon,lat])
            A = fin.readline()
        fin.close()

        # Convert to utm
        self.addfaultsxy = []
        for fault in self.addfaults:
            x,y = self.ll2xy(fault[:,0], fault[:,1])
            self.addfaultsxy.append([x,y])
        
        # All done
        return


    def trace2xy(self):
        ''' 
        Transpose the fault trace lat/lon into the UTM reference.
        '''

        # do it 
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return

    def trace2ll(self):
        ''' 
        Transpose the fault trace UTM coordinates into lat/lon.
        '''

        # do it 
        self.lon, self.lat = self.ll2xy(self.xf, self.yf)

        # All done
        return



    def trace(self, x, y, utm=False):
        ''' 
        Set the surface fault trace from Lat/Lon or UTM coordinates

        Args:
            * Lon           : Array/List containing the Lon points.
            * Lat           : Array/List containing the Lat points.
        '''

        # Set lon and lat
        if utm:
            self.xf  = np.array(x)/1000.
            self.yf  = np.array(y)/1000.
            # to lat/lon
            self.trace2ll()
        else:
            self.lon = np.array(x)
            self.lat = np.array(y)
            # utmize
            self.trace2xy()

        # All done
        return


    def file2trace(self, filename, utm=False):
        '''
        Reads the fault trace Lat/Lon directly from a text file.
        Format is:
        Lon Lat

        Args:
            * filename      : Name of the fault file.
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read the whole thing
        A = fin.readlines()

        # store these into Lon Lat
        x = []
        y = []
        for i in range(len(A)):
            x.append(np.float(A[i].split()[0]))
            y.append(np.float(A[i].split()[1]))
            
        # Create the trace 
        self.trace(x, y, utm)

        # All done
        return


    def getindex(self, p):
        '''
        Returns the index of a patch.
        '''

        # output index
        iout = None

        # Find the index of the patch
        for i in range(len(self.patch)):
            if (self.patch[i] == p).all():
                iout = i

        # All done
        return iout


    def getslip(self, p):
        '''
        Returns the slip vector for a patch.
        '''
            
        # Get patch index
        io = self.getindex(p)

        # All done
        return self.slip[io,:]


    def computeArea(self):
        '''
        Computes the area of all triangles.
        '''

        # Area
        self.area = []

        # Loop over patches
        for patch in self.patch:

            # Get vertices of patch 
            p1, p2, p3 = patch[:3]

            # Compute side lengths
            a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
            b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)
            c = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)

            # Compute area using numerically stable Heron's formula
            c,b,a = np.sort([a, b, c])
            self.area.append(0.25 * np.sqrt((a + (b + c)) * (c - (a - b)) 
                           * (c + (a - b)) * (a + (b - c))))
            
        # all done
        return


    def writeTrace2File(self, filename, ref='lonlat'):
        '''
        Writes the trace to a file.
        Args:
            * filename      : Name of the file
            * ref           : can be lonlat or utm.
        '''

        # Get values
        if ref in ('utm'):
            x = self.xf*1000.
            y = self.yf*1000.
        elif ref in ('lonlat'):
            x = self.lon
            y = self.lat

        # Open file 
        fout = open(filename, 'w')

        # Write 
        for i in range(x.shape[0]):
            fout.write('{} {} \n'.format(x[i], y[i]))

        # Close file
        fout.close()

        # All done
        return
