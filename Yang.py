'''
A Yang sub-class

Written by T. Shreve, June 2019
'''

# Import Externals stuff
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from . import triangularDisp as tdisp
from scipy.linalg import block_diag
import scipy.spatial.distance as scidis
import copy
import sys
import os
from argparse import Namespace

# Personals
from . import yangfull
from .Pressure import Pressure
from .EDKSmp import sum_layered
from .EDKSmp import dropSourcesInPatches as Patches2Sources

#sub-class Yang
class Yang(Pressure):

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Parent class implementing what is common in all pressure objects.

        Args:
            * name          : Name of the pressure source.
            * utmzone       : UTM zone  (optional, default=None)
            * ellps         : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(Yang,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=True)

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert pressure change to volume change for Yang
    def pressure2volume(self):
        '''
        Converts pressure change to volume change (m3) for Yang.

        Uses empirical formulation from:
        Battaglia, Maurizio, Cervelli, P.F., and Murray, J.R., 2013, Modeling crustal deformation near active faults and volcanic centers

        Rigorous formulation:
        deltaV = ((1-2v)/(2*(1+v)))*V*(deltaP/mu)*((p^T/deltaP)-3),
        where V is the volume of the ellipsoidal cavity and p^T is the trace of the stress inside the ellipsoidal cavity.

        Empirical formulation:
        deltaV = V*(deltaP/mu)((A^2/3)-0.7A+1.37)

        Returns:
            * deltavolume             : Volume change
        '''
        #Check if deltavolume already defined
        if self.deltavolume is None:
            self.volume = self.computeVolume()
            A = self.ellipshape['A']
            self.deltavolume = (self.volume)*(self.deltapressure/self.mu)*(((A**2)/3.)-(0.7*A)+1.37)

        # All done
        return self.deltavolume

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert volume change to pressure change for Yang
    def volume2pressure(self):
        '''
        Converts volume change (m3) to pressure change for Yang.

        Uses empirical formulation from:
        Battaglia, Maurizio, Cervelli, P.F., and Murray, J.R., 2013, Modeling crustal deformation near active faults and volcanic centers

        Empirical formulation:
        deltaP = (deltaV/V)*(mu/((A^2/3)-0.7A+1.37))

        Returns:
            * deltapressure             : Pressure change
        '''
        #Check if deltapressure already defined
        if self.deltapressure is None:
            self.volume = self.computeVolume()
            A = self.ellipshape['A']
            self.deltapressure = (self.deltavolume/self.volume)*(self.mu/(((A**2)/3.)-(0.7*A)+1.37))

        # All done
        return self.deltapressure
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Find volume of ellipsoidal cavity
    def computeVolume(self):
        '''
        Computes volume (m3) of ellipsoidal cavity, given the semimajor axis.

        Returns:
            None
        '''
        if self.volume is None:
            a = self.ellipshape['a']
            A = self.ellipshape['A']
            self.volume = (4./3.)*np.pi*a*((a*A)**2)             #Volume of ellipsoid = 4/3*pi*a*b*c

        # All done
        return self.volume


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Some building routines that can be touched... I guess
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def pressure2dis(self, data, delta="pressure"):
        '''
        Computes the surface displacement at the data location using yang. ~~~ This is where the good stuff happens ~~

        Args:
            * data          : data object from gps or insar.
            * delta      : if pressure, unit pressure is assumed. If volume, ... is this necessary?
        '''
        # Set the shear modulus and poisson's ratio
        if self.mu is None:
            self.mu        = 30e9
        if self.nu is None:
            self.nu        = 0.25

        # Set the pressure value
        if delta is "pressure":
            print("testing")
            self.deltapressure = self.mu #so that DP/self.mu = 1.0
            DP = self.deltapressure                              #Dimensionless unit pressure
        elif delta is "volume":
            self.deltapressure = self.volume2pressure()
            DP = self.deltapressure                                       ##correct ???

        # Get parameters ???
        if self.ellipshape is None:
            raise Exception("Error: Need to define shape of spheroid (run self.createShape)")
        #define dictionary entries as variables
        ellipse = Namespace(**self.ellipshape)

        # Get data position -- in m
        x = data.x*1000
        y = data.y*1000
        z = np.zeros(x.shape)   # Data are the surface
        # Run it for yang pressure source
        # ??? Do we really need to divide by shear modulus to get dimensionless pressure...
        if (DP!=0.0):
            ####x0, y0 need to be in utm and meters
            Ux,Uy,Uz = yangfull.displacement(x, y, z, ellipse.x0m*1000, ellipse.y0m*1000, ellipse.z0, ellipse.a, ellipse.A, ellipse.dip, ellipse.strike, DP/self.mu, self.nu)
        else:
            dp_dis = np.zeros((len(x), 3))
        # All done
        # concatenate into one array
        u = np.vstack([Ux, Uy, Uz]).T

        return u #, x, y

    # ----------------------------------------------------------------------
    # How to define the shape of the ellipse??? This is a placeholder until non-linear inversion implemented.
    #
    # ----------------------------------------------------------------------

    def createShape(self, x0, y0, z0, a, A, dip, strike,latlon=True):
        #Should this be done in pressure class?
        if self.mu is None:
            self.mu        = 30e9
        if self.nu is None:
            self.nu        = 0.25
        if self.ellipshape is None:
            if latlon is True:
                self.lon, self.lat = x0, y0
                self.pressure2xy()
                xf, yf = self.xf, self.yf
            else:
                self.xf, self.yf = x0, y0
        if A == 1:
            raise Exception('If semi-minor and semi-major axes are equal, use Mogi.py')
        ###NEED TO DOUBLE CHECK THIS
        if float(z0) < (float(A)*float(a))**2/float(a):
            raise Exception('radius of curvature has to be less than the depth...')
        self.ellipshape = {'x0': x0,'y0': y0,'z0': z0,'a': a,'A': A,'dip': dip,'strike': strike}                   #Or as array?

        return x0, y0, z0, a, A, dip, strike

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def writePressure2File(self, filename, add_pressure=None, scale=1.0,
                              stdh5=None, decim=1):
            '''
            Writes the pressure parameters in a file. Trying to make this as generic as possible.
            Args:
                * filename      : Name of the file.
                * add_pressure  : Put the pressure as a value for the color. Can be None, pressure or volume.
                * scale         : Multiply the pressure change by a factor.
            '''
            # Write something
            if self.verbose:
                print('Writing pressure source to file {}'.format(filename))

            # Open the file
            fout = open(filename, 'w')

            # If an h5 file is specified, open it
            if stdh5 is not None:
                import h5py
                h5fid = h5py.File(stdh5, 'r')
                samples = h5fid['samples'].value[::decim,:]

            # Select the string for the color
            string = '  '
            if add_pressure is not None:
                if stdh5 is not None:
                    slp = np.std(samples[:])
                elif add_pressure is "pressure":
                    slp = self.deltapressure*scale
                elif add_pressure is "volume":
                    slp = self.deltavolume*scale
                # Make string
                string = '-Z{}'.format(slp)

                # Put the parameter number in the file as well if it exists --what is this???
            parameter = ' '
            if hasattr(self,'index_parameter'):
                i = np.int(self.index_parameter[0])
                j = np.int(self.index_parameter[1])
                k = np.int(self.index_parameter[2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            slipstring = ' # {} '.format(slp)

            # Write the string to file
            fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Save the shape parameters we created
            fout.write(' # x0 y0 -z0 \n {} {} {} \n # a b c \n {} {} {} \n # strike dip \n {} {} -999999 \n'.format(self.ellipshape['x0'], self.ellipshape['y0'],float(self.ellipshape['z0']),float(self.ellipshape['a']),self.ellipshape['a']*self.ellipshape['A'],self.ellipshape['a']*self.ellipshape['A'],float(self.ellipshape['strike']),float(self.ellipshape['dip'])))


            # Close th file
            fout.close()

            # Close h5 file if it is open
            if stdh5 is not None:
                h5fid.close()

            # All done
            return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def readPressureFromFile(self, filename, Cm=None, inputCoordinates='lonlat', donotreadslip=False):
        '''
        Read the pressure source parameters from a GMT formatted file.
        Args:
            * filename  : Name of the file.

        '''

        # create the lists
        self.ellipshape = []
        self.Cm   = []
        if not donotreadslip:
            Slip = []

        # open the files
        fin = open(filename, 'r')

        # Assign posterior covariances
        if Cm!=None: # Slip
            self.Cm = np.array(Cm)

        # read all the lines
        A = fin.readlines()

        # depth
        D = 0.0

        # Loop over the file
        # Assert it works
        assert A[0].split()[0] is '>', 'Reformat your file...'
        # Get the slip value
        if not donotreadslip:
            if len(A[0].split())>3:
                slip = np.array([np.float(A[0].split()[3])])
                print("read from file, pressure is ", slip)
            else:
                slip = np.array([0.0])
            Slip.append(slip)
        # get the values
        if inputCoordinates in ('lonlat'):
            lon1, lat1, z1 = A[2].split()
            a, b, c = A[4].split()
            strike, dip, tmp = A[6].split()
            # Pass as floating point
            lon1 = float(lon1); lat1 = float(lat1); z1 = float(z1); a = float(a); A1 = float(b)/float(a); dip = float(dip); strike = float(strike)
            # translate to utm
            x1, y1 = self.ll2xy(lon1, lat1)
        elif inputCoordinates in ('xyz'):
            x1, y1, z1 = A[2].split()
            a, b, c = A[4].split()
            strike, dip, tmp = A[6].split()
            # Pass as floating point
            x1 = float(x1); y1 = float(y1); z1 = float(z1); a = float(a); A1 = float(b)/float(a); dip = float(dip); strike = float(strike)
            # translate to lat and lon
            lon1, lat1 = self.xy2ll(x1, y1)
        del tmp
        # Should not necessary, but including just in case someone wants to manually change the parameter file
        if float(b) != float(c):
            raise Exception('semi-minor axes (b and c) must be equal to proceed with Yang')
        ###Immediately run mogi instead?
        if float(a) == float(b):
            raise Exception('if semi-minor and semi-major axes are equal, proceed with Mogi.py')
        if float(a) < float(b):
            raise Exception('semi-minor axis is larger than semi-major axis, need to switch values')
        ###NEED TO DOUBLE CHECK THIS
        if float(z1) < float(b)**2/float(a):
            raise Exception('radius of curvature has to be less than the depth...')
        # Depth
        mm = [float(z1)]
        if D<mm:
            D=mm
        # Set parameters
        self.ellipshape = {'x0': lon1, 'x0m': x1, 'y0': lat1,'y0m': y1,'z0': z1,'a': a,'A': A1,'dip': dip,'strike': strike}                   #Or as array?

        # Close the file
        fin.close()

        # depth
        self.depth = D

        # All done
        return
