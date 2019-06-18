'''
A parent Pressure class

Written by T. Shreve, May 2019
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

from .SourceInv import SourceInv
from .EDKSmp import sum_layered
from .EDKSmp import dropSourcesInPatches as Patches2Sources

#class Pressure
class Pressure(SourceInv):

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
        super(Pressure,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the pressure source
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing pressure source {}".format(self.name))
        self.verbose = verbose

        self.type = "Pressure"

        # Set the reference point in the x,y domain (not implemented)
        self.xref = 0.0
        self.yref = 0.0



        # Allocate fault trace attributes
        self.xf   = None # original non-regularly spaced coordinates (UTM)
        self.yf   = None
        self.lon  = None
        self.lat  = None


        # Allocate depth attributes
        self.depth = None           # Depth of the center of the pressure source

        # Allocate patches
        self.deltavolume    = None  #If given pressure we can calculate volume and vice versa
        self.deltapressure  = None  #Dimensionless pressure
        self.ellipshape     = None
        self.volume    = None
        #self.Cm        = None  #Don't need this because no smoothing?
        self.mu        = None   #Shear modulus, should I set this here??
        self.nu        = None
        self.numz      = None   #This is... what?

        # Remove files
        self.cleanUp = True

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
    # Set up whats needed for a null pressure source
    def initializeEmptyPressure(self):
        '''
        Initializes what is required for a pressure source with no volume change

        Returns: None
        '''

        # Initialize
        self.deltavolume = 0
        self.initializeslip()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Returns a copy of the fault
    def duplicatePressure(self):
        '''
        Returns a full copy (copy.deepcopy) of the pressure object.

        Return:
            * pressure         : pressure object
        '''

        return copy.deepcopy(self)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Initialize the slip vector
    def initializeslip(self, values=None):
        '''
        Re-initializes the volume/pressure change to zero.

        Kwargs:
            * values        : Can be 'depth', 'strike', 'dip', 'length',
                              'width', 'area', 'index' or a numpy array
                              The array can be of size (n,3) or (n,1)

        Returns:
            None
        '''

        self.deltavolume = 0
        self.deltapressure = 0

        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pressure2xy(self):
        '''
        Transpose the initial pressure source position in lat/lon into the UTM reference.
        UTM coordinates are stored in self.xf and self.yf in km

        Returns:
            * None
        '''

        # do it
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pressure2ll(self):
        '''
        Transpose the initial pressure source position in UTM coordinates into lat/lon.
        Lon/Lat coordinates are stored in self.lon and self.lat in degrees

        Returns:
            * None
        '''

        # do it
        self.lon, self.lat = self.xy2ll(self.xf, self.yf)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def trace(self, x, y, utm=False):
        '''
        Set the initial pressure source position from Lat/Lon or UTM coordinates
        Surface initial pressure source position is stored in self.xf, self.yf (UTM) and
        self.lon, self.lat (Lon/lat)

        ??--Surface projection

        Args:
            * Lon           : Array/List containing the Lon points.
            * Lat           : Array/List containing the Lat points.

        Kwargs:
            * utm           : If False, considers x and y are lon/lat
                              If True, considers x and y are utm in km

        Returns:
            * None
        '''

        # Set lon and lat
        if utm:
            self.xf  = np.array(x)/1000.
            self.yf  = np.array(y)/1000.
            # to lat/lon
            self.pressure2ll()
        else:
            self.lon = np.array(x)
            self.lat = np.array(y)
            # utmize
            self.pressure2xy()

        # All done
        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def saveGFs(self, dtype='d', outputDir='.',
                      suffix={'pressure':'dP'}):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype       : Format of the binary data saved
                                'd' for double
                                'f' for float32
            * outputDir   : Directory to save binary data.
            * suffix      : suffix for GFs name (dictionary)

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for pressure source {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for each slip componenets
            for c in G.keys():
                if G[c] is not None:
                    g = G[c].flatten()
                    n = self.name.replace(' ', '_')
                    d = data.replace(' ', '_')
                    filename = '{}_{}_{}.gf'.format(n, d, suffix[c])
                    g = g.astype(dtype)
                    g.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def setGFsFromFile(self, data, pressure=None,
                                   custom=None, vertical=False, dtype='d'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * pressure    : File containing the Green's functions for
                              pressure source related displacements.
            * vertical      : Deal with the UP component (gps: default is false,
                              insar: it will be true anyway).
            * dtype         : Type of binary data.
                                    'd' for double/float64
                                    'f' for float32

        Returns:
            * None
        '''

        if self.verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Set up Green's functions for fault {}".format(self.name))
            print("and data {} from files: ".format(data.name))
            print("     pressure: {}".format(pressure))


        # Read the files and reshape the GFs
        Gdp = None
        if pressure is not None:
            Gdp = np.fromfile(pressure, dtype=dtype)
            ndl = int(Gdp.shape[0])

        # Create the big dictionary
        G = {'pressure': Gdp}

        # The dataset sets the Green's functions itself
        data.setGFsInFault(self, G, vertical=vertical)

        # If custom
        if custom is not None:
            self.setCustomGFs(data, custom)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveData(self, dtype='d', outputDir='.'):
        '''
        Saves the Data in binary files.

        Kwargs:
            * dtype       : Format of the binary data saved
                                'd' for double
                                'f' for float32
            * outputDir   : Directory to save binary data

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for pressure source {}'.format(self.name))

        # Loop over the data names in self.d
        for data in self.d.keys():

            # Get data
            D = self.d[data]

            # Write data file
            filename = '{}_{}.data'.format(self.name, data)
            D.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildGFs(self, data, vertical=True,
                 method='pressure', verbose=True):
        '''
        ??? How to determine if can be deltaP or deltaV ???
        Builds the Green's function matrix based on the pressure source.

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.
        Each of these entry is a dictionary that contains 'volume' or 'pressure'

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for
                              the vertical displacements in a gps object.
            * method        : Can be pressure
            * verbose       : Writes stuff to the screen (overwrites self.verbose)


        Returns:
            * None

        '''


        # Check something
        # if method in ('homogeneous', 'Homogeneous'):
        #     if self.patchType == 'rectangle':
        #         method = 'Okada'
        #     elif self.patchType == 'triangle':
        #         method = 'Meade'
        #     elif self.patchType == 'triangletent':
        #         method = 'Meade'

        # Print
        if verbose:
            print('Greens functions computation method: {}'.format(method))

        # Data type check
        if data.dtype == 'insar':
            if not vertical:
                if verbose:
                    print('---------------------------------')
                    print('---------------------------------')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('  You specified vertical=False   ')
                    print(' As this is quite dangerous, we  ')
                    print(' switched it directly to True... ')
                    print(' SAR data are very sensitive to  ')
                    print('     vertical displacements.     ')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('---------------------------------')
                    print('---------------------------------')
                vertical = True

        # Compute the Green's functions
        if method in ('pressure'):
            G = self.homogeneousGFs(data, vertical=vertical, verbose=verbose)

        # Separate the Green's functions for each type of data set
        data.setGFsInFault(self, G, vertical=vertical)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def homogeneousGFs(self, data, vertical=True, verbose=True):
        '''
        Builds the Green's functions for a homogeneous half-space.
        ??? How to determine if can be deltaP or deltaV ???
        ??? How to formulate with non-linear parameters??
        Yang's formulation is used (----).


        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for
                              the vertical displacements in a gps object.
            * verbose       : Writes stuff to the screen (overwrites self.verbose)

        Returns:
            * G             : Dictionary of the built Green's functions
        '''

        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building pressure source Green's functions for the data set ")
            print("{} of type {} in a homogeneous half-space".format(data.name,
                                                                     data.dtype))

        # Create the dictionary
        G = {'pressure':[]}

        # Create the matrices to hold the whole thing
        Gdp = np.zeros((3, len(data.x)))

        dp = self.pressure2dis(data, delta="pressure")                            ####Will have to initially solve for a unit pressure, then in the solver solve for the pressure linearly
        # Store them
        Gdp[:,:] = dp.T

        if verbose:
            print(' ')

        # Build the dictionary
        G = self._buildGFsdict(data, Gdp, vertical=vertical)

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setGFs(self, data, deltapressure=[None, None, None],
                           vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the pressure source structure.

        These GFs are organized in a dictionary structure in self.G
        Entries of self.G are the data set names (data.name).
            Entries of self.G[data.name] are 'deltapressure'.

        If you provide GPS GFs, those are organised with E, N and U in lines

        If you provide Optical GFs, those are organised with E and N in lines

        If you provide InSAR GFs, these need to be projected onto the
        LOS direction already.

        Args:
            * data          : Data structure

        Kwargs:
            * deltapressure    : List of matrices of the pressure source
                              Green's functions

        Returns:
            * None
        '''

        # Get the number of data per point
        if data.dtype == 'insar' or data.dtype == 'tsunami':
            data.obs_per_station = 1
        elif data.dtype in ('gps', 'multigps'):
            data.obs_per_station = 0
            # Check components
            if not np.isnan(data.vel_enu[:,0]).any():
                data.obs_per_station += 1
            if not np.isnan(data.vel_enu[:,1]).any():
                data.obs_per_station += 1
            if vertical:
                if np.isnan(data.vel_enu[:,2]).any():
                    raise ValueError('Vertical can only be true if all stations have vertical components')
                data.obs_per_station += 1
        elif data.dtype == 'opticorr':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if not synthetic:
            if data.dtype == 'insar':
                self.d[data.name] = data.vel
                vertical = True # Always true for InSAR
            elif data.dtype == 'tsunami':
                self.d[data.name] = data.d
                vertical = True
            elif data.dtype in ('gps', 'multigps'):
                if vertical:
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
                self.d[data.name]=self.d[data.name][np.isfinite(self.d[data.name])]
            elif data.dtype == 'opticorr':
                self.d[data.name] = np.hstack((data.east.T.flatten(),
                                               data.north.T.flatten()))
                if vertical:
                    self.d[data.name] = np.hstack((self.d[data.name],
                                                   np.zeros_like(data.east.T.ravel())))

        # Pressure
        if len(deltapressure) == 3:            # GPS case

            E_dp = deltapressure[0]
            N_dp = deltapressure[1]
            U_dp = deltapressure[2]
            dp = []
            nd = 0
            if (E_dp is not None) and (N_dp is not None):
                d = E_dp.shape[0]
                m = E_dp.shape[1]
                dp.append(E_dp)
                dp.append(N_dp)
                nd += 2
            if (U_dp is not None):
                d = U_dp.shape[0]
                m = U_dp.shape[1]
                dp.append(U_dp)
                nd += 1
            if nd > 0:
                dp = np.array(dp)
                dp = dp.reshape((nd*d, m))
                G['pressure'] = dp

        elif len(deltapressure) == 1:          # InSAR/Tsunami case
            ###deltapresure parametered defined in function
            Green_dp = deltapressure[0]
            print(min(Green_dp),max(Green_dp))
            if Green_dp is not None:
                G['pressure'] = Green_dp
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    #
    # def rotateGFs(self, data, azimuth):
    #     '''
    #     For the data set data, returns the rotated GFs so that dip slip motion
    #     is aligned with the azimuth. It uses the Greens functions stored
    #     in self.G[data.name].
    #
    #     Args:
    #         * data          : Name of the data set.
    #         * azimuth       : Direction in which to rotate the GFs
    #
    #     Returns:
    #         * rotatedGar    : GFs along the azimuth direction
    #         * rotatedGrp    : GFs in the direction perpendicular to the
    #                           azimuth direction
    #     '''
    #
    #     # Check if strike and dip slip GFs have been computed
    #     assert 'strikeslip' in self.G[data.name].keys(), \
    #                     "No strike slip Green's function available..."
    #     assert 'dipslip' in self.G[data.name].keys(), \
    #                     "No dip slip Green's function available..."
    #
    #     # Get the Green's functions
    #     Gss = self.G[data.name]['strikeslip']
    #     Gds = self.G[data.name]['dipslip']
    #
    #     # Do the rotation
    #     rotatedGar, rotatedGrp = self._rotatedisp(Gss, Gds, azimuth)
    #
    #     #Store it, it will be used to return the slip vector.
    #     self.azimuth = azimuth
    #
    #     # All done
    #     return rotatedGar, rotatedGrp
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
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

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Create a data vector
        d = np.zeros((Nd,))

        # Loop over the datasets
        el = 0
        for data in datas:

                # print
                if verbose:
                    print("Dealing with data {}".format(data.name))

                # Get the local d
                dlocal = self.d[data.name]
                Ndlocal = dlocal.shape[0]

                # Store it in d
                d[el:el+Ndlocal] = dlocal

                # update el
                el += Ndlocal

        # Store d in self
        self.dassembled = d
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleGFs(self, datas, polys=None, verbose=True,
                                 custom=False, computeNormFact=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        This method allows to specify which transformation is going
        to be estimated in the data sets, through the polys argument.

        Assembled Green's function matrix is stored in self.Gassembled

        Args:
            * datas             : list of data sets. If only one data set is
                                  used, can be a data instance only.

        Kwargs:
            * polys             : None -> nothing additional is estimated

                 For InSAR, Optical, GPS:
                       1 -> estimate a constant offset
                       3 -> estimate z = ax + by + c
                       4 -> estimate z = axy + bx + cy + d

                 For GPS only:
                       'full'                -> Estimates a rotation,
                                                translation and scaling
                                                (Helmert transform).
                       'strain'              -> Estimates the full strain
                                                tensor (Rotation + Translation
                                                + Internal strain)
                       'strainnorotation'    -> Estimates the strain tensor and a
                                                translation
                       'strainonly'          -> Estimates the strain tensor
                       'strainnotranslation' -> Estimates the strain tensor and a
                                                rotation
                       'translation'         -> Estimates the translation
                       'translationrotation  -> Estimates the translation and a
                                                rotation

            * custom            : If True, gets the additional Green's function
                                  from the dictionary self.G[data.name]['custom']

            * computeNormFact   : bool
                if True, compute new OrbNormalizingFactor
                if False, uses parameters in self.OrbNormalizingFactor

            * verbose           : Talk to me (overwrites self.verbose)

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
            print("Assembling G for pressure source {}".format(self.name))

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}

        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if (polys.__class__ is not str) and (polys is not None):
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    self.poly[data.name] = polys
        elif polys.__class__ is list:
            for data, poly in zip(datas, polys):
                print(poly.__class__,data.name)
                print((poly.__class__ is not str))
                if (poly.__class__ is not str) and (poly is not None) and (poly.__class__ is not list):
                    self.poly[data.name] = poly*data.obs_per_station
                    print(data.obs_per_station)
                else:
                    self.poly[data.name] = poly

        # Create the transformation holder
        if not hasattr(self, 'helmert'):
            self.helmert = {}
        if not hasattr(self, 'strain'):
            self.strain = {}
        if not hasattr(self, 'transformation'):
            self.transformation = {}

        # For now let's just try to invert for pressure
        ##########################################
        # if self.ellipshape is None:
        #     self.createShape()             #source location (x0,y0,z0), semimajor axis (a), geometric aspect ratio (A=b/a),  dip angle (theta, theta=90 is vertical), strike (phi, phi=0 aligned to N)
        Nps = 1                                        #Solve just for the deltaPressure parameter
        Npo = 0
        for data in datas :
            transformation = self.poly[data.name]
            if type(transformation) in (str, list):
                tmpNpo = data.getNumberOfTransformParameters(self.poly[data.name])
                Npo += tmpNpo
                if type(transformation) is str:
                    if transformation in ('full'):
                        self.helmert[data.name] = tmpNpo
                    elif transformation in ('strain', 'strainonly',
                                            'strainnorotation', 'strainnotranslation',
                                            'translation', 'translationrotation'):
                        self.strain[data.name] = tmpNpo
                else:
                    self.transformation[data.name] = tmpNpo
            elif transformation is not None:
                Npo += transformation
        Np = Nps + Npo

        # Save extra Parameters
        self.TransformationParameters = Npo

        # Custom?
        if custom:
            Npc = 0
            for data in datas:
                if 'custom' in self.G[data.name].keys():
                    Npc += self.G[data.name]['custom'].shape[1]
            Np += Npc
            self.NumberCustom = Npc
        else:
            Npc = 0

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        custstart = Nps # custom indices
        polstart = Nps + Npc # poly indices
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Elastic Green's functions

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))

            # Fill Glocal --- difference between Glocal and big G?
            ec = 0

            # for sp in sliplist:
            Nclocal = self.G[data.name]['pressure'].shape[0]
            Glocal[:,0] = self.G[data.name]['pressure'] #???
            #ec += Nclocal

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal
            # Custom
            if custom:
                # Check if data has custom GFs
                if 'custom' in self.G[data.name].keys():
                    nc = self.G[data.name]['custom'].shape[1] # Nb of custom param
                    custend = custstart + nc
                    G[el:el+Ndlocal,custstart:custend] = self.G[data.name]['custom']
                    custstart += nc

            # Polynomes and strain
            if self.poly[data.name] is not None:

                # Build the polynomial function
                if data.dtype in ('gps', 'multigps'):
                    orb = data.getTransformEstimator(self.poly[data.name])
                elif data.dtype in ('insar', 'opticorr'):
                    orb = data.getPolyEstimator(self.poly[data.name],computeNormFact=computeNormFact)
                elif data.dtype == 'tsunami':
                    orb = data.getRampEstimator(self.poly[data.name])

                # Number of columns
                nc = orb.shape[1]

                # Put it into G for as much observable per station we have
                polend = polstart + nc
                G[el:el+Ndlocal, polstart:polend] = orb
                polstart += nc

            # Update el to check where we are
            el = el + Ndlocal

        # Store G in self
        self.Gassembled = G

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleCd(self, datas, add_prediction=None, verbose=False):
        '''
        Assembles the data covariance matrices that have been built for each
        data structure.

        Args:
            * datas         : List of data instances or one data instance

        Kwargs:
            * add_prediction: Precentage of displacement to add to the Cd
                              diagonal to simulate a Cp (dirty version of
                              a prediction error covariance, see Duputel et
                              al 2013, GJI).
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        # Check
        if type(datas) is not list:
            datas = [datas]

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            # Fill in Cd
            if verbose:
                print("{0:s}: data vector shape {1:s}".format(data.name, self.d[data.name].shape))
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            # Add some Cp if asked
            if add_prediction is not None:
                Cd[st:se, st:se] += np.diag((self.d[data.name]*add_prediction/100.)**2)
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def builddummyCm(self, extra_params=None, lim=None,
                                  verbose=True):
        '''
        Builds a dummy model covariance matrix as the identity matrix.

        Model covariance is stored in self.Cm

        Args:

        Kwargs:
            * extra_params  : a list of extra parameters.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the dummy Cm matrix ")



        # Creates the principal Cm matrix
        Np = 1
        if extra_params is not None:
            Np += len(extra_params)
        Cm = np.eye(Np, Np)

        # Put the extra values
        st = 0
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]
        print(Cm)

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def _buildGFsdict(self, data, Gdp, vertical=True):
        '''
        Some ordering of the Gfs to make the computation routines simpler.

        Args:
            * data          : instance of data
            * Gdp           : Pressure greens functions (Yang)

        Kwargs:
            *vertical       : If true, assumes verticals are used for the GPS case

        Returns:
            * G             : Dictionary of GFs
        '''

        # Verticals?
        Ncomp = 3
        if not vertical:
            Ncomp = 2
            Gdp = Gdp[:2,:,:]
            Nparm = Gdp.shape[2]
            Npoints = Gdp.shape[1]

        # Get some size info
        Npoints = Gdp.shape[1]
        Ndata = Ncomp*Npoints

        # Check format
        if data.dtype in ['gps', 'opticorr', 'multigps']:
            # Flat arrays with e, then n, then u (optional)
            Gdp = Gdp.reshape((Ndata, Nparm))
        elif data.dtype in ('insar', 'insartimeseries'):
            # If InSAR, do the dot product with the los
            Gdp_los = []
            for i in range(Npoints):
                    Gdp_los.append(np.dot(data.los[i,:], Gdp[:,i]))
            Gdp = np.array(Gdp_los).reshape((Npoints))


            #plt.scatter(x, y, c=Gdp, s=100)
            #plt.colorbar()
            #plt.show()

        # Create the dictionary
        G = {'pressure':[]}

        # Reshape the Green's functions
        G['pressure'] = Gdp

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # def _rotatedisp(self, Gss, Gds, azimuth):
    #     '''
    #     A rotation function for Green function.
    #
    #     Args:
    #         * Gss           : Strike slip GFs
    #         * Gds           : Dip slip GFs
    #         * azimtuh       : Direction to rotate (degrees)
    #
    #     Return:
    #         * rotatedGar    : Displacements along azimuth
    #         * rotatedGrp    : Displacements perp. to azimuth direction
    #     '''
    #
    #     # Make azimuth positive
    #     if azimuth < 0.:
    #         azimuth += 360.
    #
    #     # Get strikes and dips
    #     #if self.patchType is 'triangletent':
    #     #    strike = super(self.__class__, self).getStrikes()
    #     #    dip = super(self.__class__, self).getDips()
    #     #else:
    #     strike, dip = self.getStrikes(), self.getDips()
    #
    #     # Convert angle in radians
    #     azimuth *= ((np.pi) / 180.)
    #     rotation = np.arctan2(np.tan(strike) - np.tan(azimuth),
    #                         np.cos(dip)*(1.+np.tan(azimuth)*np.tan(strike)))
    #
    #     # If azimuth within ]90, 270], change rotation
    #     if azimuth*(180./np.pi) > 90. and azimuth*(180./np.pi)<=270.:
    #         rotation += np.pi
    #
    #     # Store rotation angles
    #     self.rotation = rotation.copy()
    #
    #     # Rotate them (ar: along-rake; rp: rake-perpendicular)
    #     rotatedGar = Gss*np.cos(rotation) + Gds*np.sin(rotation)
    #     rotatedGrp = Gss*np.sin(rotation) - Gds*np.cos(rotation)
    #
    #     # All done
    #     return rotatedGar, rotatedGrp
