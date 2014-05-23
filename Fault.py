'''
A parent Fault class

Written by Z. Duputel, R. Jolivet, and B. Riel, March 2014
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

    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(Fault,self).__init__(name, utmzone, ellps)

        # Initialize the fault
        if verbose:
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
        self.N_slip    = None # This will be the number of slip values
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
           self.N_slip = len(self.patch)
        else:
            self.N_slip = n

        self.slip = np.zeros((self.N_slip,3))

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


    def setTrace(self,delta_depth=0.):
        '''
        Set Trace from patches (assuming positive depth)
        Arg:
            * delta_depth: The trace is made of all patch vertices at a depth smaller
                           than fault_top+trace_delta_depth
        '''
        self.xf = []
        self.yf = []
        minz = np.round(self.top+delta_depth,1)
        for p in self.patch:
            for v in p:
                if np.round(v[2],1)>=minz:
                    continue
                self.xf.append(v[0])
                self.yf.append(v[1])
        self.xf = np.array(self.xf)
        self.yf = np.array(self.yf)
        i = np.argsort(self.yf)
        self.xf = self.xf[i]
        self.yf = self.yf[i]

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


    def saveGFs(self, dtype='d', outputDir='.', suffix={'strikeslip':'SS','dipslip':'DS','tensile':'TS'}):
        '''
        Saves the Green's functions in different files
        Args:
            dtype       : Format of the binary data saved.
            outputDir   : Directory to save binary data.
            suffix      : suffix for GFs name
        '''

        # Print stuff
        print('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for each slip componenets
            for c in G.keys():
                g = G[c].flatten()
                filename = '{}_{}_{}.gf'.format(self.name, data, suffix[c])
                g = g.astype(dtype)
                g.tofile(os.path.join(outputDir, filename))

       # All done
        return


    def saveData(self, dtype='d', outputDir='.'):
        '''
        Saves the Data in binary files
        Args:
            * dtype       : Format of the binary data saved.
            * outputDir   : Directory to save binary data.
        '''

        # Print stuff
        print('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the data names in self.d
        for data in self.d.keys():

            # Get data
            D = self.d[data]

            # Write data file
            filename = '{}_{}.data'.format(self.name, data)
            D.tofile(os.path.join(outputDir, filename))

        # All done
        return


    def setGFsFromFile(self, data, strikeslip=None, dipslip=None, tensile=None,
                       vertical=False, dtype='d'):
        '''
        Sets the Green's functions from binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already)
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : File containing the Green's functions for strikeslip displacements.
            * dipslip       : File containing the Green's functions for dipslip displacements.
            * tensile       : File containing the Green's functions for tensile displacements.
            * vertical      : Deal with the UP component (gps: default is false,
                              insar: it will be true anyway).
            * dtype         : Type of binary data.
        '''

        print('---------------------------------')
        print('---------------------------------')
        print("Set up Green's functions for fault {} from files {}, {} and {}".format(self.name, strikeslip, dipslip, tensile))

        # Get the number of patches
        if self.N_slip == None:
            self.N_slip = len(self.patch)

        # Read the files and reshape the GFs
        Gss = None; Gds = None; Gts = None
        if strikeslip is not None:
            Gss = np.fromfile(strikeslip, dtype=dtype)
            ndl = int(Gss.shape[0]/self.N_slip)
            Gss = Gss.reshape((ndl, self.N_slip))
        if dipslip is not None:
            Gds = np.fromfile(dipslip, dtype=dtype)
            ndl = int(Gds.shape[0]/self.N_slip)
            Gds = Gds.reshape((ndl, self.N_slip))
        if tensile is not None:
            Gts = np.fromfile(tensile, dtype=dtype)
            ndl = int(Gts.shape[0]/self.N_slip)
            Gts = Gts.reshape((ndl, self.N_slip))

        # Get the data type
        datatype = data.dtype

        # Cut the Matrices following what data do we have and set the GFs
        if datatype is 'gpsrates':

            # Initialize
            GssE = None; GdsE = None; GtsE = None
            GssN = None; GdsN = None; GtsN = None
            GssU = None; GdsU = None; GtsU = None

            # Check components
            east  = False
            north = False
            if not np.isnan(data.vel_enu[:,0]).any():
                east = True
            if not np.isnan(data.vel_enu[:,1]).any():
                north = True
            if vertical and np.isnan(data.vel_enu[:,2]).any():
                raise ValueError('vertical can only be true if all stations have vertical components')

            # Get the values
            if strikeslip is not None:
                N = 0
                if east:
                    GssE = Gss[range(0,data.vel_enu.shape[0]),:]
                    N += data.vel_enu.shape[0]
                if north:
                    GssN = Gss[range(N,N+data.vel_enu.shape[0]),:]
                    N += data.vel_enu.shape[0]
                if vertical:
                    GssU = Gss[range(N,N+data.vel_enu.shape[0]),:]
            if dipslip is not None:
                N = 0
                if east:
                    GdsE = Gds[range(0,data.vel_enu.shape[0]),:]
                    N += data.vel_enu.shape[0]
                if north:
                    GdsN = Gds[range(N,N+data.vel_enu.shape[0]),:]
                    N += data.vel_enu.shape[0]
                if vertical:
                    GdsU = Gds[range(N,N+data.vel_enu.shape[0]),:]
            if tensile is not None:
                if east:
                    GtsE = Gts[range(0,data.vel_enu.shape[0]),:]
                    N += data.vel_enu.shape[0]
                if north:
                    GtsN = Gts[range(N,N+data.vel_enu.shape[0]),:]
                    N += data.vel_enu.shape[0]
                if vertical:
                    GtsU = Gts[range(N,N+data.vel_enu.shape[0]),:]

            # set the GFs
            self.setGFs(data, strikeslip=[GssE, GssN, GssU], dipslip=[GdsE, GdsN, GdsU],
                        tensile=[GtsE, GtsN, GtsU], vertical=vertical)

        elif datatype is 'insarrates':

            # Initialize
            GssLOS = None; GdsLOS = None; GtsLOS = None

            # Get the values
            if strikeslip is not None:
                GssLOS = Gss
            if dipslip is not None:
                GdsLOS = Gds
            if tensile is not None:
                GtsLOS = Gts

            # set the GFs
            self.setGFs(data, strikeslip=[GssLOS], dipslip=[GdsLOS], tensile=[GtsLOS],
                        vertical=True)

        elif datatype is 'cosicorrrates':
            # Don't need to do anything special here. The GF arrays we read in are
            # already in the right shape.
            self.setGFs(data, strikeslip=[Gss], dipslip=[Gds], tensile=[Gts], vertical=vertical)

        # all done
        return


    def setGFs(self, data, strikeslip=[None, None, None], dipslip=[None, None, None],
               tensile=[None, None, None], vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the fault structure.
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : List of matrices of the Strikeslip Green's functions, ordered E, N, U
            * dipslip       : List of matrices of the dipslip Green's functions, ordered E, N, U
            * tensile       : List of matrices of the tensile Green's functions, ordered E, N, U
            If you provide InSAR GFs, these need to be projected onto the LOS direction already.
        '''

        # Get the number of data per point
        if data.dtype == 'insarrates' or data.dtype == 'tsunami':
            data.obs_per_station = 1
        elif data.dtype is 'gpsrates':
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

        elif data.dtype is 'cosicorrrates':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if not synthetic:
            if data.dtype is 'insarrates':
                self.d[data.name] = data.vel
                vertical = True # Always true for InSAR
            elif data.dtype is 'tsunami':
                self.d[data.name] = data.d
                vertical = True
            elif data.dtype is 'gpsrates':
                if vertical:
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
                self.d[data.name]=self.d[data.name][-np.isnan(self.d[data.name])]
            elif data.dtype is 'cosicorrrates':
                self.d[data.name] = np.hstack((data.east.T.flatten(),
                                               data.north.T.flatten()))
                if vertical:
                    self.d[data.name] = np.hstack((self.d[data.name],
                                                   np.zeros_like(data.east.T.ravel())))

        # StrikeSlip
        if len(strikeslip) == 3:            # GPS case

            E_ss = strikeslip[0]
            N_ss = strikeslip[1]
            U_ss = strikeslip[2]
            ss = []
            nd = 0
            if (E_ss is not None) and (N_ss is not None):
                d = E_ss.shape[0]
                m = E_ss.shape[1]
                ss.append(E_ss)
                ss.append(N_ss)
                nd += 2
            if (U_ss is not None):
                d = U_ss.shape[0]
                m = U_ss.shape[1]
                ss.append(U_ss)
                nd += 1
            if nd > 0:
                ss = np.array(ss)
                ss = ss.reshape((nd*d, m))
                G['strikeslip'] = ss

        elif len(strikeslip) == 1:          # InSAR/Tsunami case

            Green_ss = strikeslip[0]
            if Green_ss is not None:
                G['strikeslip'] = Green_ss

        # DipSlip
        if len(dipslip) == 3:               # GPS case
            E_ds = dipslip[0]
            N_ds = dipslip[1]
            U_ds = dipslip[2]
            ds = []
            nd = 0
            if (E_ds is not None) and (N_ds is not None):
                d = E_ds.shape[0]
                m = E_ds.shape[1]
                ds.append(E_ds)
                ds.append(N_ds)
                nd += 2
            if (U_ds is not None):
                d = U_ds.shape[0]
                m = U_ds.shape[1]
                ds.append(U_ds)
                nd += 1
            if nd > 0:
                ds = np.array(ds)
                ds = ds.reshape((nd*d, m))
                G['dipslip'] = ds

        elif len(dipslip) == 1:             # InSAR/Tsunami case

            Green_ds = dipslip[0]
            if Green_ds is not None:
                G['dipslip'] = Green_ds

        # Tensile
        if len(tensile) == 3:               # GPS case

            E_ts = tensile[0]
            N_ts = tensile[1]
            U_ts = tensile[2]
            ts = []
            nd = 0
            if (E_ts is not None) and (N_ts is not None):
                d = E_ts.shape[0]
                m = E_ts.shape[1]
                ts.append(E_ts)
                ts.append(N_ts)
                nd += 2
            if (U_ts is not None):
                d = U_ts.shape[0]
                m = U_ts.shape[1]
                ts.append(U_ts)
                nd += 1
            if nd > 0:
                ts = np.array(ts)
                ts = ts.reshape((nd*d, m))
                G['tensile'] = ts

        elif len(tensile) == 1:             # InSAR/Tsunami Case
            Green_ts = tensile[0]
            if Green_ts is not None:
                G['dipslip'] = Green_ts

        # All done
        return


    def assembled(self, datas):
        '''
        Assembles the data vector corresponding to the stored green's functions.
        Args:
            * datas         : list of the data object involved (from gpsrates and insarrates).
        '''

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


    def assembleGFs(self, datas, polys=0, slipdir='sd', verbose=True):
        '''
        Assemble the Green's functions that have been built using build GFs.
        This routine spits out the General G and the corresponding data vector d.
        Args:
            * datas         : data sets to use as inputs (from gpsrates and insarrates).
            * polys         : 0 -> nothing additional is estimated
                              1 -> estimate a constant offset
                              3 -> estimate z = ax + by + c
                              4 -> estimate z = axy + bx + cy + d
                              'full' -> For GPS, estimates a rotation, translation and scaling
                              with respect to the center of the network (Helmert transform).
                              'strain' -> For GPS, estimate the full strain tensor (Rotation
                              + Translation + Internal strain)
                              'strainnorotation' -> For GPS, estimate the strain tensor, no rotation).
            * slipdir       : directions of slip to include. can be any combination of s,d,t.
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print("Assembling G for fault {}".format(self.name))

        # Store the assembled slip directions
        self.slipdir = slipdir

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}

        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if polys.__class__ is not str:
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    if data.dtype is 'gpsrates':
                        self.poly[data.name] = polys
                    else:
                        print('Only GPS data have particular transformations implemented so far...')
                        return
        elif polys.__class__ is list:
            for d in range(len(datas)):
                if polys[d].__class__ is not str:
                    self.poly[datas[d].name] = polys[d]*datas[d].obs_per_station
                else:
                    if datas[d].dtype is 'gpsrates':
                        self.poly[datas[d].name] = polys[d]
                    else:
                        print('Only GPS data have particular transformations implemented so far...')
                        return

        # Create the transformation holder
        if not hasattr(self, 'helmert'):
            self.helmert = {}
        if not hasattr(self, 'strain'):
            self.strain = {}

        # Get the number of parameters
        if self.N_slip == None:
            self.N_slip = len(self.patch)
        Nps = self.N_slip*len(slipdir)
        Npo = 0
        for data in datas :
            transformation = self.poly[data.name]
            if type(transformation) is str:
                tmpNpo = data.getNumberOfTransformParameters(self.poly[data.name])
                Npo += tmpNpo
                if transformation in ('full'):
                    self.helmert[data.name] = tmpNpo
                elif transformation in ('strain', 'strainonly', 'strainnorotation', 'strainnotranslation'):
                    self.strain[data.name] = tmpNpo
            else:
                Npo += (self.poly[data.name])
        Np = Nps + Npo

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Build the desired slip list
        sliplist = []
        if 's' in slipdir:
            sliplist.append('strikeslip')
        if 'd' in slipdir:
            sliplist.append('dipslip')
        if 't' in slipdir:
            sliplist.append('tensile')

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        polstart = Nps
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))

            # Fill Glocal
            ec = 0
            for sp in sliplist:
                Glocal[:,ec:ec+self.N_slip] = self.G[data.name][sp]
                ec += self.N_slip

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal

            # Build the polynomial function
            if data.dtype is 'gpsrates':
                orb = data.getTransformEstimator(self.poly[data.name])) 
            elif data.dtype in ('insarrates', 'cosicorrrates'):
                orb = getPolyEstimator(self.poly[data.name])

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

    def assembleCd(self, datas, add_prediction=None, verbose=False):
        '''
        Assembles the data covariance matrixes that have been built by each data structure.
        add_prediction: Precentage of displacement to add to the Cd diagonal to simulate a Cp (prediction error).
        '''

        # Check if the Green's function are ready
        if self.Gassembled is None:
            print("You should assemble the Green's function matrix first")
            return

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


    def buildCmGaussian(self, sigma, extra_params=None):
        '''
        Builds a diagonal Cm with sigma values on the diagonal.
        sigma is a list of numbers, as long as you have components of slip (1, 2 or 3).
        Args:
           * extra_params : a list of extra parameters.
        '''

        # Get the number of slip directions
        slipdir = len(self.slipdir)
        if self.N_slip==None:
            self.N_slip = len(self.patch)

        # Number of parameters
        Np = self.N_slip * slipdir
        if extra_params is not None:
            Np += len(extra_params)

        # Create Cm
        Cm = np.zeros((Np, Np))

        # Loop over slip dir
        for i in range(slipdir):
            Cmt = np.diag(sigma[i] * np.ones(self.N_slip,))
            Cm[i*self.N_slip:(i+1)*self.N_slip,i*self.N_slip:(i+1)*self.N_slip] = Cmt

        # Put the extra parameter sigma values
        st = self.N_slip * slipdir
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Stores Cm
        self.Cm = Cm

        # all done
        return


    def buildCm(self, sigma, lam, lam0=None, extra_params=None, lim=None, verbose=True):
        '''
        Builds a model covariance matrix using the equation described in Radiguet et al 2010.
        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.
            * lam0          : Normalizing distance (if None, lam0=min(distance between patches)).
            * extra_params  : Add some extra values on the diagonal.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(lam))

        # Need the patch geometry
        if self.patch is None:
            print("You should build the patches and the Green's functions first.")
            return

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = ((np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())
                / (np.unique(self.centers[:,0]).size))
            yd = ((np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())
                / (np.unique(self.centers[:,1]).size))
            zd = ((np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())
                / (np.unique(self.centers[:,2]).size))
            lam0 = np.sqrt(xd**2 + yd**2 + zd**2)
        if verbose:
            print("Lambda0 = {}".format(lam0))
        C = (sigma * lam0 / lam)**2

        # Creates the principal Cm matrix
        if self.N_slip==None:
            self.N_slip = len(self.patch)
        Np = self.N_slip * len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((self.N_slip, self.N_slip))
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        if self.N_slip == len(self.patch):
            flag_patch = True
            vertices = None
        elif self.patchType == 'triangle' and self.N_slip == len(self.gocad_vertices):
            flag_patch = False
            vertices = self.gocad_vertices
        else:
            raise NotImplementedError('Error: Inconsistent number of slip values')

        for i in range(self.N_slip):
            distances = np.zeros((self.N_slip,))
            if flag_patch:
                p1 = self.patch[i]
            else:
                v1 = vertices[i]
            for j in range(self.N_slip):
                if j == i:
                    continue
                if flag_patch:
                    p2 = self.patch[j]
                    distances[j] = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)
                else:
                    v2 = vertices[j]
                    distances[j] = distanceVertexToVertex(v1, v2, lim=lim)
            Cmt[i,:] = C * np.exp(-distances / lam)

        # Store that into Cm
        st = 0
        for i in range(len(slipdir)):
            se = st + self.N_slip
            Cm[st:se, st:se] = Cmt
            st += self.N_slip

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return


    def estimateSeismicityRate(self, earthquake, extra_div=1.0, epsilon=0.00001):
        '''
        Counts the number of earthquakes per patches and divides by the area of the patches.
        Args:
            * earthquake    : seismiclocation object
            * extra_div     : Extra divider to get the seismicity rate.
            * epsilon       : Epsilon value for precision of earthquake location.
        '''

        # Make sure the area of the fault patches is computed
        self.computeArea()

        # Project the earthquakes on fault patches 
        ipatch = earthquake.getEarthquakesOnPatches(self, epsilon=epsilon)

        # Count
        number = np.zeros(len(self.patch))

        # Loop 
        for i in range(len(self.patch)):
            number[i] = len(ipatch[i].tolist())/(self.area[i]*extra_div)

        # Store that in the fault
        self.earthquakesInPatch = ipatch
        self.seismicityRate = number

        # All done
        return

#EOF
