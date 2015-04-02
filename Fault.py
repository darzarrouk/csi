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
import scipy.spatial.distance as scidis
import copy
import sys
import os

# Personals
from .SourceInv import SourceInv
from .EDKS import sum_layered_sub
from .EDKS import dropSourcesInPatches as Patches2Sources

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

    def initializeslip(self, n=None, values=None):
        '''
        Re-initializes the fault slip array to zero values.
        Args:
            * n     : Number of slip values. If None, it'll take the number of patches.
            * values: Can be depth, strike, dip, length, area or a numpy array
        '''

        # Shape
        if n is None:
           self.N_slip = len(self.patch)
        else:
            self.N_slip = n

        self.slip = np.zeros((self.N_slip,3))
        
        # Values
        if values is not None:
            # string type
            if type(values) is str:
                if values == 'depth':
                    values = np.array([self.getpatchgeometry(p, center=True)[2] for p in self.patch])
                elif values == 'strike':
                    values = np.array([self.getpatchgeometry(p, center=True)[5] for p in self.patch])
                elif values == 'dip':
                    values = np.array([self.getpatchgeometry(p, center=True)[6] for p in self.patch])
                elif values == 'length':
                    values = np.array([self.getpatchgeometry(p, center=True)[4] for p in self.patch])
                elif values == 'width':
                    values = np.array([self.getpatchgeometry(p, center=True)[3] for p in self.patch])
                elif values == 'area':
                    self.computeArea()
                    values = self.area
                elif values == 'index':
                    values = np.array([np.float(self.getindex(p)) for p in self.patch])
                self.slip[:,0] = values
            # Numpy array 
            if type(values) is np.ndarray:
                try:
                    self.slip[:,:] = values
                except:
                    try:
                        self.slip[:,0] = values
                    except:
                        print('Wrong size for the slip array provided')
                        return

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
        self.lon, self.lat = self.xy2ll(self.xf, self.yf)

        # All done
        return


    def patch2ll(self):
        '''
        Takes all the patches in self.patch and convert them to lonlat.
        '''

        # Create list
        patchll = []

        # Iterate
        for patch in self.patch:
            # Create a patch
            pll = []
            # Iterate again
            for p in patch.tolist():
                lon, lat = self.xy2ll(p[0], p[1])
                pll.append([lon, lat, p[2]])
            patchll.append(np.array(pll))

        # Save
        self.patchll = patchll

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

        # Set top
        if self.top is None:
            depth = [[p[2] for p in patch] for patch in self.patch]
            depth = np.unique(np.array(depth).flatten())
            self.top = np.min(depth)
            self.depth = np.max(depth)

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

        # Set lon lat
        self.trace2ll()

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

    def discretize(self, every=2, tol=0.01, fracstep=0.2, xaxis='x', cum_error=True): 
        '''
        Refine the surface fault trace prior to divide it into patches. (Fault cannot be north-south)
        Args:
            * every         : Spacing between each point.
            * tol           : Tolerance in the spacing.
            * fracstep      : fractional step in x for the discretization optimization
            * xaxis         : x axis for the discretization ('x'= use x as the x axis, 'y'= use y as the x axis)
            * cum_error     : if True, account for cumulated error to define the x axis bound for the last patch
        '''

        # Check if the fault is in UTM coordinates
        if self.xf is None:
            self.trace2xy()

        if xaxis=='x':
            xf = self.xf
            yf = self.yf
        else:
            yf = self.xf
            xf = self.yf            

        # Import the interpolation routines
        import scipy.interpolate as scint   

        # Build the interpolation
        od = np.argsort(xf)
        f_inter = scint.interp1d(xf[od], yf[od], bounds_error=False)
    
        # Initialize the list of equally spaced points
        xi = [xf[od][0]]                               # Interpolated x fault
        yi = [yf[od][0]]                               # Interpolated y fault
        xlast = xf[od][-1]                             # Last point
        ylast = yf[od][-1]

        # First guess for the next point
        xt = xi[-1] + every * fracstep 
        yt = f_inter(xt)
        # Check if first guess is in the domain
        if xt>xlast-tol:
            xt = xlast
            xi.append(xt)
            yi.append(f_inter(xt))
        # While the last point is not the last wanted point
        total_error = 0.
        mod_error   = 0.
        while (xi[-1] < xlast):
            # I compute the distance between me and the last accepted point
            d = np.sqrt( (xt-xi[-1])**2 + (yt-yi[-1])**2 )
            # Check if I am in the tolerated range
            if np.abs(d-every)<tol:
                xi.append(xt)
                yi.append(yt)
            else:
                # While I am to far away from my goal and I did not pass the last x
                while ((np.abs(d-every)>tol) and (xt<xlast)):
                    # I add the distance*frac that I need to go
                    xt += (every-d)*fracstep                    
                    # If I passed the last point (accounting for error in previous steps) 
                    if (np.round(xt,decimals=2)>=np.round(xlast-mod_error-tol,decimals=2)):   
                        xt = xlast                            
                    elif (xt<xi[-1]):  # If I passed the previous point
                        xt = xi[-1] + every
                    # I compute the corresponding yt
                    yt = f_inter(xt)
                    # I compute the corresponding distance
                    d = np.sqrt( (xt-xi[-1])**2 + (yt-yi[-1])**2 )                    
                # When I stepped out of that loop, append
                if cum_error:
                    total_error += every - d
                    mod_error    = np.abs(total_error)%(0.5*every)
                xi.append(xt)
                yi.append(yt)
            # Next guess for the loop
            xt = xi[-1] + every * fracstep

        # Store the result in self
        if xaxis=='x': 
            self.xi = np.array(xi)
            self.yi = np.array(yi)
        else:
            self.yi = np.array(xi)
            self.xi = np.array(yi)

        # Compute the lon/lat
        self.loni, self.lati = self.putm(self.xi*1000., self.yi*1000., inverse=True)

        # All done
        return

    def cumdistance(self, discretized=False):
        '''
        Computes the distance between the first point of the fault and every other 
        point, when you walk along the fault.
        Args:   
            * discretized           : if True, use the discretized fault trace 
                                      (default False)
        '''

        # Get the x and y positions
        if discretized:
            x = self.xi
            y = self.yi
        else:
            x = self.xf
            y = self.yf

        # initialize
        dis = np.zeros((x.shape[0]))

        # Loop 
        for i in range(1,x.shape[0]):
            d = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            dis[i] = dis[i-1] + d

        # all done 
        return dis

    def distance2trace(self, lon, lat, discretized=False):
        '''
        Computes the distance between a point and the trace of a fault.
        This is a slow method, so it has been recoded in a few places throughout the code.
        Args:
            * lon               : Longitude of the point.
            * lat               : Latitude of the point.
            * discretized       : Uses the discretized trace (default=False)
        '''
        
        # Get the cumulative distance along the fault
        cumdis = self.cumdistance(discretized=discretized)

        # ll2xy
        x, y = self.ll2xy(lon, lat)

        # Fault coordinates
        if discretized:
            xf = self.xi
            yf = self.yi
        else:
            xf = self.xf
            yf = self.yf

        # Compute the distance between the point and all the points
        d = scidis.cdist([[x,y]], [[xf[i], yf[i]] for i in range(len(xf))])[0]

        # Get the two closest points
        imin1 = d.argmin()
        dmin1 = d[imin1]
        d[imin1] = 999999.
        imin2 = d.argmin()
        dmin2 = d[imin2]
        d[imin2] = 999999.
        dtot = dmin1+dmin2

        # Along the fault?
        xc = (xf[imin1]*dmin1 + xf[imin2]*dmin2)/dtot
        yc = (yf[imin1]*dmin1 + yf[imin2]*dmin2)/dtot

        # Distance
        if dmin1<dmin2:
            jm = imin1
        else:
            jm = imin2
        dalong = cumdis[jm] + np.sqrt( (xc-xf[jm])**2 + (yc-yf[jm])**2 )
        dacross = np.sqrt((xc-x)**2 + (yc-y)**2)

        # All done
        return dalong, dacross

    def getindex(self, p):
        '''
        Returns the index of a patch.
        '''

        # output index
        iout = None

        # Find the index of the patch
        for i in range(len(self.patch)):
            try:
                if (self.patch[i] == p).all():
                    iout = i
            except:
                if self.patch[i]==p:
                    iout = i

        # All done
        return iout


    def getslip(self, p):
        '''
        Returns the slip vector for a patch or tent
        Args:
            * p         : patch or tent
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


    def saveGFs(self, dtype='d', outputDir='.', suffix={'strikeslip':'SS','dipslip':'DS','tensile':'TS', 'coupling': 'Coupling'}):
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
                n = self.name.replace(' ', '_')
                d = data.replace(' ', '_')
                filename = '{}_{}_{}.gf'.format(n, d, suffix[c])
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
    
    def buildGFs(self, data, vertical=True, slipdir='sd', method='homogeneous', verbose=True):
        '''
        Builds the Green's function matrix based on the discretized fault.
        Args:
            * data      : data object from gpsrates or insarrates.
            * vertical  : if True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir   : direction of the slip along the patches. can be any combination of s (strikeslip), d (dipslip) and t (tensile).
            * method    : Can be okada (Okada, 1982) (rectangular patches only)
                                 meade (Meade 2007) (triangular patches only)
                                 edks (Zhao & Rivera, 2002) 
                                 homogeneous (Okada for rectangles, Meade for triangles)
        The Green's function matrix is stored in a dictionary. 
        Each entry of the dictionary is named after the corresponding dataset. 
        Each of these entry is a dictionary that contains 'strikeslip', 'dipslip' and/or 'tensile'.

        **********************
        TODO: Implement the homogeneous case for the Node-based triangular GFs
        **********************
        '''

        # Chech something
        if self.patchType == 'triangletent':
            assert method is 'edks', 'Homogeneous case not implemented for {} faults'.format(self.patchType)

        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))

        # Check something
        if method in ('homogeneous', 'Homogeneous'):
            if self.patchType == 'rectangle':
                method = 'Okada'
            elif self.patchType == 'triangle':
                method = 'Meade'
            elif self.patchType == 'triangletent':
                method = 'Meade'
        
        # Print
        if verbose:
            print('Greens functions computation method: {}'.format(method))

        # Data type check
        if data.dtype == 'insarrates':
            if not vertical:
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
        if method in ('okada', 'Okada', 'OKADA', 'ok92', 'meade', 'Meade', 'MEADE'):
            G = self.homogeneousGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose)
        elif method in ('edks', 'EDKS'):
            G = self.edksGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose)

        # Separate the Green's functions for each type of data set
        data.setGFsInFault(self, G, vertical=vertical)

        # All done
        return

    def homogeneousGFs(self, data, vertical=True, slipdir='sd', verbose=True):
        '''
        Builds the Green's functions for a homogeneous half-space.
        If your patches are rectangular, Okada's formulation is used (Okada, 1982)
        If your patches are triangular, Meade's formulation is used (Meade, 2007)
        Args:
            * data      : data object from gpsrates or insarrates.
            * vertical  : if True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir   : direction of the slip along the patches. can be any combination of s (strikeslip), d (dipslip) and t (tensile).
        '''

        assert self.patchType != 'triangletent', 'Need to run EDKS for that particular type of fault'

        # Initialize the slip vector
        SLP = []
        if 's' in slipdir:              # If strike slip is aksed
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)
        if 'd' in slipdir:              # If dip slip is asked
            SLP.append(1.0) 
        else:                           # Else
            SLP.append(0.0)
        if 't' in slipdir:              # If tensile is asked
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)

        # Create the dictionary
        G = {'strikeslip':[], 'dipslip':[], 'tensile':[]}

        # Loop over each patch
        for p in range(len(self.patch)):
            if verbose:
                sys.stdout.write('\r Patch: {} / {} '.format(p+1,len(self.patch)))
                sys.stdout.flush()
            
            # get the surface displacement corresponding to unit slip
            # ss,ds,op will all have shape (Nd,3) for 3 components
            ss, ds, op = self.slip2dis(data, p, slip=SLP)
            Nd = ss.shape[0]

            # Do we keep the verticals
            if not vertical:
                # Just get horizontal components
                ss = ss[:,0:2]
                ds = ds[:,0:2]
                op = op[:,0:2]

            # Organize the response
            if data.dtype in ['gpsrates', 'cosicorrrates', 'multigps']:
                # If GPS type, construct a flat vector with east displacements first, then
                # north, then vertical
                ss = ss.T.flatten()
                ds = ds.T.flatten()
                op = op.T.flatten()

            elif data.dtype == 'insarrates':
                # If InSAR, do the dot product with the los
                ss_los = []
                ds_los = []
                op_los = []
                for i in range(Nd):
                    ss_los.append(np.dot(data.los[i,:], ss[i,:]))
                    ds_los.append(np.dot(data.los[i,:], ds[i,:]))
                    op_los.append(np.dot(data.los[i,:], op[i,:]))
                ss = np.array(ss_los)
                ds = np.array(ds_los)
                op = np.array(op_los)

            # Store these guys in the corresponding G slot
            if 's' in slipdir:
                G['strikeslip'].append(ss)
            if 'd' in slipdir:
                G['dipslip'].append(ds)
            if 't' in slipdir:
                G['tensile'].append(op)

        # Easily get the number of data
        Nd = ss.shape[0]
        Np = len(self.patch)

        # Reshape the Green's functions
        if 's' in slipdir:
            G['strikeslip'] = np.array(G['strikeslip']).reshape((Np, Nd)).T
        else:
            G['strikeslip'] = None
        if 'd' in slipdir:
            G['dipslip'] = np.array(G['dipslip']).reshape((Np, Nd)).T
        else:
            G['dipslip'] = None
        if 't' in slipdir:
            G['tensile'] = np.array(G['tensile']).reshape((Np, Nd)).T
        else:
            G['tensile'] = None

        # Clean the screen 
        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # All done
        return G

    def edksGFs(self, data, vertical=True, slipdir='sd', verbose=True):
        '''
        Builds the Green's functions based on the solution by Zhao & Rivera 2002.
        The Corresponding functions are in the EDKS code that needs to be installed and 
        the executables should be in the environment variable EDKS_BIN.
        A few variables need to be set in self before:
        self.kernelsEDKS    : Filename of the EDKS kernels.
            One of the Three:
        self.sourceSpacing  : Spacing between the sources in each patch.
        self.sourceNumber   : Number of sources per patches.
        self.sourceArea     : Maximum Area of the sources.
        Args:
            * data      : data object from gpsrates or insarrates.
            * vertical  : if True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir   : direction of the slip along the patches. can be any combination of s (strikeslip), d (dipslip), t (tensile).
        '''

        # Check if we can find kernels
        if not hasattr(self, 'kernelsEDKS'):
            print('---------------------------------')
            print('---------------------------------')
            print(' WARNING WARNING WARNING WARNING ')
            print('   Kernels for computation of')
            print('stratified Greens functions not ')
            print('    set in {}.kernelsEDKS'.format(self.name))
            print('   Looking for default kernels')
            print('---------------------------------')
            print('---------------------------------')
            self.kernelsEDKS = 'kernels.edks'
            print('No Kernels provided, trying with kernels.edks')
        stratKernels = self.kernelsEDKS
        if not os.path.isfile(stratKernels):
            print('Kernels for EDKS not found...')
            print('I give up...')
            sys.exit(1)
        else:
            print('Kernels used: {}'.format(stratKernels))
        
        # Check if we can find mention of the spacing between points
        if not hasattr(self, 'sourceSpacing') and not hasattr(self, 'sourceNumber')\
                and not hasattr(self, 'sourceArea'):
            print('---------------------------------')
            print('---------------------------------')
            print(' WARNING WARNING WARNING WARNING ')
            print('  Cannot find sourceSpacing nor  ')
            print('   sourceNumber nor sourceArea   ')
            print('         for stratified          ')
            print('   Greens function computation   ')
            print('           computation           ')
            print('          Dying here...          ')
            print('              Arg...             ')
            sys.exit(1)
        
        # Receivers to meters
        xr = data.x * 1000.
        yr = data.y * 1000.

        # Prefix for the files
        prefix = '{}_{}'.format(self.name.replace(' ','-'), data.name.replace(' ','-'))

        # Check something
        if not hasattr(self, 'keepTrackOfSources'):  
            if self.patchType == 'triangletent':
                self.keepTrackOfSources = True
            else:
                self.keepTrackOfSources = False

        # If we have already done that step
        if self.keepTrackOfSources and hasattr(self, 'edksSources'):
            if verbose:
                print('Get sources from saved sources')
            Ids, xs, ys, zs, strike, dip, Areas = self.edksSources[:7]
        # Else, drop sources in the patches
        else:
            if verbose:
                print('Subdividing patches into point sources')
            Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(self, verbose=verbose)
            # All these guys need to be in meters
            xs *= 1000.
            ys *= 1000.
            zs *= 1000.
            Areas *= 1e6
            # Strike and dip in degrees
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            # Keep track?
            if self.keepTrackOfSources:
                self.edksSources = [Ids, xs, ys, zs, strike, dip, Areas]

        # Get the slip vector
        if self.patchType in ('triangle', 'rectangle'):
            slip = np.ones(dip.shape)
        if self.patchType == 'triangletent':
            # If saved, good
            if self.keepTrackOfSources and hasattr(self, 'edksSources') and (len(self.edksSources)>7):
                slip = self.edksSources[7]
            # Else, we have to re-organize the Ids from facet to nodes
            else:
                self.Facet2Nodes()
                Ids, xs, ys, zs, strike, dip, Areas, slip = self.edksSources

        # Informations
        if verbose:
            print('{} sources for {} patches and {} data points'.format(len(Ids), len(self.patch), len(xr)))
        
        # Run EDKS
        if 's' in slipdir:
            if verbose:
                print('Running Strike Slip component for data set {}'.format(data.name))
            Gss = sum_layered_sub(Ids, xs, ys, zs, \
                                  strike, dip, np.zeros(dip.shape), slip, \
                                  Areas,\
                                  xr, yr, stratKernels, prefix, BIN_EDKS='EDKS_BIN')
            Gss = np.array(Gss)
        if 'd' in slipdir:
            if verbose:
                print('Running Dip Slip component for data set {}'.format(data.name))
            Gds = sum_layered_sub(Ids, xs, ys, zs, \
                                  strike, dip, np.ones(dip.shape)*90.0, slip, \
                                  Areas,\
                                  xr, yr, stratKernels, prefix, BIN_EDKS='EDKS_BIN')
            Gds = np.array(Gds)
        if 't' in slipdir:
            if verbose:
                print('Running tensile component for data set {}'.format(data.name))
            Gts = sum_layered_sub(Ids, xs, ys, zs,
                                  strike, dip, np.zeros(dip.shape), slip,
                                  Areas, xr, yr, stratKernels, prefix,
                                  BIN_EDKS='EDKS_BIN', tensile=True)
            Gts = np.array(Gts)
                             
        
        # Verticals?
        Ncomp = 3
        if not vertical:
            Ncomp = 2
            if 'd' in slipdir:
                Gds = Gds[:2,:,:]
            if 's' in slipdir:
                Gss = Gss[:2,:,:]
            if 't' in slipdir:
                Gts = Gts[:2,:,:]
        
        # Numbers
        Ndata = Ncomp*xr.shape[0]
        Nparm = self.slip.shape[0]

        # Check format
        if data.dtype in ['gpsrates', 'cosicorrrates', 'multigps']:
            # Flat arrays with e, then n, then u (optional)
            if 's' in slipdir:
                Gss = Gss.reshape((Ndata, Nparm))
            if 'd' in slipdir:
                Gds = Gds.reshape((Ndata, Nparm))
            if 't' in slipdir:
                Gts = Gts.reshape((Ndata, Nparm))
        elif data.dtype == 'insarrates':
            # If InSAR, do the dot product with the los
            if 's' in slipdir:
                Gss_los = []
            if 'd' in slipdir:
                Gds_los = []
            if 't' in slipdir:
                Gts_los = []
            for i in range(xr.shape[0]):
                for j in range(Nparm):
                    if 's' in slipdir:
                        Gss_los.append(np.dot(data.los[i,:], Gss[:,i,j]))
                    if 'd' in slipdir:
                        Gds_los.append(np.dot(data.los[i,:], Gds[:,i,j]))
                    if 't' in slipdir:
                        Gts_los.append(np.dot(data.los[i,:], Gts[:,i,j]))
            if 's' in slipdir:
                Gss = np.array(Gss_los).reshape((xr.shape[0], Nparm))
            if 'd' in slipdir:
                Gds = np.array(Gds_los).reshape((xr.shape[0], Nparm))
            if 't' in slipdir:
                Gts = np.array(Gts_los).reshape((xr.shape[0], Nparm))

        # Create the dictionary
        G = {'strikeslip':[], 'dipslip':[], 'tensile':[]}

        # Reshape the Green's functions
        if 's' in slipdir:
            G['strikeslip'] = Gss
        else:
            G['strikeslip'] = None
        if 'd' in slipdir:
            G['dipslip'] = Gds
        else:
            G['dipslip'] = None
        if 't' in slipdir:
            G['tensile'] = Gts
        else:
            G['tensile'] = None

        # All done
        return G

    def setGFsFromFile(self, data, strikeslip=None, dipslip=None, tensile=None, coupling=None,
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
            * coupling      : File containing the Green's functions for coupling.
            * vertical      : Deal with the UP component (gps: default is false,
                              insar: it will be true anyway).
            * dtype         : Type of binary data.
        '''

        print('---------------------------------')
        print('---------------------------------')
        print("Set up Green's functions for fault {} from files {}, {} and {}".format(self.name, strikeslip, dipslip, tensile))

        # Get the number of patches
        if self.N_slip == None:
            self.N_slip = self.slip.shape[0]

        # Read the files and reshape the GFs
        Gss = None; Gds = None; Gts = None; Gcp = None
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
        if coupling is not None:
            Gcp = np.fromfile(coupling, dtype=dtype)
            ndl = int(Gcp.shape[0]/self.N_slip)
            Gcp = Gcp.reshape((ndl, self.N_slip))
        
        # Create the big dictionary
        G = {'strikeslip': Gss,
             'dipslip': Gds,
             'tensile': Gts, 
             'coupling': Gcp}

        # The dataset sets the Green's functions itself
        data.setGFsInFault(self, G, vertical=vertical)

        # all done
        return

    def setGFs(self, data, strikeslip=[None, None, None], dipslip=[None, None, None],
               tensile=[None, None, None], coupling=[None, None, None], vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the fault structure.
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : List of matrices of the Strikeslip Green's functions, ordered E, N, U
            * dipslip       : List of matrices of the dipslip Green's functions, ordered E, N, U
            * tensile       : List of matrices of the tensile Green's functions, ordered E, N, U
            * coupling      : List of matrices of the coupling Green's function, ordered E, N, U
            If you provide InSAR GFs, these need to be projected onto the LOS direction already.
        '''

        # Get the number of data per point
        if data.dtype == 'insarrates' or data.dtype == 'tsunami':
            data.obs_per_station = 1
        elif data.dtype in ('gpsrates', 'multigps'):
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
        elif data.dtype == 'cosicorrrates':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if not synthetic:
            if data.dtype == 'insarrates':
                self.d[data.name] = data.vel
                vertical = True # Always true for InSAR
            elif data.dtype == 'tsunami':
                self.d[data.name] = data.d
                vertical = True
            elif data.dtype in ('gpsrates', 'multigps'):
                if vertical:
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
                self.d[data.name]=self.d[data.name][-np.isnan(self.d[data.name])]
            elif data.dtype == 'cosicorrrates':
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
                G['tensile'] = Green_ts

        # Coupling
        if len(coupling) == 3:               # GPS case

            E_cp = coupling[0]
            N_cp = coupling[1]
            U_cp = coupling[2]
            cp = []
            nd = 0
            if (E_cp is not None) and (N_cp is not None):
                d = E_cp.shape[0]
                m = E_cp.shape[1]
                cp.append(E_cp)
                cp.append(N_cp)
                nd += 2
            if (U_cp is not None):
                d = U_cp.shape[0]
                m = U_cp.shape[1]
                cp.append(U_cp)
                nd += 1
            if nd > 0:
                cp = np.array(cp)
                cp = ts.reshape((nd*d, m))
                G['coupling'] = cp

        elif len(coupling) == 1:             # InSAR/Tsunami Case
            Green_cp = coupling[0]
            if Green_cp is not None:
                G['coupling'] = Green_cp
        # All done
        return

    def rotateGFs(self, data, convergence, returnGFs=False):
        '''
            For the data set data, rotates the Greens' functions so that dip slip motion is aligned with
        the convergence vector.
        These Greens' functions are not stored in self.G or returned, given arguments.

        Args:
            * data          : Name of the data set.
            * convergence   : Convergence vector, or list/array of convergence vector with
                                shape = (Number of fault patches, 2). 
        '''
        
        # Get the name of the data
        data = data.name

        # Assert Green's functions have been computed 
        assert data in self.G.keys(), 'Need to compute regular GFs first...'
        assert 'dipslip' in self.G[data].keys(), 'Need to compute GFs for unitary dip slip first...'
        assert 'strikeslip' in self.G[data].keys(), 'Need to compute GFs for unitary strike slip first...'

        # Number of slip parameters
        nSlip = self.slip.shape[0]

        # Get the convergence vector
        if len(convergence)==2:
            Conv = np.ones((nSlip, 2))
            Conv[:,0] *= convergence[0]
            Conv[:,1] *= convergence[1]
            self.convergence = Conv
        elif len(convergence)==nSlip:
            if type(convergence) is list:
                self.convergence = np.array(convergence)
        else:
            print('Convergence vector is of wrong format...')
            sys.exit()

        # Get the fault strike
        strike = self.getStrikes()

        # Get the green's functions
        Gss = self.G[data]['strikeslip']
        Gds = self.G[data]['dipslip']

        # Get the strike and dip vectors
        strikeVec = np.vstack((np.sin(strike), np.cos(strike))).T
        dipVec = np.vstack((np.sin(strike+np.pi/2.), np.cos(strike+np.pi/2.))).T

        # Project the convergence along strike and dip
        Sbr = (self.convergence*strikeVec).sum(axis=1)
        Dbr = (self.convergence*dipVec).sum(axis=1)

        # Optional return
        if returnGFs:
            return Sbr, Dbr
        else:
            self.G[data]['dipslip'] = Dbr
            self.G[data]['strikeslip'] = Sbr

        # All done
        return

    def computeCouplingGFs(self, data, convergence, initializeCoupling=True):

        '''
            For the data set data, computes the Green's Function for coupling, using the formula
        described in Francisco Ortega's PhD, pages 106 to 108.

                    !!!! YOU NEED TO COMPUTE THE GREEN'S FUNCTIONS BEFORE !!!!

            The corresponding GFs are stored in the GFs dictionary, under the name of the data
        set and are named 'coupling'. When inverting for coupling, we suggest building these 
        functions and assembling with slipdir='c'.
        
        Args:
            * data          : Name of the data set.
            * convergence   : Convergence vector, or list/array of convergence vector with
                                shape = (Number of fault patches, 2). 
        '''
        
        # Get the green's functions
        Gss = self.G[data.name]['strikeslip']
        Gds = self.G[data.name]['dipslip']

        # Rotates the Greens' functions
        Sbr, Dbr = self.rotateGFs(data, convergence, returnGFs=True)

        # Multiply and sum
        Gc = -1.0*((np.multiply(-1.0*Gss, Sbr) + np.multiply(Gds, Dbr)))
        # Precision: (the -1.0* is because we use a different convention from that of Francisco)

        # Store those new GFs
        self.G[data.name]['coupling'] = Gc

        # Initialize a coupling vector
        if initializeCoupling:
            self.coupling = np.zeros((self.slip.shape[0],))

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

    def assembleGFs(self, datas, polys=None, slipdir='sd', verbose=True):
        '''
        Assemble the Green's functions that have been built using build GFs.
        This routine spits out the General G and the corresponding data vector d.
        Args:
            * datas         : data sets to use as inputs (from gpsrates and insarrates).
            
            * polys         : None -> nothing additional is estimated

                        For all datasets:
                              1 -> estimate a constant offset
                              3 -> estimate z = ax + by + c
                              4 -> estimate z = axy + bx + cy + d
             
                        For GPS only:
                              'full'                -> Estimates a rotation, translation and scaling
                                                       with respect to the center of the network (Helmert transform).
                              'strain'              -> Estimates the full strain tensor (Rotation + Translation + Internal strain)
                              'strainnorotation'    -> Estimates the strain tensor + translation
                              'strainonly'          -> Estimates the strain tensor
                              'strainnotranslation' -> Estimates the strain tensor + rotation
                              'translation'         -> Estimates the translation
                              'translationrotation  -> Estimates the translation + rotation
            
            * slipdir       : directions of slip to include. can be any combination of s,d,t or c
                              s: strike slip
                              d: dip slip
                              t: tensile
                              c: coupling
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
                if (polys.__class__ is not str) and (polys is not None):
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    self.poly[data.name] = polys
        elif polys.__class__ is list:
            for data, poly in zip(datas, polys):
                if (poly.__class__ is not str) and (poly is not None) and (poly.__class__ is not list):
                    self.poly[data.name] = poly*data.obs_per_station
                else:
                    self.poly[data.name] = poly

        # Create the transformation holder
        if not hasattr(self, 'helmert'):
            self.helmert = {}
        if not hasattr(self, 'strain'):
            self.strain = {}
        if not hasattr(self, 'transformation'):
            self.transformation = {}

        # Get the number of parameters
        if self.N_slip == None:
            self.N_slip = self.slip.shape[0]
        Nps = self.N_slip*len(slipdir)
        Npo = 0
        for data in datas :
            transformation = self.poly[data.name]
            if type(transformation) in (str, list):
                tmpNpo = data.getNumberOfTransformParameters(self.poly[data.name])
                Npo += tmpNpo
                if type(transformation) is str:
                    if transformation in ('full'):
                        self.helmert[data.name] = tmpNpo
                    elif transformation in ('strain', 'strainonly', 'strainnorotation', 'strainnotranslation', 'translation', 'translationrotation'):
                        self.strain[data.name] = tmpNpo
                else:
                    self.transformation[data.name] = tmpNpo
            elif transformation is not None:
                Npo += transformation
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
        if 'c' in slipdir:
            sliplist.append('coupling')

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

            # Elastic Green's functions

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

            # Polynomes and strain
            if self.poly[data.name] is not None:

                # Build the polynomial function
                if data.dtype in ('gpsrates', 'multigps'):
                    orb = data.getTransformEstimator(self.poly[data.name]) 
                elif data.dtype in ('insarrates', 'cosicorrrates'):
                    orb = data.getPolyEstimator(self.poly[data.name])

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
            self.N_slip = self.slip.shape[0]

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

    def buildCmLaplacian(self, lam, extra_params=None, sensitivity=True):
        '''
        Implements the Laplacian smoothing with sensitivity (optional)
        Description can be found in F. Ortega-Culaciati's PhD thesis.
        Args:
            * lam               : Damping factor (list of size of slipdirections)
            * extra_params      : what sigma to allow to ramp parameters.
            * sensitivity       : Weights the Laplacian by Sensitivity (default True)
        '''

        # lambda
        if type(lam) is float:
            lam = [lam for i in range(len(self.slipdir))]

        # Get the number of patches
        nSlip = self.N_slip
        if extra_params is not None:
            nExtra = len(extra_params)
        else:
            nExtra = 0

        # How many parameters
        Np = self.N_slip * len(self.slipdir)
        if extra_params is not None:
            Np += nExtra

        # Create the matrix
        Cm = np.zeros((Np, Np))

        # Build the laplacian
        D = self.buildLaplacian(verbose=True)

        Sensitivity = {}

        # Loop over directions:
        for i in range(len(self.slipdir)):

            # Start/Stop
            ist = nSlip*i
            ied = ist+nSlip

            if sensitivity:

                # Compute sensitivity matrix (see Loveless & Meade, 2011)
                G = self.Gassembled[:,ist:ied]
                S = np.diag(np.dot(G.T, G))
                Sensitivity[self.slipdir[i]] = S

                # Weight Laplacian by sensitivity (see F. Ortega-Culaciati PhD Thesis)
                iS = np.sqrt(1./S)
                D = D*iS[:,np.newaxis]

            # Cm
            DtD = np.dot(D.T, D)
            localCm = 1./lam[i]*np.linalg.inv(DtD)

            # Put it into Cm
            Cm[ist:ied, ist:ied] = localCm

        # Add extra params
        if nExtra>0:
            CmRamp = np.diag(extra_params)
            Cm[-nExtra:, -nExtra:] = CmRamp

        # Set inside the fault
        self.Cm = Cm
        self.Laplacian = D
        self.Sensitivity = Sensitivity

        # All done
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
            self.N_slip = self.slip.shape[0]
        Np = self.N_slip * len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        distances = self.distanceMatrix(distance='center', lim=lim)
        Cmt = C * np.exp(-distances / lam)

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
    
    def buildCmSlipDirs(self, sigma, lam, lam0=None, extra_params=None, lim=None):
        '''
        Builds a model covariance matrix using the equation described in Radiguet et al 2010.
        Here, Sigma and Lambda are lists specifying values for the slip directions
        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.
            * lam0          : Normalizing distance (if None, lam0=min(distance between patches)).
            * extra_params  : Add some extra values on the diagonal.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Assembling the Cm matrix ")
        print ("Sigma = {}".format(sigma))
        print ("Lambda = {}".format(lam))

        # Need the patch geometry
        if self.patch is None:
            print("You should build the patches and the Green's functions first.")
            return

        # Get slip
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = (np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )

        # Creates the principal Cm matrix
        Np = self.N_slip*len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((self.N_slip, self.N_slip))
        Cm = np.zeros((Np, Np))

        # Build the sigma and lambda lists
        if type(sigma) is not list:
            s = []; l = []
            for sl in range(len(slipdir)):
                s.append(sigma)
                l.append(lam)
            sigma = s
            lam = l
        assert (type(sigma) is list), 'Sigma is not a list, why???'
        if type(sigma) is list:
            assert(len(sigma)==len(lam)), 'Sigma and lambda must have the same length'
            assert(len(sigma)==len(slipdir)), 'Need one value of sigma and one value of lambda per slip direction'

        # Loop over the slipdirections
        st = 0
        for sl in range(len(slipdir)):
            # pick the right values
            la = lam[sl]
            C = (sigma[sl]*lam0/la)**2
            # Get distance matrix
            distance = self.distanceMatrix(distance='center', lim=lim)
            # Compute Cmt
            Cmt = C * np.exp( -1.0*distance/la)
            # Store that into Cm
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

    def buildCmSensitivity(self, sigma, lam, lam0=None, extra_params=None, lim=None):
        '''
        Builds a model covariance matrix using the equation described in Radiguet et al 2010.
        Then correlation length is weighted by the sensitivity matrix described in Ortega's PhD thesis.
                     ==>       S = diag(G'G)
        Here, Sigma and Lambda are lists specifying values for the slip directions
        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.
            * lam0          : Normalizing distance (if None, lam0=min(distance between patches)).
            * extra_params  : Add some extra values on the diagonal.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Assembling the Cm matrix ")
        print ("Sigma = {}".format(sigma))
        print ("Lambda = {}".format(lam))

        # Assert
        assert hasattr(self, 'Gassembled'), "Need to assemble the Green's functions"

        # Need the patch geometry
        if self.patch is None:
            print("You should build the patches and the Green's functions first.")
            return

        # Set
        self.N_slip = self.slip.shape[0]

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = (np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )

        # Creates the principal Cm matrix
        Np = self.N_slip*len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((self.N_slip, self.N_slip))
        lambdast = np.zeros((self.N_slip, self.N_slip))
        Cm = np.zeros((Np, Np))
        Lambdas = np.zeros((Np, Np))

        # Build the sigma and lambda lists
        if type(sigma) is not list:
            s = []; l = []
            for sl in range(len(slipdir)):
                s.append(sigma)
                l.append(lam)
            sigma = s
            lam = l
        if type(sigma) is list:
            assert(len(sigma)==len(lam)), 'Sigma and lambda must have the same length'
            assert(len(sigma)==len(slipdir)), 'Need one value of sigma and one value of lambda per slip direction'

        # Loop over the slipdirections
        st = 0
        for sl in range(len(slipdir)):

            # Update a counter
            se = st + self.N_slip
            
            # Get the greens functions and build sensitivity
            G = self.Gassembled[:,st:se]
            S = np.diag(np.dot(G.T, G)).copy()
            ss = S.max()
            S /= ss
            
            # pick the right values
            la = lam[sl]

            # Loop over the patches
            distance = self.distanceMatrix(distance='center', lim=lim)

            # Weight Lambda by the relative sensitivity
            s1, s2 = np.meshgrid(S, S)
            L = la/np.sqrt(s1*s2)
            # Compute Cm
            Cmt = ((sigma[sl]*lam0/L)**2) * np.exp( -1.0*distance/L)

            # Store that into Cm
            Cm[st:se, st:se] = Cmt
            Lambdas[st:se, st:se] = lambdast
            st += self.N_slip

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm
        self.Lambdas = Lambdas

        # All done
        return

    def sumPatches(self, iPatches, finalPatch):
        '''
        Takes a list of patches, sums the green's functions,
        and modifies the list self.patch.
        
        Args:
            * patches       : List of the patches to sum (indexes).
            * finalPatch    : Geometry of the final patch.
        '''

        # Needs to have Greens functions  
        assert len(self.G.keys())>0, 'Need some Greens functions, otherwise this function is pointless'

        # Loop over the data sets
        for data in self.G:

            # Get it
            G = self.G[data]

            # Loop over the Green's functions
            for comp in G:

                # Get the matrix
                gf = G[comp]

                # Sum the columns
                col = np.sum(gf[:,iPatches], axis=1)

                # New matrix
                gf = np.delete(gf, iPatches[1:], axis=1)
                gf[:,iPatches[0]] = col

                # Set it 
                G[comp] = gf
        
        # Replace the first of the patches by the new patch
        self.replacePatch(finalPatch, iPatches[0])

        # Delete the other patches
        self.deletepatches(iPatches[1:])

        # Equivalent Patches
        if self.patchType == 'rectangle':
            self.computeEquivRectangle()

        # Check 
        self.N_slip = len(self.patch)

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

    def gaussianSlipSmoothing(self, length):
        '''
        Smoothes the slip distribution using a Gaussian filter.
        Args:
            * length        : Correlation length.
        '''

        # Number of patches
        nP = self.slip.shape[0]

        # Build the smoothing matrix
        S = self.distanceMatrix(distance='center', lim=None)**2

        # Compute
        S = np.exp(-0.5*S/(length**2))
        div = 1./S.sum(axis=0)
        S = np.multiply(S, div)
        self.Smooth = S
    
        # Smooth
        self.slip[:,0] = np.dot(S, self.slip[:,0])
        self.slip[:,1] = np.dot(S, self.slip[:,1])
        self.slip[:,2] = np.dot(S, self.slip[:,2])

        # All done
        return

#EOF
