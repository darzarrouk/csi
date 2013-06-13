'''
A class that deals with dipping faults.

Written by R. Jolivet, April 2013
'''
import copy
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import okada4py as ok

class dippingfault(object):

    def __init__(self, name, utmzone=None):
        '''
        Args:
            * name          : Name of the fault.
        '''

        # Initialize the fault
        self.name = name

        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initializing fault %s"%self.name)

        # Set the reference point in the x,y domain (not implemented)
        self.xref = 0.0
        self.yref = 0.0

        # Set the utm zone
        self.utmzone = utmzone
        if self.utmzone is not None:
            self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

        # allocate some things
        self.xf = None
        self.yf = None
        self.xi = None
        self.yi = None
        self.loni = None
        self.lati = None

        # Allocate depth and number of patches
        self.top = None             # Depth of the top of the fault
        self.depth = None           # Depth of the bottom of the fault
        self.numz = None            # Number of patches at depth
        self.dip = None             # Fault dip

        # Allocate patches
        self.patch = None
        self.slip = None
        self.totalslip = None

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # All done
        return

    def initializeslip(self, n=None):
        '''
        Re-initializes the fault slip array.
        Args:
            * n     : Number of slip values. If None, it'll take the number of patches.
        '''

        if n is None:
            n = len(self.patch)

        self.slip = np.array(())

        # All done
        return

    def trace(self, Lon, Lat):
        ''' 
        Set the surface fault trace.

        Args:
            * Lon           : Array/List containing the Lon points.
            * Lat           : Array/List containing the Lat points.
        '''

        # Set lon and lat
        self.lon = np.array(Lon)
        self.lat = np.array(Lat)

        # All done
        return

    def setdip(self, dip):
        '''
        Sets the dip of the fault.
        Args:
            * dip           : Dip angle.
        '''

        # Set dip
        self.dip = dip

        # if depth is set 
        if self.depth is not None:
            self.vertdepth = self.depth/np.cos(np.pi/2. - self.dip*np.pi/180.)

        # All done
        return

    def setvertdepth(self, vertdepth, top=0, num=5):
        '''
        Set the maximum depth of the fault patches.

        Args:
            * vertdepth         : Depth of the fault patches before rotation.
            * num           : Number of fault patches at depth.
        '''

        # Set depth
        self.top = top
        self.vertdepth = vertdepth
        self.numz = num

        # if dip is set
        if self.dip is not None:
            self.depth = self.vertdepth*np.cos(np.pi/2. - self.dip*np.pi/180.)

        # All done
        return


    def setdepth(self, depth, top=0, num=5):
        '''
        Set the maximum depth of the fault patches.

        Args:
            * depth         : Depth of the fault patches.
            * num           : Number of fault patches at depth.
        '''

        # Set depth
        self.top = top
        self.depth = depth
        self.numz = num

        # if dip is set
        if self.dip is not None:
            self.vertdepth = self.depth/np.cos(np.pi/2. - self.dip*np.pi/180.)

        # All done
        return

    def file2trace(self, filename):
        '''
        Reads the fault trace directly from a file.
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
        Lon = []
        Lat = []
        for i in range(len(A)):
            Lon.append(np.float(A[i].split()[0]))
            Lat.append(np.float(A[i].split()[1]))
            
        # Create the trace 
        self.trace(Lon, Lat)

        # All done
        return

    def utmzone(self, utmzone):
        '''
        Set the utm zone of the fault.

        Args:
            * utm           : UTM zone of the fault.
        '''

        # Set utmzone
        self.utmzone = utmzone
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

        # All done
        return

    def trace2xy(self):
        ''' 
        Transpose the surface trace of the fault into the UTM reference.
        '''

        # do it 
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return

    def ll2xy(self, lon, lat):
        '''
        Do the lat lon 2 utm transform
        '''

        # Transpose 
        x, y = self.putm(lon, lat)

        # Put it in Km
        x = x/1000.
        y = y/1000.

        # All done
        return x, y

    def xy2ll(self, x, y):
        '''
        Do the utm to lat lon transform
        '''

        # Transpose and return
        return self.putm(x*1000., y*1000., inverse=True)

    def extrapolate(self, length_added=50, tol=2., fracstep=5., extrap='ud'):
        ''' 
        Extrapolates the surface trace. This is usefull when building deep patches for interseismic loading.
        Args:
            * length_added  : Length to add when extrapolating.
            * tol           : Tolerance to find the good length.
            * fracstep      : control each jump size.
            * extrap        : if u in extrap -> extrapolates at the end
                              if d in extrap -> extrapolates at the beginning
                              default is 'ud'
        '''

        # print 
        print ("Extrapolating the fault for %f km"%length_added)

        # Check if the fault has been interpolated before
        if self.xi is None:
            print ("Run the discretize() routine first")
            return

        # Build the interpolation routine
        import scipy.interpolate as scint
        fi = scint.interp1d(self.xi, self.yi)

        # Build the extrapolation routine
        fx = self.extrap1d(fi)

        # make lists
        self.xi = self.xi.tolist()
        self.yi = self.yi.tolist()

        if 'd' in extrap:
        
            # First guess for first point
            xt = self.xi[0] - length_added/2.
            yt = fx(xt)
            d = np.sqrt( (xt-self.xi[0])**2 + (yt-self.yi[0])**2)

            # Loop to find the best distance
            while np.abs(d-length_added)>tol:
                # move up or down
                if (d-length_added)>0:
                    xt = xt + d/fracstep
                else:
                    xt = xt - d/fracstep
                # Get the corresponding yt
                yt = fx(xt)
                # New distance
                d = np.sqrt( (xt-self.xi[0])**2 + (yt-self.yi[0])**2) 
        
            # prepend the thing
            self.xi.reverse()
            self.xi.append(xt)
            self.xi.reverse()
            self.yi.reverse()
            self.yi.append(yt)
            self.yi.reverse()

        if 'u' in extrap:

            # First guess for the last point
            xt = self.xi[-1] + length_added/2.
            yt = fx(xt)
            d = np.sqrt( (xt-self.xi[-1])**2 + (yt-self.yi[-1])**2)

            # Loop to find the best distance
            while np.abs(d-length_added)>tol:
                # move up or down
                if (d-length_added)<0:
                    xt = xt + d/fracstep
                else:
                    xt = xt - d/fracstep
                # Get the corresponding yt
                yt = fx(xt)
                # New distance
                d = np.sqrt( (xt-self.xi[-1])**2 + (yt-self.yi[-1])**2)

            # Append the result
            self.xi.append(xt)
            self.yi.append(yt)

        # Make them array again
        self.xi = np.array(self.xi)
        self.yi = np.array(self.yi)

        # Build the corresponding lon lat arrays
        self.loni, self.lati = self.xy2ll(self.xi*1000., self.yi*1000.)

        # All done
        return

    def extrap1d(self,interpolator):
        '''
        Linear extrapolation routine. Found on StackOverflow by sastanin.
        '''

        # import a bunch of stuff
        from scipy import arange, array, exp

        xs = interpolator.x
        ys = interpolator.y
        def pointwise(x):
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                return interpolator(x)
        def ufunclike(xs):
            return pointwise(xs) #array(map(pointwise, array(xs)))
        return ufunclike
    
    def discretize(self, every=2, tol=0.5, method='linear', fracstep=5.):
        '''
        Refine the surface fault trace prior to divide it into patches.
        Args:
            * every         : Spacing between each point.
            * tol           : Tolerance in the spacing.
            * method        : Interpolation method. Can be 'linear','nearest', 'zero', 'slinear', 'quadratic' or 'cubic'. Default is 'linear'.
            * fracstep      : Jumping factor. The smallest, the bigger the jumps.
        '''

        # print
        print("Discretizing the fault %s every %f km"%(self.name, every))

        # Check if the fault is in UTM coordinates
        if self.xf is None:
            self.trace2xy()

        # Import the interpolation routines
        import scipy.interpolate as scint   

        # Build the interpolation
        od = np.argsort(self.xf)
        self.inter = scint.interp1d(self.xf[od], self.yf[od], bounds_error=False)
    
        # Loop to get equally spaced points
        xi = [self.xf[od][0]]                             # Interpolated x fault
        yi = [self.yf[od][0]]                             # Interpolated y fault
        xt = xi[-1]                                       # Starting point
        while xt<self.xf[od][-1]:
            # First guess
            xt = xi[-1] + every/2.
            # First guess of y
            yt = self.inter(xt)
            # Compute the distance between the two successive points
            d = np.sqrt((xt-xi[-1])**2 + (yt-yi[-1])**2)
            while np.abs(d-every)>tol:
                if (xt>xi[-1]) and ((d-every)>0):       # In that case, reduce xt
                    xt = xt - d/fracstep
                elif (xt>xi[-1]) and ((d-every)<0):     # In that case, increase xt
                    xt = xt + d/fracstep
                # Get corresponding yt
                yt = self.inter(xt)
                # New distance
                d = np.sqrt((xt-xi[-1])**2 + (yt-yi[-1])**2)
            # When d-every<tol, store the xt and yt values, if they are non-nan
            if np.isfinite(xt) and np.isfinite(yt):
                xi.append(xt)
                yi.append(yt)

        # Store the result in self
        self.xi = np.array(xi)
        self.yi = np.array(yi)

        # Compute the lon/lat
        if self.utmzone is not None:
            self.loni, self.lati = self.putm(self.xi*1000., self.yi*1000., inverse=True)

        # All done
        return

    def build_patches(self):
        '''
        Builds rectangular patches from the discretized fault.
        A patch is a list of 4 corners.
        '''

        # If the maximum depth and the number of patches is not set
        assert self.vertdepth is not None, "Please set dip, depth and number of patches."

        print ("Build patches for fault %s between depths: %f, %f"%(self.name, self.top, self.vertdepth))

        # Define the depth vector
        z = np.linspace(self.top, self.vertdepth, num=self.numz+1)
        self.z_patches = z

        # If the discretization is not done
        if self.xi is None:
            self.discretize()

        # Define a patch list
        self.patch = []
        self.patchll = []
        self.slip = []

        # Iterate over the surface discretized fault points
        for i in range(len(self.xi)-1):
            # First corner
            x1 = self.xi[i]
            y1 = self.yi[i]
            lon1 = self.loni[i]
            lat1 = self.lati[i]
            # Second corner
            x2 = self.xi[i]
            y2 = self.yi[i]
            lon2 = self.loni[i]
            lat2 = self.lati[i]
            # Third corner
            x3 = self.xi[i+1]
            y3 = self.yi[i+1]
            lon3 = self.loni[i+1]
            lat3 = self.lati[i+1]
            # Fourth corner
            x4 = self.xi[i+1]
            y4 = self.yi[i+1]
            lon4 = self.loni[i+1]
            lat4 = self.lati[i+1]
            # iterate at depth
            for j in range(len(z)-1):
                p = np.zeros((4,3))
                pll = np.zeros((4,3))
                p[0,:] = [x1, y1, z[j]]
                pll[0,:] = [lon1, lat1, z[j]]
                p[1,:] = [x2, y2, z[j+1]]
                pll[1,:] = [lon2, lat2, z[j]]
                p[2,:] = [x3, y3, z[j+1]]
                pll[2,:] = [lon3, lat3, z[j+1]]
                p[3,:] = [x4, y4, z[j]]
                pll[3,:] = [lon4, lat4, z[j+1]]
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])

        # Say that the fault is vertical
        self.vertical = True

        # Translate slip to np.array
        self.slip = np.array(self.slip)

        # Rotate patches to have the good dip
        self.rotatepatches()
        self.vertical = False

        # All done
        return

    def rotatepatches(self):
        '''
        Rotates the existing patches.
        '''

        # Assert dip is set
        assert self.dip, "Please specify dip angle."

        # build the unit vector of the axis of rotation
        D, P = self.surfacetrace2director()

        # Rotation
        theta = np.pi/2 - self.dip*np.pi/180.

        # Rotate all the points of all the patches
        self.patchvert = copy.copy(self.patch)
        self.patchrot = []
        for p in self.patch:
            pr = []
            for pp in p:
                pr.append(self.rotpoint(pp, D, P, theta))
            self.patchrot.append(np.array(pr))

        # replace patch by the rotated
        self.patch = self.patchrot
        del self.patchrot

        # All done
        return

    def surfacetrace2director(self):
        '''
        Computes an average vector director for the surface trace.
        '''

        # Gets the equation
        G = np.vstack((self.xf, np.ones((self.xf.shape)))).T
        m = np.linalg.lstsq(G, self.yf)
        a = m[0][0]
        b = m[0][1]

        # Take two points
        A = [self.xf.min(), a*self.xf.min()+b, 0]
        B = [self.xf.max(), a*self.xf.max()+b, 0]

        # Vector director
        u = A[0] - B[0]
        v = A[1] - B[1]
        w = 0

        # Norm 
        n = np.sqrt(u**2 + v**2 + w**2)

        # Normalize
        u /= n
        v /= n
        w = 0

        # All done
        return [u, v, w], A

    def deletepatch(self, patch):
        '''
        Deletes a patch.
        Args:   
            * patch     : index of the patch to remove.
        '''

        # Remove the patch
        if patch.__class__ is int:
            del self.patch[patch]
            self.slip = np.delete(self.slip, patch, axis=0)
        elif patch.__class__ is list:
            for p in patch:
                del self.patch[p]
                self.slip = np.delete(self.slip, p, axis=0)

        # All done
        return

    def addpatch(self, patch, slip=[0, 0, 0]):
        '''
        Adds a patch to the list.
        Args:
            * patch     : Geometry of the patch to add
            * slip      : List of the strike, dip and tensile slip.
        '''

        # append the patch
        self.patch.append(patch)

        # modify the slip
        sh = self.slip.shape
        nl = sh[0] + 1
        nc = 3
        tmp = np.zeros((nl, nc))
        if nl > 1:                      # Case where slip is empty
            tmp[:nl-1,:] = self.slip
        tmp[-1,:] = slip
        self.slip = tmp

        # All done
        return

    def getpatchgeometry(self, patch, center=False):
        '''
        Returns the patch geometry as needed for okada85.
        Args:
            * patch         : index of the wanted patch.
            * center        : if true, returns the coordinates of the center of the patch. if False, returns the UL corner.
        '''

        # Get the patch
        p = self.patch[patch]

        # Get the UL corner of the patch
        if center:
            x1 = (p[1,0] + p[2,0])/2.
            x2 = (p[1,1] + p[2,1])/2.
            x3 = (p[1,2] + p[0,2])/2.
        else:
            x1 = p[0,0]
            x2 = p[0,1]
            x3 = p[0,2]

        # Get the patch width (this fault is vertical for now)
        width = p[2,2] - p[0,2]

        # Get the length
        length = np.sqrt( (p[2,0] - p[1,0])**2 + (p[2,1] - p[1,1])**2 )

        # Get the strike
        strike = np.arctan( (p[2,0] - p[1,0])/(p[2,1] - p[1,1]) ) + np.pi

        # Set the dip
        dip = np.pi*90. / 180.

        # All done
        return x1, x2, x3, width, length, strike, dip

    def slip2dis(self, data, patch, slip=None):
        '''
        Computes the surface displacement at the data location using okada.

        Args:
            * data          : data object from gpsrates or insarrates.
            * patch         : number of the patch that slips
            * slip          : if a number is given, that is the amount of slip along strike
                              if three numbers are given, that is the amount of slip along strike, along dip and opening
                              if None, values from slip are taken
        '''

        # Set the slip values
        if slip is None:
            SLP = [self.slip[patch,0], self.slip[patch,1], self.slip[patch,2]]
        elif slip.__class__ is float:
            SLP = [slip, 0.0, 0.0]
        elif slip.__class__ is list:
            SLP = slip

        # Get patch geometry
        x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(patch)

        # Get data position
        x = data.x
        y = data.y

        # Allocate displacement lists
        ss_dis = []
        ds_dis = []
        op_dis = []

        for i in range(len(x)):

            # Run okada for strike slip
            ss = ok.displacement(x[i], y[i], dip, x1, x2, x3, length, width, strike, 1)
            ss_dis.append(ss*SLP[0])

            # Run okada for dip slip
            ds = ok.displacement(x[i], y[i], dip, x1, x2, x3, length, width, strike, 2)
            ds_dis.append(ds*SLP[1])

            # Run okada for opening
            op = ok.displacement(x[i], y[i], dip, x1, x2, x3, length, width, strike, 3)
            op_dis.append(op*SLP[2])

        # Make arrays
        ss_dis = np.array(ss_dis)
        ds_dis = np.array(ds_dis)
        op_dis = np.array(op_dis)

        # All done
        return ss_dis, ds_dis, op_dis

    def buildGFs(self, data, vertical=True, slipdir='sd'):
        '''
        Builds the Green's function matrix based on the discretized fault.
        Args:
            * data      : data object from gpsrates or insarrates.
            * vertical  : if True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir   : direction of the slip along the patches. can be any combination of s (strikeslip), d (dipslip) and t (tensile).

        The Green's function matrix is stored in a dictionary. Each entry of the dictionary is named after the corresponding dataset. Each of these entry is a dictionary that contains 'strikeslip', 'dipslip' and/or 'tensile'.
        '''

        print ("Building Green's functions for the data set %s of type %s"%(data.name, data.dtype))

        # Get the number of data
        Nd = data.lon.shape[0]
        if data.dtype is 'insarrates':
            Ndt = Nd
            data.obs_per_station = 1
        elif data.dtype is 'gpsrates':
            Ndt = data.lon.shape[0]*2
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station = 3
                Ndt += data.lon.shape[0]

        # Get the number of parameters
        Np = len(self.patch)
        Npt = len(self.patch)*len(slipdir)

        # Initializes a space in the dictionary to store the green's function
        self.G[data.name] = {}
        G = self.G[data.name]
        if 's' in slipdir:
            G['strikeslip'] = np.zeros((Ndt, Np))
        if 'd' in slipdir:
            G['dipslip'] = np.zeros((Ndt, Np))
        if 't' in slipdir:
            G['tensile'] = np.zeros((Ndt, Np))

        # Initializes the data vector
        if data.dtype is 'insarrates':
            self.d[data.name] = data.vel
            vertical = True                 # In InSAR, you need to use the vertical, no matter what....
        elif data.dtype is 'gpsrates':
            if vertical:
                self.d[data.name] = data.vel_enu.T.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].T.flatten()

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

        # import something
        import sys

        # Loop over each patch
        for p in range(len(self.patch)):
            sys.stdout.write('\r Patch: %i / %i '%(p+1,len(self.patch)))
            sys.stdout.flush()
            
            # get the surface displacement corresponding to unit slip
            ss, ds, op = self.slip2dis(data, p, slip=SLP)

            # Do we keep the verticals
            if not vertical:
                ss = ss[:,0:2]
                ds = ds[:,0:2]
                op = op[:,0:2]

            # Organize the response
            if data.dtype is 'gpsrates':            # If GPS type, just put them in the good format
                ss = ss.T.flatten()
                ds = ds.T.flatten()
                op = op.T.flatten()
            elif data.dtype is 'insarrates':        # If InSAR, do the dot product with the los
                ss_los = []
                ds_los = []
                op_los = []
                for i in range(Nd):
                    ss_los.append(np.dot(data.los[i,:], ss[i,:]))
                    ds_los.append(np.dot(data.los[i,:], ds[i,:]))
                    op_los.append(np.dot(data.los[i,:], op[i,:]))
                ss = ss_los
                ds = ds_los
                op = op_los

            # Store these guys in the corresponding G slot
            if 's' in slipdir:
                G['strikeslip'][:,p] = ss
            if 'd' in slipdir:
                G['dipslip'][:,p] = ds
            if 't' in slipdir:
                G['tensile'][:,p] = op

        # Clean the screen 
        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return

    def setGFs(self, data, strikeslip=[None, None, None], dipslip=[None, None, None], tensile=[None, None, None], vertical=True):
        '''
        Stores the Green's functions matrices into the fault structure.
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : List of matrices of the Strikeslip Green's functions, ordered E, N, U
            * dipslip       : List of matrices of the dipslip Green's functions, ordered E, N, U
            * tensile       : List of matrices of the tensile Green's functions, ordered E, N, U
        '''

        # Clean up the Green's functions
        self.G = {}

        # Create the storage for that dataset
        self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if data.dtype is 'insarrates':
            self.d[data.name] = data.vel
            vertical = True                 # In InSAR, you need to use the vertical, no matter what....
        elif data.dtype is 'gpsrates':
            if vertical:
                self.d[data.name] = data.vel_enu.T.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].T.flatten()

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

        elif len(strikeslip) == 1:          # InSAR case

            LOS_ss = strikeslip[0]
            if LOS_ss is not None:
                G['strikeslip'] = LOS_ss

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
                ds = np.array(ss)
                ds = ds.reshape((nd*d, m))
                G['dipslip'] = ds
        
        elif len(dipslip) == 1:             # InSAR case

            LOS_ds = dipslip[0]
            if LOS_ds is not None:
                G['dipslip'] = LOS_ds

        # StrikeSlip
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

        elif len(tensile) == 1:             # InSAR Case

            LOS_ts = tensile[0]
            if LOS_ts is not None:
                G['dipslip'] = LOS_ds

        # All done
        return

    def assembleGFs(self, datas, polys=0, slipdir='sd'):
        '''
        Assemble the Green's functions that have been built using build GFs.
        This routine spits out the General G and the corresponding data vector d.
        Args:
            * datas         : data sets to use as inputs (from gpsrates and insarrates).
            * polys         : 0 -> nothing additional is estimated
                              1 -> estimate a constant offset
                              3 -> estimate z = ax + by + c
                              4 -> estimate z = axy + bx + cy + d
            * slipdir       : which directions of slip to include. can be any combination of s, d and t.
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print("Assembling G for fault %s"%self.name)

        # Store the assembled slip directions
        self.slipdir = slipdir

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}

        # Set poly right
        if (polys.__class__ is float) or (polys.__class__ is int):
            for data in datas:
                self.poly[data.name] = polys*data.obs_per_station
        elif polys.__class__ is list:
            for d in range(len(datas)):
                self.poly[datas[d].name] = polys[d]*data.obs_per_station

        # Get the number of parameters
        N = len(self.patch)
        Nps = N*len(slipdir)
        Npo = 0
        for data in datas :
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

        # loop over the datasets
        el = 0
        polstart = Nps
        for data in datas:

            # print
            print("Dealing with %s of type %s"%(data.name, data.dtype))

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))
            
            # Fill Glocal
            ec = 0
            for sp in sliplist:
                Glocal[:,ec:ec+N] = self.G[data.name][sp]
                ec += N

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal

            # Build the polynomial function
            if self.poly[data.name] > 0:
                if data.dtype is 'gpsrates':
                    orb = np.zeros((Ndlocal, self.poly[data.name]))
                    nn = Ndlocal/data.obs_per_station
                    orb[:nn, 0] = 1.0
                    orb[nn:2*nn, 1] = 1.0
                    if data.obs_per_station == 3:
                        orb[2*nn:3*nn, 2] = 1.0
                elif data.dtype is 'insarrates':
                    orb = np.zeros((Ndlocal, self.poly[data.name]))
                    orb[:] = 1.0
                    if self.poly[data.name] >= 3:
                        orb[:,1] = data.x/np.abs(data.x).max()
                        orb[:,2] = data.y/np.abs(data.y).max()
                    if self.poly[data.name] >= 4:
                        orb[:,3] = (data.x/np.abs(data.x).max())*(data.y/np.abs(data.y).max())

                # Put it into G for as much observable per station we have
                polend = polstart + self.poly[data.name]
                G[el:el+Ndlocal, polstart:polend] = orb
                polstart += self.poly[data.name]

            # Update el to check where we are
            el = el + Ndlocal
            
        # Store G in self
        self.Gassembled = G

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
                print("Dealing with data %s"%data.name)

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

    def assembleCd(self, datas):
        '''
        Assembles the data covariance matrixes that have been built by each data structure.
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
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return

    def buildCm(self, sigma, lam, lam0=None, extra_params=None, lim=None):
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
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Assembling the Cm matrix ")
        print ("Sigma = %f"%sigma)
        print ("Lambda = %f"%lam)

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
            xd = (np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )
        print ("Lambda0 = %f"%lam0)
        C = (sigma*lam0/lam)**2

        # Creates the principal Cm matrix
        Np = len(self.patch)*len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((len(self.patch), len(self.patch)))
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        i = 0
        for p1 in self.patch:
            j = 0
            for p2 in self.patch:
                # Compute the distance
                d = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)
                # Compute Cm
                Cmt[i,j] = C * np.exp( -1.0*d/lam)
                Cmt[j,i] = C * np.exp( -1.0*d/lam)
                # Upgrade counter
                j += 1
            # upgrade counter
            i += 1

        # Store that into Cm
        st = 0
        for i in range(len(slipdir)):
            se = st + len(self.patch)
            Cm[st:se, st:se] = Cmt
            st += len(self.patch)

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return

    def distancePatchToPatch(self, patch1, patch2, distance='center', lim=None):
        '''
        Measures the distance between two patches.
        Args:
            * patch1    : geometry of the first patch.
            * patch2    : geometry of the second patch.
            * distance  : distance estimation mode
                            center : distance between the centers of the patches.
                            no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].
        '''

        if distance is 'center':

            # Get the centers
            x1, y1, z1 = self.getcenter(patch1)
            x2, y2, z2 = self.getcenter(patch2)

            # Compute the distance
            dis = np.sqrt( (x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis>lim[0]:
                    dis = lim[1]

        # All done
        return dis

    def writeEDKSsubParams(self, data, edksfilename, amax=None, plot=False):
        '''
        Write the subParam file needed for the interpolation of the green's function in EDKS.
        Francisco's program cuts the patches into small patches, interpolates the kernels to get the GFs at each point source, 
        then averages the GFs on the pacth. To decide the size of the minimum patch, it uses St Vernant's principle.
        If amax is specified, the minimum size is fixed.
        Args:
            * data          : Data object from gpsrates or insarrates.
            * edksfilename  : Name of the file containing the kernels.
            * amax          : Specifies the minimum size of the divided patch. If None, uses St Vernant's principle.
            * plot          : Activates plotting.
        Returns:
            * filename      : Name of the subParams file created.
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Write the EDKS files and run the GFs interpolation")

        # Write the geometry to the EDKS file
        self.writeEDKSgeometry()

        # Write the data to the EDKS file
        data.writeEDKSdata()

        # Open the EDKSsubParams.py file
        filename = 'EDKSParams_%s_%s.py'%(self.name, data.name)
        fout = open(filename, 'w')

        # Write in it
        fout.write("# File with the rectangles properties\n")
        fout.write("RectanglesPropFile = 'edks_%s.END'\n"%self.name)
        fout.write("# File with id, E[km], N[km] coordinates of the receivers.\n")
        fout.write("ReceiverFile = 'edks_%s.idEN'\n"%data.name)
        fout.write("# read receiver direction (# not yet implemented)\n")
        if data.dtype is 'insarrates':
            fout.write("useRecvDir = True # True for InSAR, uses LOS information\n")
        else:
            fout.write("useRecvDir = False # False for GPS, uses ENU displacements\n")

        fout.write("# Maximum Area to subdivide triangles. If None, uses Saint-Venant's principle.\n")
        if amax is None:
            fout.write("Amax = None # None computes Amax automatically. \n")
        else:
            fout.write("Amax = %f # Minimum size for the patch division.\n"%amax)

        fout.write("EDKSunits = 1000.0 # to convert from kilometers to meters\n")
        fout.write("EDKSfilename = '%s'\n"%edksfilename)
        fout.write("prefix = 'edks_%s_%s'\n"%(self.name, data.name))
        fout.write("plotGeometry = %s # set to False if you are running in a remote Workstation\n"%plot)
        
        # Close the file
        fout.close()

        # All done
        return filename

    def writeEDKSgeometry(self, ref=None):
        '''
        This routine spits out 2 files:
        filename.lonlatdepth: Lon center | Lat Center | Depth Center (km) | Strike | Dip | Length (km) | Width (km) | patch ID
        filename.END: Easting (km) | Northing (km) | Depth Center (km) | Strike | Dip | Length (km) | Width (km) | patch ID

        These files are to be used with /home/geomod/dev/edks/MPI_EDKS/calcGreenFunctions_EDKS_subRectangles.py

        Args:
            * ref           : Lon and Lat of the reference point. If None, the patches positions is in the UTM coordinates.
        '''

        # Filename
        filename = 'edks_%s'%self.name

        # Open the output file
        flld = open(filename+'.lonlatdepth','w')
        flld.write('#lon lat Dep[km] strike dip length(km) width(km) ID\n')
        fend = open(filename+'.END','w')
        fend.write('#Easting[km] Northing[km] Dep[km] strike dip length(km) width(km) ID\n')

        # Reference
        if ref is not None:
            refx, refy = self.putm(ref[0], ref[1])
            refx /= 1000.
            refy /= 1000.

        # Loop over the patches
        for p in range(len(self.patch)):
            x, y, z, width, length, strike, dip = self.getpatchgeometry(p)
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            lon, lat = self.xy2ll(x,y)
            if ref is not None:
                x -= refx
                y -= refy
            flld.write('%f %f %f %f %f %f %f %5i \n'%(lon,lat,z,strike,dip,length,width,p))
            fend.write('%f %f %f %f %f %f %f %5i \n'%(x,y,z,strike,dip,length,width,p))

        # Close the files
        flld.close()
        fend.close()

        # All done
        return

    def getcenters(self):
        '''
        Get the center of the patches.
        '''

        # Get the patches
        patch = self.patch

        # Initialize a list
        center = []

        # loop over the patches
        for p in patch:
            x, y, z = self.getcenter(p)
            center.append([x, y, z])

        # All done
        return center

    def getcenter(self, p):
        ''' 
        Get the center of one patch.
        Args:
            * p    : Patch geometry.
        '''
    
        # Get center
        x = (p[1][0] + p[2][0])/2.
        y = (p[1][1] + p[2][1])/2.
        z = (p[1][2] + p[0][2])/2.

        # All done
        return x,y,z

    def surfacesimulation(self, box=None, disk=None, err=None, npoints=None):
        ''' 
        Takes the slip vector and computes the surface displacement that corresponds on a regular grid.
        Args:
            * box       : Can be a list of [minlon, maxlon, minlat, maxlat].
            * disk      : list of [xcenter, ycenter, radius, n]
        '''

        # create a fake gps object
        from .gpsrates import gpsrates
        self.sim = gpsrates('simulation', utmzone=self.utmzone)

        # Create a lon lat grid
        if (box is None) and (disk is None) :
            lon = np.linspace(self.lon.min(), self.lon.max(), 100)
            lat = np.linspace(self.lat.min(), self.lat.max(), 100)
            lon, lat = np.meshgrid(lon,lat)
            lon = lon.flatten()
            lat = lat.flatten()
        elif (box is not None):
            lon = np.linspace(box[0], box[1], 100)
            lat = np.linspace(box[2], box[3], 100)
            lon, lat = np.meshgrid(lon,lat)
            lon = lon.flatten()
            lat = lat.flatten()
        elif (disk is not None):
            lon = []; lat = []
            xd, yd = self.ll2xy(disk[0], disk[1])
            print 'Center: %f ; %f'%(xd,yd)
            print 'Radius: %f'%(disk[2])
            xmin = xd-disk[2]; xmax = xd+disk[2]; ymin = yd-disk[2]; ymax = yd+disk[2]
            ampx = (xmax-xmin) 
            ampy = (ymax-ymin)
            n = 0
            while n<disk[3]:
                x, y = np.random.rand(2)
                x *= ampx; x -= ampx/2.; x += xd
                y *= ampy; y -= ampy/2.; y += yd
                if ((x-xd)**2 + (y-yd)**2) <= (disk[2]**2):
                    lo, la = self.xy2ll(x,y)
                    lon.append(lo); lat.append(la)
                    n += 1
            lon = np.array(lon); lat = np.array(lat)

        if (lon.max()>360.) or (lon.min()<-180.0) or (lat.max()>90.) or (lat.min()<-90):
            self.sim.x = lon
            self.sim.y = lat
        else:
            self.sim.lon = lon
            self.sim.lat = lat
            # put these in x y utm coordinates
            self.sim.ll2xy()

        # Initialize the vel_enu array
        self.sim.vel_enu = np.zeros((lon.size, 3))

        # Create the station name array
        self.sim.station = []
        for i in range(len(self.sim.x)):
            name = '%04i'%i
            self.sim.station.append(name)
        self.sim.station = np.array(self.sim.station)

        # Create an error array
        if err is not None:
            self.sim.err_enu = []
            for i in range(len(self.sim.x)):
                x,y,z = np.random.rand(3)
                x *= err
                y *= err
                z *= err
                self.sim.err_enu.append([x,y,z])
            self.sim.err_enu = np.array(self.sim.err_enu)

        # import stuff
        import sys

        # Loop over the patches
        for p in range(len(self.patch)):
            sys.stdout.write('\r Patch %i / %i '%(p+1,len(self.patch)))
            sys.stdout.flush()
            # Get the surface displacement due to the slip on this patch
            ss, ds, op = self.slip2dis(self.sim, p)
            # Sum these to get the synthetics
            self.sim.vel_enu += ss
            self.sim.vel_enu += ds
            self.sim.vel_enu += op

        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return

    def computetotalslip(self):
        '''
        Computes the total slip.
        '''

        # Computes the total slip
        self.totalslip = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2 + self.slip[:,2]**2)
    
        # All done
        return

    def plot(self,ref='utm', figure=134):
        '''
        Plot the available elements of the fault.
        
        Args:
            * ref           : Referential for the plot ('utm' or 'lonlat').
            * figure        : Number of the figure.
        '''

        # Import necessary things
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figure)
        ax = fig.add_subplot(111, projection='3d')

        # Set the axes
        if ref is 'utm':
            ax.set_xlabel('Easting (km)')
            ax.set_ylabel('Northing (km)')
        else:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth (km)')

        # Plot the surface trace
        if ref is 'utm':
            if self.xf is None:
                self.trace2xy()
            ax.plot(self.xf, self.yf, '-b')
        else:
            ax.plot(self.lon, self.lat,'-b')

        # Plot the discretized trace
        if self.xi is not None:
            if ref is 'utm':
                ax.plot(self.xi, self.yi, '.r')
            else:
                if self.loni is None:
                    self.loni, self.lati = self.putm(self.xi*1000., self.yi*1000., inverse=True)
                ax.plot(loni, lati, '.r')

        # Compute the total slip
        self.computetotalslip()

        # Plot the patches
        if self.patch is not None:
            
            # import stuff
            import mpl_toolkits.mplot3d.art3d as art3d
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            
            # set z axis
            ax.set_zlim3d([-1.0*(self.vertdepth+5), 0])
            zticks = []
            zticklabels = []
            for z in self.z_patches:
                zticks.append(-1.0*z)
                zticklabels.append(z)
            ax.set_zticks(zticks)
            ax.set_zticklabels(zticklabels)
            
            # set color business
            cmap = plt.get_cmap('jet')
            cNorm  = colors.Normalize(vmin=0, vmax=self.totalslip.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

            for p in range(len(self.patch)):
                ncorners = len(self.patch[0])
                x = []
                y = []
                z = []
                xv = []
                yv = []
                zv = []
                for i in range(ncorners):
                    if ref is 'utm':
                        x.append(self.patch[p][i][0])
                        y.append(self.patch[p][i][1])
                        z.append(-1.0*self.patch[p][i][2])
                        if self.vertical is False:
                            xv.append(self.patchvert[p][i][0])
                            yv.append(self.patchvert[p][i][1])
                            zv.append(-1.0*self.patchvert[p][i][2])
                    else:
                        x.append(self.patchll[p][i][0])
                        y.append(self.patchll[p][i][1])
                        z.append(-1.0*self.patchll[p][i][2])
                verts = [zip(x, y, z)]
                rect = art3d.Poly3DCollection(verts)
                rect.set_color(scalarMap.to_rgba(self.totalslip[p]))
                rect.set_edgecolors('k')
                ax.add_collection3d(rect)
                if self.vertical is False:
                    vverts = [zip(xv, yv, zv)]
                    rect = art3d.Poly3DCollection(vverts)
                    rect.set_color(scalarMap.to_rgba(self.totalslip[p]))
                    rect.set_edgecolors('k')
                    ax.add_collection3d(rect)

            # put up a colorbar        
            scalarMap.set_array(self.totalslip)
            plt.colorbar(scalarMap)

        # show
        plt.show()

        # All done
        return

    def rotpoint(self,A,D,P,theta):
        '''
        Rotates the point A.
        The rotation is around a line defined by its director vector <u,v,w>
        and a point <a,b,c>.
        The rotation angle is thetha.
        Args:
            * A     : POint to rotate.
            * D     : Vector director (3 values).
            * P     : Point (3 values).
            * theta : Rotation angle, in rad.
        '''

        # Define values
        u = D[0]
        v = D[1]
        w = D[2]

        a = P[0]
        b = P[1]
        c = P[2]

        x = A[0]
        y = A[1]
        z = A[2]

        L = u**2 + v**2 + w**2

        # Rotated point
        xr = (a*(v**2 + w**2) - u*(b*v + c*w - u*x - v*y - w*z)) * (1-np.cos(theta)) + L*x*np.cos(theta) + np.sqrt(L)*(b*w - c*v + v*z -w*y)*np.sin(theta)
        xr /= L
        yr = (b*(u**2 + w**2) - v*(a*u + c*w - u*x - v*y - w*z)) * (1-np.cos(theta)) + L*y*np.cos(theta) + np.sqrt(L)*(c*u - a*w + w*x -u*z)*np.sin(theta)
        yr /= L
        zr = (c*(u**2 + v**2) - w*(a*u + b*v - u*x - v*y - w*z)) * (1-np.cos(theta)) + L*z*np.cos(theta) + np.sqrt(L)*(a*v - b*u + u*y - v*x)*np.sin(theta)
        zr /= L

        # All done
        return [xr, yr, zr]

