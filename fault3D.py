'''
A class that deals with vertical faults.

Written by R. Jolivet, April 2013
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

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

class fault3D(object):

    def __init__(self, name, utmzone=None):
        '''
        Args:
            * name          : Name of the fault.
        '''

        # Initialize the fault
        self.name = name

        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initializing fault {}".format(self.name))

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
        self.numz = None

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

    def duplicateFault(self):
        '''
        Returns a copy of the fault.
        '''

        return copy.deepcopy(self)

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

        # utmize
        self.trace2xy()

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

    def setdepth(self, nump, width, top=0):
        '''
        Set the maximum depth of the fault patches.

        Args:
            * nump          : Number of fault patches at depth.
            * width         : Width of the fault patches
            * top           : depth of the top row
        '''

        # Set depth
        self.top = top
        self.numz = nump
        self.width = width

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

    def dipatZ(self, interp, z):
        '''
        Uses the interpolator to return the dip angle evolution along strike at depth z.
        The interpolation scheme is piecewise linear.
        Args:
            * interp        : Dip interpolation function
            * z             : Depth.
        '''

        # Create a structure
        self.dip = []

        # Set a distance counter
        dis = 0

        # Set the previous x,y
        xp = self.xi[0]
        yp = self.yi[0]

        # Loop along the discretized fault trace
        for i in range(self.xi.shape[0]):

            # Update the distance
            dis += np.sqrt( (self.xi[i]-xp)**2 + (self.yi[i]-yp)**2 )

            # get the dip
            d = interp(dis, z[i])

            # store it
            self.dip.append(d)

            # Update previous xp, yp
            xp = self.xi[i]; yp = self.yi[i]

        # Array it
        self.dip = np.array(self.dip)

        # all done
        return

    def buildPatches(self, dip, dipdirection, every=10, minpatchsize=0.00001):
        '''
        Builds a dipping fault.
        Args:
            * dip           : Dip angle evolution [[alongstrike, depth, dip], [alongstrike, depth, dip], ..., [alongstrike, depth, dip]]
            * dipdirection  : Direction towards which the fault dips.

            Example: dip = [[0, 20], [10, 30], [80, 90]] means that from the origin point of the 
            fault (self.xi[0], self.yi[0]), the dip is 20 deg at 0 km, 30 deg at km 10 and 90 deg 
            at km 80. The routine starts by discretizing the surface trace, then defines a dip 
            evolution as a function of distance from the fault origin and drapes the fault down to
            depth.
        '''

        # Print
        print("Building a dipping fault")
        print("         Dip Angle       : from {} to {} degrees".format(dip[0], dip[-1]))
        print("         Dip Direction   : {} degrees From North".format(dipdirection))

        # Initialize the structures
        self.patch = []
        self.patchll = []
        self.slip = []
        self.patchdip = []

        # Build a 2d dip interpolator
        import scipy.interpolate as sciint
        xy = np.array([ [dip[i][0], dip[i][1]] for i in range(len(dip))])
        dips = np.array([dip[i][2] for i in range(len(dip))])
        dipinterpolator = sciint.LinearNDInterpolator(xy, dips, fill_value=90.)      # If the points are not inside the area provided by the user, the dip will be 90 deg (vertical)

        # Discretize the surface trace of the fault
        self.discretize(every=every)

        # degree to rad
        dipdirection = (-1.0*dipdirection+90)*np.pi/180.

        # initialize the depth of the top row
        self.zi = np.ones((self.xi.shape))*self.top

        # set a marker
        D = []

        # Loop over the depths
        for i in range(self.numz):

            # Get the depth of the top of the row
            zt = self.zi

            # Compute the dips for this row (it updates xi and yi at the same time)
            self.dipatZ(dipinterpolator, zt)
            self.dip *= np.pi/180.

            # Get the top of the row
            xt = self.xi
            yt = self.yi
            lont, latt = self.putm(xt*1000., yt*1000., inverse=True)
            zt = self.zi

            # Compute the bottom row
            xb = xt + self.width*np.cos(self.dip)*np.cos(dipdirection)
            yb = yt + self.width*np.cos(self.dip)*np.sin(dipdirection)
            lonb, latb = self.putm(xb*1000., yb*1000., inverse=True)
            zb = zt + self.width*np.sin(self.dip)

            # fill D
            D.append(zb.max())

            # Build the patches by linking the points together
            for j in range(xt.shape[0]-1):
                # 1st corner
                x1 = xt[j]
                y1 = yt[j]
                z1 = zt[j]
                lon1 = lont[j]
                lat1 = latt[j]
                # 2nd corner
                x2 = xt[j+1]
                y2 = yt[j+1]
                z2 = zt[j+1]
                lon2 = lont[j+1]
                lat2 = latt[j+1]
                # 3rd corner
                x3 = xb[j+1]
                y3 = yb[j+1]
                z3 = zb[j+1]
                lon3 = lonb[j+1]
                lat3 = latb[j+1]
                # 4th corner 
                x4 = xb[j]
                y4 = yb[j]
                z4 = zb[j]
                lon4 = lonb[j]
                lat4 = latb[j]
                # Set points
                if y1>y2:
                    p2 = [x1, y1, z1]; p2ll = [lon1, lat1, z1]
                    p1 = [x2, y2, z2]; p1ll = [lon2, lat2, z2]
                    p4 = [x3, y3, z3]; p4ll = [lon3, lat3, z3]
                    p3 = [x4, y4, z4]; p3ll = [lon4, lat4, z4]
                else:
                    p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
                    p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
                    p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
                    p4 = [x4, y4, z4]; p4ll = [lon4, lat4, z4]
                # Store these
                psize = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
                if psize<minpatchsize:           # Increase the size of the previous patch
                    continue                # Breaks the loop and trashes the patch
                p = [p1, p2, p3, p4]
                pll = [p1ll, p2ll, p3ll, p4ll]
                p = np.array(p)
                pll = np.array(pll)
                # fill in the lists
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(dip)

            # upgrade xi
            self.xi = xb
            self.yi = yb
            self.zi = zb

        # set depth
        D = np.array(D)
        self.z_patches = D
        self.depth = D.max()

        # Translate slip into an array
        self.slip = np.array(self.slip)

        # Re-discretoze to get the original fault
        self.discretize(every=every)

        # Compute the equivalent rectangles
        self.computeEquivRectangle()
    
        # All done
        return

    def discretize(self, every=2, tol=0.5, fracstep=0.2): 
        '''
        Refine the surface fault trace prior to divide it into patches.
        Args:
            * every         : Spacing between each point.
            * tol           : Tolerance in the spacing.
        '''

        # Check if the fault is in UTM coordinates
        if self.xf is None:
            self.trace2xy()

        # Import the interpolation routines
        import scipy.interpolate as scint   

        # Build the interpolation
        od = np.argsort(self.xf)
        self.inter = scint.interp1d(self.xf[od], self.yf[od], bounds_error=False)
    
        # Initialize the list of equally spaced points
        xi = [self.xf[od][0]]                               # Interpolated x fault
        yi = [self.yf[od][0]]                               # Interpolated y fault
        xlast = self.xf[od][-1]                             # Last point
        ylast = self.yf[od][-1]

        # First guess for the next point
        xt = xi[-1] + every * fracstep 
        # Check if first guess is in the domain
        if xt>xlast:
            xt = xlast
        # Get the corresponding yt
        yt = self.inter(xt)
            
        # While the last point is not the last wanted point
        while (xi[-1] < xlast):
            if (xt==xlast):         # I am at the end
                xi.append(xt)
                yi.append(yt)
            else:                   # I am not at the end
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
                        xt -= (d-every)*fracstep
                        if (xt>xlast):          # If I passed the last point
                            xt = xlast
                        elif (xt<xi[-1]):       # If I passed the previous point
                            xt = xi[-1] + every
                        # I compute the corresponding yt
                        yt = self.inter(xt)
                        # I compute the corresponding distance
                        d = np.sqrt( (xt-xi[-1])**2 + (yt-yi[-1])**2 )
                    # When I stepped out of that loop, append
                    xi.append(xt)
                    yi.append(yt)
            # Next guess for the loop
            xt = xi[-1] + every * fracstep

        # Store the result in self
        self.xi = np.array(xi)
        self.yi = np.array(yi)

        # Compute the lon/lat
        self.loni, self.lati = self.putm(self.xi*1000., self.yi*1000., inverse=True)

        # All done
        return

    def importPatches(self, filename, origin=[45.0, 45.0]):
        '''
        Builds a patch geometry and the corresponding files from a relax co-seismic file type.
        Args:
            filename    : Input from Relax (See Barbot and Cie on the CIG website).
            origin      : Origin of the reference frame used by relax. [lon, lat]
        '''

        # Create lists
        self.patch = []
        self.patchll = []
        self.slip = []

        # origin
        x0, y0 = self.ll2xy(origin[0], origin[1])

        # open/read/close the input file
        fin = open(filename, 'r')
        Text = fin.readlines()
        fin.close()

        # Depth array
        D = []

        # Loop over the patches
        for text in Text:

            # split
            text = text.split()

            # check if continue
            if not text[0]=='#':

                # Get values
                slip = np.float(text[1])
                xtl = np.float(text[2]) + x0
                ytl = np.float(text[3]) + y0
                depth = np.float(text[4])
                length = np.float(text[5])
                width = np.float(text[6])
                strike = np.float(text[7])*np.pi/180.
                rake = np.float(text[9])*np.pi/180.

                D.append(depth)

                # Build a patch with that
                x1 = xtl
                y1 = ytl
                z1 = depth + width

                x2 = xtl 
                y2 = ytl
                z2 = depth

                x3 = xtl + length*np.cos(strike) 
                y3 = ytl + length*np.sin(strike)
                z3 = depth

                x4 = xtl + length*np.cos(strike)
                y4 = ytl + length*np.sin(strike)
                z4 = depth + width

                # Convert to lat lon
                lon1, lat1 = self.xy2ll(x1, y1)
                lon2, lat2 = self.xy2ll(x2, y2)
                lon3, lat3 = self.xy2ll(x3, y3)
                lon4, lat4 = self.xy2ll(x4, y4)

                # Fill the patch
                p = np.zeros((4, 3))
                pll = np.zeros((4, 3))
                p[0,:] = [x1, y1, z1]
                p[1,:] = [x2, y2, z2]
                p[2,:] = [x3, y3, z3]
                p[3,:] = [x4, y4, z4]
                pll[0,:] = [lon1, lat1, z1]
                pll[1,:] = [lon2, lat2, z2]
                pll[2,:] = [lon3, lat3, z3]
                pll[3,:] = [lon4, lat4, z4]
                self.patch.append(p)
                self.patchll.append(pll)

                # Slip
                ss = slip*np.cos(rake)
                ds = slip*np.sin(rake)
                ts = 0.
                self.slip.append([ss, ds, ts])

        # Translate slip to np.array
        self.slip = np.array(self.slip)

        # Depth 
        D = np.unique(np.array(D))
        self.z_patches = D
        self.depth = D.max()

        # Create a trace
        dmin = D.min()
        self.lon = []
        self.lat = []
        for p in self.patchll:
            d = p[1][2]
            if d==dmin:
                self.lon.append(p[1][0])
                self.lat.append(p[1][1])
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
    
        # All done
        return

    def readPatchesFromFile(self, filename):
        '''
        Read the patches from a GMT formatted file.
        Args:   
            * filename  : Name of the file.
        '''

        # create the lists
        self.patch = []
        self.patchll = []
        self.index_parameter = []
        self.slip = []

        # open the file
        fin = open(filename, 'r')

        # read all the lines
        A = fin.readlines()

        # depth
        D = 0.0

        # Loop over the file
        i = 0
        while i<len(A):
            
            # Assert it works
            assert A[i].split()[0] is '>', 'Not a patch, reformat your file...'
            # Get the Patch Id
            self.index_parameter.append([np.int(A[i].split()[3]),np.int(A[i].split()[4]),np.int(A[i].split()[5])])
            # Get the slip value
            if len(A[i].split())>7:
                slip = np.array([np.float(A[i].split()[7]), np.float(A[i].split()[8]), np.float(A[i].split()[9])])
            else:
                slip = np.array([0.0, 0.0, 0.0])
            self.slip.append(slip)
            # get the values
            lon1, lat1, z1 = A[i+1].split()
            lon2, lat2, z2 = A[i+2].split()
            lon3, lat3, z3 = A[i+3].split()
            lon4, lat4, z4 = A[i+4].split()
            # Depth
            mm = min([float(z1), float(z2), float(z3), float(z4)])
            if D<mm:
                D=mm
            # Pass as floating point
            lon1 = float(lon1); lat1 = float(lat1); z1 = float(z1)
            lon2 = float(lon2); lat2 = float(lat2); z2 = float(z2)
            # Pass as floating point
            lon1 = float(lon1); lat1 = float(lat1); z1 = float(z1)
            lon2 = float(lon2); lat2 = float(lat2); z2 = float(z2)
            lon3 = float(lon3); lat3 = float(lat3); z3 = float(z3)
            lon4 = float(lon4); lat4 = float(lat4); z4 = float(z4)
            # translate to utm
            x1, y1 = self.ll2xy(lon1, lat1)
            x2, y2 = self.ll2xy(lon2, lat2)
            x3, y3 = self.ll2xy(lon3, lat3)
            x4, y4 = self.ll2xy(lon4, lat4)
            # Set points
            if y1>y2:
                p2 = [x1, y1, z1]; p2ll = [lon1, lat1, z1]
                p1 = [x2, y2, z2]; p1ll = [lon2, lat2, z2]
                p4 = [x3, y3, z3]; p4ll = [lon3, lat3, z3]
                p3 = [x4, y4, z4]; p3ll = [lon4, lat4, z4]
            else:
                p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
                p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
                p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
                p4 = [x4, y4, z4]; p4ll = [lon4, lat4, z4]
            # Store these
            p = [p1, p2, p3, p4]
            pll = [p1ll, p2ll, p3ll, p4ll]
            p = np.array(p)
            pll = np.array(pll)
            # Store these in the lists
            self.patch.append(p)
            self.patchll.append(pll)
            # increase i
            i += 5

        # Close the file
        fin.close()

        # depth
        self.depth = D
        self.z_patches = np.linspace(0,D,5)

        # Translate slip to np.array
        self.slip = np.array(self.slip)
        self.index_parameter = np.array(self.index_parameter)

        # Compute equivalent patches
        self.computeEquivRectangle()

        # All done
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

    def writePatches2File(self, filename, add_slip=None, scale=1.0, patch='normal'):
        '''
        Writes the patch corners in a file that can be used in psxyz.
        Args:
            * filename      : Name of the file.
            * add_slip      : Put the slip as a value for the color. Can be None, strikeslip, dipslip, total.
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
        '''

        # Write something
        print('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        for p in range(len(self.patchll)):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip is 'strikeslip':
                    slp = self.slip[p,0]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'dipslip':
                    slp = self.slip[p,1]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'total':
                    slp = np.sqrt(self.slip[p,0]**2 + self.slip[p,1]**2)*scale
                    string = '-Z{}'.format(slp)

            # Put the parameter number in the file as well if it exists
            parameter = ' ' 
            if hasattr(self,'index_parameter'):
                i = np.int(self.index_parameter[p,0])
                j = np.int(self.index_parameter[p,1])
                k = np.int(self.index_parameter[p,2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            slipstring = ' # {} {} {} '.format(self.slip[p,0], self.slip[p,1], self.slip[p,2])

            # Write the string to file
            fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Write the 4 patch corners (the order is to be GMT friendly)
            if patch in ('normal'):
                p = self.patchll[p]
            elif patch in ('equiv'):
                p = self.equivpatchll[p]
            pp=p[1]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[0]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[3]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[2]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))

        # Close th file
        fout.close()

        # All done 
        return

    def getslip(self, p):
        '''
        Returns the slip vector for a patch.
        '''
        
        # output index
        io = None

        # Find the index of the patch
        for i in range(len(self.patch)):
            if (self.patch[i] == p).all():
                io = i

        # All done
        return self.slip[io,:]

    def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0, neg_depth=False):
        '''
        Write a psxyz compatible file to draw lines starting from the center of each patch, 
        indicating the direction of slip.
        Tensile slip is not used...
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Copmute the slip direction
        self.computeSlipDirection(scale=scale, factor=factor)

        # Write something
        print('Writing slip direction to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        for p in self.slipdirection:
            
            # Write the > sign to the file
            fout.write('> \n')

            # Get the center of the patch
            xc, yc, zc = p[0]
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))

            # Get the end of the vector
            xc, yc, zc = p[1]
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))

        # Close file
        fout.close()

        # all done
        return

    def computeSlipDirection(self, scale=1.0, factor=1.0):
        '''
        Computes the segment indicating the slip direction.
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Create the array
        self.slipdirection = []

        # Loop over the patches
        for p in range(len(self.patch)):  
            
            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)                                   
            # Get the slip vector
            slip = self.getslip(self.patch[p]) 
            rake = np.arctan(slip[1]/slip[0])

            # Compute the vector
            x = np.sin(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake) 
            y = np.cos(strike)*np.cos(rake) - np.sin(strike)*np.cos(dip)*np.sin(rake)
            z = np.sin(dip)*np.sin(rake)
        
            # Scale these
            if scale.__class__ is float:
                sca = scale
            elif scale.__class__ is str:
                if scale is 'total':
                    sca = np.sqrt(slip[0]**2 + slip[1]**2 + slip[2]**2)*factor
                elif scale is 'strikeslip':
                    sca = slip[0]*factor
                elif scale is 'dipslip':
                    sca = slip[1]*factor
                elif scale is 'tensile':
                    sca = slip[2]*factor
                else:
                    print('Unknown Slip Direction in computeSlipDirection')
                    sys.exit(1)
            x *= sca
            y *= sca
            z *= sca
        
            # update point 
            xe = xc + x
            ye = yc + y
            ze = zc + z                                                                          
 
            # Append
            self.slipdirection.append([[xc, yc, zc],[xe, ye, ze]])

        # All done
        return

    def deletepatch(self, patch):
        '''
        Deletes a patch.
        Args:   
            * patch     : index of the patch to remove.
        '''

        # Remove the patch
        del self.patch[patch]
        del self.patchll[patch]
        self.slip = np.delete(self.slip, patch, axis=0)

        # All done
        return

    def deletepatches(self, tutu):
        '''
        Deletes a list of patches.
        '''

        while len(tutu)>0:

            # Get index to delete
            i = tutu.pop()

            # delete it
            self.deletepatch(i)

            # Upgrade list
            for u in range(len(tutu)):
                if tutu[u]>i:
                    tutu[u] -= 1

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

    def computeArea(self):
        '''
        Computes the area of all rectangles.
        '''

        # Area
        self.area = []

        # Loop
        for p in self.equivpatch:

            # get points
            p1 = p[0]
            p2 = p[1]
            p3 = p[2]

            # computes distances
            d1 = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )
            d2 = np.sqrt( (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2 + (p3[2]-p2[2])**2 )
            self.area.append(d1*d2)

        # all done
        return

    def computeEquivRectangle(self, strikeDir='south'):
        '''
        Computes the equivalent rectangle patches and stores these into self.equivpatch
        '''
        
        # Initialize the equivalent structure
        self.equivpatch = []
        self.equivpatchll = []

        # Loop on the patches
        for u in range(len(self.patch)):
            p = self.patch[u]
            p1, p2, p3, p4 = self.patch[u]
            # 1. Get the two top points
            pt1 = p[0]; x1, y1, z1 = pt1 
            pt2 = p[1]; x2, y2, z2 = pt2
            # 2. Get the azimuth of this patch
            az = np.arctan2(y2-y1, x2-x1)
            # 3. Get the dip of this patch 
            dip1 = np.arcsin((p4[2] - p1[2]) / np.sqrt((p1[0] - p4[0])**2 
                           + (p1[1] - p4[1])**2 + (p1[2] - p4[2])**2))
            dip2 = np.arcsin((p3[2] - p2[2]) / np.sqrt( (p2[0] - p3[0])**2 
                           + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2))
            dip = 0.5 * (dip1 + dip2)
            # 4. compute the position of the bottom corners  
            width = np.sqrt((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2 + (p1[2] - p4[2])**2)
            wc = width * np.cos(dip)
            ws = width * np.sin(dip)
            halfPi = 0.5 * np.pi
            x3 = x2 + wc * np.cos(az + halfPi)
            y3 = y2 + wc * np.sin(az + halfPi)
            z3 = z2 + ws
            x4 = x1 + wc * np.cos(az + halfPi)
            y4 = y1 + wc * np.sin(az + halfPi)
            z4 = z1 + ws
            pt3 = [x3, y3, z3]
            pt4 = [x4, y4, z4]
            # set up the patch
            self.equivpatch.append(np.array([pt1, pt2, pt3, pt4]))
            # Deal with the lon lat
            lon1, lat1 = self.putm(x1*1000., y1*1000., inverse=True)
            lon2, lat2 = self.putm(x2*1000., y2*1000., inverse=True)
            lon3, lat3 = self.putm(x3*1000., y3*1000., inverse=True)
            lon4, lat4 = self.putm(x4*1000., y4*1000., inverse=True)
            pt1 = [lon1, lat1, z1]
            pt2 = [lon2, lat2, z2]
            pt3 = [lon3, lat3, z3]
            pt4 = [lon4, lat4, z4]
            # set up the patchll
            self.equivpatchll.append(np.array([pt1, pt2, pt3, pt4]))

        # All done
        return

    def getpatchgeometry(self, patch, center=False):
        '''
        Returns the patch geometry as needed for okada85.
        Args:
            * patch         : index of the wanted patch or patch;
            * center        : if true, returns the coordinates of the center of the patch. 
                              if False, returns the UL corner.

        When we build the fault, the patches are not exactly rectangular. Therefore, 
        this routine will return the rectangle that matches with the two shallowest 
        points and that has an average dip angle to match with the other corners.
        '''

        # Get the patch
        if patch.__class__ is int:
            u = patch
        else:
            u = [self.patch == patch]

        # Get the four corners of the rectangle
        p1, p2, p3, p4 = self.equivpatch[u]

        # Get the UL corner of the patch
        if center:
            x1, x2, x3 = self.getcenter(self.equivpatch[u])
        else:
            x1 = p2[0]
            x2 = p2[1]
            x3 = p2[2]

        # Get the patch width (this fault is vertical for now)
        width = np.sqrt( (p4[0] - p1[0])**2 + (p4[1] - p1[1])**2 + (p4[2] - p1[2])**2 )   

        # Get the length
        length = np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

        # Get the strike
        strike = np.arctan2( (p1[0] - p2[0]),(p1[1] - p2[1]) )

        # Set the dip
        dip = np.arcsin( (p4[2] - p1[2])/width )

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

        print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))

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
        elif data.dtype is 'cosicorrrates':
            Ndt = 2 * Nd
            data.obs_per_station = 2
            if vertical:
                Ndt += Nd
                data.obs_per_station += 1

        # Get the number of parameters
        Np = len(self.patch)
        Npt = len(self.patch)*len(slipdir)

        # Initializes a space in the dictionary to store the green's function
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]
        if 's' in slipdir:
            G['strikeslip'] = np.zeros((Ndt, Np))
        if 'd' in slipdir:
            G['dipslip'] = np.zeros((Ndt, Np))
        if 't' in slipdir:
            G['tensile'] = np.zeros((Ndt, Np))

        # Initializes the data vector and the data covariance
        if data.dtype is 'insarrates':
            self.d[data.name] = data.vel
            vertical = True                 # In InSAR, you need to use the vertical, no matter what....
        elif data.dtype is 'gpsrates':
            if vertical:
                self.d[data.name] = data.vel_enu.T.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
        elif data.dtype is 'cosicorrrates':
            self.d[data.name] = np.hstack((data.east.flatten(), data.north.flatten()))
            if vertical:
                self.d[data.name] = np.hstack((self.d[data.name], np.zeros((Nd,))))
            assert self.d[data.name].shape[0] == Ndt, 'd vector and numObs do not match'

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
            sys.stdout.write('\r Patch: {} / {} '.format(p+1,len(self.patch)))
            sys.stdout.flush()
            
            # get the surface displacement corresponding to unit slip
            # ss,ds,op will all have shape (Nd,3) for 3 components
            ss, ds, op = self.slip2dis(data, p, slip=SLP)

            # Do we keep the verticals
            if not vertical:
                # Just get horizontal components
                ss = ss[:,0:2]
                ds = ds[:,0:2]
                op = op[:,0:2]

            # Organize the response
            if data.dtype in ['gpsrates', 'cosicorrrates']:
                # If GPS type, construct a flat vector with east displacements first, then
                # north, then vertical
                ss = ss.T.flatten()
                ds = ds.T.flatten()
                op = op.T.flatten()

            elif data.dtype is 'insarrates':
                # If InSAR, do the dot product with the los
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

    def saveGFs(self, dtype='d', outputDir='.'):
        '''
        Saves the Green's functions in different files
        Args:
            dtype       : Format of the binary data saved.
            outputDir   : Directory to save binary data.
        '''

        # Print stuff
        print('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys(): 

            # Get the Green's function
            G = self.G[data]

            # StrikeSlip Component
            if 'strikeslip' in G.keys():
                gss = G['strikeslip'].flatten()
                filename = '{}_{}_SS.gf'.format(self.name, data)
                gss = gss.astype(dtype)
                gss.tofile(os.path.join(outputDir, filename))

            # DipSlip Component
            if 'dipslip' in G.keys():
                gds = G['dipslip'].flatten()
                filename = '{}_{}_DS.gf'.format(self.name, data)
                gds = gds.astype(dtype)
                gds.tofile(os.path.join(outputDir, filename))

            # Tensile
            if 'tensile' in G.keys():
                gts = G['tensile'].flatten()
                filename = '{}_{}_TS.gf'.format(self.name, data)
                gts = gts.astype(dtype)
                gts.tofile(os.path.join(outputDir, filename))

        # All done
        return

    def setGFsFromFile(self, data, strikeslip=None, dipslip=None, tensile=None, vertical=False, dtype='d'):
        '''
        Sets the Green's functions from binary files. Be carefull, these have to be in the 
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and 
        if insar, GF are projected already)
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : File containing the Green's functions for strikeslip displacements.
            * dipslip       : File containing the Green's functions for dipslip displacements.
            * tensile       : File containing the Green's functions for tensile displacements.
            * vertical      : Deal with the UP component (gps: default is false, insar, it will be true anyway).
            * dtype         : Type of binary data.
        '''

        print('---------------------------------')
        print('---------------------------------')
        print("Set up Green's functions for fault {} from files {}, {} and {}".format(self.name, strikeslip, dipslip, tensile))

        # Get the number of patches
        Npatchs = len(self.patch)

        # Read the files and reshape the GFs
        Gss = None; Gds = None; Gts = None
        if strikeslip is not None:
            Gss = np.fromfile(strikeslip, dtype=dtype)
            ndl = int(Gss.shape[0]/Npatchs)
            Gss = Gss.reshape((ndl, Npatchs))
        if dipslip is not None:
            Gds = np.fromfile(dipslip, dtype=dtype)
            ndl = int(Gds.shape[0]/Npatchs)
            Gds = Gds.reshape((ndl, Npatchs))
        if tensile is not None:
            Gts = np.fromfile(tensile, dtype=dtype)
            ndl = int(Gts.shape[0]/Npatchs)
            Gts = Gts.reshape((ndl, Npatchs))

        # Get the data type
        datatype = data.dtype

        # Cut the Matrices following what data do we have and set the GFs
        if datatype is 'gpsrates':
         
            # Initialize
            GssE = None; GdsE = None; GtsE = None
            GssN = None; GdsN = None; GtsN = None
            GssU = None; GdsU = None; GtsU = None

            # Get the values
            if strikeslip is not None:
                GssE = Gss[range(0,data.vel_enu.shape[0]),:]
                GssN = Gss[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GssU = Gss[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]
            if dipslip is not None:
                GdsE = Gds[range(0,data.vel_enu.shape[0]),:]
                GdsN = Gds[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GdsU = Gds[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]
            if tensile is not None:
                GtsE = Gts[range(0,data.vel_enu.shape[0]),:]
                GtsN = Gts[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GtsU = Gts[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]

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
               tensile=[None, None, None], vertical=False):
        '''
        Stores the Green's functions matrices into the fault structure.
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : List of matrices of the Strikeslip Green's functions, ordered E, N, U
            * dipslip       : List of matrices of the dipslip Green's functions, ordered E, N, U
            * tensile       : List of matrices of the tensile Green's functions, ordered E, N, U
            If you provide InSAR GFs, these need to be projected onto the LOS direction already.
        '''

        # Get the number of data per point
        if data.dtype is 'insarrates':
            data.obs_per_station = 1
        elif data.dtype is 'gpsrates':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station = 3
        elif data.dtype is 'cosicorrrates':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
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
        elif data.dtype is 'cosicorrrates':
            self.d[data.name] = np.hstack((data.east.T.flatten(), data.north.T.flatten()))
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
                ds = np.array(ds)
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
                              'full' -> Only for GPS, estimates a rotation, translation and scaling with 
                                        respect to the center of the network (Helmert transform).
            * slipdir       : which directions of slip to include. can be any combination of s, d and t.
        '''

        # print
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
                        print('Data type must be gpsrates to implement a Helmert transform')
                        return
        elif polys.__class__ is list:
            for d in range(len(datas)):
                if polys[d].__class__ is not str:
                    self.poly[datas[d].name] = polys[d]*datas[d].obs_per_station
                else:
                    if datas[d].dtype is 'gpsrates':
                        self.poly[datas[d].name] = polys[d]
                    else:
                        print('Data type must be gpsrates to implement a Helmert transform')
                        return

        # Get the number of parameters
        N = len(self.patch)
        Nps = N*len(slipdir)
        Npo = 0
        for data in datas :
            if self.poly[data.name] is 'full':
                if not hasattr(self, 'helmert'):
                    self.helmert = {}
                if data.obs_per_station==3:
                    Npo += 7                    # 3D Helmert transform is 7 parameters
                    self.helmert[data.name] = 7
                else:   
                    Npo += 4                    # 2D Helmert transform is 4 parameters
                    self.helmert[data.name] = 4
            elif self.poly[data.name] is 'strain':
                if not hasattr(self, 'strain'):
                    self.strain = {}
                if data.obs_per_station==2:
                    Npo += 6
                    self.strain[data.name] = 6
                else:
                    print('3d strain has not been implemented')
                    return
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
            print("Dealing with {} of type {}".format(data.name, data.dtype))

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
            if self.poly[data.name].__class__ is not str:
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
                        orb[:,0] = 1.0
                        if self.poly[data.name] >= 3:
                            # Compute normalizing factors
                            if not hasattr(self, 'OrbNormalizingFactor'):
                                self.OrbNormalizingFactor = {}
                            self.OrbNormalizingFactor[data.name] = {}
                            x0 = data.x[0]
                            y0 = data.y[0]
                            normX = np.abs(data.x - x0).max()
                            normY = np.abs(data.y - y0).max()
                            # Save them for later
                            self.OrbNormalizingFactor[data.name]['x'] = normX
                            self.OrbNormalizingFactor[data.name]['y'] = normY
                            self.OrbNormalizingFactor[data.name]['ref'] = [x0, y0]
                            # Fill in functionals
                            orb[:,1] = (data.x - x0) / normX
                            orb[:,2] = (data.y - y0) / normY
                        if self.poly[data.name] >= 4:
                            orb[:,3] = orb[:,1] * orb[:,2]
                        # Scale everything by the data factor
                        orb *= data.factor

                    elif data.dtype is 'cosicorrrates':

                        basePoly = self.poly[data.name] / data.obs_per_station
                        assert basePoly == 3 or basePoly == 6, """
                            only support 3rd or 4th order poly for cosicorr
                            """

                        # In case vertical is True, make sure we only include polynomials
                        # for horizontals
                        if basePoly == 3:
                            self.poly[data.name] = 6
                        else:
                            self.poly[data.name] = 12

                        # Compute normalizing factors
                        numPoints = Ndlocal // data.obs_per_station
                        if not hasattr(self, 'OrbNormalizingFactor'):
                            self.OrbNormalizingFactor = {}
                        x0 = data.x[0]
                        y0 = data.y[0]
                        normX = np.abs(data.x - x0).max()
                        normY = np.abs(data.y - y0).max()
                        # Save them for later
                        self.OrbNormalizingFactor[data.name] = {}
                        self.OrbNormalizingFactor[data.name]['ref'] = [x0, y0]
                        self.OrbNormalizingFactor[data.name]['x'] = normX
                        self.OrbNormalizingFactor[data.name]['y'] = normY

                        # Pre-compute position-dependent functional forms
                        f1 = data.factor * np.ones((numPoints,))
                        f2 = data.factor * (data.x - x0) / normX
                        f3 = data.factor * (data.y - y0) / normY
                        f4 = data.factor * (data.x - x0) * (data.y - y0) / (normX*normY)
                        f5 = data.factor * (data.x - x0)**2 / normX**2
                        f6 = data.factor * (data.y - y0)**2 / normY**2
                        polyFuncs = [f1, f2, f3, f4, f5, f6]

                        # Fill in orb matrix given an order
                        orb = np.zeros((numPoints, basePoly))
                        for ind in xrange(basePoly):
                            orb[:,ind] = polyFuncs[ind]

                        # Block diagonal for both components
                        orb = block_diag(orb, orb)

                        # Check to see if we're including verticals
                        if data.obs_per_station == 3:
                            orb = np.vstack((orb, np.zeros((numPoints, 2*basePoly))))

                    # Put it into G for as much observable per station we have
                    polend = polstart + self.poly[data.name]
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += self.poly[data.name]

            else:
                if self.poly[data.name] is 'full':
                    orb = self.getHelmertMatrix(data)
                    if data.obs_per_station==3:
                        nc = 7
                    elif data.obs_per_station==2:
                        nc = 4
                    # Put it into G for as much observable per station we have
                    polend = polstart + nc
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += nc
                if self.poly[data.name] is 'strain':
                    orb = self.get2DstrainEst(data)
                    if data.obs_per_station == 2:
                        nc = 6
                    polend = polstart + nc
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += nc
            # Update el to check where we are
            el = el + Ndlocal
            
        # Store G in self
        self.Gassembled = G

        # All done
        return

    def get2DstrainEst(self, data):
        '''
        Returns the matrix to estimate the full 2d strain tensor.
        '''

        # Check
        assert (data.dtype is 'gpsrates')

        # Get the number of gps stations
        ns = data.station.shape[0]

        # Get the data vector size
        nd = self.d[data.name].shape[0]

        # Get the number of parameters to look for
        if data.obs_per_station==2:
            nc = 6
        else:
            print('Not implemented')
            return

        # Check something
        assert data.obs_per_station*ns==nd

        # Get the center of the network
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)

        # Compute the baselines
        base_x = data.x - x0
        base_y = data.y - y0

        # Normalize the baselines 
        base_max = np.max([np.abs(base_x).max(), np.abs(base_y).max()])
        base_x /= base_max
        base_y /= base_max

        # Store the normalizing factor
        if not hasattr(self, 'StrainNormalizingFactor'):
            self.StrainNormalizingFactor = {}
        self.StrainNormalizingFactor[data.name] = base_max

        # Allocate a Base
        H = np.zeros((data.obs_per_station,nc))

        # Put the transaltion in the base
        H[:,:data.obs_per_station] = np.eye(data.obs_per_station)

        # Allocate the full matrix
        Hf = np.zeros((nd,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,data.obs_per_station:] = 0.0 

            # Get the values
            x1, y1 = base_x[i], base_y[i]

            # Store them
            H[0,2] = x1
            H[0,3] = 0.5*y1
            H[0,5] = 0.5*y1
            H[1,3] = 0.5*x1
            H[1,4] = y1
            H[1,5] = -0.5*y1

            # Put the lines where they should be
            Hf[i,:] = H[0,:]
            Hf[i+ns,:] = H[1,:]

        # All done
        return Hf

    def getHelmertMatrix(self, data):
        '''
        Returns a Helmert matrix for a gps data set.
        '''

        # Check
        assert (data.dtype is 'gpsrates')

        # Get the number of stations
        ns = data.station.shape[0]

        # Get the data vector size
        nd = self.d[data.name].shape[0]

        # Get the number of helmert transform parameters
        if data.obs_per_station==3:
            nc = 7
        else:
            nc = 4

        # Check something
        assert data.obs_per_station*ns==nd

        # Get the position of the center of the network
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)
        z0 = 0              # We do not deal with the altitude of the stations yet (later)

        # Compute the baselines
        base_x = data.x - x0
        base_y = data.y - y0
        base_z = 0

        # Normalize the baselines
        base_x_max = np.abs(base_x).max(); base_x /= base_x_max
        base_y_max = np.abs(base_y).max(); base_y /= base_y_max

        # Allocate a Helmert base
        H = np.zeros((data.obs_per_station,nc))
        
        # put the translation in it (that part never changes)
        H[:,:data.obs_per_station] = np.eye(data.obs_per_station)

        # Allocate the full matrix
        Hf = np.zeros((nd,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,data.obs_per_station:] = 0.0

            # Put the rotation components and the scale components
            x1, y1, z1 = base_x[i], base_y[i], base_z
            if nc==7:
                H[:,3:6] = np.array([[0.0, -z1, y1],
                                     [z1, 0.0, -x1],
                                     [-y1, x1, 0.0]])
                H[:,7] = np.array([x1, y1, z1])
            else:
                H[:,2] = np.array([y1, -x1])
                H[:,3] = np.array([x1, y1])

            # put the lines where they should be
            Hf[i,:] = H[0]
            Hf[i+ns,:] = H[1]
            if nc==7:
                Hf[i+2*ns,:] = H[2]

        # all done 
        return Hf

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

    def assembleCd(self, datas, add_prediction=None):
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
        extra_params is a list of extra parameters.
        '''
    
        # Get the number of slip directions
        slipdir = len(self.slipdir)
        patch = len(self.patch)

        # Number of parameters
        Np = patch*slipdir
        if extra_params is not None:
            Np += len(extra_params)

        # Create Cm
        Cm = np.zeros((Np, Np))

        # Loop over slip dir
        for i in range(slipdir):
            Cmt = np.diag(sigma[i] * np.ones(len(self.patch),))
            Cm[i*patch:(i+1)*patch,i*patch:(i+1)*patch] = Cmt

        # Put the extra parameter sigma values
        st = patch*slipdir
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Stores Cm
        self.Cm = Cm

        # all done
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
            xd = (np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )
        print ("Lambda0 = {}".format(lam0))
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
        print ("Write the EDKS files for fault {} and data {}".format(self.name, data.name))

        # Write the geometry to the EDKS file
        self.writeEDKSgeometry()

        # Write the data to the EDKS file
        data.writeEDKSdata()

        # Create the variables
        if len(self.name.split())>1:
            fltname = self.name.split()[0]
            for s in self.name.split()[1:]:
                fltname = fltname+'_'+s
        else:
            fltname = self.name
        RectanglePropFile = 'edks_{}.END'.format(fltname)
        if len(data.name.split())>1:
            datname = data.name.split()[0]
            for s in data.name.split()[1:]:
                datname = datname+'_'+s
        else:
            datname = data.name
        ReceiverFile = 'edks_{}.idEN'.format(datname)

        if data.dtype is 'insarrates':
            useRecvDir = True # True for InSAR, uses LOS information
        else:
            useRecvDir = False # False for GPS, uses ENU displacements
        EDKSunits = 1000.0
        EDKSfilename = '{}'.format(edksfilename)
        prefix = 'edks_{}_{}'.format(fltname, datname)
        plotGeometry = '{}'.format(plot)

        # Open the EDKSsubParams.py file
        filename = 'EDKSParams_{}_{}.py'.format(fltname, datname)
        fout = open(filename, 'w')

        # Write in it
        fout.write("# File with the rectangles properties\n")
        fout.write("RectanglesPropFile = '{}'\n".format(RectanglePropFile))
        fout.write("# File with id, E[km], N[km] coordinates of the receivers.\n")
        fout.write("ReceiverFile = '{}'\n".format(ReceiverFile))
        fout.write("# read receiver direction (# not yet implemented)\n")
        fout.write("useRecvDir = {} # True for InSAR, uses LOS information\n".format(useRecvDir))
        fout.write("# Maximum Area to subdivide triangles. If None, uses Saint-Venant's principle.\n")
        if amax is None:
            fout.write("Amax = None # None computes Amax automatically. \n")
        else:
            fout.write("Amax = {} # Minimum size for the patch division.\n".format(amax))

        fout.write("EDKSunits = 1000.0 # to convert from kilometers to meters\n")
        fout.write("EDKSfilename = '{}'\n".format(edksfilename))
        fout.write("prefix = '{}'\n".format(prefix))
        fout.write("plotGeometry = {} # set to False if you are running in a remote Workstation\n".format(plot))
        
        # Close the file
        fout.close()

        # Build usefull outputs
        parNames = ['useRecvDir', 'Amax', 'EDKSunits', 'EDKSfilename', 'prefix']
        parValues = [ useRecvDir ,  amax ,  EDKSunits ,  EDKSfilename ,  prefix ]
        method_par = dict(zip(parNames, parValues))

        # All done
        return filename, RectanglePropFile, ReceiverFile, method_par

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
        if len(self.name.split())>1:
            fltname = self.name.split()[0]
            for s in self.name.split()[1:]:
                fltname = fltname+'_'+s
        else:
            fltname = self.name
        print(fltname)
        filename = 'edks_{}'.format(fltname)

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
            x, y, z, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            lon, lat = self.xy2ll(x,y)
            if ref is not None:
                x -= refx
                y -= refy
            flld.write('{} {} {} {} {} {} {} {:5d} \n'.format(lon,lat,z,strike,dip,length,width,p))
            fend.write('{} {} {} {} {} {} {} {:5d} \n'.format(x,y,z,strike,dip,length,width,p))

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
        patch = self.equivpatch

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
        Get the center of one rectangular patch.
        Args:
            * p    : Patch geometry.
        '''
    
        # Get center
        p1, p2, p3, p4 = p

        # Compute the center
        x = p1[0] + (p3[0] - p1[0])/2.
        y = p1[1] + (p3[1] - p1[1])/2.
        z = p1[2] + (p3[2] - p1[2])/2.

        # All done
        return x,y,z

    def surfacesimulation(self, box=None, disk=None, err=None, npoints=None, lonlat=None,
                          slipVec=None):
        ''' 
        Takes the slip vector and computes the surface displacement that corresponds on a regular grid.
        Args:
            * box       : Can be a list of [minlon, maxlon, minlat, maxlat].
            * disk      : list of [xcenter, ycenter, radius, n]
            * lonlat    : Arrays of lat and lon. [lon, lat]
        '''

        # create a fake gps object
        from .gpsrates import gpsrates
        self.sim = gpsrates('simulation', utmzone=self.utmzone)

        # Create a lon lat grid
        if lonlat is None:
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
        else:
            lon = np.array(lonlat[0])
            lat = np.array(lonlat[1])

        # Clean it
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
            name = '{:04d}'.format(i)
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

        # Load the slip values if provided
        if slipVec is not None:
            nPatches = len(self.patch)
            print(nPatches, slipVec.shape)
            assert slipVec.shape == (nPatches,3), 'mismatch in shape for input slip vector'
            self.slip = slipVec

        # Loop over the patches
        for p in range(len(self.patch)):
            sys.stdout.write('\r Patch {} / {} '.format(p+1,len(self.patch)))
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

    def plot(self,ref='utm', figure=134, add=False, maxdepth=None, axis='equal', value_to_plot='total', equiv=False):
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

        if add and (ref is 'utm'):
            for fault in self.addfaultsxy:
                ax.plot(fault[:,0], fault[:,1], '-k')
        elif add and (ref is not 'utm'):
            for fault in self.addfaults:
                ax.plot(fault[:,0], fault[:,1], '-k')

        # Plot the discretized trace
        if self.xi is not None:
            if ref is 'utm':
                ax.plot(self.xi, self.yi, '.r')
            else:
                if self.loni is None:
                    self.loni, self.lati = self.putm(self.xi*1000., self.yi*1000., inverse=True)
                ax.plot(loni, lati, '.r')

        # Compute the total slip
        if value_to_plot=='total':
            self.computetotalslip()
            plotval = self.totalslip
        elif value_to_plot=='index':
            plotval = np.linspace(0, len(self.patch)-1, len(self.patch))

        # Plot the patches
        if self.patch is not None:
            
            # import stuff
            import mpl_toolkits.mplot3d.art3d as art3d
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            
            # set z axis
            ax.set_zlim3d([-1.0*(self.depth+5), 0])
            zticks = []
            zticklabels = []
            for z in self.z_patches:
                zticks.append(-1.0*z)
                zticklabels.append(z)
            ax.set_zticks(zticks)
            ax.set_zticklabels(zticklabels)
            
            # set color business
            cmap = plt.get_cmap('jet')
            cNorm  = colors.Normalize(vmin=0, vmax=plotval.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

            for p in range(len(self.patch)):
                ncorners = len(self.patch[0])
                x = []
                y = []
                z = []
                for i in range(ncorners):
                    if ref is 'utm':
                        x.append(self.patch[p][i][0])
                        y.append(self.patch[p][i][1])
                        z.append(-1.0*self.patch[p][i][2])
                    else:
                        x.append(self.patchll[p][i][0])
                        y.append(self.patchll[p][i][1])
                        z.append(-1.0*self.patchll[p][i][2])
                verts = [zip(x, y, z)]
                rect = art3d.Poly3DCollection(verts)
                rect.set_color(scalarMap.to_rgba(plotval[p]))
                rect.set_edgecolors('k')
                ax.add_collection3d(rect)

            if equiv:
                for p in range(len(self.equivpatch)): 
                    ncorners = len(self.equivpatch[0])                                                                
                    x = []                                                                                       
                    y = []
                    z = []
                    for i in range(ncorners):                                                                    
                        if ref is 'utm':
                            x.append(self.equivpatch[p][i][0])                                                        
                            y.append(self.equivpatch[p][i][1])                                                        
                            z.append(-1.0*self.equivpatch[p][i][2])                                                   
                        else: 
                            x.append(self.equivpatchll[p][i][0])                                                      
                            y.append(self.equivpatchll[p][i][1])                                                      
                            z.append(-1.0*self.equivpatchll[p][i][2])
                    verts = [zip(x, y, z)]
                    rect = art3d.Poly3DCollection(verts)                                                         
                    rect.set_color(scalarMap.to_rgba(plotval[p]))
                    rect.set_edgecolors('r')                                                                     
                    ax.add_collection3d(rect)               

            # put up a colorbar        
            scalarMap.set_array(plotval)
            plt.colorbar(scalarMap)

        # Depth
        if maxdepth is not None:
            ax.set_zlim3d([-1.0*maxdepth, 0])

        # show
        plt.show()

        # All done
        return

