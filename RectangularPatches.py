'''
A parent class that deals with rectangular patches fault

Started by R. Jolivet, November 2013

Main contributors:
R. Jolivet, CalTech, USA
Z. Duputel, Univ. de Strasbourg, France,
B. Riel, CalTech, USA
F. Ortega-Culaciati, Univ. de Santiago, Chile
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys
import os

# Personals
from .Fault import Fault
from .stressfield import stressfield
from . import okadafull

class RectangularPatches(Fault):
    
    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''
        
        # Base class init
        super(RectangularPatches,self).__init__(name,utmzone,ellps, verbose=verbose)

        # Specify the type of patch
        self.patchType = 'rectangle'

        # Allocate depth and number of patches
        self.numz = None            # Number of patches along dip

        # All done
        return


    def setdepth(self, nump, width, top=0):
        '''
        Set depth patch attributes

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
        print ("Extrapolating the fault for {} km".format(length_added))

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
        self.loni, self.lati = self.xy2ll(self.xi, self.yi)

        # All done
        return

    def splitPatch(self, patch):
        '''
        Splits a patch in 4 patches and returns 4 new patches.
        Args:
            * patch         : item of the list of patch
        '''

        # Gets the 4 corners
        c1, c2, c3, c4 = patch
        c1 = c1.tolist()
        c2 = c2.tolist()
        c3 = c3.tolist()
        c4 = c4.tolist()

        # Compute the center
        xc, yc, zc = self.getcenter(patch)
        center = [xc, yc, zc]

        # Compute middle of segments
        c12 = [c1[0] + (c2[0]-c1[0])/2.,
               c1[1] + (c2[1]-c1[1])/2.,
               c1[2] + (c2[2]-c1[2])/2.]
        c23 = [c2[0] + (c3[0]-c2[0])/2.,
               c2[1] + (c3[1]-c2[1])/2.,
               c2[2] + (c3[2]-c2[2])/2.]
        c34 = [c3[0] + (c4[0]-c3[0])/2.,
               c3[1] + (c4[1]-c3[1])/2.,
               c3[2] + (c4[2]-c3[2])/2.]
        c41 = [c4[0] + (c1[0]-c4[0])/2.,
               c4[1] + (c1[1]-c4[1])/2.,
               c4[2] + (c1[2]-c4[2])/2.]

        # make patches
        p1 = np.array([c1, c12, center, c41])
        p2 = np.array([c12, c2, c23, center])
        p3 = np.array([center, c23, c3, c34])
        p4 = np.array([c41, center, c34, c4])

        # All done
        return p1, p2, p3, p4

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

    def surfacePatches2Trace(self):
        '''
        Takes the surface patches and set the fault trace with that
        '''

        # Create lists
        xf = []
        yf = []

        # Iterate on patches
        for p in self.patch:
            for c in p:
                if c[2]==0.0:
                    xf.append(c[0])
                    yf.append(c[1])
        
        # Make arrays
        xf = np.array(xf)
        yf = np.array(yf)

        # Compute lonlat
        lonf, latf = self.xy2ll(xf, yf)

        # Set values
        self.xf = xf
        self.yf = yf
        self.lonf = lonf
        self.latf = latf

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
            self.area.append(self.patchArea(p))

        # all done
        return

    def patchArea(self, p):
        ''' 
        Computes the area of one patch.
        Args:
            * p      : One item of self.patch
        '''

        # get points
        p1 = p[0]
        p2 = p[1]
        p3 = p[2]

        # computes distances
        d1 = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )
        d2 = np.sqrt( (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2 + (p3[2]-p2[2])**2 )
        area = d1*d2

        # All done
        return area

    def patchesUtm2LonLat(self):
        '''
        Perform the utm to lonlat conversion for all patches.
        '''

        # iterate over the patches
        for i in range(len(self.patch)):

            # Get the patch
            patch = self.patch[i]

            # Get x, y
            x1 = patch[0][0]
            y1 = patch[0][1]
            x2 = patch[1][0]
            y2 = patch[1][1]
            zu = patch[0][2]
            zd = patch[2][2]

            # Convert
            lon1, lat1 = self.xy2ll(x1, y1)
            lon2, lat2 = self.xy2ll(x2, y2)

            # Build a ll patch
            patchll = [ [lon1, lat1, zu],
                        [lon2, lat2, zu], 
                        [lon2, lat2, zd], 
                        [lon1, lat1, zd] ]

            # Put it in the list
            self.patchll[i] = patchll

        # All done
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

    def lonlat2patch(self, lon, lat, depth, strike, dip, length, width):
        '''
        From a list of arguments, returns a patch and a patchll.
        Args:
            * lon       : Longitude of the center of the patch
            * lat       : Latitude of the center of the patch
            * depth     : Depth of the center of the fault (km)
            * strike    : Strike of the fault (degree)
            * dip       : Dip of the fault (degree)
            * length    : Length of the fault (km)
            * width     : Width of the fault (km)
        '''

        # Create lists
        patch = []
        patchll = []

        # Convert angles
        dip *= np.pi/180.
        strike *= np.pi/180.

        # Convert Lon Lat to X Y
        xc, yc = self.ll2xy(lon,lat)
        zc = -1.0*depth

        # Calculate the center of the upper segment
        xcu = xc - width/2.*np.cos(dip)*np.cos(strike)
        ycu = yc + width/2.*np.cos(dip)*np.sin(strike)
        zcu = zc + width/2.*np.sin(dip)

        # Calculate the center of the lower segment
        xcd = xc + width/2.*np.cos(dip)*np.cos(strike)
        ycd = yc - width/2.*np.cos(dip)*np.sin(strike)
        zcd = zc - width/2.*np.sin(dip)
        
        # Calculate the 2 upper corners
        x1 = xcu - length/2.*np.sin(strike)
        y1 = ycu - length/2.*np.cos(strike)
        z1 = zcu
        p1 = [x1, y1, -1.0*z1]
        lon1, lat1 = self.xy2ll(x1, y1)
        pll1 = [lon1, lat1, -1.0*z1]

        x2 = xcu + length/2.*np.sin(strike)
        y2 = ycu + length/2.*np.cos(strike)
        z2 = zcu
        p2 = [x2, y2, -1.0*z2]
        lon2, lat2 = self.xy2ll(x2, y2)
        pll2 = [lon2, lat2, -1.0*z2]

        # Calculate the 2 lower corner
        x3 = xcd + length/2.*np.sin(strike)
        y3 = ycd + length/2.*np.cos(strike)
        z3 = zcd
        p3 = [x3, y3, -1.0*z3]
        lon3, lat3 = self.xy2ll(x3, y3)
        pll3 = [lon3, lat3, -1.0*z3]

        x4 = xcd - length/2.*np.sin(strike)
        y4 = ycd - length/2.*np.cos(strike)
        z4 = zcd
        p4 = [x4, y4, -1.0*z4]
        lon4, lat4 = self.xy2ll(x4, y4)
        pll4 = [lon4, lat4, -1.0*z4]

        # Set up patch
        patch = [p1, p2, p3, p4]
        patchll = [pll1, pll2, pll3, pll4]

        # All done
        return patch, patchll

    def geometry2patch(self, Lon, Lat, Depth, Strike, Dip, Length, Width, initializeSlip=True):
        '''
        Builds the list of patches from lists of lon, lat, depth...
        Args:
            * Lon       : List of longitudes
            * Lat       : List of Latitudes
            * Depth     : List of depths
            * Strike    : List of strike angles (degree)
            * Dip       : List of dip angles (degree)
            * Length    : List of length (km)
            * Width     : List of width (km)
        '''

        # Create the patch lists
        self.patch = []
        self.patchll = []

        # Iterate
        for lon, lat, depth, strike, dip, length, width in zip(Lon, Lat, Depth, Strike, Dip, Length, Width):
            patch, patchll = self.lonlat2patch(lon, lat, depth, strike, dip, length, width)
            self.patch.append(patch)
            self.patchll.append(patchll)
        
        # Initialize Slip
        if initializeSlip:
            self.initializeslip()

        # Depth things
        depth = np.unique([[p[2] for p in patch] for patch in self.patch])
        self.z_patches = depth.tolist()
        self.top = np.min(depth)
        self.depth = np.max(depth)

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

    def readPatchesFromFile(self, filename, Cm=None, readpatchindex=True, inputCoordinates='lonlat', donotreadslip=False):
        '''
        Read the patches from a GMT formatted file.
        Args:   
            * filename  : Name of the file.
        '''

        # create the lists
        self.patch = []
        self.patchll = []
        if readpatchindex:
            self.index_parameter = []
        self.Cm   = []
        if not donotreadslip:
            self.slip = []

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
        i = 0
        while i<len(A):
            
            # Assert it works
            assert A[i].split()[0] is '>', 'Not a patch, reformat your file...'
            # Get the Patch Id
            if readpatchindex:
                self.index_parameter.append([np.int(A[i].split()[3]),np.int(A[i].split()[4]),np.int(A[i].split()[5])])
            # Get the slip value
            if not donotreadslip:
                if len(A[i].split())>7:
                    slip = np.array([np.float(A[i].split()[7]), np.float(A[i].split()[8]), np.float(A[i].split()[9])])
                else:
                    slip = np.array([0.0, 0.0, 0.0])
                self.slip.append(slip)
            # get the values
            if inputCoordinates in ('lonlat'):
                lon1, lat1, z1 = A[i+1].split()
                lon2, lat2, z2 = A[i+2].split()
                lon3, lat3, z3 = A[i+3].split()
                lon4, lat4, z4 = A[i+4].split()
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
            elif inputCoordinates in ('xyz'):
                x1, y1, z1 = A[i+1].split()
                x2, y2, z2 = A[i+2].split()
                x3, y3, z3 = A[i+3].split()
                x4, y4, z4 = A[i+4].split()
                # Pass as floating point
                x1 = float(x1); y1 = float(y1); z1 = float(z1)
                x2 = float(x2); y2 = float(y2); z2 = float(z2)
                x3 = float(x3); y3 = float(y3); z3 = float(z3)
                x4 = float(x4); y4 = float(y4); z4 = float(z4)
                # translate to utm
                lon1, lat1 = self.xy2ll(x1, y1)
                lon2, lat2 = self.xy2ll(x2, y2)
                lon3, lat3 = self.xy2ll(x3, y3)
                lon4, lat4 = self.xy2ll(x4, y4)
            # Depth
            mm = min([float(z1), float(z2), float(z3), float(z4)])
            if D<mm:
                D=mm
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
        if not donotreadslip:
            self.slip = np.array(self.slip)        
        if readpatchindex:
            self.index_parameter = np.array(self.index_parameter)

        # Compute equivalent patches
        self.computeEquivRectangle()

        # All done
        return

    def writePatches2File(self, filename, add_slip=None, scale=1.0, patch='normal',
                          stdh5=None, decim=1):
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

        # If an h5 file is specified, open it
        if stdh5 is not None:
            import h5py
            h5fid = h5py.File(stdh5, 'r')
            samples = h5fid['samples'].value[::decim,:]

        # Loop over the patches
        nPatches = len(self.patch)
        for p in range(nPatches):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip is 'strikeslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,p])
                    else:
                        slp = self.slip[p,0]*scale
                elif add_slip is 'dipslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,p+nPatches])
                    else:
                        slp = self.slip[p,1]*scale
                elif add_slip is 'total':
                    if stdh5 is not None:
                        slp = np.std(samples[:,p]**2 + samples[:,p+nPatches]**2)
                    else:
                        slp = np.sqrt(self.slip[p,0]**2 + self.slip[p,1]**2)*scale
                elif add_slip is 'normaltraction':
                    slp = self.Normal
                elif add_slip is 'strikesheartraction':
                    slp = self.ShearStrike
                elif add_slip is 'dipsheartraction':
                    slp = self.ShearDip
                # Make string
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
            elif patch in ('xyz'):
                p = self.patch[p]
            pp=p[1]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[0]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[3]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[2]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))

        # Close th file
        fout.close()

        # Close h5 file if it is open
        if stdh5 is not None:
            h5fid.close()

        # All done 
        return


    def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0, neg_depth=False, ellipse=False, flipstrike=False):
        '''
        Write a psxyz compatible file to draw lines starting from the center of each patch, 
        indicating the direction of slip.
        Tensile slip is not used...
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Copmute the slip direction
        self.computeSlipDirection(scale=scale, factor=factor, ellipse=ellipse, flipstrike=flipstrike)

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

        if ellipse:
            # Open the file
            fout = open('ellipse_'+filename, 'w')

            # Loop over the patches
            for e in self.ellipse:
                
                # Get ellipse points
                ex, ey, ez = e[:,0],e[:,1],e[:,2]
                
                # Depth
                if neg_depth:
                    ez = -1.0 * ez

                # Conversion to geographical coordinates
                lone,late = self.putm(ex*1000.,ey*1000.,inverse=True)
                
                # Write the > sign to the file
                fout.write('> \n')

                for lon,lat,z in zip(lone,late,ez):
                    fout.write('{} {} {} \n'.format(lon, lat, z))

            # Close file
            fout.close()            

        # All done
        return

    def getEllipse(self,patch,ellipseCenter=None,Npoints=10,factor=1.0):
        '''
        Compute the ellipse error given Cm for a given patch
        args:
               (optional) center  : center of the ellipse
               (optional) Npoints : number of points on the ellipse
        '''

        # Get Cm
        Cm = np.diag(self.Cm[patch,:2])
        Cm[0,1]=Cm[1,0]=self.Cm[patch,2]
        
        # Get strike and dip
        xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(patch, center=True) 
        if ellipseCenter!=None:
            xc,yc,zc = ellipseCenter
        
        # Compute eigenvalues/eigenvectors
        D,V=np.linalg.eig(Cm)
        v1 = V[:,0]
        a  = np.sqrt(np.abs(D[0]))
        b  = np.sqrt(np.abs(D[1]))
        phi   = np.arctan2(v1[1],v1[0])
        theta = np.linspace(0,2*np.pi,Npoints);
    
        # The ellipse in x and y coordinates
        Ex = a*np.cos(theta)*factor
        Ey = b*np.sin(theta)*factor
    
        # Correlation Rotation     
        R  = np.array([[np.cos(phi),-np.sin(phi)],
                       [np.sin(phi),np.cos(phi)]])
        RE = np.dot(R,np.array([Ex,Ey]))    
        
        # Strike/Dip rotation
        ME = np.array([RE[0,:], RE[1,:] * np.cos(dip), RE[1,:]*np.sin(dip)])
        R  = np.array([[np.sin(strike),-np.cos(strike),0.],
                       [np.cos(strike),np.sin(strike) ,0.],
                       [      0.      ,      0.       ,1.]])
        RE = np.dot(R,ME).T
        
        # Translation on Fault
        RE[:,0] += xc
        RE[:,1] += yc
        RE[:,2] += zc
        
        # All done
        return RE

    def computeSlipDirection(self, scale=1.0, factor=1.0, ellipse=False, flipstrike=False):
        '''
        Computes the segment indicating the slip direction.
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Create the array
        self.slipdirection = []
        
        # Check Cm if ellipse
        if ellipse:
            self.ellipse = []
            assert(self.Cm!=None), 'Provide Cm values'

        # Loop over the patches
        for p in range(len(self.patch)):  
            
            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True) 
            # Get the slip vector
            slip = self.getslip(self.patch[p]) 
            rake = np.arctan2(slip[1],slip[0])

            # Compute the vector
            if flipstrike:
                x = (np.sin(strike+np.pi)*np.cos(rake) + np.sin(strike+np.pi)*np.cos(dip)*np.sin(rake))
                y = (np.cos(strike+np.pi)*np.cos(rake) - np.cos(strike+np.pi)*np.cos(dip)*np.sin(rake))
                z = np.sin(dip)*np.sin(rake)
            else:
                x = (np.sin(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake))
                y = (np.cos(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake))
                z = -1.0*np.sin(dip)*np.sin(rake)
        
            # Scale these
            if scale.__class__ is float:
                sca = scale
            elif scale.__class__ is int:
                sca = scale*1.0
            elif scale.__class__ is str:
                if scale in ('total'):
                    sca = np.sqrt(slip[0]**2 + slip[1]**2 + slip[2]**2)*factor
                elif scale in ('strikeslip'):
                    sca = slip[0]*factor
                elif scale in ('dipslip'):
                    sca = slip[1]*factor
                elif scale in ('tensile'):
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
 
            # Append ellipse 
            if ellipse:
                self.ellipse.append(self.getEllipse(p,ellipseCenter=[xe, ye, ze],factor=factor*ellipse))

            # Append slip direction
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
        if len(self.equivpatch)==len(self.patch):
            del self.equivpatch[patch]
        self.slip = np.delete(self.slip, patch, axis=0)
        if hasattr(self, 'index_parameter'):
            self.index_parameter = np.delete(self.index_parameter, patch, axis=0)

        # All done
        return

    def deletepatches(self, titi):
        '''
        Deletes a list of patches.
        '''

        tutu = copy.deepcopy(titi)

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

        # Build the ll patch
        lon1, lat1 = self.xy2ll(patch[0][0], patch[0][1])
        z1 = patch[0][2]
        lon2, lat2 = self.xy2ll(patch[1][0], patch[1][1])
        z2 = patch[1][2]
        lon3, lat3 = self.xy2ll(patch[2][0], patch[2][1])
        z3 = patch[2][2]
        lon4, lat4 = self.xy2ll(patch[3][0], patch[3][1])
        z4 = patch[3][2]

        # append the ll patch
        patchll = [ [lon1, lat1, z1],
                    [lon2, lat2, z2],
                    [lon3, lat3, z3],
                    [lon4, lat4, z4] ]
        self.patchll.append(patchll)

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

    def addPatchesFromOtherFault(self, fault, indexes=None):
        '''
        Adds the patches of one fault to another.
        '''

        # Set indexes
        if indexes is None:
            indexes = range(len(fault.patch))

        # Loop on patches
        for i in indexes:
            p = fault.patch[i]
            slip = fault.slip[i,:]
            self.addpatch(p, slip)

        # Build the equivalent patches
        self.computeEquivRectangle()

        # All done
        return

    def chCoordinates(self,p,p_ref,angle_rad):
        '''
        Change the coordinate system:
           reference at p_ref
           rotation by angle_rad
        '''
        
        # Translation
        t_p = [p[0]-p_ref[0],p[1]-p_ref[1],p[2]-p_ref[2]]
        
        # Rotation 
        r_p     = [t_p[0]*np.cos(angle_rad) - t_p[1] * np.sin(angle_rad)]
        r_p.append(t_p[0]*np.sin(angle_rad) + t_p[1] * np.cos(angle_rad))
        r_p.append(t_p[2])
        
        # All done
        return r_p

    def getpatchgeometry(self, patch, center=False, checkindex=True):
        '''
        Returns the patch geometry as needed for okada92.
        Args:
            * patch         : index of the wanted patch or patch;
            * center        : if true, returns the coordinates of the center of the patch. 
                              if False, returns the UL corner.
            * checkindex    : Checks the index of the patch

        When we build the fault, the patches are not exactly rectangular. Therefore, 
        this routine will return the rectangle that matches with the two shallowest 
        points and that has an average dip angle to match with the other corners.
        '''

        # Get the patch
        u = None
        if patch.__class__ is int:
            u = patch
        else:
            if checkindex:
                for i in range(len(self.patch)):
                    if (self.patch[i]==patch).all():
                        u = i
        if u is not None:
            patch = self.equivpatch[u]

        # Get the four corners of the rectangle
        p1, p2, p3, p4 = patch

        # Get the UL corner of the patch
        if center:
            x1, x2, x3 = self.getcenter(patch)
        else:
            x1 = p2[0]
            x2 = p2[1]
            x3 = p2[2]        

        # Get the patch width (this fault is vertical for now)
        width = np.sqrt( (p4[0] - p1[0])**2 + (p4[1] - p1[1])**2 + (p4[2] - p1[2])**2 )   

        # Get the length
        length = np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

        # Get the strike assuming dipping to the east
        strike = np.arctan2( (p2[0] - p1[0]),(p2[1] - p1[1]) )

        # Change coordinates (origin at p1 and rotation by strike)
        r_p1 = self.chCoordinates(p1,p1,strike) 
        r_p4 = self.chCoordinates(p4,p1,strike) 
        
        # If it dip to the west, change update strike
        if r_p4[0] < 0.:
            strike += np.pi
        
        # Set the dip
        dip = np.arcsin( (p4[2] - p1[2])/width )

        # All done
        return x1, x2, x3, width, length, strike, dip

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
        x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(patch, center=True)

        # Get data position
        x = data.x
        y = data.y
        z = np.zeros(x.shape)   # Data are the surface

        # Run it for the strike slip component
        if (SLP[0]!=0.0):
            ss_dis = okadafull.displacement(x, y, z, x1, x2, x3, width, length, strike, dip, SLP[0], 0.0, 0.0)
        else:
            ss_dis = np.zeros((len(x), 3))

        # Run it for the dip slip component
        if (SLP[1]!=0.0):
            ds_dis = okadafull.displacement(x, y, z, x1, x2, x3, width, length, strike, dip, 0.0, SLP[1], 0.0)
        else:
            ds_dis = np.zeros((len(x), 3))

        # Run it for the tensile component
        if (SLP[2]!=0.0):
            ts_dis = okadafull.displacement(x, y, z, x1, x2, x3, width, length, strike, dip, 0.0, 0.0, SLP[2])
        else:
            ts_dis = np.zeros((len(x), 3))

        # All done
        return ss_dis, ds_dis, ts_dis

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

    def read3DsquareGrid(self, filename):
        '''
        This routine read the square fault geometry
        Format: lon lat E[km] N[km] Dep[km] strike dip Area ID

        These files are to be used with edks/MPI_EDKS/calcGreenFunctions_EDKS_subRectangles.py
        or edks/MPI_EDKS/calcGreenFunctions_EDKS_subSquares.py
        '''

        # Open the output file
        flld = open(filename,'r')
                
        # Loop over the patches
        self.patch     = []
        self.patchll   = []
        self.z_patches = []
        for l in flld:
            if l.strip()[0]=='#':
                continue
            items  = l.strip().split()
            
            # Get patch properties
            lonc   = float(items[0])
            latc   = float(items[1])
            zc     = float(items[4])
            strike = float(items[5])
            dip    = float(items[6])
            area   = float(items[7])
            PID    = int(items[8])
            #if strike<0.:
            #    strike += 360.
            length = np.sqrt(area)
            width  = np.sqrt(area)
            
            xc,yc = self.ll2xy(lonc,latc)
            
            # Build a patch with that
            strike_rad = strike*np.pi/180.
            dip_rad    = dip*np.pi/180.
            dstrike_x  =  0.5 * length * np.sin(strike_rad)
            dstrike_y  =  0.5 * length * np.cos(strike_rad)
            ddip_x     =  0.5 * width  * np.cos(dip_rad) * np.cos(strike_rad)
            ddip_y     = -0.5 * width  * np.cos(dip_rad) * np.sin(strike_rad)
            ddip_z     =  0.5 * width  * np.sin(dip_rad)    

            x1 = xc - dstrike_x - ddip_x
            y1 = yc - dstrike_y - ddip_y
            z1 = zc - ddip_z
            
            x2 = xc + dstrike_x - ddip_x
            y2 = yc + dstrike_y - ddip_y
            z2 = zc - ddip_z            

            x3 = xc + dstrike_x + ddip_x
            y3 = yc + dstrike_y + ddip_y
            z3 = zc + ddip_z

            x4 = xc - dstrike_x + ddip_x
            y4 = yc - dstrike_y + ddip_y
            z4 = zc + ddip_z     
            
            if self.top == None:
                self.top = z2
            elif self.top > z2:
                self.top = z2
            if self.depth == None:
                self.depth = z1
            elif self.depth > z1:
                self.depth = z2

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
            p1,p2,p3,p4 = p
            self.patch.append(p)
            self.patchll.append(pll)            
            self.z_patches.append(z1)            
            
            
        # Close the files
        flld.close()
        self.equivpatch   = self.patch
        self.equivpatchll = self.patchll
        # All done
        return


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

    def computetotalslip(self):
        '''
        Computes the total slip.
        '''

        # Computes the total slip
        self.totalslip = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2 + self.slip[:,2]**2)
    
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
            self.sim.lonlat2xy()

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

    def AverageAlongStrikeOffsets(self, name, insars, filename, discretized=True, smooth=None):
        '''
        If the profiles have the lon lat vectors as the fault, 
        This routines averages it and write it to an output file.
        '''

        if discretized:
            lon = self.loni
            lat = self.lati
        else:
            lon = self.lon
            lat = self.lat

        # Check if good
        for sar in insars:
            dlon = sar.AlongStrikeOffsets[name]['lon']
            dlat = sar.AlongStrikeOffsets[name]['lat']
            assert (list(dlon)==list(lon)), '{} dataset rejected'.format(sar.name)
            assert (list(dlat)==list(lat)), '{} dataset rejected'.format(sar.name)

        # Get distance
        x = insars[0].AlongStrikeOffsets[name]['distance']

        # Initialize lists
        D = []; AV = []; AZ = []; LO = []; LA = []

        # Loop on the distance
        for i in range(len(x)):

            # initialize average
            av = 0.0
            ni = 0.0
            
            # Get values
            for sar in insars:
                o = sar.AlongStrikeOffsets[name]['offset'][i]
                if np.isfinite(o):
                    av += o
                    ni += 1.0
        
            # if not only nan
            if ni>0:
                d = x[i]
                av /= ni
                az = insars[0].AlongStrikeOffsets[name]['azimuth'][i]
                lo = lon[i]
                la = lat[i]
            else:
                d = np.nan
                av = np.nan
                az = np.nan
                lo = lon[i]
                la = lat[i]

            # Append
            D.append(d)
            AV.append(av)
            AZ.append(az)
            LO.append(lo)
            LA.append(la)


        # smooth?
        if smooth is not None:
            # Arrays
            D = np.array(D); AV = np.array(AV); AZ = np.array(AZ); LO = np.array(LO); LA = np.array(LA)
            # Get the non nans
            u = np.flatnonzero(np.isfinite(AV))
            # Gaussian Smoothing
            dd = np.abs(D[u][:,None] - D[u][None,:])
            dd = np.exp(-0.5*dd*dd/(smooth*smooth))
            norm = np.sum(dd, axis=1)
            dd = dd/norm[:,None]
            AV[u] = np.dot(dd,AV[u])
            # List 
            D = D.tolist(); AV = AV.tolist(); AZ = AZ.tolist(); LO = LO.tolist(); LA = LA.tolist()

        # Open file and write header
        fout = open(filename, 'w')
        fout.write('# Distance (km) || Offset || Azimuth (rad) || Lon || Lat \n')

        # Write to file
        for i in range(len(D)):
            d = D[i]; av = AV[i]; az = AZ[i]; lo = LO[i]; la = LA[i]
            fout.write('{} {} {} {} {} \n'.format(d,av,az,lo,la))

        # Close the file
        fout.close()

        # All done
        return

    def horizshrink1patch(self, ipatch, fixedside='south', finallength=25.):
        '''
        Takes an existing patch and shrinks its size in the horizontal direction.
        Args:
            * ipatch        : Index of the patch of concern.
            * fixedside     : One side has to be fixed, takes the southernmost if 'south', 
                                                        takes the northernmost if 'north'
            * finallength   : Length of the final patch.
        '''

        # Get the patch
        patch = self.patch[ipatch]
        patchll = self.patchll[ipatch]

        # Find the southernmost points
        y = np.array([patch[i][1] for i in range(4)])
        imin = y.argmin()
        
        # Take the points we need to move
        if fixedside is 'south':
            fpts = np.flatnonzero(y==y[imin])
            mpts = np.flatnonzero(y!=y[imin])
        elif fixedside is 'north':
            fpts = np.flatnonzero(y!=y[imin])
            mpts = np.flatnonzero(y==y[imin])

        # Find which depths match
        d = np.array([patch[i][2] for i in range(4)])

        # Deal with the shallow points
        isf = fpts[d[fpts].argmin()]      # Index of the shallow fixed point
        ism = mpts[d[mpts].argmin()]      # Index of the shallow moving point        
        x1 = patch[isf][0]; y1 = patch[isf][1]
        x2 = patch[ism][0]; y2 = patch[ism][1]
        DL = np.sqrt( (x1-x2)**2 + (y1-y2)**2 ) # Distance between the original points
        Dy = y1 - y2                            # Y-Distance between the original points
        Dx = x1 - x2                            # X-Distance between the original points
        dy = finallength*Dy/DL                  # Y-Distance between the new points
        dx = finallength*Dx/DL                  # X-Distance between the new points
        patch[ism][0] = patch[isf][0] - dx
        patch[ism][1] = patch[isf][1] - dy

        # Deal with the deep points
        idf = fpts[d[fpts].argmax()]      # Index of the deep fixed point
        idm = mpts[d[mpts].argmax()]      # Index of the deep moving point
        x1 = patch[idf][0]; y1 = patch[idf][1]
        x2 = patch[idm][0]; y2 = patch[idm][1]
        DL = np.sqrt( (x1-x2)**2 + (y1-y2)**2 ) # Distance between the original points
        Dy = y1 - y2                            # Y-Distance between the original points
        Dx = x1 - x2                            # X-Distance between the original points
        dy = finallength*Dy/DL                  # Y-Distance between the new points
        dx = finallength*Dx/DL                  # X-Distance between the new points
        patch[idm][0] = patch[idf][0] - dx
        patch[idm][1] = patch[idf][1] - dy

        # Rectify the lon lat patch
        for i in range(4):
            x, y = patch[i][0], patch[i][1]
            lon, lat = self.xy2ll(x, y)
            patchll[i][0] = lon
            patchll[i][1] = lat

        # All done
        return

    def ExtractAlongStrikeVariationsOnDiscretizedFault(self, depth=0.5, filename=None, discret=0.5, interpolation='linear'):
        '''
        Extracts the Along Strike variations of the slip at a given depth, resampled along the discretized fault trace.
        Args:
            depth       : Depth at which we extract the along strike variations of slip.
            discret     : Discretization length
            filename    : Saves to a file.
            interpolation : Interpolation method
        '''

        # Import things we need
        import scipy.spatial.distance as scidis

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the list where we store things
        # [lon, lat, strike-slip, dip-slip, tensile, distance, xi, yi]
        Var = []

        # Open the output file if needed
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Distance to origin (km) | Position (x,y) (km)\n')

        # Discretize the fault
        if discret is not None:
            self.discretize(every=discret, tol=0.05, fracstep=0.02)
        nd = self.xi.shape[0]

        # Compute the cumulative distance along the fault
        dis = self.cumdistance(discretized=True)

        # Get the patches concerned by the depths asked
        dPatches = []
        sPatches = []
        for p in self.patch:
            # Check depth
            if ((p[0][2]<=depth) and (p[2][2]>=depth)):
                # Get patch
                sPatches.append(self.getslip(p))
                # Put it in dis
                xc, yc = self.getcenter(p)[:2]
                d = scidis.cdist([[xc, yc]], [[self.xi[i], self.yi[i]] for i in range(self.xi.shape[0])])[0]
                imin1 = d.argmin()
                dmin1 = d[imin1] 
                d[imin1] = 99999999.
                imin2 = d.argmin()  
                dmin2 = d[imin2]  
                dtot=dmin1+dmin2
                # Put it along the fault
                xcd = (self.xi[imin1]*dmin1 + self.xi[imin2]*dmin2)/dtot
                ycd = (self.yi[imin1]*dmin1 + self.yi[imin2]*dmin2)/dtot
                # Distance
                if dmin1<dmin2:
                    jm = imin1
                else:
                    jm = imin2
                dPatches.append(dis[jm] + np.sqrt( (xcd-self.xi[jm])**2 + (ycd-self.yi[jm])**2) )

        # Create the interpolator
        ssint = sciint.interp1d(dPatches, [sPatches[i][0] for i in range(len(sPatches))], kind=interpolation, bounds_error=False)
        dsint = sciint.interp1d(dPatches, [sPatches[i][1] for i in range(len(sPatches))], kind=interpolation, bounds_error=False)
        tsint = sciint.interp1d(dPatches, [sPatches[i][2] for i in range(len(sPatches))], kind=interpolation, bounds_error=False)

        # Interpolate
        for i in range(self.xi.shape[0]):
            x = self.xi[i]
            y = self.yi[i]
            lon = self.loni[i]
            lat = self.lati[i]
            d = dis[i]
            ss = ssint(d)
            ds = dsint(d)
            ts = tsint(d)
            Var.append([lon, lat, ss, ds, ts, d, x, y])
            # Write things if asked
            if filename is not None:
                fout.write('{} {} {} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, d, x, y))

        # Store it in AlongStrike
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # Close fi needed
        if filename is not None:
            fout.close()

        # All done
        return

    def ExtractAlongStrikeVariations(self, depth=0.5, origin=None, filename=None, orientation=0.0):
        '''
        Extract the Along Strike Variations of the creep at a given depth
        Args:
            depth   : Depth at which we extract the along strike variations of slip.
            origin  : Computes a distance from origin. Give [lon, lat].
            filename: Saves to a file.
            orientation: defines the direction of positive distances.
        '''

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the List where we will store things
        # For each patch, it will be [lon, lat, strike-slip, dip-slip, tensile, distance]
        Var = []

        # Creates the orientation vector
        Dir = np.array([np.cos(orientation*np.pi/180.), np.sin(orientation*np.pi/180.)])

        # initialize the origin
        x0, y0 = self.getpatchgeometry(0, center=True)[:2]
        if origin is not None:
            x0, y0 = self.ll2xy(origin[0], origin[1])

        # open the output file
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Patch Area (km2) | Distance to origin (km) \n')

        # compute area, if not done yet
        if not hasattr(self,'area'):
            self.computeArea()

        # Loop over the patches
        for p in self.patch:

            # Get depth range
            dmin = np.min([p[i][2] for i in range(4)])
            dmax = np.max([p[i][2] for i in range(4)])

            # If good depth, keep it
            if ((depth>=dmin) & (depth<=dmax)):

                # Get index
                io = self.getindex(p)

                # Get the slip and area
                slip = self.slip[io,:]
                area = self.area[io]

                # Get patch center
                xc, yc, zc = self.getcenter(p)
                lonc, latc = self.xy2ll(xc, yc)

                # Computes the horizontal distance
                vec = np.array([xc-x0, yc-y0])
                sign = np.sign( np.dot(Dir,vec) )
                dist = sign * np.sqrt( (xc-x0)**2 + (yc-y0)**2 )

                # Assemble
                o = [lonc, latc, slip[0], slip[1], slip[2], area, dist]

                # write output
                if filename is not None:
                    fout.write('{} {} {} {} {} {} {} \n'.format(lonc, latc, slip[0], slip[1], slip[2], area, dist))

                # append
                Var.append(o)

        # Close the file
        if filename is not None:
            fout.close()

        # Stores it 
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # all done 
        return

    def ExtractAlongStrikeAllDepths(self, filename=None, discret=0.5):
        '''
        Extracts the Along Strike Variations of the creep at all depths for the discretized version.
        '''

        # Dictionnary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # If filename provided, create it
        if filename is not None:
            fout = open(filename, 'w')

        # Create the list of depths
        depths = np.unique(np.array([[self.patch[i][u][2] for u in range(4)] for i in range(len(self.patch))]).flatten())
        depths = depths[:-1] + (depths[1:] - depths[:-1])/2.

        # Discretize the fault
        self.discretize(every=discret, tol=discret/10., fracstep=discret/12.)

        # For a list of depths, iterate
        for d in depths.tolist():

            # Get the values
            self.ExtractAlongStrikeVariationsOnDiscretizedFault(depth=d, filename=None, discret=None)
        
            # If filename, write to it
            if filename is not None:
                fout.write('> # Depth = {} \n'.format(d))
                Var = self.AlongStrike['Depth {}'.format(d)]
                for i in range(Var.shape[0]):
                    lon = Var[i,0]
                    lat = Var[i,1]
                    ss = Var[i,2]
                    ds = Var[i,3]
                    ts = Var[i,4]
                    dist = Var[i,5]
                    x = Var[i,6]
                    y = Var[i,7]
                    fout.write('{} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, dist, x, y))

        # Close file if done
        if filename is not None:
            fout.close()

        # All done
        return

    def getFaultVector(self, i, discretized=True, normal=False):
        '''
        Returns the vector tangential to the fault at the i-th point of the fault (discretized if True).
        if normal is True, returns a normal vector (+90 counter-clockwise wrt. tangente)
        '''

        # Get fault trace
        if discretized:
            x = self.xi
            y = self.yi
        else:
            x = self.xf
            y = self.yf

        # Starting point
        st = max(0, i-5)
        ed = min(len(x), i+5)

        # Get slope
        G = np.vstack((x[st:ed], np.ones(x[st:ed].shape))).T
        Gg = np.dot(np.linalg.inv(np.dot(G.T, G)),G.T)
        m = np.dot(Gg,y[st:ed])
        slope = m[0]
        const = m[1]

        # Make unit vector from slope
        b = np.sqrt( 1./(1.+slope**2) )
        a = b*slope
        vect = np.array([a, b])

        if normal:
            # Rotate 90deg counter-clokwise
            R = np.array([ [0.0, -1.0],
                           [1.0,  0.0]])
            vect = np.dot(R,vect)

        # All done
        return vect

    def getPatchPositionAlongStrike(self, p, discretized=True):
        '''
        Returns the position of a patch along strike (distance to the first point of the fault).
        p is the patch index or the patch.
        '''

        import scipy.spatial.distance as scidis

        # Convert patch to index 
        if type(p) is not int:
            p = getindex(p)

        # Compute the cumulative distance
        if self.xi is None:
            self.discretize(every=0.5, tol=0.05, fracstep=0.02)
        dis = self.cumdistance(discretized=discretized)

        # Get the fault
        if discretized:
            x = self.xi
            y = self.yi
        else:
            x = self.xf
            y = self.yf

        # get the patch center
        xc, yc = self.getpatchgeometry(p, center=True)[:2]

        # compute the distance
        d = scidis.cdist([[xc, yc]], [[x[i], y[i]] for i in range(x.shape[0])])[0]
        
        # Get the two closest points
        imin1 = d.argmin()
        dmin1 = d[imin1]
        d[imin1] = 9999999999.
        imin2 = d.argmin()
        dmin2 = d[imin2]
        dtot = dmin1+dmin2
        xcd = (x[imin1]*dmin1 + x[imin2]*dmin2)/dtot
        ycd = (y[imin1]*dmin1 + y[imin2]*dmin2)/dtot
        if dmin1<dmin2:
            jm = imin1
        else:
            jm = imin2

        # All done
        return dis[jm] + np.sqrt( (xcd-x[jm])**2 + (ycd-y[jm])**2)

    def computeTractionOnEachFaultPatch(self, factor=0.001, mu=30e9, nu=0.25):
        '''
        Uses okada92 to compute the traction change on each patch.
        Args:
            * factor        : Conversion fator between slip units and distance units. In a regular case, distance 
                              units are km. If the slip is in mm, then factor is 10e-6.
            * mu            : Shear Modulus (default is 30e9 Pa).
            * nu            : Poisson's ratio (default is 0.25).
        '''

        # How many fault patches
        nP = len(self.patch)

        # Create a stress object
        stress = stressfield('Stress', utmzone=self.utmzone)

        # Compute the stress on the fault
        stress.Fault2Stress(self, factor=factor, mu=mu, nu=nu, slipdirection='sdt', stressonpatches=True)
        stress.total2deviatoric()

        # Create the arrays
        self.T = np.zeros((3,nP))
        self.ShearStrike = np.zeros((nP,))
        self.ShearDip = np.zeros((nP,))
        self.Normal = np.zeros((nP,))
        
        # Get the strike and dip
        angles = np.array([self.getpatchgeometry(p, center=True)[-2:] for p in self.patch])
        strike = angles[:,0]
        dip = angles[:,1]

        # Compute the tractions on the patch
        n1, n2, n3, T, Sigma, TauStrike, TauDip = stress.computeTractions(strike, dip)

        # Store these
        self.Stress = stress.Stress
        self.flag = stress.flag
        self.flag2 = stress.flag2
        self.T = T
        self.Normal = Sigma
        self.ShearStrike = TauStrike
        self.ShearDip = TauDip
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        # All done
        return

    def computeCoulombOnPatches(self, friction=0.6, sign='strike'):
        '''
        Computes the Coulomb failure stress change on patches.
        On the routine, the normal stress is positive away from the center of the patch.
        '''
    
        # Check if Tractions have been computed
        assert hasattr(self, 'Normal'), 'Compute Tractions first...'

        # Sign
        if sign in ('strike'):
            Sign = np.sign(self.ShearStrike)
        elif sign in ('dip'):
            Sign = np.sign(self.ShearDip)

        # Compute the Coulomb failure stress (-1.0 for Normal stress because positive must be compressive)
        self.Coulomb = Sign*np.sqrt(self.ShearStrike**2 + self.ShearDip**2) - friction*self.Normal*-1.0

        # All done 
        return
        
    def computeImposedTractionOnEachFaultPatch(self, factor=0.001, mu=30e9, nu=0.25):
        '''
        Uses okada92 to compute the traction change on each patch from slip on the other patches.
        Args:
            * factor        : Conversion fator between slip units and distance units. In a regular case, distance 
                              units are km. If the slip is in mm, then factor is 10e-6.
            * mu            : Shear Modulus (default is 30e9 Pa).
            * nu            : Poisson's ratio (default is 0.25).
        '''

        # How many fault patches
        nP = len(self.patch)

        # Save the slip
        Slip = copy.deepcopy(self.slip)

        # Create the arrays
        self.T = np.zeros((3,nP))
        self.ShearStrike = np.zeros((nP,))
        self.ShearDip = np.zeros((nP,))
        self.Normal = np.zeros((nP,))

        # Create a stress object 
        stress = stressfield('Stress', utmzone=self.utmzone)

        # Loop on each fault patch
        for p in self.patch:

            # Get the patch index
            ii = self.getindex(p)

            # Write out
            sys.stdout.write('\r Patch: {}/{}'.format(ii, nP))
            sys.stdout.flush()

            # Get the patch characteristics
            xc, yc, depthc, widthc, lengthc, strike, dip = self.getpatchgeometry(p, center=True)
            stress.setXYZ(xc, yc, depthc)

            # Set the slip of the fault
            self.slip = Slip
            self.slip[ii,:] = 0.0

            # compute the stress from that fault
            stress.Fault2Stress(self, mu=mu, slipdirection='sdt', factor=factor)
            stress.total2deviatoric()

            # compute the tractions on the patch
            n1, n2, n3, T, Sigma, TauStrike, TauDip = stress.computeTractions(strike, dip)

            # Store these
            self.T[:,ii] = T[0]
            self.ShearStrike[ii] = TauStrike[0]
            self.ShearDip[ii] = TauDip[0]
            self.Normal[ii] = Sigma[0]

        # Restore slip correctly on that fault
        self.slip = Slip

        # Clean up screen 
        print('')

        # all done
        return

    def plot(self,ref='utm', figure=134, add=False, maxdepth=None, axis='equal', value_to_plot='total', equiv=False, show=True, axesscaling=True, Norm=None, linewidth=1.0):
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
        ax = fig.add_subplot(111, projection='3d', aspect='equal')

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
        elif value_to_plot=='strikeslip':
            plotval = self.slip[:,0]
        elif value_to_plot=='dipslip':
            plotval = self.slip[:,1]
        elif value_to_plot=='tensile':
            plotval = self.slip[:,2]
        elif value_to_plot=='normaltraction':
            plotval = self.Normal
        elif value_to_plot=='strikesheartraction':
            plotval = self.ShearStrike
        elif value_to_plot=='dipsheartraction':
            plotval = self.ShearDip
        elif value_to_plot=='coulomb':
            plotval = self.Coulomb
        elif value_to_plot=='index':
            plotval = np.linspace(0, len(self.patch)-1, len(self.patch))

        # Plot the patches
        if self.patch is not None:
            
            # import stuff
            import mpl_toolkits.mplot3d.art3d as art3d
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            
            # set z axis
            if axesscaling:
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
            if Norm is not None:
                cNorm = colors.Normalize(vmin=Norm[0], vmax=Norm[1])
            else:
                cNorm  = colors.Normalize(vmin=plotval.min(), vmax=plotval.max())
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
                rect.set_linewidth(linewidth)
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

        # Store it 
        self.ax = ax

        # show
        if show:
            plt.show()

        # All done
        return

    def getPatchesThatAreUnder(self, xf, yf, ref='utm', tolerance=0.5):
        '''
        For a list of positions, returns the patches that are directly underneath.
        Only works if you have a vertical fault.
        '''

        # Check reference
        if ref in ('lonlat'):
            xf, yf = self.ll2xy(xf, yf)

        # Make an array
        points = np.vstack((xf, yf)).T

        # Create a list of empty lists (one per point you want to test)
        List = [[] for u in range(len(xf))]
        sscheck = False
        dscheck = False
        tscheck = False
        if hasattr(self, 'index_parameter'):
            if self.index_parameter[0][0]<9999999:
                sscheck = True
                SSList = [[] for u in range(len(xf))]
            if self.index_parameter[0][1]<9999999:
                dscheck = True
                DSList = [[] for u in range(len(xf))]
            if self.index_parameter[0][2]<9999999:
                tscheck = True
                TSList = [[] for u in range(len(xf))]

        # Iterate over patches
        for p in self.patch:

            # Get the index
            i = self.getindex(p)

            # Get the patch geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)

            # make a vector Director and a Normal
            D = np.array([np.sin(strike), np.cos(strike)])
            N = np.array([np.sin(strike+np.pi/2.), np.cos(strike+np.pi/2.)])

            # Center
            center = np.array([xc, yc])

            # Build path
            p1 = center + D*length/2. + D*tolerance - N*tolerance
            p2 = center + D*length/2. + D*tolerance + N*tolerance
            p3 = center - D*length/2. - D*tolerance + N*tolerance
            p4 = center - D*length/2. - D*tolerance - N*tolerance
            pp = path.Path([p1, p2, p3, p4], closed=False)

            # Yes or no?
            check = pp.contains_points(points)

            # Append the index of the patch in the corresponding list in List
            for u in np.flatnonzero(check).tolist():
                List[u].append(i)
                if sscheck:
                    SSList[u].append(self.index_parameter[i][0])
                if dscheck:
                    DSList[u].append(self.index_parameter[i][1])
                if tscheck:
                    TSList[u].append(self.index_parameter[i][2])

 #           if i==252:
 #               plt.figure()
 #               plt.plot(xc, yc, '+k', markersize=20.)
 #               plt.plot([xc, xc+D[0]], [yc, yc+D[1]], '-b')
 #               plt.plot(xf, yf, '.r')
 #               plt.plot(xf[check], yf[check], '.b')
 #               plt.plot([p[i][0] for i in range(4)], [p[i][1] for i in range(4)], '.g')
 #               plt.plot(p1[0], p1[1], '+r')
 #               plt.plot(p2[0], p2[1], '+g')
 #               plt.plot(p3[0], p3[1], '+b')
 #               plt.plot(p4[0], p4[1], '+k')
 #               plt.show()

        # all done
        if sscheck and not dscheck and not tscheck:
            return List, SSList
        elif sscheck and dscheck and not tscheck:
            return List, SSList, DSList
        elif sscheck and dscheck and tscheck:
            return List, SSList, DSList, TSList
        elif not sscheck and dscheck and not tscheck:
            return List, DSList
        elif not sscheck and not dscheck and tscheck:
            return List, TSList
        elif not sscheck and dscheck and tscheck:
            return List, DSList, TSList
        elif sscheck and not dscheck and tscheck:
            return List, SSList, TSList
        else:
            return List

    def mapFault2Fault(self, Map, fault):
        '''
        User provides a Mapping function np.array((len(self.patch), len(fault.patch))) and a fault and the slip from the argument
        fault is mapped into self.slip.
        Function just does:
        self.slip[:,0] = np.dot(Map,fault.slip)
        ...
        '''

        # Get the number of patches
        nPatches = len(self.patch)
        nPatchesExt = len(fault.patch)

        # Assert the Mapping function is correct
        assert(Map.shape==(nPatches,nPatchesExt)), 'Mapping function has the wrong size...'

        # Map the slip
        self.slip[:,0] = np.dot(Map, fault.slip[:,0])
        self.slip[:,1] = np.dot(Map, fault.slip[:,1])
        self.slip[:,2] = np.dot(Map, fault.slip[:,2])

        # all done
        return

    def mapUnder2Above(self, deepfault, extrapolate=10.):
        '''
        This routine is very very particular. It only works with 2 vertical faults.
        It Builds the mapping function from one fault to another, when these are vertical.
        These two faults must have the same surface trace. If the deep fault has more than one raw of patches, 
        it might go wrong and give some unexpected results.
        Args:
            * deepfault     : Deep section of the fault.
        '''

        # Assert faults are compatible
        assert ( (self.lon==deepfault.lon).all() and (self.lat==deepfault.lat).all()), 'Surface traces are different...'

        # Check that all patches are verticals
        dips = np.array([self.getpatchgeometry(i)[-1]*180./np.pi for i in range(len(self.patch))])
        assert((dips == 90.).all()), 'Not viable for non-vertical patches, fault {}....'.format(self.name)
        deepdips = np.array([deepfault.getpatchgeometry(i)[-1]*180./np.pi for i in range(len(deepfault.patch))])
        assert((deepdips == 90.).all()), 'Not viable for non-vertical patches, fault {}...'.format(deepfault.name)

        # Get the number of patches
        nPatches = len(self.patch)
        nDeepPatches = len(deepfault.patch)

        # Create the map from under to above
        Map = np.zeros((nPatches, nDeepPatches)) 

        # Set the top row as the surface trace
        self.surfacePatches2Trace()

        # Discretize the surface trace quite finely
        self.discretize(every=0.5, tol=0.05, fracstep=0.02)

        # Compute the cumulative distance along the fault
        dis = self.cumdistance(discretized=True)

        # Compute the cumulative distance between the beginning of the fault and the corners of the patches
        distance = []
        for p in self.patch:
            D = []
            # for each corner
            for c in p:
                # Get x,y
                x = c[0]
                y = c[1]
                # Get the index of the nearest xi value under x
                i = np.flatnonzero(x>=self.xi)[-1]
                # Get the corresponding distance along the fault
                d = dis[i] + np.sqrt( (x-self.xi[i])**2 + (y-self.yi[i])**2 )
                # Append 
                D.append(d)
            # Array unique
            D = np.unique(np.array(D))
            # append
            distance.append(D)

        # Do the same for the deep patches
        deepdistance = []
        for p in deepfault.patch:
            D = []
            for c in p:
                x = c[0]
                y = c[1]
                i = np.flatnonzero(x>=self.xi)
                if len(i)>0:
                    i = i[-1]
                    d = dis[i] + np.sqrt( (x-self.xi[i])**2 + (y-self.yi[i])**2 )
                else:
                    d = 99999999.
                D.append(d)
            D = np.unique(np.array(D))
            deepdistance.append(D)

        # Numpy arrays
        distance = np.array(distance)
        deepdistance = np.array(deepdistance)

        self.disdis = distance 
        self.deepdis = deepdistance

        # Loop over the patches to find out which are over which 
        for p in range(len(self.patch)):

            # Get the patch distances
            d1 = distance[p,0]
            d2 = distance[p,1]

            # Get the index for the points
            i1 = np.intersect1d(np.flatnonzero((d1>=deepdistance[:,0])), np.flatnonzero((d1<deepdistance[:,1])))[0]
            i2 = np.intersect1d(np.flatnonzero((d2>deepdistance[:,0])), np.flatnonzero((d2<=deepdistance[:,1])))[0]

            # two cases possible:
            if i1==i2:              # The shallow patch is fully inside the deep patch
                Map[p,i1] = 1.0     # All the slip comes from this patch
            else:                   # The shallow patch is on top of several patches
                # two cases again
                if np.abs(i2-i1)==1:       # It covers the boundary between 2 patches 
                    delta1 = np.abs(d1-deepdistance[i1][1])
                    delta2 = np.abs(d2-deepdistance[i2][0])
                    total = delta1 + delta2
                    delta1 /= total
                    delta2 /= total
                    Map[p,i1] = delta1
                    Map[p,i2] = delta2
                else:                       # It is larger than the boundary between 2 patches and covers several deep patches
                    delta = []
                    delta.append(np.abs(d1-deepdistance[i1][1]))
                    for i in range(i1+1,i2):
                        delta.append(np.abs(deepdistance[i][1]-deepdistance[i][0]))
                    delta.append(np.abs(d2-deepdistance[i2][0]))
                    delta = np.array(delta)
                    total = np.sum(delta)
                    delta = delta/total
                    for i in range(i1,i2+1):
                        Map[p,i] = delta

        # All done
        return Map

    def mapSlipPlane2Plane(self, fault, interpolation='linear', verbose=False, addlimits=True, smooth=0.1):
        '''
        Maps the slip distribution from fault onto self.
        Mapping is built by computing the best plane between the two faults, 
        projecting the center of patches on that plane and doing a simple resampling.
        The closer the two faults, the better...
        Args:
            * fault         : Fault that has a slip distribution
            * interpolation : Type of interpolation method. Can be 'linear', 'cubic' or 'quintic'
            * verbose       : if True, the routine says a few things.
            * addlimits     : Adds the upper and lower bounds of the fault in the interpolation scheme.
        '''

        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Map Slip from fault {} into fault {}'.format(fault.name, self.name))
            print('Build the best plane')

        # 1. Find the best fitting plane for self
        # 1.1 Compute the average strike and the average dip of the fault
        selfstrike = np.mean([self.getpatchgeometry(i)[5] for i in range(len(self.patch))])
        selfdip = np.mean([self.getpatchgeometry(i)[6] for i in range(len(self.patch))])

        # 1.2 Get the center of the fault
        selfxc = np.mean([self.getpatchgeometry(i, center=True)[0] for i in range(len(self.patch))])
        selfyc = np.mean([self.getpatchgeometry(i, center=True)[1] for i in range(len(self.patch))])
        selfzc = np.mean([self.getpatchgeometry(i, center=True)[2] for i in range(len(self.patch))])

        # 2. Find the best fitting plane for fault
        # 2.1 Compute the average strike and dip of the fault
        faultstrike = np.mean([fault.getpatchgeometry(i)[5] for i in range(len(fault.patch))])
        faultdip = np.mean([fault.getpatchgeometry(i)[6] for i in range(len(fault.patch))])

        # 2.2 Get the center of the fault
        faultxc = np.mean([fault.getpatchgeometry(i, center=True)[0] for i in range(len(fault.patch))])
        faultyc = np.mean([fault.getpatchgeometry(i, center=True)[1] for i in range(len(fault.patch))])
        faultzc = np.mean([fault.getpatchgeometry(i, center=True)[2] for i in range(len(fault.patch))])

        # 3. Compute the average plane P
        # 3.1 Compute the average strike and dip of the plane
        strike = (selfstrike + faultstrike)/2.
        dip = (selfdip + faultdip)/2.

        # 3.2 Compute the average center of the plane
        xc = (faultxc + selfxc)/2.
        yc = (faultyc + selfyc)/2.
        zc = (faultzc + selfzc)/2.

        # 3.3 Get the unit vectors of that plane
        n1, n2, n3 = self.strikedip2normal(strike, dip)

        if verbose:
            print('Project on the best plane')

        # 4. Project patch centers from self on the plane P
        # 4.1 Build vectors from new plane center to each patch center
        selfvector = np.array([self.getpatchgeometry(i, center=True)[:3] for i in range(len(self.patch))])
        selfvector[:,0] -= xc
        selfvector[:,1] -= yc
        selfvector[:,2] -= zc

        # 4.2 Project on the unit vectors
        self.x1 = np.dot(selfvector, n2).reshape((len(self.patch),))
        self.x2 = np.dot(selfvector, n3).reshape((len(self.patch),))

        # 5. Project patch centers from fault on the plane P
        # 5.1 Build vectors from the new plane center to each patch center
        faultvector = np.array([fault.getpatchgeometry(i, center=True)[:3] for i in range(len(fault.patch))])
        size = len(fault.patch)

        # 5.1.bis Add Limits
        if addlimits:
            # Get depths
            depths = np.array([[fault.patch[i][0][2], fault.patch[i][2][2]] for i in range(len(fault.patch))])
            # Get min and max depths
            mindepth = np.min(np.unique(depths))
            maxdepth = np.max(np.unique(depths))
            # Get which patches are concerned
            uu = np.flatnonzero(depths[:,0]==mindepth)
            vv = np.flatnonzero(depths[:,1]==maxdepth)
            # Add top row
            addvector0 = np.array([fault.patch[i][0] for i in uu])
            addvector1 = np.array([fault.patch[i][1] for i in uu])
            faultvector = np.vstack(( np.vstack((faultvector, addvector0)), addvector1 ))
            size = size + addvector0.shape[0] + addvector1.shape[0]
            # Add bottom row
            addvector0 = np.array([fault.patch[i][2] for i in vv])
            addvector1 = np.array([fault.patch[i][3] for i in vv])
            faultvector = np.vstack(( np.vstack((faultvector, addvector0)), addvector1 ))
            size = size + addvector0.shape[0] + addvector1.shape[0]

        faultvector[:,0] -= xc
        faultvector[:,1] -= yc
        faultvector[:,2] -= zc

        # 5.2 Project on the unit vectors
        fault.x1 = np.dot(faultvector, n2).reshape((size,))
        fault.x2 = np.dot(faultvector, n3).reshape((size,))

        if verbose:
            print('Run the interpolation')

        # 6. Now, resample the slip that is on 
        # 6.0 import scipy
        import scipy.interpolate as sciint

        # 6.1 Get slip
        inslip = fault.slip
        if addlimits:
            addslip = fault.slip[uu,:]
            inslip = np.vstack((inslip, addslip))
            inslip = np.vstack((inslip, addslip))
            addslip = fault.slip[vv,:]
            inslip = np.vstack((inslip, addslip))
            inslip = np.vstack((inslip, addslip))
        fault.inslip = inslip

        # 6.1 Create an interpolator
        minx1 = np.min(fault.x1); normx1 = np.max(fault.x1) - minx1; 
        minx2 = np.min(fault.x2); normx2 = np.max(fault.x2) - minx2; 
        fStrikeSlip = sciint.Rbf((fault.x1-minx1)/normx1, (fault.x2-minx2)/normx2, inslip[:,0], function=interpolation, smooth=smooth)
        fDipSlip = sciint.Rbf((fault.x1-minx1)/normx1, (fault.x2-minx2)/normx2, inslip[:,1], function=interpolation, smooth=smooth)
        fTensile = sciint.Rbf((fault.x1-minx1)/normx1, (fault.x2-minx2)/normx2, inslip[:,2], function=interpolation, smooth=smooth)

        # 6.2 Interpolate
        for i in range(len(self.patch)):
            self.slip[i,0] = fStrikeSlip((self.x1[i]-minx1)/normx1, (self.x2[i]-minx2)/normx2)
            self.slip[i,1] = fDipSlip((self.x1[i]-minx1)/normx1, (self.x2[i]-minx2)/normx2)
            self.slip[i,2] = fTensile((self.x1[i]-minx1)/normx1, (self.x2[i]-minx2)/normx2)

        # All done
        return

    def strikedip2normal(self, strike, dip):
        '''
        Returns a vector normal to a plane with a given strike and dip.
        strike and dip in radians...
        '''
        
        # Compute normal
        n1 = np.array([np.sin(dip)*np.cos(strike), -1.0*np.sin(dip)*np.sin(strike), np.cos(dip)])

        # Along Strike
        n2 = np.array([np.sin(strike), np.cos(strike), np.zeros(strike.shape)])

        # Along Dip
        n3 = np.cross(n1, n2, axisa=0, axisb=0).T

        # All done
        if len(n1.shape)==1:
            return n1.reshape((3,1)), n2.reshape((3,1)), n3.reshape((3,1))
        else:
            return n1, n2, n3

# Things that should become obsolete soon

    def writeEDKSsubParams(self, data, edksfilename, amax=None, plot=False, w_file=True):
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
            * w_file        : if False, will not write the subParam fil (default=True)
        Returns:
            * filename         : Name of the subParams file created (only if w_file==True)
            * RectanglePropFile: Name of the rectangles properties file
            * ReceiverFile     : Name of the receiver file
            * method_par       : Dictionary including useful EDKS parameters
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

        # Build usefull outputs
        parNames = ['useRecvDir', 'Amax', 'EDKSunits', 'EDKSfilename', 'prefix']
        parValues = [ useRecvDir ,  amax ,  EDKSunits ,  EDKSfilename ,  prefix ]
        method_par = dict(zip(parNames, parValues))

        # Open the EDKSsubParams.py file        
        if w_file:
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

            # All done
            return filename, RectanglePropFile, ReceiverFile, method_par
        else:
            return RectanglePropFile, ReceiverFile, method_par

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
        fltname = self.name.replace(' ','_')
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

    def computeAdjacencyMat(self, verbose=False):
        """
        Computes the adjacency matrix for the fault geometry provided by ndip x nstrike. Values of 0
        indicate no adjacency while values of 1 indicate patches share an edge.
        """
        if verbose:
            print('Computing adjacency matrix for fault %s' % self.name)

        # Get numbers
        npatch = len(self.patch)
        if self.numz is None:
            print('We try a wild guess for the number of patches along dip')
            width = np.mean([self.getpatchgeometry(p, center=True)[3] for p in self.patch])
            depths = [ [p[j][2] for j in range(4)] for p in self.patch]
            depthRange = np.max(depths)-np.min(depths)
            self.numz = np.rint(depthRange/width)

        # Get number of Patches along strike
        nstrike = np.int(npatch // self.numz)

        # Create the matrix
        Jmat = np.zeros((npatch,npatch), dtype=int)

        # Set diagonal k = 1
        template = np.ones((nstrike,), dtype=int)
        template[-1] = 0
        repvec = np.tile(template, (1,self.numz)).flatten()[:-1]
        Jmat[range(0,npatch-1),range(1,npatch)] = repvec
        
        # Set diagonal k = nstrike
        nd = np.diag(Jmat, k=nstrike).size
        Jmat[range(0,npatch-nstrike),range(nstrike,npatch)] = np.ones((nd,), dtype=int)
        
        # Return symmetric part to fill lower triangular part
        self.adjacencyMat = Jmat + Jmat.T

        # All done
        return

    def buildLaplacian(self, verbose=False):
        """
        Build normalized Laplacian smoothing array.
        This routine is not designed for unevenly paved faults.
        It does not account for the variations in size of the patches.
        """

        # Adjacency Matrix
        self.computeAdjacencyMat(verbose=verbose)
        Jmat = self.adjacencyMat
        npatch = Jmat.shape[0]
        assert Jmat.shape[1] == npatch, 'adjacency matrix is not square'

        # Build Laplacian by looping over each patch
        D = np.zeros((npatch, npatch))
        for p in range(npatch):
            adjacents = Jmat[p,:].nonzero()[0]
            nadj = len(adjacents)
            for ind in adjacents:
                D[p,ind] = 1.0
            D[p,p] = -4.0

        return D

#EOF
