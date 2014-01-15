'''
A class that deals with vertical faults. (not finished)

Written by Z. Duputel, R. Jolivet and B. Riel, April 2013
'''

## Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys

## Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

# Rectangular patches Fault class
from .RectangularPatches import RectangularPatches



class faultwithdipkinematic(RectangularPatches):

    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name      : Name of the fault.
            * utmzone   : UTM zone.
        '''
        
        super(self.__class__,self).__init__(name,utmzone,ellps)
        
        hypo_lat = None
        hypo_lon = None
        
        # All done
        return

    def buildPatches(self, dip, dipdirection, every=10, trace_tol=0.1, trace_fracstep=0.2, trace_xaxis='x'):
        '''
        Builds a dipping fault.
        Args:
            * dip           : Dip angle
            * dipdirection  : Direction towards which the fault dips.
        '''

        # Print
        print("Building a dipping fault")
        print("         Dip Angle       : {} degrees".format(dip))
        print("         Dip Direction   : {} degrees From North".format(dipdirection))

        # Initialize the structures
        self.patch = []
        self.patchll = []
        self.slip = []
        self.patchdip = []

        # Discretize the surface trace of the fault
        self.discretize(every,trace_tol,trace_fracstep,trace_xaxis)

        # degree to rad
        dip = (dip)*np.pi/180.
        dipdirection_rad = dipdirection*np.pi/180.

        # initialize the depth of the top row
        self.zi = np.ones((self.xi.shape))*self.top

        # set a marker
        D = [self.top]

        # Loop over the depths
        for i in range(self.numz):

            # Get the top of the row
            xt = self.xi
            yt = self.yi
            lont, latt = self.putm(xt*1000., yt*1000., inverse=True)
            zt = self.zi

            # Compute the bottom row
            xb = xt + self.width*np.cos(dip)*np.sin(dipdirection_rad)
            yb = yt + self.width*np.cos(dip)*np.cos(dipdirection_rad)
            lonb, latb = self.putm(xb*1000., yb*1000., inverse=True)
            zb = zt + self.width*np.sin(dip)

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
        self.discretize(every,trace_tol,trace_fracstep,trace_xaxis)

        # Compute the equivalent rectangles
        self.computeEquivRectangle()
    
        # All done
        return

    def buildKinematicGFs(self, data, rake, slip=1., slipdir=['parallel','perpendicular']):
        '''
        Builds the Kinematic Green's function matrix on the discretized fault
        Args:
        '''
        
        # Get the number of patches
        Np  = len(self.patch)

        # Initializes a space in the dictionary to store the green's function
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]        
        if 's' in slipdir:
            assert 'parallel' not in G.keys(), 'parallel GFs already assigned for {}'%(data.name)
            G['parallel'] ## See that later

        # Loop over each patch
        for p in range(Np):
            sys.stdout.write('\r Patch: {} / {} '.format(p+1,len(self.patch)))
            sys.stdout.flush()
            
            # Get the position, size, strike and dip of the patch
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            
            # Assign the source location to the waveform engine
            

            # Calculate synthetics
            
