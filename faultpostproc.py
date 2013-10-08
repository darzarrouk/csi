'''
A class the allows to compute various things using a fault object.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import shapely.geometry as geom
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import sys
import os

class faultpostproc(object):

    def __init__(self, name, fault, Mu=24e9, utmzone='10'):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * fault         : Fault object
            * Mu            : Shear modulus. Default is 24e9 GPa, because it is the PREM value for the upper 15km.
            * utmzone       : UTM zone. Default is 10 (Western US).
        '''

        # Initialize the data set 
        self.name = name
        self.fault = fault
        self.utmzone = utmzone
        self.Mu = Mu

        # Determine number of patches along-strike and along-dip
        self.numPatches = len(self.fault.patch)
        if self.fault.numz is not None:
            self.numDepthPatches = self.fault.numz
            self.numStrikePatches = self.numPatches / self.numDepthPatches

        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize Post Processing object {} on fault {}".format(self.name, fault.name))

        # Initialize the UTM transformation
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

        # All done
        return

    def lonlat2xy(self, lon, lat):
        '''
        Uses the transformation in self to convert  lon/lat vector to x/y utm.
        Args:
            * lon           : Longitude array.
            * lat           : Latitude array.
        '''

        x, y = self.putm(lon,lat)
        x /= 1000.
        y /= 1000.

        return x, y

    def xy2lonlat(self, x, y):
        '''
        Uses the transformation in self to convert x.y vectors to lon/lat.
        Args:
            * x             : Xarray
            * y             : Yarray
        '''

        lon, lat = self.putm(x*1000., y*1000., inverse=True)
        return lon, lat

    def patchNormal(self, p):
        '''
        Returns the Normal to a patch.
        Args:
            * p             : Index of the desired patch.
        '''

        # Get the geometry of the patch
        x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

        # Normal
        n1 = -1.0*np.sin(dip)*np.sin(strike)
        n2 = np.sin(dip)*np.cos(strike)
        n3 = -1.0*np.cos(dip)
        N = np.sqrt(n1**2+ n2**2 + n3**2)

        # All done
        return np.array([n1/N, n2/N, n3/N])

    def slipVector(self, p):
        '''
        Returns the slip vector in the cartesian space for the patch p.
        We do not deal with the opening component.
        Args:
            * p             : Index of the desired patch.
        '''

        # Get the geometry of the patch
        x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

        # Get the slip
        strikeslip, dipslip, tensile = self.fault.slip[p,:]
        slip = np.sqrt(strikeslip**2 + dipslip**2)

        # Get the rake
        rake = np.arctan2(dipslip, strikeslip)

        # Vectors
        ux = slip*(np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike))
        uy = slip*(np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike))
        uz = -1.0*slip*np.sin(rake)*np.sin(dip)

        # All done
        return np.array([ux, uy, uz])

    def computePatchMoment(self, p) :
        '''
        Computes the Moment tensor for one patch.
        Args:
            * p             : patch index
        '''

        # Get the normal
        n = self.patchNormal(p).reshape((3,1))

        # Get the slip vector
        u = self.slipVector(p).reshape((3,1))

        # Compute the moment density
        mt = self.Mu * (np.dot(u, n.T) + np.dot(n, u.T))

        # Multiply by the area
        mt *= self.fault.area[p]*1000000.

        # All done
        return mt

    def computeMomentTensor(self):
        '''
        Computes the full seismic (0-order) moment tensor from the slip distribution.
        '''

        # Compute the area of each patch
        if not hasattr(self.fault, 'area'):
            self.fault.computeArea()

        # Create a new tensor
        M = np.zeros((3,3))

        # Compute the tensor for each patch and sum it into M
        for p in range(len(self.fault.patch)):
            # Compute the moment of one patch
            mt = self.computePatchMoment(p)
            # Add it up to the full tensor
            M += mt

        # Check if symmetric
        self.checkSymmetric(M)

        # Store it (Aki convention)
        self.Maki= M

        # Convert it to Harvard
        self.Aki2Harvard()

        # All done
        return

    def computeScalarMoment(self):
        '''
        Computes the scalar seismic moment.
        '''

        # check 
        assert hasattr(self,'Maki'), 'Compute the Moment Tensor first'

        # Get the moment tensor
        M = self.Maki

        # get the norm
        Mo = np.sqrt( np.sum(M**2)/2. )

        # Store it
        self.Mo = Mo

        # All done
        return Mo

    def computeMagnitude(self):
        '''
        Computes the moment magnitude.
        '''

        # check
        if not hasattr(self, 'Mo'):
            self.computeScalarMoment()

        # Mw
        Mw = 2./3.*(np.log10(self.Mo) - 9.1)

        # Store 
        self.Mw = Mw

        # All done
        return Mw

    def Aki2Harvard(self):
        '''
        Transform the patch from the Aki convention to the Harvard convention.
        '''

        # Create new tensor
        M = np.zeros((3,3))

        # Get Maki 
        Maki = self.Maki

        # Shuffle things around following Aki & Richard, Second edition, pp 113
        M[0,0] = Maki[2,2]
        M[1,0] = M[0,1] = Maki[0,2]
        M[2,0] = M[0,2] = -1.0*Maki[1,2]
        M[1,1] = Maki[0,0]
        M[2,1] = M[1,2] = -1.0*Maki[1,0]
        M[2,2] = Maki[1,1]

        # Store it
        self.Mharvard = M

        # All done 
        return

    def computeCentroidLonLatDepth(self):
        '''
        Computes the equivalent centroid location.
        Take from Theoretical Global Seismology, Dahlen & Tromp. pp. 169
        '''

        # Check
        assert hasattr(self, 'Mharvard'), 'Compute the Moment tensor first'

        # Get the scalar moment
        Mo = self.computeScalarMoment()

        # Get the total Moment
        M = self.Maki

        # initialize centroid loc.
        xc, yc, zc = 0.0, 0.0, 0.0

        # Loop on the patches
        for p in range(len(self.fault.patch)):

            # Get patch info 
            x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

            # Get the moment tensor
            dS = self.computePatchMoment(p)

            # Compute the normalized scalar moment density
            m = 0.5 / (Mo**2) * np.sum(M * dS)

            # Add it up to the centroid location
            xc += m*x
            yc += m*y
            zc += m*z

        # Store the x, y, z location
        self.centroid = [xc, yc, zc]

        # Convert to lon lat
        lonc, latc = self.putm(xc*1000., yc*1000., inverse=True)
        self.centroidll = [lonc, latc, zc]

        return lonc, latc, zc

    def checkSymmetric(self, M):
        '''
        Check if a matrix is symmetric.
        '''

        # Check
        assert (M==M.T).all(), 'Matrix is not symmetric'

        # all done
        return

    def computeBetaMagnitude(self):
        '''
        Computes the magnitude with a simple approximation.
        '''

        # Initialize moment
        Mo = 0.0

        # Loop on patches
        for p in range(len(self.fault.patch)):

            # Get area
            S = self.fault.area[p]*1000000.

            # Get slip
            strikeslip, dipslip, tensile = self.fault.slip[p,:]

            # Add to moment
            Mo += self.Mu * S * np.sqrt(strikeslip**2 + dipslip**2)

        # Compute magnitude
        Mw = 2./3.*(np.log10(Mo) - 9.1)

        # All done
        return Mo, Mw

    def integratedMomentWithDepth(self, plotOutput=None):
        '''
        Computes the cumulative moment with depth by summing the moment per row of
        patches.
        '''
        depths = []; moments = []
        for depthIndex in range(self.numDepthPatches):
            # Sum moment for current row of patches
            rowMoment = 0.0
            for strikeIndex in range(self.numStrikePatches):
                patchIndex = depthIndex * self.numStrikePatches + strikeIndex
                rowMoment += self.computePatchMoment(patchIndex)
            # Convert to scalar moment
            rowMo = np.sqrt(0.5 * np.sum(rowMoment**2))
            # Add to list of moments
            moments.append(rowMo)
            # Store the patch center depth
            patch_z = self.fault.getpatchgeometry(patchIndex, center=True)[2]
            depths.append(patch_z)

        # Compute cumulative sums and convert to moment magnitude
        moments = np.array(moments)
        Mws = 2.0 / 3.0 * (np.log10(moments) - 9.1)
        sumMoments = np.cumsum(moments)
        sumMws = 2.0 / 3.0 * (np.log10(sumMoments) - 9.1)

        if plotOutput is not None:
            fig = plt.figure(figsize=(12,8))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            for ax,dat in [(ax1, Mws), (ax2, sumMws)]:
                ax.plot(dat, depths, '-o')
                ax.grid(True)
                ax.set_xlabel('Moment magnitude', fontsize=16)
                ax.set_ylabel('Depth (km)', fontsize=16)
                ax.tick_params(labelsize=16)
                ax.set_ylim(ax.get_ylim()[::-1])
            ax1.set_title('Moment magnitude vs. depth', fontsize=18)
            ax2.set_title('Integrated moment magnitude vs. depth', fontsize=18)
            fig.savefig(os.path.join(plotOutput, 'depthMomentDistribution.png'),
                        dpi=400, bbox_inches='tight')

        return depths, sumMoments, sumMws

    def write2GCMT(self, form='full', filename=None):
        '''
        Writes in GCMT style
        Args:
            * form          : format is either 'full' to match with Zacharie binary
                                            or 'line' to match with the option -Sm in GMT

        Example of 'full':
         PDE 2006  1  1  7 11 57.00  31.3900  140.1300  10.0 5.3 5.0 SOUTHEAST OF HONSHU, JAP                
        event name:     200601010711A  
        time shift:     10.4000
        half duration:   1.5000
        latitude:       31.5100
        longitude:     140.0700
        depth:          12.0000
        Mrr:       3.090000e+24
        Mtt:      -2.110000e+24
        Mpp:      -9.740000e+23
        Mrt:      -6.670000e+23
        Mrp:      -5.540000e+23
        Mtp:      -5.260000e+23
        '''

        # Check
        assert hasattr(self,'Mharvard'), 'Compute the Moment tensor first'

        # Get the moment
        M = self.Mharvard

        # Get lon lat
        lon, lat, depth = self.computeCentroidLonLatDepth()

        # Check filename
        if filename is not None:
            fout = open(filename, 'w')
        else:
            fout = sys.stdout

        if form is 'full':
            # Write the BS header
            fout.write(' PDE 1999  1  1  9 99 99.00  99.9900   99.9900  99.0 5.3 5.0 BULLSHIT \n')
            fout.write('event name:    thebigbaoum \n')
            fout.write('time shift:    99.9999     \n')
            fout.write('half duration: 99.9999     \n')
            fout.write('latitude:       {}     \n'.format(lat))
            fout.write('longitude:      {}     \n'.format(lon))
            fout.write('depth:          {}     \n'.format(depth))
            fout.write('Mrr:           {:7e}       \n'.format(M[0,0]*1e7))
            fout.write('Mtt:           {:7e}       \n'.format(M[1,1]*1e7))
            fout.write('Mpp:           {:7e}       \n'.format(M[2,2]*1e7))
            fout.write('Mrt:           {:7e}       \n'.format(M[0,1]*1e7))
            fout.write('Mrp:           {:7e}       \n'.format(M[0,2]*1e7))
            fout.write('Mtp:           {:7e}       \n'.format(M[1,2]*1e7))
        elif form is 'line':
            # get the largest mantissa
            mantissa = 0
            A = [M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]]
            for i in range(6):
                if np.abs(A[i])>0.0:
                    exp = int(np.log10(np.abs(A[i])))
                    if exp > mantissa:
                        mantissa = exp
            mrr = (M[0,0])/10**mantissa
            mtt = (M[1,1])/10**mantissa
            mpp = (M[2,2])/10**mantissa
            mrt = (M[0,1])/10**mantissa
            mrp = (M[0,2])/10**mantissa
            mtp = (M[1,2])/10**mantissa
            fout.write('{} {} {} {:3f} {:3f} {:3f} {:3f} {:3f} {:3f} {:d} \n'.format(
                lon, lat, depth, mrr, mtt, mpp, mrt, mrp, mtp, mantissa+7))

        # Close file
        if filename is not None:
            fout.close()
        else:
            fout.flush()

        # All done
        return




