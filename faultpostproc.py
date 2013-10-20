'''
A class the allows to compute various things using a fault object.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import copy
import shapely.geometry as geom
import matplotlib.pyplot as plt
import sys
import os

class faultpostproc(object):

    def __init__(self, name, fault, Mu=24e9, utmzone='10', samplesh5=None):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * fault         : Fault object
            * Mu            : Shear modulus. Default is 24e9 GPa, because it is the PREM value for the upper 15km.
            * utmzone       : UTM zone. Default is 10 (Western US).
        '''

        # Initialize the data set 
        self.name = name
        self.fault = copy.deepcopy(fault) # we don't want to modify fault slip
        self.utmzone = utmzone
        self.Mu = Mu
        self.patchDepths = None

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

        # Check to see if we're reading in an h5 file for posterior samples
        self.samplesh5 = samplesh5

        # All done
        return

    def h5_init(self, decim=1):
        '''
        If the attribute self.samplesh5 is not None, we open the h5 file specified by 
        self.samplesh5 and copy the slip values to self.fault.slip (hopefully without loading 
        into memory).

        kwargs:
            decim                       decimation factor for skipping samples
        '''

        if self.samplesh5 is None:
            return
        else:
            try:
                import h5py
            except ImportError:
                print('Cannot import h5py. Computing scalar moments only')
                return
            self.hfid = h5py.File(self.samplesh5, 'r')
            samples = self.hfid['samples']
            nsamples = np.arange(0, samples.shape[0], decim).size
            self.fault.slip = np.zeros((self.numPatches,3,nsamples))
            self.fault.slip[:,0,:] = samples[::decim,:self.numPatches].T
            self.fault.slip[:,1,:] = samples[::decim,self.numPatches:2*self.numPatches].T

        return

    def h5_finalize(self):
        '''
        Close the (potentially) open h5 file.
        '''
        if hasattr(self, 'hfid'):
            self.hfid.close()

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
        Returns the slip vector in the cartesian space for the patch p. We do not deal with 
        the opening component. The fault slip may be a 3D array for multiple samples of slip.
        Args:
            * p             : Index of the desired patch.
        '''

        # Get the geometry of the patch
        x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

        # Get the slip
        strikeslip, dipslip, tensile = self.fault.slip[p,:,...]
        slip = np.sqrt(strikeslip**2 + dipslip**2)

        # Get the rake
        rake = np.arctan2(dipslip, strikeslip)

        # Vectors
        ux = slip*(np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike))
        uy = slip*(np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike))
        uz = -1.0*slip*np.sin(rake)*np.sin(dip)

        # All done
        if isinstance(ux, np.ndarray):
            outArr = np.zeros((3,1,ux.size))
            outArr[0,0,:] = ux
            outArr[1,0,:] = uy
            outArr[2,0,:] = uz
            return outArr
        else:
            return np.array([[ux], [uy], [uz]])

    def computePatchMoment(self, p) :
        '''
        Computes the Moment tensor for one patch.
        Args:
            * p             : patch index
        '''

        # Get the normal
        n = self.patchNormal(p).reshape((3,1))

        # Get the slip vector
        u = self.slipVector(p)

        # Compute the moment density
        if u.ndim == 2:
            mt = self.Mu * (np.dot(u, n.T) + np.dot(n, u.T))
        elif u.ndim == 3:
            n = np.tile(n, (1,1,u.shape[2]))
            nT = np.transpose(n, (1,0,2))
            uT = np.transpose(u, (1,0,2))
            # Tricky 3D multiplication
            mt = self.Mu * ((u[:,:,None]*nT).sum(axis=1) + (n[:,:,None]*uT).sum(axis=1))

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

        # Initialize an empty moment
        M = 0.0

        # Compute the tensor for each patch
        for p in range(len(self.fault.patch)):
            # Compute the moment of one patch
            mt = self.computePatchMoment(p)
            # Add it up to the full tensor
            M += mt
            
        # Check if symmetric
        self.checkSymmetric(M)

        # Store it (Aki convention)
        self.Maki = M

        # Convert it to Harvard
        self.Aki2Harvard()

        # All done
        return

    def computeScalarMoment(self):
        '''
        Computes the scalar seismic moment.
        '''

        # check 
        assert hasattr(self, 'Maki'), 'Compute the Moment Tensor first'

        # Get the moment tensor
        M = self.Maki

        # get the norm
        Mo = np.sqrt(0.5 * np.sum(M**2, axis=(0,1)))

        # Store it
        self.Mo = Mo

        # All done
        return Mo

    def computeMagnitude(self, plotHist=None):
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

        # Plot histogram of magnitudes
        if plotHist is not None:
            assert isinstance(Mw, np.ndarray), 'cannot make histogram with one value'
            fig = plt.figure(figsize=(14,8))
            ax = fig.add_subplot(111)
            ax.hist(Mw, bins=100)
            ax.grid(True)
            ax.set_xlabel('Moment magnitude', fontsize=18)
            ax.set_ylabel('Normalized count', fontsize=18)
            ax.tick_params(labelsize=18)
            fig.savefig(os.path.join(plotHist, 'momentMagHist.png'), dpi=400, 
                        bbox_inches='tight')
            fig.clf()

        # All done
        return Mw

    def Aki2Harvard(self):
        '''
        Transform the patch from the Aki convention to the Harvard convention.
        '''
 
        # Get Maki 
        Maki = self.Maki

        # Create new tensor
        M = np.zeros_like(Maki)

        # Shuffle things around following Aki & Richard, Second edition, pp 113
        M[0,0,...] = Maki[2,2,...]
        M[1,0,...] = M[0,1,...] = Maki[0,2,...]
        M[2,0,...] = M[0,2,...] = -1.0*Maki[1,2,...]
        M[1,1,...] = Maki[0,0,...]
        M[2,1,...] = M[1,2,...] = -1.0*Maki[1,0,...]
        M[2,2,...] = Maki[1,1,...]

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
        for p in range(self.numPatches):

            # Get patch info 
            x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

            # Get the moment tensor
            dS = self.computePatchMoment(p)

            # Compute the normalized scalar moment density
            m = 0.5 / (Mo**2) * np.sum(M * dS, axis=(0,1))

            # Add it up to the centroid location
            xc += m*x
            yc += m*y
            zc += m*z

        # Store the x, y, z locations
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
        if M.ndim == 2:
            MT = M.T
        else:
            MT = np.transpose(M, (1,0,2))
        assert (M == MT).all(), 'Matrix is not symmetric'

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
            strikeslip, dipslip, tensile = self.fault.slip[p,:,...]

            # Add to moment
            Mo += self.Mu * S * np.sqrt(strikeslip**2 + dipslip**2)

        # Compute magnitude
        Mw = 2./3.*(np.log10(Mo) - 9.1)

        # All done
        return Mo, Mw

    def integratedPotencyWithDepth(self, plotOutput=None, numDepthBins=5):
        '''
        Computes the cumulative moment with depth by summing the moment per row of
        patches. If the moments were computed with mutiple samples, we form histograms of 
        potency vs. depth. Otherwise, we just compute a depth profile.

        kwargs:
            plotOutput                      output directory for figures
            numDepthBins                    number of bins to group patch depths
        '''

        # Check to see we have compute moment tensor
        assert hasattr(self, 'Maki'), 'Compute the Moment Tensor first'

        # Collect all patch depths
        patchDepths = np.empty((self.numPatches,))
        for pIndex in range(self.numPatches):
            patchDepths[pIndex] = self.fault.getpatchgeometry(pIndex, center=False)[2]

        # Determine depth bins for grouping
        zmin, zmax = patchDepths.min(), patchDepths.max()
        zbins = np.linspace(zmin, zmax, numDepthBins+1)
        binDepths = 0.5 * (zbins[1:] + zbins[:-1])
        dz = abs(zbins[1] - zbins[0])

        # Loop over depth bins
        potencyDict = {}; scalarPotencyList = []; meanLogPotency = []
        for i in range(numDepthBins):

            # Get the patch indices that fall in this bin
            zstart, zend = zbins[i], zbins[i+1]
            ind = patchDepths >= zstart
            ind *= patchDepths <= zend
            ind = ind.nonzero()[0]

            # Sum the total moment for the depth bin
            M = 0.0
            for patchIndex in ind:
                M += self.computePatchMoment(int(patchIndex))
            # Convert to scalar potency
            potency = np.sqrt(0.5 * np.sum(M**2, axis=(0,1))) / self.Mu
            logPotency = np.log10(potency)
            meanLogPotency.append(np.log10(np.mean(potency)))

            # Create and store histogram for current bin
            if self.samplesh5 is not None:
                n, bins = np.histogram(logPotency, bins=100, density=True)
                binCenters = 0.5 * (bins[1:] + bins[:-1])
                zbindict = {}
                zbindict['count'] = n
                zbindict['bins'] = binCenters
                key = 'depthBin_%03d' % (i)
                potencyDict[key] = zbindict
            else:
                scalarPotencyList.append(potency)

        if plotOutput is not None:

            if self.samplesh5 is None:

                fig = plt.figure(figsize=(12,8))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                scalarPotency = np.array(scalarPotencyList)
                logPotency = np.log10(scalarPotency)
                sumLogPotency = np.log10(np.cumsum(scalarPotencyList))
                for ax,dat in [(ax1, logPotency), (ax2, sumLogPotency)]:
                    ax.plot(dat, binDepths, '-o')
                    ax.grid(True)
                    ax.set_xlabel('Log Potency', fontsize=16)
                    ax.set_ylabel('Depth (km)', fontsize=16)
                    ax.tick_params(labelsize=16)
                    ax.set_ylim(ax.get_ylim()[::-1])
                ax1.set_title('Potency vs. depth', fontsize=18)
                ax2.set_title('Integrated Potency vs. depth', fontsize=18)
                fig.savefig(os.path.join(plotOutput, 'depthPotencyDistribution.png'),
                            dpi=400, bbox_inches='tight')

            else:
      
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111) 
                
                for depthIndex in range(numDepthBins):
                    # Get the histogram for the current depth
                    key = 'depthBin_%03d' % (depthIndex)
                    zbindict = potencyDict[key]
                    n, bins = zbindict['count'], zbindict['bins']
                    # Shift the histogram to the current depth and scale it
                    n /= n.max() / (0.5 * dz)
                    n -= binDepths[depthIndex]
                    # Plot normalized histogram
                    ax.plot(bins, -n)

                # Also draw the means
                ax.plot(meanLogPotency, binDepths, '-ob', linewidth=2)

                ax.set_ylim(ax.get_ylim()[::-1])
                ax.set_xlabel('Log potency', fontsize=18)
                ax.set_ylabel('Depth (km)', fontsize=18)
                ax.tick_params(labelsize=18)
                ax.grid(True)
                fig.savefig(os.path.join(plotOutput, 'depthPotencyDistribution.png'),
                            dpi=400, bbox_inches='tight')

        return

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




