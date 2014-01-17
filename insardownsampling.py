'''
A class that deal with downsampling the insar data.

Written by R. Jolivet, January 2014.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import matplotlib.path as path
import copy
import sys
import os

# Personals
import insarrates

class Downsampler(object):

    def __init__(self, name, insar, faults, verbose=True):
        '''
        Args:
            * name      : Name of the downsampler.
            * insar     : InSAR data set to be downsampled.
            * faults    : List of faults.
        '''

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR downsampling tools {}".format(name))

        self.verbose = verbose

        # Set the name
        self.name = name

        # Set the transformation
        self.utmzone = insar.utmzone
        self.putm = insar.putm
        self.ll2xy = insar.ll2xy
        self.xy2ll = insar.xy2ll

        # Check if the faults are in the same utm zone
        self.faults = []
        for fault in faults:
            assert (fault.utmzone==self.utmzone), 'Fault {} not in utm zone #{}'.format(fault.name, self.utmzone)
            self.faults.append(fault)

        # Save the insar
        self.insar = insar

        # Incidence and heading need to be defined
        assert hasattr(self.insar, 'heading'), 'No Heading precised for insar object'
        assert hasattr(self.insar, 'incidence'), 'No Incidence precised for insar object'
        self.incidence = self.insar.incidence
        self.heading = self.insar.heading

        # Create the initial box
        xmin = np.floor(insar.x.min())
        xmax = np.floor(insar.x.max())+1.
        ymin = np.floor(insar.y.min())
        ymax = np.floor(insar.y.max())+1.
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.box = [[xmin, ymin],
                    [xmin, ymax],
                    [xmax, ymax],
                    [xmax, ymin]]
        lonmin = insar.lon.min()
        lonmax = insar.lon.max()
        latmin = insar.lat.min()
        latmax = insar.lat.max()
        self.lonmin = lonmin; self.latmax = latmax
        self.latmin = latmin; self.lonmax = lonmax
        self.boxll = [[lonmin, latmin],
                      [lonmin, latmax],
                      [lonmax, latmax],
                      [lonmax, latmin]]
        
        # Get the original pixel spacing
        self.spacing = distance.cdist([[insar.x[0], insar.y[0]]], [[insar.x[i], insar.y[i]] for i in range(1, insar.x.shape[0])])[0]
        self.spacing = self.spacing.min()

        # Deduce the original pixel area
        self.pixelArea = self.spacing**2

        # All done
        return

    def initialstate(self, startingsize, minimumsize, tolerance=0.5, plot=False):
        '''
        Does the first cut onto the data.
        Args:
            * startingsize  : Size of the first regular downsampling (it'll be the effective maximum size of windows)
            * minimumsize   : Minimum Size of the blocks.
            * tolerance     : Between 0 and 1. If 1, all the pixels must have a value so that the box is kept, 
                                               If 0, no pixels are needed... Default is 0.5
        '''

        # Set the tolerance
        self.tolerance = tolerance
        self.minsize = minimumsize

        # Define Edges
        xLeftEdges = np.arange(self.xmin-startingsize, self.xmax+startingsize, startingsize)[:-1].tolist()
        yUpEdges = np.arange(self.ymin-startingsize, self.ymax+startingsize, startingsize)[1:].tolist()

        # Make blocks
        blocks = []
        for x in xLeftEdges:
            for y in yUpEdges:
                block = [ [x, y],
                          [x+startingsize, y],
                          [x+startingsize, y-startingsize],
                          [x, y-startingsize] ]
                blocks.append(block)

        # Generate the sampling to test
        self.downsample(blocks, plot=plot)

        # All done
        return

    def downsample(self, blocks, plot=False):
        '''
        From the saved list of blocks, computes the downsampled data set and the informations that come along.
        '''

        # Create the new insar object
        newsar = insarrates.insarrates('Downsampled {}'.format(self.insar.name), utmzone=self.utmzone, verbose=False)

        # Save the blocks
        self.blocks = blocks
        
        # Build the list of blocks in lon, lat
        blocksll = []
        for block in blocks:
            c1, c2, c3, c4 = block
            blockll = [ self.xy2ll(c1[0], c1[1]), 
                        self.xy2ll(c2[0], c2[1]),
                        self.xy2ll(c3[0], c3[1]),
                        self.xy2ll(c4[0], c4[1]) ]
            blocksll.append(blockll)
        self.blocksll = blocksll

        # Create the variables
        newsar.vel = []
        newsar.lon = []
        newsar.lat = []
        newsar.x = []
        newsar.y = []
        newsar.err = []

        # Store the factor
        newsar.factor = self.insar.factor

        # Build the previous geometry
        SARXY = np.vstack((self.insar.x, self.insar.y)).T

        # Keep track of the blocks to trash
        blocks_to_remove = []

        # Over each block, we average the position and the phase to have a new point
        for i in range(len(blocks)):
            block = blocks[i]
            # Create a path
            p = path.Path(block, closed=False)
            # Find those who are inside
            ii = p.contains_points(SARXY)
            # Check if total area is suffucient
            blockarea = self.getblockarea(block)
            coveredarea = np.flatnonzero(ii).shape[0]*self.pixelArea
            if (coveredarea/blockarea >= self.tolerance):
                # Get Mean, Std, x, y, ...
                vel = np.mean(self.insar.vel[ii])
                err = np.std(self.insar.vel[ii])
                x = np.mean(self.insar.x[ii])
                y = np.mean(self.insar.y[ii])
                lon, lat = self.xy2ll(x, y)
                # Store that
                newsar.vel.append(vel)
                newsar.err.append(err)
                newsar.x.append(x)
                newsar.y.append(y)
                newsar.lon.append(lon)
                newsar.lat.append(lat)
            else:
                blocks_to_remove.append(i)

        # Clean up useless blocks
        self.trashblocks(blocks_to_remove)

        # Convert
        newsar.vel = np.array(newsar.vel)
        newsar.err = np.array(newsar.err)
        newsar.x = np.array(newsar.x)
        newsar.y = np.array(newsar.y)
        newsar.lon = np.array(newsar.lon)
        newsar.lat = np.array(newsar.lat)

        # LOS
        newsar.inchd2los(self.incidence, self.heading)

        # Store newsar
        self.newsar = newsar

        # plot y/n
        if plot:
            self.plotDownsampled()

        # All done
        return

    def cutblockinfour(self, block):
        '''
        From a block, returns 4 equal blocks.
        Args:
            * block         : block as defined in initialstate.
        '''

        # Get the four corners
        c1, c2, c3, c4 = block
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
        x4, y4 = c4

        # Compute the position of the center
        xc = x1 + (x2 - x1)/2.
        yc = y1 + (y4 - y1)/2.
        
        # Form the 4 blocks
        b1 = [ [x1, y1],
               [xc, y1],
               [xc, yc],
               [x1, yc] ]
        b2 = [ [xc, y2],
               [x2, y2],
               [x2, yc],
               [xc, yc] ]
        b3 = [ [x4, yc],
               [xc, yc],
               [xc, y4],
               [x4, y4] ]
        b4 = [ [xc, yc],
               [x3, yc],
               [x3, y3],
               [xc, y3] ]

        # all done
        return b1, b2, b3, b4

    def ResolutionBasedIterations(self, threshold, damping, slipdirection='s', plot=False):
        '''
        Iteratively downsamples the dataset until value compute inside each block is lower than the threshold.
        Args:
            * threshold     : Threshold.
            * sigma         : Amplitude of the Smoothing.
            * lam           : Smoothing characteristic length.
            * slipdirection : Which direction to accout for to build the slip Green's functions.
        '''
        
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Downsampling Iterations")

        # Creates the list of booleans
        Check = [False]*len(self.blocks)
        do_cut = False

        # counter
        it = 0

        # Check if block size is minimum
        Bsize = self._is_minimum_size(self.blocks)

        # Loops until done
        while not all(Check):

            # Cut if asked
            if do_cut:
                # New list of blocks
                newblocks = []
                # Iterate over blocks
                for j in range(len(self.blocks)):
                    block = self.blocks[j]
                    if not Check[j] and not Bsize[j]:
                        b1, b2, b3, b4 = self.cutblockinfour(block)
                        newblocks.append(b1)
                        newblocks.append(b2)
                        newblocks.append(b3)
                        newblocks.append(b4)
                    else:
                        newblocks.append(block)
                # Do the downsampling
                self.downsample(newblocks, plot=plot)
            else:
                do_cut = True

            # Iteration #
            it += 1
            if self.verbose:
                sys.stdout.write('\r Iteration {} testing {} data samples '.format(it, len(self.blocks)))
                sys.stdout.flush()

            # Create the Greens function 
            G = None

            # Compute the greens functions for each fault and cat these together
            for fault in self.faults:
                # build GFs
                fault.buildGFs(self.newsar, vertical=True, slipdir=slipdirection, verbose=False)
                fault.assembleGFs([self.newsar], polys=0, slipdir=slipdirection, verbose=False)
                # Cat GFs
                if G is None:
                    G = fault.Gassembled
                else:
                    G = np.hstack((G, fault.Gassembled))

            # Compute the data resolution matrix
            Npar = G.shape[1]
            Ginv = np.dot(np.linalg.inv(np.dot(G.T,G)+ damping*np.eye(Npar)),G.T)
            Rd = np.dot(G, Ginv)
            self.Rd = copy.deepcopy(np.diag(Rd))

            # Blocks that have a minimum size, don't check these
            Bsize = self._is_minimum_size(self.blocks)
            self.Rd[np.where(Bsize)] = 0.0
    
            # Find the blocks that are fine
            Check = self.Rd<threshold

        if self.verbose:
            print(" ")

        # All done
        return

    def getblockarea(self, block):
        '''
        Returns the total area of a block.
        Args:
            * block : Block as defined in initialstate.
        '''
        
        # All done in one line
        return np.abs(block[0][0]-block[1][0]) * np.abs(block[0][1] - block[2][1])

    def trashblock(self, j):
        '''
        Deletes one block.
        '''

        del self.blocks[j]
        del self.blocksll[j]

        # all done
        return

    def trashblocks(self, jj):
        '''
        Deletes the blocks corresponding to indexes in the list jj.
        '''

        while len(jj)>0:

            # Get index
            j = jj.pop()

            # delete it
            self.trashblock(j)

            # upgrade list
            for i in range(len(jj)):
                if jj[i]>j:
                    jj[i] -= 1

        # all done
        return

    def plotDownsampled(self, figure=145, axis='equal', ref='utm', Norm=None):
        '''
        Plots the downsampling as it is at this step.
        Args:
            * figure    : Figure ID.
            * axis      : Axis argument from matplotlib.
            * Norm      : [colormin, colormax]
            * ref       : Can be 'utm' or 'lonlat'.
        '''

        # Create the figure
        fig = plt.figure(figure)
        full = fig.add_subplot(121)
        down = fig.add_subplot(122)

        # Set the axes
        if ref is 'utm':
            full.set_xlabel('Easting (km)')
            full.set_ylabel('Northing (km)')
            down.set_xlabel('Easting (km)')
            down.set_ylabel('Northing (km)')
        else:
            full.set_xlabel('Longitude')
            full.set_ylabel('Latitude')
            down.set_xlabel('Longitude')
            down.set_ylabel('Latitude')

        # Get the datasets
        original = self.insar
        downsampled = self.newsar

        # Vmin, Vmax
        if Norm is not None:
            vmin, vmax = Norm
        else:
            vmin = original.vel.min()
            vmax = original.vel.max()

        # Prepare the colormaps
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # Plot original dataset
        if ref is 'utm':
            # insar
            sca = full.scatter(original.x, original.y, s=10, c=original.vel, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.)
            # Faults
            for fault in self.faults:
                full.plot(fault.xf, fault.yf, '-k')
        else:
            # insar
            sca = full.scatter(original.lon, original.lat, s=10, c=original.vel, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.) 
            # Faults
            for fault in self.faults:
                full.plot(fault.lon, fault.lat, '-k')

        # Patches
        import matplotlib.collections as colls

        # Plot downsampled
        if ref is 'utm':
            # Insar
            for i in range(len(self.blocks)):
                # Get block
                block = self.blocks[i]
                # Get value
                val = downsampled.vel[i]
                # Build patch
                x = [block[j][0] for j in range(4)]
                y = [block[j][1] for j in range(4)]
                verts = [zip(x, y)]
                patch = colls.PolyCollection(verts)
                # Set its color
                patch.set_color(scalarMap.to_rgba(val))
                patch.set_edgecolors('k')
                down.add_collection(patch)
            # Faults
            for fault in self.faults:
                down.plot(fault.xf, fault.yf, '-k')
        else:
            # InSAR
            for i in range(len(self.blocks)):
                # Get block
                block = self.blockll[i]
                # Get value
                val = downsampled.vel[i]
                # Build patch
                x = [block[j][0] for j in range(4)]
                y = [block[j][1] for j in range(4)] 
                verts = [zip(x, y)] 
                patch = colls.PolyCollection(verts)
                # Set its color  
                patch.set_color(scalarMap.to_rgba(val))
                patch.set_edgecolors('k')    
                down.add_collection(patch)
            # Faults
            for fault in faults:
                down.plot(fault.lon, fault.lat, '-k')

        # Axes
        down.axis(axis)
        full.axis(axis)

        if ref is 'utm':
            full.set_xlim([self.xmin, self.xmax])
            down.set_xlim([self.xmin, self.xmax])
        else:
            full.set_xlim([self.lonmin, self.lonmax])
            down.set_xlim([self.lonmin, self.lonmax])

        # All done
        plt.show()
        return
        
    def _is_minimum_size(self, blocks):
        '''
        Returns a Boolean array.
        True if block is minimum size, 
        False either.
        '''

        # Initialize
        Bsize = []

        # loop
        for block in self.blocks:
            w = block[1][0] - block[0][0]
            if w<=self.minsize:
                Bsize.append(True)
            else:
                Bsize.append(False)

        # All done
        return Bsize
