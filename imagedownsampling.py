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
from .insarrates import insarrates
from .cosicorrrates import cosicorrrates

class imagedownsampling(object):

    def __init__(self, name, image, faults, verbose=True):
        '''
        Args:
            * name      : Name of the downsampler.
            * image    : InSAR or Cosicorr data set to be downsampled.
            * faults    : List of faults.
        '''

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR downsampling tools {}".format(name))

        self.verbose = verbose

        # Set the name
        self.name = name
        self.datatype = image.dtype

        # Set the transformation
        self.utmzone = image.utmzone
        self.putm = image.putm
        self.ll2xy = image.ll2xy
        self.xy2ll = image.xy2ll

        # Check if the faults are in the same utm zone
        self.faults = []
        for fault in faults:
            assert (fault.utmzone==self.utmzone), 'Fault {} not in utm zone #{}'.format(fault.name, self.utmzone)
            self.faults.append(fault)

        # Save the image
        self.image = image

        # Incidence and heading need to be defined
        if self.datatype is 'insarrates':
            assert hasattr(self.image, 'heading'), 'No Heading precised for image object'
            assert hasattr(self.image, 'incidence'), 'No Incidence precised for image object'
            self.incidence = self.image.incidence
            self.heading = self.image.heading

        # Create the initial box
        xmin = np.floor(image.x.min())
        xmax = np.floor(image.x.max())+1.
        ymin = np.floor(image.y.min())
        ymax = np.floor(image.y.max())+1.
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.box = [[xmin, ymin],
                    [xmin, ymax],
                    [xmax, ymax],
                    [xmax, ymin]]
        lonmin = image.lon.min()
        lonmax = image.lon.max()
        latmin = image.lat.min()
        latmax = image.lat.max()
        self.lonmin = lonmin; self.latmax = latmax
        self.latmin = latmin; self.lonmax = lonmax
        self.boxll = [[lonmin, latmin],
                      [lonmin, latmax],
                      [lonmax, latmax],
                      [lonmax, latmin]]
        
        # Get the original pixel spacing
        self.spacing = distance.cdist([[image.x[0], image.y[0]]], [[image.x[i], image.y[i]] for i in range(1, image.x.shape[0])])[0]
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

        # Create the new image object
        if self.datatype is 'insarrates':
            newimage = insarrates('Downsampled {}'.format(self.image.name), utmzone=self.utmzone, verbose=False)
        elif self.datatype is 'cosicorrrates':
            newimage = cosicorrrates('Downsampled {}'.format(self.image.name), utmzone=self.utmzone, verbose=False)

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
        if self.datatype is 'insarrates':
            newimage.vel = []
            newimage.err = []
        elif self.datatype is 'cosicorrrates':
            newimage.east = []
            newimage.north = []
            newimage.err_east = []
            newimage.err_north = []
        newimage.lon = []
        newimage.lat = []
        newimage.x = []
        newimage.y = []
        newimage.wgt = []

        # Store the factor
        newimage.factor = self.image.factor

        # Build the previous geometry
        PIXXY = np.vstack((self.image.x, self.image.y)).T

        # Keep track of the blocks to trash
        blocks_to_remove = []

        # Over each block, we average the position and the phase to have a new point
        for i in range(len(blocks)):
            block = blocks[i]
            # Create a path
            p = path.Path(block, closed=False)
            # Find those who are inside
            ii = p.contains_points(PIXXY)
            # Check if total area is sufficient
            blockarea = self.getblockarea(block)
            coveredarea = np.flatnonzero(ii).shape[0]*self.pixelArea
            if (coveredarea/blockarea >= self.tolerance):
                # Get Mean, Std, x, y, ...
                wgt = len(np.flatnonzero(ii))
                if self.datatype is 'insarrates':
                    vel = np.mean(self.image.vel[ii])
                    err = np.std(self.image.vel[ii])
                elif self.datatype is 'cosicorrrates':
                    east = np.mean(self.image.east[ii])
                    north = np.mean(self.image.north[ii])
                    err_east = np.std(self.image.east[ii])
                    err_north = np.std(self.image.north[ii])
                x = np.mean(self.image.x[ii])
                y = np.mean(self.image.y[ii])
                lon, lat = self.xy2ll(x, y)
                # Store that
                if self.datatype is 'insarrates':
                    newimage.vel.append(vel)
                    newimage.err.append(err)
                elif self.datatype is 'cosicorrrates':
                    newimage.east.append(east)
                    newimage.north.append(north)
                    newimage.err_east.append(err_east)
                    newimage.err_north.append(err_north)
                newimage.x.append(x)
                newimage.y.append(y)
                newimage.lon.append(lon)
                newimage.lat.append(lat)
                newimage.wgt.append(wgt)
            else:
                blocks_to_remove.append(i)

        # Clean up useless blocks
        self.trashblocks(blocks_to_remove)

        # Convert
        if self.datatype is 'insarrates':
            newimage.vel = np.array(newimage.vel)
            newimage.err = np.array(newimage.err)
        elif self.datatype is 'cosicorrrates':
            newimage.east = np.array(newimage.east)
            newimage.north = np.array(newimage.north)
            newimage.err_east = np.array(newimage.err_east)
            newimage.err_north = np.array(newimage.err_north)
        newimage.x = np.array(newimage.x)
        newimage.y = np.array(newimage.y)
        newimage.lon = np.array(newimage.lon)
        newimage.lat = np.array(newimage.lat)
        newimage.wgt = np.array(newimage.wgt)

        # LOS
        if self.datatype is 'insarrates':
            newimage.inchd2los(self.incidence, self.heading)

        # Store newimage
        self.newimage = newimage

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

    def ResolutionBasedIterations(self, threshold, damping, slipdirection='s', plot=False, verboseLevel='minimum'):
        '''
        Iteratively downsamples the dataset until value compute inside each block is lower than the threshold.
        Args:
            * threshold     : Threshold.
            * damping       : Damping coefficient (damping is made through an identity matrix).   
            * slipdirection : Which direction to accout for to build the slip Green's functions.
        '''
        
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Downsampling Iterations")

        # Creates the variable that is supposed to stop the loop
        # Check = [False]*len(self.blocks)
        self.Rd = np.ones(len(self.blocks),)*(threshold+1.)
        do_cut = False

        # counter
        it = 0

        # Check if block size is minimum
        Bsize = self._is_minimum_size(self.blocks)

        # Loops until done
        while not (self.Rd<threshold).all():

            # Check 
            assert self.Rd.shape[0]==len(self.blocks), 'Resolution matrix has a size different than number of blocks'

            # Cut if asked
            if do_cut:
                # New list of blocks
                newblocks = []
                # Iterate over blocks
                for j in range(len(self.blocks)):
                    block = self.blocks[j]
                    if (self.Rd[j]>threshold) and not Bsize[j]:
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
                fault.buildGFs(self.newimage, vertical=False, slipdir=slipdirection, verbose=False)
                fault.assembleGFs([self.newimage], polys=0, slipdir=slipdirection, verbose=False)
                # Cat GFs
                if G is None:
                    G = fault.Gassembled
                else:
                    G = np.hstack((G, fault.Gassembled))

            # Compute the data resolution matrix
            Npar = G.shape[1]
            Ndat = G.shape[0]/2 # vertical is False
            Ginv = np.dot(np.linalg.inv(np.dot(G.T,G)+ damping*np.eye(Npar)),G.T)
            Rd = np.dot(G, Ginv)
            self.Rd = np.diag(Rd).copy()

            # If we are dealing with cosicorr data, the diagonal is twice as long as the umber of blocks
            if self.datatype is 'cosicorrrates':
                self.Rd = np.sqrt( self.Rd[:Ndat]**2 + self.Rd[-Ndat:]**2 )

            # Blocks that have a minimum size, don't check these
            Bsize = self._is_minimum_size(self.blocks)
            self.Rd[np.where(Bsize)] = 0.0

            if self.verbose and verboseLevel is not 'minimum':
                sys.stdout.write(' ===> Resolution from {} to {}, Mean = {} +- {} \n'.format(self.Rd.min(), self.Rd.max(), self.Rd.mean(), self.Rd.std()))
                sys.stdout.flush()
    
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

    def plotDownsampled(self, figure=145, axis='equal', ref='utm', Norm=None, data2plot='north'):
        '''
        Plots the downsampling as it is at this step.
        Args:
            * figure    : Figure ID.
            * axis      : Axis argument from matplotlib.
            * Norm      : [colormin, colormax]
            * ref       : Can be 'utm' or 'lonlat'.
            * data2plot : used if datatype is cosicorrrates: can be north or east.
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
        original = self.image
        downsampled = self.newimage

        # Get what should be plotted
        if self.datatype is 'insarrates':
            data = original.vel
        elif self.datatype is 'cosicorrrates':
            if data2plot is 'north':
                data = original.north
            elif data2plot is 'east':
                data = original.east
                
        # Vmin, Vmax
        if Norm is not None:
            vmin, vmax = Norm
        else:
            vmin = data.min()
            vmax = data.max()

        # Prepare the colormaps
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # Plot original dataset
        if ref is 'utm':
            # image
            sca = full.scatter(original.x, original.y, s=10, c=data, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.)
            # Faults
            for fault in self.faults:
                full.plot(fault.xf, fault.yf, '-k')
        else:
            # image
            sca = full.scatter(original.lon, original.lat, s=10, c=data, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.) 
            # Faults
            for fault in self.faults:
                full.plot(fault.lon, fault.lat, '-k')

        # Patches
        import matplotlib.collections as colls

        # Get downsampled data
        if self.datatype is 'insarrates':
            downdata = downsampled.vel
        elif self.datatype is 'cosicorrrates':
            if data2plot is 'north':
                downdata = downsampled.north
            elif data2plot is 'east':
                downdata = downsampled.east

        # Plot downsampled
        if ref is 'utm':
            # Image
            for i in range(len(self.blocks)):
                # Get block
                block = self.blocks[i]
                # Get value
                val = downdata[i]
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
            # Image
            for i in range(len(self.blocks)):
                # Get block
                block = self.blockll[i]
                # Get value
                val = downdata[i]
                # Build patch
                x = [blockll[j][0] for j in range(4)]
                y = [blockll[j][1] for j in range(4)] 
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
        
    def writeDownsampled2File(self, prefix, rsp=False):
        '''
        Writes the downsampled image data to a file.
        The file will be called prefix.txt.
        If rsp is True, then it writes a file called prefix.rsp 
        containing the boxes of the downsampling.
        '''

        # Open files
        ftxt = open(prefix+'.txt', 'w')
        if rsp:
            frsp = open(prefix+'.rsp', 'w')

        # Write the header
        if self.datatype is 'insarrates':
            ftxt.write('Number xind yind east north data err wgt Elos Nlos Ulos\n')
        elif self.datatype is 'cosicorrrates':
            ftxt.write('Number Lon Lat East North EastErr NorthErr \n') 
        ftxt.write('********************************************************\n')
        if rsp:
            frsp.write('xind yind UpperLeft-x,y DownRight-x,y\n')
            frsp.write('********************************************************\n')

        # Loop over the samples
        for i in xrange(len(self.newimage.x)):

            # Write in txt
            wgt = self.newimage.wgt[i]
            x = int(self.newimage.x[i])
            y = int(self.newimage.y[i])
            lon = self.newimage.lon[i]
            lat = self.newimage.lat[i]
            if self.datatype is 'insarrates':
                vel = self.newimage.vel[i]
                err = self.newimage.err[i]
                elos = self.newimage.los[i,0]
                nlos = self.newimage.los[i,1]
                ulos = self.newimage.los[i,2]
                strg = '{:4d} {:4d} {:4d} {:3.6f} {:3.6f} {} {} {} {} {} {}\n'\
                    .format(i, x, y, lon, lat, vel, err, wgt, elos, nlos, ulos) 
            elif self.datatype is 'cosicorrrates':
                east = self.newimage.east[i]
                north = self.newimage.north[i]
                err_east = self.newimage.err_east[i]
                err_north = self.newimage.err_north[i]
                strg = '{:4d} {:3.6f} {:3.6f} {} {} {} {} \n'\
                        .format(i, lon, lat, east, north, err_east, err_north)
            ftxt.write(strg)

            # Write in rsp
            if rsp:
                ulx = self.blocks[i][0][0]
                uly = self.blocks[i][0][1]
                drx = self.blocks[i][2][0]
                dry = self.blocks[i][2][1]
                ullon = self.blocksll[i][0][0]
                ullat = self.blocksll[i][0][1]
                drlon = self.blocksll[i][2][0]
                drlat = self.blocksll[i][2][1]
                strg = '{:4d} {:4d} {} {} {} {} {} {} {} {} \n'\
                        .format(x, y, ulx, uly, drx, dry, ullon, ullat, drlon, drlat)
                frsp.write(strg)

        # Close the files
        ftxt.close()
        if rsp:
            frsp.close()

        # All done
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
