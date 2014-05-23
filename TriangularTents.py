'''
A parent class that deals with piece-wise linear triangular fault discretization

Written by Junle Jiang Apr, 2014

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
from .TriangularPatches import TriangularPatches


class TriangularTents(TriangularPatches):

    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(TriangularTents, self).__init__(name, utmzone, ellps)

        # Specify the type of patch
        self.patchType = 'triangletent'
        self.area = None
        self.area_tent = None

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

        raise NotImplementedError('do not need this')

        # Set depth
        self.top = top
        self.numz = nump
        self.width = width

        # All done
        return


    def getTentInfo(self, tent):
        '''
        Returns the geometry info related to vertex-based tent parameterization
        Args:
            * tent         : index of the wanted tent or tent;
        '''

        # Get the patch
        u = None
        if tent.__class__ is int:
            u = tent
        else:
            for i in range(len(self.tent)):
                if (self.tent[i]==tent).all():
                    u = i

        x, y, z = self.tent[u]
        strike, dip = 0, 0
        nbr_faces = self.adjacencyMapVT[self.tentid[u]]

        for pid in nbr_faces:
            xc, yc, zc, w, l, stk, dp = self.getpatchgeometry(pid, center=True)
            strike += stk
            dip += dp

        strike /= sum(nbr_faces)
        dip /= sum(nbr_faces)

        # All done
        return x, y, z, strike, dip


    def deleteTent(self, tent):
        '''
        Deletes a tent.
        Args:
            * tent     : index of the tent to remove.
        '''

        # Remove the patch
        del self.tent[tent]
        del self.tentll[tent]
        del self.tentid[tent]

        if self.N_slip!=None or self.N_slip==len(self.tent):
            self.slip = np.delete(self.slip, tent, axis=0)
            self.N_slip = len(self.slip)
            self.numtent -= 1
        else:
            raise NotImplementedError('Only works for len(slip)==len(tent)', self.N_slip, len(self.slip), len(self.tent))

        # All done
        return


    def deleteTents(self, tents):
        '''
        Deletes a list of tents.
        '''

        while len(tents)>0:

            # Get index to delete
            i = tents.pop()

            # delete it
            self.deleteTent(i)

            # Upgrade list
            for u in range(len(tents)):
                if tents[u]>i:
                    tents[u] -= 1

        # All done
        return


    def addTent(self, tent, slip=[0, 0, 0]):
        '''
        Append a tent to the current list.
        Args:
            * tent      : tent to add
            * slip      : List of the strike, dip and tensile slip.
        '''

        # append the patch
        self.tent.append(tent)
        z = tent[2]
        lon, lat = self.xy2ll(tent[0], tent[1])
        self.tentll.append([lon, lat, z])

        # modify the slip
        if self.N_slip!=None and self.N_slip==len(self.tent)-1:
            sh = self.slip.shape
            nl = sh[0] + 1
            nc = 3
            tmp = np.zeros((nl, nc))
            if nl > 1:                      # Case where slip is empty
                tmp[:nl-1,:] = self.slip
            tmp[-1,:] = slip
            self.slip = tmp
            self.N_slip = len(self.slip)
        else:
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # All done
        return


    def addTents(self, tents, slip=None):
        '''
        Adds a tent to the list.
        Args:
            * tents      : tent to add
            * slip      : List of the strike, dip and tensile slip.
        '''
        if (slip is None) or (slip == [0, 0, 0]):
            slip = np.zeros((len(tents),3))

        for i in range(len(tents)):
            self.addTent(tents[i], list(slip[i,:]))

        # All done
        return


    def writePatches2File(self, filename, add_slip=None, scale=1.0, stdh5=None, decim=1):
        '''
        Writes the patch corners in a file that can be used in psxyz.
        Args:
            * filename      : Name of the file.
            * add_slip      : Put the slip as a value for the color.
                              Can be None, strikeslip, dipslip, total.
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
        '''

        # Check size
        if add_slip is not None:
            if self.N_slip!=None and self.N_slip!=len(self.patch):
                raise NotImplementedError('Only works for len(slip)==len(patch)')

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
        for pIndex in range(nPatches):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip is 'strikeslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex])
                    else:
                        slp = self.slip[pIndex,0]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'dipslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex+nPatches])
                    else:
                        slp = self.slip[pIndex,1]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'total':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex]**2 + samples[:,pIndex+nPatches]**2)
                    else:
                        slp = np.sqrt(self.slip[pIndex,0]**2 + self.slip[pIndex,1]**2)*scale
                    string = '-Z{}'.format(slp)

            # Put the slip value
            slipstring = ''
            if add_slip is not None:
                slipstring = ' # {} {} {} '.format(self.slip[pIndex,0],
                                                   self.slip[pIndex,1], self.slip[pIndex,2])
            parameter = ''

            # Write the string to file
            fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Write the 3 patch corners (the order is to be GMT friendly)
            p = self.patchll[pIndex]
            pp = p[0]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp = p[1]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp = p[2]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))

        # Close the file
        fout.close()

        # Close h5 file if it is open
        if stdh5 is not None:
            h5fid.close()

        # All done
        return


    def writeNodes2File(self, filename, add_slip=None, scale=1.0, stdh5=None, decim=1):
        '''
        Writes the tent node in a file that can be used in psxyz.
        Args:
            * filename      : Name of the file.
            * add_slip      : Put the slip as a value for the color.
                              Can be None, strikeslip, dipslip, total.
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.tent):
            raise NotImplementedError('Only works for len(slip)==len(tent)')

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
        nTents = len(self.tent)
        for tIndex in range(nTents):

            # Select the string for the color
            if add_slip is not None:
                if add_slip is 'strikeslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,tIndex])
                    else:
                        slp = self.slip[tIndex,0]*scale
                elif add_slip is 'dipslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,tIndex+nPatches])
                    else:
                        slp = self.slip[tIndex,1]*scale
                elif add_slip is 'total':
                    if stdh5 is not None:
                        slp = np.std(samples[:,tIndex]**2 + samples[:,tIndex+nPatches]**2)
                    else:
                        slp = np.sqrt(self.slip[tIndex,0]**2 + self.slip[tIndex,1]**2)*scale

            # Write the node
            p = self.tentll[tIndex]
            fout.write('{} {} {} {}\n'.format(p[0], p[1], p[2], slp))

        # Close the file
        fout.close()

        # Close h5 file if it is open
        if stdh5 is not None:
            h5fid.close()

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

        raise NotImplementedError('fault should be pre-discretized')


    def computeTentArea(self):
        '''
        Computes the effective area for each node (1/3 of the summed area of all neighbor triangles)
        '''

        # Area
        if self.area is None:
            self.computeArea()

        self.area_tent = []

        areas = np.array(self.area)
        # Loop over vertices
        for i in range(self.numtent):
            vid = self.tentid[i]

            # find the triangle neighbors for each vertex
            nbr_triangles = self.adjacencyMapVT[vid]
            area = 1./3 * np.sum(areas[nbr_triangles])
            self.area_tent.append(area)

        # All done
        return

    def readGocadPatches(self, filename, neg_depth=False, utm=False, factor_xy=1.0,
                         factor_depth=1.0, box=None):
        """
        Load a triangulated Gocad surface file. Vertices must be in geographical coordinates.
        Args:
            * filename:  tsurf file to read
            * neg_depth: if true, use negative depth
            * utm: if true, input file is given as utm coordinates (if false -> lon/lat)
            * factor_xy: if utm==True, multiplication factor for x and y
            * factor_depth: multiplication factor for z
        """
        # Initialize the lists of patches
        self.patch   = []
        self.patchll = []
        self.tent   = []
        self.tentll = []
        self.tentid   = []

        # Factor to correct input negative depths (we want depths to be positive)
        if neg_depth:
            negFactor = -1.0
        else:
            negFactor =  1.0

        # Get the geographic vertices and connectivities from the Gocad file
        with open(filename, 'r') as fid:
            vertices = []
            vids     = []
            faces    = []
            for line in fid:
                if line.startswith('VRTX'):
                    items = line.split()
                    name, vid, x, y, z = items[:5]
                    vids.append(vid)
                    vertices.append([float(x), float(y), negFactor*float(z)])
                elif line.startswith('TRGL'):
                    name, p1, p2, p3 = line.split()
                    faces.append([int(p1), int(p2), int(p3)])
            fid.close()
            vids = np.array(vids,dtype=int) - 1
            i    = np.argsort(vids)
            vertices = np.array(vertices, dtype=float)[i,:]
            faces = np.array(faces, dtype=int) - 1

        # Resample vertices to UTM
        if utm:
            vx = vertices[:,0].copy()*factor_xy
            vy = vertices[:,1].copy()*factor_xy
            vertices[:,0],vertices[:,1] = self.xy2ll(vx,vy)
        else:
            vx, vy = self.ll2xy(vertices[:,0], vertices[:,1])
        vz = vertices[:,2]*factor_depth
        self.gocad_vertices = np.column_stack((vx, vy, vz))
        self.gocad_vertices_ll = vertices
        self.gocad_faces = faces
        print('min/max depth: {} km/ {} km'.format(vz.min(),vz.max()))
        print('min/max lat: {} deg/ {} deg'.format(vertices[:,1].min(),vertices[:,1].max()))
        print('min/max lon: {} deg/ {} deg'.format(vertices[:,0].min(),vertices[:,0].max()))
        print('min/max x: {} km/ {} km'.format(vx.min(),vx.max()))
        print('min/max y: {} km/ {} km'.format(vy.min(),vy.max()))

        # Loop over faces and create a triangular patch consisting of coordinate tuples
        self.numpatch = faces.shape[0]
        for i in range(self.numpatch):
            # Get the indices of the vertices
            v1, v2, v3 = faces[i,:]
            # Get the coordinates
            x1, y1, lon1, lat1, z1 = vx[v1], vy[v1], vertices[v1,0], vertices[v1,1], vz[v1]
            x2, y2, lon2, lat2, z2 = vx[v2], vy[v2], vertices[v2,0], vertices[v2,1], vz[v2]
            x3, y3, lon3, lat3, z3 = vx[v3], vy[v3], vertices[v3,0], vertices[v3,1], vz[v3]
            # Make the coordinate tuples
            p1 = [x1, y1, z1]; pll1 = [lon1, lat1, z1]
            p2 = [x2, y2, z2]; pll2 = [lon2, lat2, z2]
            p3 = [x3, y3, z3]; pll3 = [lon3, lat3, z3]
            # Store the patch
            self.patch.append([p1, p2, p3])
            self.patchll.append([pll1, pll2, pll3])

        # Loop over vetices and create a node-based tent consisting of coordinate tuples
        self.numtent = vertices.shape[0]
        for i in range(self.numtent):
            # Get the coordinates
            x, y, lon, lat, z = vx[i], vy[i], vertices[i,0], vertices[i,1], vz[i]
            # Make the coordinate tuples
            p = [x, y, z]; pll = [lon, lat, z]
            # Store the patch
            self.tent.append(p)
            self.tentll.append(pll)
            self.tentid.append(i)

        # Update the depth of the bottom of the fault
        if neg_depth:
            self.top   = np.max(vz)
            self.depth = np.min(vz)
        else:
            self.top   = np.min(vz)
            self.depth = np.max(vz)
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # All done
        return


    def buildGFs(self, data, vertical=True, slipdir='sd', verbose=True):
        '''
        Builds the Green's function matrix based on the discretized fault.
        Args:
            * data      : data object from gpsrates or insarrates.
            * vertical  : if True, will produce green's functions for the vertical
                          displacements in a gps object.
            * slipdir   : direction of the slip along the patches. can be any combination
                          of s (strikeslip), d (dipslip) and t (tensile).

        The Green's function matrix is stored in a dictionary. Each entry of the
        dictionary is named after the corresponding dataset. Each of these entry is a
        dictionary that contains 'strikeslip', 'dipslip' and/or 'tensile'.
        '''

        if verbose:
            print("Building Green's functions for the data set {} of type {}".format(
                    data.name, data.dtype))

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
            vertical = True # In InSAR, you need to use the vertical, no matter what....
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
        if 's' in slipdir:              # If strike slip is asked
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
            if verbose:
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
        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # All done
        return


    def buildAdjacencyMapVT(self, verbose=True):
        """
        For each triangle vertex, find the indices of the adjacent triangles.
        """
        if verbose:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Finding the adjacent triangles for all vertices")

        self.adjacencyMapVT = []

        # Cache the vertices and faces arrays
        vertices, faces = self.gocad_vertices, self.gocad_faces

        # First find adjacent triangles for all triangles
        numvert = len(vertices)
        numface = len(faces)

        for i in range(numvert):

            sys.stdout.write('%i / %i\r' % (i + 1, numvert))
            sys.stdout.flush()

            # Find triangles that share an edge
            adjacents = []
            for j in range(numface):
                if i in faces[j,:]:
                    adjacents.append(j)

            self.adjacencyMapVT.append(adjacents)

        print('\n')
        return


    def buildLaplacian(self, extra_params=None, verbose=True):
        """
        Build a discrete Laplacian smoothing matrix.
        """
        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the Laplacian matrix")

        if self.adjacencyMap is None or len(self.adjacencyMap) != len(self.patch):
            assert False, 'Must run self.buildAdjacencyMap() first'

        # Pre-compute patch centers
        centers = self.getcenters()

        # Cache the vertices and faces arrays
        vertices, faces = self.gocad_vertices, self.gocad_faces

        # Allocate array for Laplace operator
        npatch = len(self.patch)
        D = np.zeros((npatch,npatch))

        # Loop over patches
        for i in range(npatch):

            sys.stdout.write('%i / %i\r' % (i, npatch))
            sys.stdout.flush()

            # Center for current patch
            refCenter = np.array(centers[i])

            # Compute Laplacian using adjacent triangles
            hvals = []
            adjacents = self.adjacencyMap[i]
            for index in adjacents:
                pcenter = np.array(centers[index])
                dist = np.linalg.norm(pcenter - refCenter)
                hvals.append(dist)
            if len(hvals) == 3:
                h12, h13, h14 = hvals
                D[i,adjacents[0]] = h13*h14
                D[i,adjacents[1]] = h12*h14
                D[i,adjacents[2]] = h12*h13
                sumProd = h13*h14 + h12*h14 + h12*h13
            elif len(hvals) == 2:
                h12, h13 = hvals
                # Make a virtual patch
                h14 = max(h12, h13)
                D[i,adjacents[0]] = h13*h14
                D[i,adjacents[1]] = h12*h14
                sumProd = h13*h14 + h12*h14 + h12*h13
            D[i,i] = -sumProd

        print('\n')
        D = D / np.max(np.abs(np.diag(D)))
        return D


    def writeEDKSsubParams(self, data, edksfilename, amax=None, plot=False, w_file=True):
        '''
        Write the subParam file needed for the interpolation of the green's function in EDKS.
        Francisco's program cuts the patches into small patches, interpolates the kernels to get the GFs at each point source,
        then averages the GFs on the pacth. To decide the size of the minimum patch, it uses St Vernant's principle.
        If amax is specified, the minimum size is fixed.
        Args:
            * data          : Data object from gpsrates or insarrates.
            * edksfilename  : Name of the file containing the kernels
            * amax          : Specifies the minimum size of the divided patch. If None, uses St Vernant's principle (default=None)
            * plot          : Activates plotting (default=False)
            * w_file        : if False, will not write the subParam fil (default=True)
        Returns:
            * filename         : Name of the subParams file created (only if w_file==True)
            * TrianglePropFile : Name of the triangle properties file
            * PointCoordFile   : Name of the Point coordinates file
            * ReceiverFile     : Name of the receiver file
            * method_par       : Dictionary including useful EDKS parameters
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Write the EDKS files for fault {} and data {}".format(self.name, data.name))

        # Write the geometry to the EDKS file
        fltname,TrianglePropFile, PointCoordFile = self.writeEDKSgeometry()

        # Write the data to the EDKS file
        datname,ReceiverFile = data.writeEDKSdata()

        # Assign some EDKS parameters
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
            fout.write("# File with the triangle properties\n")
            fout.write("TriPropFile = '{}'\n".format(TrianglePropFile))
            fout.write("# File with the Triangles' Points (vertex) coordinates \n")
            fout.write("TriPointsFile = '{}'\n".format(PointCoordFile))
            fout.write("# File with id, E[km], N[km] coordinates of the receivers.\n")
            fout.write("ReceiverFile = '{}'\n".format(ReceiverFile))
            fout.write("# read receiver direction (# not yet implemented)\n")
            fout.write("useRecvDir = {} # leace this to False for now\n".format(useRecvDir))
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
            return filename, TrianglePropFile, PointCoordFile, ReceiverFile, method_par
        else:
            return TrianglePropFile, PointCoordFile, ReceiverFile, method_par

    def writeEDKSgeometry(self, ref=None):
        '''
        This routine spits out 2 files:
        filename.TriangleProp: Patch ID | Lon (deg) | Lat | East (km) | North | Depth (km) | Strike (deg) | Dip  | Area (km^2) | Vertice ids
        (coordinates are given for the center of the patch)
        filename.PointCoord: Vertice ID | Lon (deg) | Lat | East (km) | North | Depth (km)

        These files are to be used with edks/MPI_EDKS/calcGreenFunctions_EDKS_subTriangles.py
        '''

        # Filename
        if len(self.name.split())>1:
            fltname = self.name.split()[0]
            for s in self.name.split()[1:]:
                fltname = fltname+'_'+s
        else:
            fltname = self.name
        filename = 'edks_{}'.format(fltname)
        TrianglePropFile = filename+'.TriangleProp'
        PointCoordFile   = filename+'.PointCoord'

        # Open the output file and write headers
        TriP = open(TrianglePropFile,'w')
        h_format ='%-6s %10s %10s %10s %10s %10s %10s %10s %10s %6s %6s %6s\n'
        h_tuple = ('%Tid','lon','lat','E[km]','N[km]','dep[km]','strike','dip',
                  'Area[km2]','idP1','idP2','idP3')
        TriP.write(h_format%h_tuple)
        PoC = open(PointCoordFile,'w')
        PoC.write('%-6s %10s %10s %10s %10s %10s\n'%('Pid','lon','lat','E[km]','N[km]','dep[km]'))

        # Reference
        if ref is not None:
            refx, refy = self.putm(ref[0], ref[1])
            refx /= 1000.
            refy /= 1000.

        # Loop over the patches
        TriP_format = '%-6d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %6d %6d %6d\n'
        for p in range(self.numpatch):
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            lonc, latc = self.xy2ll(xc,yc)
            if ref is not None:
                xc -= refx
                yc -= refy
            verts = copy.deepcopy(self.patch[p])
            v1, v2, v3 = self.gocad_faces[p,:]
            TriP_tuple = (p,lonc,latc,xc,yc,zc,strike,dip,self.area[p],v1,v2,v3)
            TriP.write(TriP_format%TriP_tuple)

        PoC_format =  '%-6d %10.4f %10.4f %10.4f %10.4f %10.4f\n'
        for v in range(self.gocad_vertices.shape[0]):
            xv,yv,zv = self.gocad_vertices[v,:]
            lonv,latv,zv2 = self.gocad_vertices_ll[v,:]
            assert zv == zv2*1.0e-3, 'inconsistend depth in gocad_vertices and gocad_vertices_ll'
            if ref is not None:
                xv -= refx
                yv -= refy
            PoC.write(PoC_format%(v,lonv,latv,xv,yv,zv))

        # Close the files
        TriP.close()
        PoC.close()

        # All done
        return fltname,TrianglePropFile,PointCoordFile


    def getEllipse(self, tent, ellipseCenter=None, Npoints=10, factor=1.0):
        '''
        Compute the ellipse error given Cm for a given tent
        args:
               (optional) center  : center of the ellipse
               (optional) Npoints : number of points on the ellipse
        '''

        # Get Cm
        Cm = np.diag(self.Cm[tent,:2])
        Cm[0,1] = Cm[1,0] = self.Cm[tent,2]

        # Get strike and dip
        xc, yc, zc, strike, dip = self.getTentInfo(tent)
        dip *= np.pi/180.
        strike *= np.pi/180.
        if ellipseCenter!=None:
            xc, yc, zc = ellipseCenter

        # Compute eigenvalues/eigenvectors
        D,V = np.linalg.eig(Cm)
        v1 = V[:,0]
        a = np.sqrt(np.abs(D[0]))
        b = np.sqrt(np.abs(D[1]))
        phi = np.arctan2(v1[1],v1[0])
        theta = np.linspace(0,2*np.pi,Npoints);

        # The ellipse in x and y coordinates
        Ex = a * np.cos(theta) * factor
        Ey = b * np.sin(theta) * factor

        # Correlation Rotation
        R  = np.array([[np.cos(phi), -np.sin(phi)],
                       [np.sin(phi), np.cos(phi)]])
        RE = np.dot(R,np.array([Ex,Ey]))

        # Strike/Dip rotation
        ME = np.array([RE[0,:], RE[1,:] * np.cos(dip), RE[1,:]*np.sin(dip)])
        R  = np.array([[np.sin(strike), -np.cos(strike), 0.0],
                       [np.cos(strike), np.sin(strike), 0.0],
                       [0.0, 0.0, 1.]])
        RE = np.dot(R,ME).T

        # Translation on Fault
        RE[:,0] += xc
        RE[:,1] += yc
        RE[:,2] += zc

        # All done
        return RE


    def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0,
                                neg_depth=False, ellipse=False):
        '''
        Write a psxyz compatible file to draw lines starting from the center of each patch,
        indicating the direction of slip.
        Tensile slip is not used...
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Copmute the slip direction
        self.computeSlipDirection(scale=scale, factor=factor, ellipse=ellipse)

        # Write something
        print('Writing slip direction to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        for p in self.slipdirection:

            # Write the > sign to the file
            fout.write('> \n')

            # Get the start of the vector (node)
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
                    fout.write('{} {} {} \n'.format(lon, lat, -1.*z))
            # Close file
            fout.close()

        # All done
        return


    def computeSlipDirection(self, scale=1.0, factor=1.0, ellipse=False):
        '''
        Computes the segment indicating the slip direction.
            * scale : can be a real number or a string in 'total', 'strikeslip',
                                    'dipslip' or 'tensile'
        '''

        # Create the array
        self.slipdirection = []

        # Check Cm if ellipse
        if ellipse:
            self.ellipse = []
            assert(self.Cm!=None), 'Provide Cm values'

        # Loop over the patches
        if self.N_slip == None:
            self.N_slip = self.numtent

        for tid in range(self.N_slip):
            # Get some geometry
            xc, yc, zc, strike, dip = self.getTentInfo(tid)

            # Get the slip vector
            #slip = self.getslip(self.patch[p]) # This is weird
            slip = self.slip[tid, :]
            rake = np.arctan2(slip[1], slip[0])

            x = (np.sin(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake))
            y = (np.cos(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake))
            z =  1.0*np.sin(dip)*np.sin(rake)

            # Scale these
            if scale.__class__ is float:
                sca = scale
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
                self.ellipse.append(self.getEllipse(tid, ellipseCenter=[xe, ye, ze], factor=factor))

            # Append slip direction
            self.slipdirection.append([[xc, yc, zc], [xe, ye, ze]])

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



    def surfacesimulation(self, box=None, disk=None, err=None, npoints=None, lonlat=None,
                          slipVec=None):
        '''
        Takes the slip vector and computes the surface displacement that corresponds on a regular grid.
        Args:
            * box       : Can be a list of [minlon, maxlon, minlat, maxlat].
            * disk      : list of [xcenter, ycenter, radius, n]
            * lonlat    : Arrays of lat and lon. [lon, lat]
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

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
        for i in xrange(1,x.shape[0]):
            d = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            dis[i] = dis[i-1] + d

        # all done
        return dis

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
            assert (dlon==lon).all(), '{} dataset rejected'.format(sar.name)
            assert (dlat==lat).all(), '{} dataset rejected'.format(sar.name)

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

    def ExtractAlongStrikeVariationsOnDiscretizedFault(self, depth=0.5, filename=None, discret=0.5):
        '''
        Extracts the Along Strike variations of the slip at a given depth, resampled along the discretized fault trace.
        Args:
            depth       : Depth at which we extract the along strike variations of slip.
            discret     : Discretization length
            filename    : Saves to a file.
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
            self.discretize(every=discret, tol=discret/10., fracstep=discret/12.)
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
        ssint = sciint.interp1d(dPatches, [sPatches[i][0] for i in range(len(sPatches))], kind='linear', bounds_error=False)
        dsint = sciint.interp1d(dPatches, [sPatches[i][1] for i in range(len(sPatches))], kind='linear', bounds_error=False)
        tsint = sciint.interp1d(dPatches, [sPatches[i][2] for i in range(len(sPatches))], kind='linear', bounds_error=False)

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

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the List where we will store things
        # For each patch, it will be [lon, lat, strike-slip, dip-slip, tensile, distance]
        Var = []

        # Creates the orientation vector
        Dir = np.array([np.cos(orientation*np.pi/180.), np.sin(orientation*np.pi/180.)])

        # initialize the origin
        x0 = 0
        y0 = 0
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
                vec = np.array([x0-xc, y0-yc])
                sign = np.sign( np.dot(Dir,vec) )
                dist = sign * np.sqrt( (xc-x0)**2 + (yc-y0)**2 )

                # Assemble
                o = [lonc, latc, slip[0], slip[1], slip[2], area, dist]

                # write output
                if filename is not None:
                    fout.write('{} {} {} {} {} {} \n'.format(lonc, latc, slip[0], slip[1], slip[2], area, dist))

                # append
                Var.append(o)

        # Close the file
        if filename is not None:
            fout.close()

        # Stores it
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # all done
        return


    def plot(self, ref='utm', figure=134, add=False, maxdepth=None, axis='equal',
             value_to_plot='total', neg_depth=False):
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

        # Sign factor for negative depths
        if neg_depth:
            negFactor = 1.0
        else:
            negFactor = -1.0

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
            ax.set_zlim3d([negFactor * (self.depth - negFactor * 5), 0])
            zticks = []
            zticklabels = []
            for z in self.z_patches:
                zticks.append(negFactor * z)
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
                        z.append(negFactor * self.patch[p][i][2])
                    else:
                        x.append(self.patchll[p][i][0])
                        y.append(self.patchll[p][i][1])
                        z.append(negFactor * self.patchll[p][i][2])
                verts = [zip(x, y, z)]
                rect = art3d.Poly3DCollection(verts)
                rect.set_color(scalarMap.to_rgba(plotval[p]))
                rect.set_edgecolors('k')
                ax.add_collection3d(rect)

            # put up a colorbar
            scalarMap.set_array(plotval)
            plt.colorbar(scalarMap)

        # Depth
        if maxdepth is not None:
            ax.set_zlim3d([neg_factor * maxdepth, 0])

        # show
        plt.show()

        # All done
        return

#EOF
