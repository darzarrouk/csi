'''
A parent class that deals with piece-wise linear triangular fault discretization

Written by Junle Jiang Apr, 2014

'''

# Externals
import numpy as np
import pyproj as pp
import inspect
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from . import triangularDisp as tdisp
from scipy.linalg import block_diag
import copy
import sys
import os

# Personals
from .TriangularPatches import TriangularPatches
from .geodeticplot import geodeticplot as geoplot

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

    def initializeFromFault(self, fault, adjMap=True):
        '''
        Initializes the tent fault object from a triangularPatches or a triangularTents instance.
        Args:
            * fault     : Instance of triangular fault.
            * adjMap    : Build the adjacency map (True/False).
        '''

        # Assert
        assert type(fault) in (TriangularPatches, type(self)), 'Input fault type must be {} or {}'.format(TriangularPatches, type(self))

        # Create a list of the attributes we want to copy
        Attributes = []
        for attr in dir(fault):
            if (attr[:2] not in ('__')) and (not inspect.ismethod(fault.__getattribute__(attr))):
                Attributes.append(attr)

        # Loop 
        for attr in Attributes:
            self.__setattr__(attr, fault.__getattribute__(attr))

        # patchType
        self.patchType = 'triangletent'

        # Vertices2Tents
        self.vertices2tents()

        # Build the adjacency map
        if adjMap:
            self.buildAdjacencyMap()

        # Slip
        self.initializeslip()

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

    def getStrikes(self):
        '''
        Returns the strikes of each nodes.
        '''

        # All done in one line
        return np.array([self.getTentInfo(t)[3] for t in self.tent])

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
            u = self.getTentindex(tent)

        x, y, z = self.tent[u]
        strike, dip = 0, 0
        nbr_faces = self.adjacencyMap[self.tentid[u]]

        for pid in nbr_faces:
            xc, yc, zc, w, l, stk, dp = self.getpatchgeometry(pid, center=True)
            strike += stk
            dip += dp

        strike /= len(nbr_faces)
        dip /= len(nbr_faces)

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
        Deletes a list of tent (indices)
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


    def chooseTents(self, tents):
        '''
        Choose a subset of tents (indices)
        '''
        tent_new = []
        tentll_new = []
        tentid_new = []

        for it in tents:
            tent_new.append(self.tent[it])
            tentll_new.append(self.tentll[it])
            tentid_new.append(self.tentid[it])

        self.tent = tent_new
        self.tentll = tentll_new
        self.tentid = tentid_new

        self.slip = self.slip[tents,:]
        self.N_slip = len(self.slip)
        self.numtent = len(tents)

        # All done
        return

    def getcenters(self):
        '''
        Returns a list of nodes.
        '''

        # All done
        return self.tent

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

    def initializeslip(self, n=None, values=None):
        '''
        Re-initializes the fault slip array to zero values.
        This function over-writes the function in the parent class Fault.
        Args:
            * n     : Number of slip values. If None, it'll take the number of patches.
            * values: Can be depth, strike, dip, length, area or a numpy array
        '''

        # Shape
        if n is None:
           self.N_slip = len(self.tent)
        else:
            self.N_slip = n

        self.slip = np.zeros((self.N_slip,3))
        
        # Values
        if values is not None:
            # string type
            if type(values) is str:
                if values is 'depth':
                    values = np.array([self.getTentInfo(t)[2] for t in self.tent])
                elif values is 'strike':
                    values = np.array([self.getTentInfo(t)[3] for t in self.tent])
                elif values is 'dip':
                    values = np.array([self.getTentInfo(t)[4] for t in self.tent])
                elif values is 'index':
                    values = np.array([np.float(self.getTentindex(t)) for t in self.tent])
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

    def distanceMatrix(self, distance='center', lim=None):
        '''
        Returns a matrix of the distances between Nodes.
        Args:
            * distance  : distance estimation mode
                            center : distance between the centers of the patches.
                            no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].
        '''

        # Assert 
        assert distance is 'center', 'No other method implemented than center'

        # Check
        if self.N_slip==None:
            self.N_slip = self.slip.shape[0]

        # Loop
        Distances = np.zeros((self.N_slip, self.N_slip))
        for i in range(self.N_slip):
            v1 = self.Vertices[i]
            for j in range(self.N_slip):
                if j == i:
                    continue
                v2 = self.Vertices[j]
                Distances[i,j] = self.distanceVertexToVertex(v1, v2, lim=lim)

        # All done
        return Distances

    def getTentindex(self, tent):
        '''
        Returns the index of a tent.
        This function over-writes that from the parent class Fault.
        '''

        # Output index
        iout = None

        # Find it
        for t in range(len(self.tent)):
            if (self.tent[t] == tent):
                iout = t
        
        # All done
        return iout
                    
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
            nbr_triangles = self.adjacencyMap[vid]
            area = 1./3 * np.sum(areas[nbr_triangles])
            self.area_tent.append(area)

        # All done
        return

    def readGocadPatches(self, filename, neg_depth=False, utm=False, factor_xy=1.0,
                         factor_depth=1.0, box=None, verbose=False):
        """
        Load a triangulated Gocad surface file. Vertices must be in geographical coordinates.
        Args:
            * filename:  tsurf file to read
            * neg_depth: if true, use negative depth
            * utm: if true, input file is given as utm coordinates (if false -> lon/lat)
            * factor_xy: if utm==True, multiplication factor for x and y
            * factor_depth: multiplication factor for z
        """

        # Run the upper level routine 
        super(TriangularTents, self).readGocadPatches(filename, neg_depth=neg_depth, utm=utm,
                    factor_xy=factor_xy, factor_depth=factor_depth, box=box, verbose=verbose)

        # Vertices to Tents
        self.vertices2tents()

        # All done
        return

    def vertices2tents(self):
        '''
        Takes the list of vertices and builds the tents.
        '''

        # Initialize the lists of patches
        self.tent   = []
        self.tentll = []
        self.tentid   = []

        # Get vertices
        vertices = self.Vertices
        verticesll = self.Vertices_ll
        vx, vy, vz = vertices[:,0], vertices[:,1], vertices[:,2]

        # Loop over vetices and create a node-based tent consisting of coordinate tuples
        self.numtent = vertices.shape[0]
        for i in range(self.numtent):
            # Get the coordinates
            x, y, lon, lat, z = vx[i], vy[i], verticesll[i,0], verticesll[i,1], vz[i]
            # Make the coordinate tuples
            p = [x, y, z]; pll = [lon, lat, z]
            # Store the patch
            self.tent.append(p)
            self.tentll.append(pll)
            self.tentid.append(i)

        # Create the adjacency map
        self.buildAdjacencyMap(verbose=False)

        # All done
        return

    def Facet2Nodes(self):
        '''
        Transfers the edksSources list into the node based setup.
        '''
    
        # Get the faces and Nodes
        Faces = np.array(self.Faces)
        Vertices = self.Vertices

        # Get the surrounding triangles
        Nodes = {}

        # Loop for that 
        for nId in self.tentid:
            Nodes[nId] = {'nTriangles': 0, 'idTriangles': []}
            for idFace in range(self.Faces.shape[0]):
                ns = self.Faces[idFace,:].tolist()
                if nId in ns:
                    Nodes[nId]['nTriangles'] += 1
                    Nodes[nId]['idTriangles'].append(idFace)

        # Save the nodes
        self.Nodes = Nodes

        # Create the new edksSources
        Ids = []
        xs = []; ys = []; zs = []
        strike = []; dip = []
        areas = []; slip = []

        # Iterate on the nodes to derive the weights
        for mainNode in Nodes:
            # Loop over each of these triangles
            for tId in Nodes[mainNode]['idTriangles']:
                # Find the sources in edksSources
                iS = np.flatnonzero(self.edksSources[0]==(tId)).tolist()
                # Get the three nodes
                tNodes = Faces[tId]
                # Affect the master node and the two outward nodes
                nodeOne, nodeTwo = tNodes[np.where(tNodes!=mainNode)]
                # Get weights (Attention: Vertices are in km, edksSources in m)
                Wi = self._getWeights(Vertices[mainNode], 
                                      Vertices[nodeOne], 
                                      Vertices[nodeTwo], 
                                      self.edksSources[1][iS]/1000., 
                                      self.edksSources[2][iS]/1000., 
                                      self.edksSources[3][iS]/1000.)
                # Save each source
                Ids += (np.ones((len(iS),))*mainNode).tolist()
                xs += self.edksSources[1][iS].tolist()
                ys += self.edksSources[2][iS].tolist()
                zs += self.edksSources[3][iS].tolist()
                strike += self.edksSources[4][iS].tolist()
                dip += self.edksSources[5][iS].tolist()
                areas += self.edksSources[6][iS].tolist()
                slip += Wi.tolist()

        # Set the new edksSources
        self.edksFacetSources = copy.deepcopy(self.edksSources)
        self.edksSources = [np.array(Ids), np.array(xs), np.array(ys), np.array(zs), 
                np.array(strike), np.array(dip), np.array(areas), np.array(slip)]

        # All done
        return

    def buildAdjacencyMap(self, verbose=True):
        """
        For each triangle vertex, find the indices of the adjacent triangles.
        This function overwrites that from the parent class TriangularPatches.
        """
        if verbose:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Finding the adjacent triangles for all vertices")

        self.adjacencyMap = []

        # Cache the vertices and faces arrays
        vertices, faces = self.Vertices, np.array(self.Faces)

        # First find adjacent triangles for all triangles
        numvert = len(vertices)
        numface = len(faces)

        for i in range(numvert):
            if verbose:
                sys.stdout.write('%i / %i\r' % (i + 1, numvert))
                sys.stdout.flush()

            # Find triangles that share an edge
            adjacents = []
            for j in range(numface):
                if i in faces[j,:]:
                    adjacents.append(j)

            self.adjacencyMap.append(adjacents)

        if verbose:
            print('\n')
        return

    def buildLaplacian(self, extra_params=None, verbose=True):
        """
        Build a discrete Laplacian smoothing matrix.
        """
        
        # No Laplacian method implemented so far
        raise NotImplementedError('Laplacian method not implemented for class {}'.format(self.__class__))

        # All done
        return 

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

            # print("strike, dip, rake, x, y, z: ", strike, dip, rake, x, y, z)

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

    def plot(self, ref='utm', figure=134, add=False, maxdepth=None, axis='equal',
             slip='total', neg_depth=False):
        '''
        Plot the available elements of the fault.

        Args:
            * ref           : Referential for the plot ('utm' or 'lonlat').
            * figure        : Number of the figure.
        '''
        
        # Create a geodetic plot
        fig = geoplot(figure=figure, ref=ref)

        # Trace
        if hasattr(self, 'lon') or hasattr(self, 'xf'):
            fig.faulttrace(self)

        # 3D geometry
        x, y, z, slip = fig.faultTents(self, slip=slip, Norm=None, colorbar=True, plot_on_2d=False, npoints=40)

        # Show
        fig.show(showFig=['fault'])

        # All done
        return x, y, z, slip

    def _getWeights(self, mainNode, nodeOne, nodeTwo, x, y, z):
        '''
        For a triangle given by the coordinates of its summits, compute the weight of
        the points given by x, y, and z positions (these guys need to be inside the triangle).
        Args:
            * mainNode  : [x,y,z] of the main Node
            * nodeOne   : [x,y,z] of the first Node
            * nodeTwo   : [x,y,z] of the second Node
            * x         : X position of all the subpoints
            * y         : Y position of all the subpoints
            * z         : Z position of all the subpoints
        '''

        # Calculate the three vectors of the sides of the triangle
        v1 = nodeOne - mainNode
        v2 = nodeTwo - mainNode

        # Barycentric coordinates: Less lines. Implemented nicely.
        Area = 0.5 * np.sqrt(np.sum(np.cross(v1, v2)**2))
        Is = np.array([x,y,z])
        S1 = nodeOne[:,np.newaxis] - Is
        S2 = nodeTwo[:,np.newaxis] - Is
        Areas = 0.5 * np.sqrt(np.sum(np.cross(S1, S2, axis=0)**2, axis=0))
        Wi = Areas/Area

        # Vectorial Method: More Lines, same results.
        ## Calculate the height of the triangle
        #v3 = Vertices[nodeTwo] - Vertices[nodeOne]
        #if np.dot(v1, v3)==0:
        #    h = v1
        #elif np.dot(v2, v3)==0:
        #    h = v2
        #else:
        #    v4 = np.cross(v1, v2)
        #    C = Vertices[nodeOne][2] - Vertices[mainNode][2]
        #    h = np.ones((3,))
        #    h[2] = C
        #    h[1] = C * (v4[2]*v3[0] - v3[2]*v4[0]) / (v3[1]*v4[0] - v3[0]*v4[1])
        #    h[0] = C * (v4[2]*v3[1] - v3[2]*v4[1]) / (v3[0]*v4[1] - v4[0]*v3[1])
        ## Normalize it
        #h = h/np.sqrt(np.sum(h**2))
        ## Compute the vectors between the mainNode and each subPoint
        #Is = np.array([self.edksSources[1][iS], self.edksSources[2][iS], self.edksSources[3][iS]])
        #Vi = Is - Vertices[mainNode][:,np.newaxis]
        ## Compute the scalar product (which is the distance we want)
        #d = np.dot(Vi.T, h[:,None])
        ## Compute the distance max
        #Dmax = np.dot(Vertices[nodeOne] - Vertices[mainNode], h) 
        ## Compute the weights
        #Wi = 1. - d/Dmax

        return Wi

#EOF
