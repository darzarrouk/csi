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
                    values = np.array([self.getTentInfo(t)[5] for t in self.tent])
                elif values is 'dip':
                    values = np.array([self.getTentInfo(t)[6] for t in self.tent])
                elif values is 'index':
                    values = np.array([np.float(self.getindex(p)) for t in self.tent])
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

    def getindex(self, tent):
        '''
        Returns the index of a tent.
        This function over-writes that from the parent class Fault.
        '''

        # Output index
        iout = None

        # Find it
        for t in range(len(self.tent)):
            try:
                if (self.tent[t] == tent).all():
                    iout = t
            except:
                if (self.tent[t]==tent):
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

        # Create the adjacency map
        self.buildAdjacencyMapVT(verbose=False)

        # All done
        return

    def Facet2Nodes(self):
        '''
        Transfers the edksSources list into the node based setup.
        '''
    
        # Get the faces and Nodes
        Faces = self.gocad_faces
        Vertices = self.gocad_vertices

        # Get the surrounding triangles
        Nodes = {}

        # Loop for that 
        for nId in self.tentid:
            Nodes[nId] = {'nTriangles': 0, 'idTriangles': []}
            for idFace in range(self.gocad_faces.shape[0]):
                ns = self.gocad_faces[idFace,:].tolist()
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
                # Calculate the three vectors of the sides of the triangle
                v1 = Vertices[nodeOne] - Vertices[mainNode]
                v2 = Vertices[nodeTwo] - Vertices[mainNode]
                # Barycentric coordinates: Less lines. Implemented nicely.
                Area = 0.5 * np.sqrt(np.sum(np.cross(v1, v2)**2))
                Is = np.array([self.edksSources[1][iS], self.edksSources[2][iS], self.edksSources[3][iS]])
                S1 = Vertices[nodeOne][:,np.newaxis] - Is
                S2 = Vertices[nodeTwo][:,np.newaxis] - Is
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
            if verbose:
                sys.stdout.write('%i / %i\r' % (i + 1, numvert))
                sys.stdout.flush()

            # Find triangles that share an edge
            adjacents = []
            for j in range(numface):
                if i in faces[j,:]:
                    adjacents.append(j)

            self.adjacencyMapVT.append(adjacents)

        if verbose:
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
        # dip *= np.pi/180.
        # strike *= np.pi/180.
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
