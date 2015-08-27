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
from . import csiutils as utils

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
            if (attr[:2] not in ('__')) and (not inspect.ismethod(fault.__getattribute__(attr))) and attr!='name':
                Attributes.append(attr)

        # Loop 
        for attr in Attributes:
            self.__setattr__(attr, copy.deepcopy(fault.__getattribute__(attr)))

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
        nbr_faces = self.adjacencyMap[self.tentid[u]]
        strike = []
        dip = []

        for pid in nbr_faces:
            xc, yc, zc, w, l, stk, dp = self.getpatchgeometry(pid, center=True)
            strike.append(stk)
            dip.append(dp)

        strike = np.sum(np.array(strike) %(2*np.pi)) / len(strike)
        dip = np.mean(dip)

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
            * add_slip      : Will be set to None
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
        '''

        # Set add_slip to none
        if add_slip is not None:
            print('Slip vector is not the same length as the number of patches for TriangularTent type.')
            print('Setting add_slip to None')
            add_slip = None

        # Run the method from the parent class
        super(TriangularTents, self).writePatches2File(filename, add_slip=add_slip, scale=scale, stdh5=stdh5, decim=decim)

        # All done
        return

    def writeSources2Grd(self, filename, npoints=10, slip='strikeslip', increments=None, nSamples=None, outDir='./', mask=False):
        '''
        Writes the values of slip in two grd files:
                -> z_{filename}
                -> {slip}_{filename}

        Args:
            * filename      : Name of the grdfile (should end by grd or nc)
            * npoints       : Number of points inside each patch.
            * dlon          : Longitude increment of the output file
            * dlat          : Latitude increment of the output file
            * slip          : Slip value to store.
            * mask          : If true,builds a mask based on the outter boundary of the fault.
        '''
    
        # Assert
        assert not ( (increments is None) and (nSamples is None) ), 'Specify increments or nSamples...'

        # Import needed stuff
        import scipy.interpolate as sciint

        # Get the sources
        gp = geoplot(figure=None)
        lon, lat, z, s = gp.faultTents(self, slip=slip, method='scatter',
                                            npoints=npoints)
        gp.close()
        del gp

        # Mask?
        if mask:
            mask = self._getFaultContour()
        else:
            mask = None

        # write 2 grd
        utils.write2netCDF('{}/z_{}'.format(outDir, filename), 
                lon, lat, -1.0*z, increments=increments, nSamples=nSamples, mask=mask)
        utils.write2netCDF('{}/{}_{}'.format(outDir, slip, filename),
                lon, lat, s, increments=increments, nSamples=nSamples, mask=mask)

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
    
    def slipIntegrate(self, slip=None):
        '''
        Integrates slip on the patch.
        Args:
            * slip  : slip vector
                    Can be strikeslip, dipslip, tensile, coupling or
                    a list/array of floats.
        '''

        # Slip
        if type(slip) is str:
            if slip=='strikeslip':
                slip = self.slip[:,0]
            elif slip=='dipslip':
                slip = self.slip[:,1]
            elif slip=='tensile':
                slip = self.slip[:,2]
            elif slip=='coupling':
                slip = self.coupling
        elif type(slip) in (np.ndarray, list):
            assert len(slip)==len(self.tent), 'Slip vector is the wrong size'
        else:
            slip = np.ones((len(self.tent),))

        # Compute Volumes
        self.computeTentArea()
        self.volume = self.area_tent*slip/3.

        # All done
        return

    def computeTentArea(self):
        '''
        Computes the area for each node 
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
            area = np.sum(areas[nbr_triangles])
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

    def rotateGFs(self, G, convergence):
        '''
            For the data set data, returns the rotation operator so that dip slip motion is aligned with
        the convergence vector.
            These Greens' functions are stored in self.G or returned, given arguments.

        Args:
            * G             : Dictionarry of strike and dip slip green's functions
            * convergence   : Convergence vector, or list/array of convergence vector with
                                shape = (Number of fault patches, 2). 
        '''
        
        # Get the Green's functions
        Gss = G['strikeslip']
        Gds = G['dipslip']

        # Number of parameters
        nSlip = Gss.shape[1]

        # Get the convergence vector
        if len(convergence)==2:
            Conv = np.ones((nSlip, 2))
            Conv[:,0] *= convergence[0]
            Conv[:,1] *= convergence[1]
            self.convergence = Conv
        elif len(convergence)==nSlip:
            if type(convergence) is list:
                self.convergence = np.array(convergence)
        else:
            print('Convergence vector is of wrong format...')
            sys.exit()

        # Get the fault strike (for the facets)
        ss = super(TriangularTents, self).getStrikes()

        # Organize strike
        strike = np.zeros((nSlip,))
        for iNode in self.Nodes:
            Triangles = self.Nodes[iNode]['idTriangles']
            Sources = self.Nodes[iNode]['subSources']
            for source,triangle in zip(Sources, Triangles):
                strike[source] = ss[triangle]

        # Get the strike and dip vectors
        strikeVec = np.vstack((np.sin(strike), np.cos(strike))).T
        dipVec = np.vstack((np.sin(strike+np.pi/2.), np.cos(strike+np.pi/2.))).T

        # Project the convergence along strike and dip
        Sbr = (self.convergence*strikeVec).sum(axis=1)
        Dbr = (self.convergence*dipVec).sum(axis=1)

        # Rotate the Green's functions
        bigGss = np.multiply(-1.0*Gss, Sbr)
        bigGds = np.multiply(Gds, Dbr)

        # All done
        return bigGss, bigGds

    def computeCouplingGFs(self, data, convergence, initializeCoupling=True, vertical=True, verbose=True, keepRotatedGFs=True):
        '''
            For the data set data, computes the Green's Function for coupling, 
            using the formula described in Francisco Ortega's PhD, pages 106 to 108.
            This function re-computes the Green's functions. No need to pre-compute them.

            The corresponding GFs are stored in the GFs dictionary, under 
            the name of the data set and are named 'coupling'. 
            When inverting for coupling, we suggest building these functions and 
            assembling with slipdir='c'.
        
        Args:
            * data                  : Name of the data set.
            * convergence           : Convergence vector, or list/array of convergence vector with
                                        shape = (Number of fault patches, 2). 
            * initializeCoupling    : Do you initialize the coupling vector in fault (True/False)
            * vertical              : Use the verticals?
            * keepRotatedGFs        : Store the dip and strike rotated GFs?
        '''

        # 0. Initialize?
        if initializeCoupling:
            self.coupling = np.zeros((len(self.tent),))

        # 1. Compute the Green's function by keeping triangles separated
        G = self.edksGFs(data, vertical=vertical, slipdir='sd', verbose=verbose, TentCouplingCase=True)

        # 2. Rotate these green's functions (this is the rotation matrix for the node based GFs)
        bigGss, bigGds = self.rotateGFs(G, convergence)

        # 3. Compute the coupling GFs
        bigGc = -1.0*(bigGss + bigGds)
        # Precision: (the -1.0* is because we use a different convention from that of Francisco)

        # 3. Sum the triangles that need to be summed
        Gc = []; Gss=[]; Gds=[]
        for iNode in self.Nodes:
            iTriangles = self.Nodes[iNode]['subSources']
            Gc.append(bigGc[:,iTriangles].sum(axis=1))
            Gss.append(bigGss[:,iTriangles].sum(axis=1))
            Gds.append(bigGds[:,iTriangles].sum(axis=1))
        Gc = np.array(Gc).T
        Gss = np.array(Gss).T
        Gds = np.array(Gds).T

        # 6. Set the GFs
        G = {'coupling': Gc}
        if keepRotatedGFs:
             G['strikeslip'] = Gss
             G['dipslip'] = Gds
        data.setGFsInFault(self, G, vertical=vertical)

        # All done
        return

    def Facet2Nodes(self, homogeneousStrike=False, homogeneousDip=False, keepFacetsSeparated=False):
        '''
        Transfers the edksSources list into the node based setup.
        Args:
            * honogeneousStrike     : In a tent, the strike varies among the faces. This variation
                                      can be a problem if the variation in strikes is too large, 
                                      with slip that can partially cancel each other.
                                      If True, the strike of each of the point is equal to the strike
                                      of the main node of the tent.
            * homogeneousDip        : Same thing for the dip angle.
            * keepFacetsSeparated   : If True, each facet of each node will have a different identifier (int).
                                      This is needed when computing the Green's function for the Node based case. 
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

        # Initialize counter
        iSource = -1

        # Iterate on the nodes to derive the weights
        for mainNode in Nodes:
            if homogeneousDip:
                mainDip = self.getTentInfo(mainNode)[4]
            if homogeneousStrike:
                mainStrike = self.getTentInfo(mainNode)[3]
            if keepFacetsSeparated:
                Nodes[mainNode]['subSources'] = []
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
                # Source Ids
                if keepFacetsSeparated:
                    iSource += 1
                    Nodes[mainNode]['subSources'].append(iSource)
                else:
                    iSource = mainNode
                # Save each source
                Ids += (np.ones((len(iS),))*iSource).astype(int).tolist()
                xs += self.edksSources[1][iS].tolist()
                ys += self.edksSources[2][iS].tolist()
                zs += self.edksSources[3][iS].tolist()
                areas += self.edksSources[6][iS].tolist()
                slip += Wi.tolist()
                if homogeneousStrike:
                    strike += (np.ones((len(iS),))*mainStrike).tolist()
                else:
                    strike += self.edksSources[4][iS].tolist()
                if homogeneousDip:
                    dip += (np.ones((len(iS),))*mainDip).tolist()
                else:
                    dip += self.edksSources[5][iS].tolist()

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

    def buildTentAdjacencyMap(self, verbose=True):
        '''
        For each triangle vertex, finds the indices of the surrounding vertices.
        This function runs typically after buildAdjacencyMap.
        '''

        if verbose:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Finding the adjacent vertices for all vertices.")

        # Check adjacency Map
        if not hasattr(self, 'adjacencyMap'):
            self.buildAdjacencyMap(verbose=verbose)

        # Cache adjacencyMap
        adjacency = self.adjacencyMap 
        faces = self.Faces

        # Create empty lists
        adjacentTents = []

        # Iterate over adjacency map
        for adj, iVert in zip(adjacency, range(len(adjacency))):
            # Create a list for that tent
            tent = []
            # Iterate over the surrounding triangles
            for iTriangle in adj:
                face = faces[iTriangle]
                face = face[face!=iVert]
                tent.append(face)
            # Clean up tent
            tent = np.unique(np.concatenate(tent)).tolist()
            # Append
            adjacentTents.append(tent)

        # Save
        self.adjacentTents = adjacentTents

        # All don
        return

    def buildLaplacian(self, verbose=True, method='distance'):
        """
        Build a discrete Laplacian smoothing matrix.
        Args:
            * verbose       : if True, displays stuff.
            * method        : Method to estimate the Laplacian operator
                'count'     --> The diagonal is 2-times the number of surrounding nodes.
                                Off diagonals are -2/(number of surrounding nodes) for the surrounding nodes, 0 otherwise.

                'distance'  --> Computes the scale-dependent operator based on Desbrun et al 1999.
                                
                                Mathieu Desbrun, Mark Meyer, Peter Schr\"oder, and Alan Barr, 1999. 
                                Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow,  
                                Proceedings of SIGGRAPH.
        """
 
        # Build the tent adjacency map
        self.buildTentAdjacencyMap(verbose=verbose)

        # Get the vertices
        vertices = self.Vertices

        # Allocate an array
        D = np.zeros((len(vertices), len(vertices)))

        # Normalize the distances
        if method=='distance':
            self.Distances = []
            for adja, i in zip(self.adjacentTents, range(len(vertices))):
                x0, y0, z0 = vertices[i,0], vertices[i,1], vertices[i,2]
                xv, yv, zv = vertices[adja,0], vertices[adja,1], vertices[adja,2] 
                distances = np.array([np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2) 
                    for x, y, z in zip(xv, yv, zv)])
                self.Distances.append(distances)
            normalizer = np.max([d.max() for d in self.Distances])

        # Iterate over the vertices
        i = 0
        for adja in self.adjacentTents:
            # Counting Laplacian
            if method=='count':
                D[i,i] = 2*float(len(adja))
                D[i,adja] = -2./float(len(adja))
            # Distance-based
            elif method=='distance':
                distances = self.Distances[i]/normalizer
                E = np.sum(distances)
                D[i,i] = float(len(adja))*2./E * np.sum(1./distances)
                D[i,adja] = -2./E * 1./distances

            # Increment 
            i += 1

        # All done
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

    def plot(self, figure=134, slip='total', equiv=False, 
             show=True, axesscaling=True, Norm=None, linewidth=1.0, plot_on_2d=True, 
             method='scatter', npoints=10, colorbar=True,
             drawCoastlines=True, expand=0.2):
        '''
        Plot the available elements of the fault.
        
        Args:
            * ref           : Referential for the plot ('utm' or 'lonlat').
            * figure        : Number of the figure.
        '''

        # Get lons lats
        lonmin = np.min([p[:,0] for p in self.patchll])-expand
	
        if lonmin<0: 
		    lonmin += 360
        lonmax = np.max([p[:,0] for p in self.patchll])+expand
        if lonmax<0:
            lonmax+= 360
        latmin = np.min([p[:,1] for p in self.patchll])-expand
        latmax = np.max([p[:,1] for p in self.patchll])+expand


        # lon = np.unique([p[:,0] for p in self.patchll])
        # lon[lon<0.] += 360.
        # lat = np.unique([p[:,1] for p in self.patchll])
        # lonmin = lon.min()-expand
        # lonmax = lon.max()+expand
        # latmin = lat.min()-expand
        # latmax = lat.max()+expand

        # Create a figure
        fig = geoplot(figure=figure, lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)

        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(drawLand=False, parallels=5, meridians=5, drawOnFault=True)

        # Draw the fault
        x, y, z, slip = fig.faultTents(self, slip=slip, Norm=Norm, colorbar=colorbar, 
                plot_on_2d=plot_on_2d, npoints=npoints,
                method=method)

        # show
        if show:
            showFig = ['fault']
            if plot_on_2d:
                showFig.append('map')
            fig.show(showFig=showFig)

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

    def _getSlipOnSubSources(self, Ids, X, Y, Z, slip):
        '''
        From a slip distribution in slip at each Node, interpolate onto the sources defined by Ids, X, Y and Z.
        '''

        # Create array
        Slip = np.zeros(X.shape)

        # Get Vertices
        vertices = self.Vertices

        # Compute the slip value at each subpoint
        for iPatch in range(len(self.patch)):
            nodeOne, nodeTwo, nodeThree = self.Faces[iPatch]
            slipOne = slip[nodeOne]
            slipTwo = slip[nodeTwo]
            slipThree = slip[nodeThree]
            vertOne = np.array(vertices[nodeOne])
            vertTwo = np.array(vertices[nodeTwo])
            vertThree = np.array(vertices[nodeThree])
            ids = np.flatnonzero(Ids==iPatch)
            w1 = self._getWeights(vertOne, vertTwo, vertThree, X[ids], Y[ids], Z[ids])
            w2 = self._getWeights(vertTwo, vertOne, vertThree, X[ids], Y[ids], Z[ids])
            w3 = self._getWeights(vertThree, vertTwo, vertOne, X[ids], Y[ids], Z[ids])
            weightedSlip = w1*slipOne + w2*slipTwo + w3*slipThree
            Slip[ids] = weightedSlip

        # All Done
        return Slip

    def _getFaultContour(self):
        '''
        Returns the outer-edge of the fault.
        '''

        # Check
        if not hasattr(self, 'adjacentTents'):
            self.buildTentAdjacencyMap()

        # Initiate a list of points
        contour = []

        # Get the lon, lat and depth
        lon = np.array([t[0] for t in self.tentll])
        lat = np.array([t[1] for t in self.tentll])
        depth = np.array([t[2] for t in self.tentll])

        # Take the first point as the shallowest
        uu = np.argmin(depth)
        ustart = copy.deepcopy(uu)
        contour.append([lon[uu], lat[uu], depth[uu], uu])

        # Since we have taken the shallowest point as a start, the second point probably is
        # as shallow or very close (and there is no such thing as a horizontal fault) 
        # This is a cheap way, but it should work 99% of the time
        adj = self.adjacentTents[uu]
        ii = np.argmin(depth[adj]-depth[uu])
        uu = adj[ii]
        contour.append([lon[uu], lat[uu], depth[uu], uu])

        # Follow the edge until we find the same point
        while uu!=ustart:

            # Get the position of the last 2 points
            lon2, lat2, d2, u2 = contour[-1]
            lon1, lat1, d1, u1 = contour[-2]
            
            # In XY plane
            x1, y1 = self.tent[u1][:2]
            x2, y2 = self.tent[u2][:2]

            # Compute the vector
            vec = np.array([x2-x1, y2-y1])
            nvec = np.linalg.norm(vec)

            # Get the adjacent tents
            adj = copy.deepcopy(self.adjacentTents[u2])

            # Remove the point we had before
            adj.remove(u1)

            # Compute the vectors
            vecs = [np.array([x2-self.tent[a][0], y2-self.tent[a][1]]) for a in adj]

            # Compute the scalar product and get the angle
            angles = []
            for v in vecs:
                angle = np.arccos(np.dot(vec,v)/(np.linalg.norm(v)*nvec))
                vprod = np.cross(vec, v)
                if vprod<0.:
                    angle = 360. - angle
                angles.append(angle)

            # test 
            #plt.plot(lon, lat, '.k', markersize=10, zorder=0)
            #plt.plot([c[0] for c in contour], [c[1] for c in contour], '-r', linewidth=2, zorder=1)
            #plt.scatter(lon[adj], lat[adj], s=50, c=np.array(angles)*180./np.pi, linewidth=0.1, zorder=2)
            #plt.colorbar()
            #plt.show()

            # Get the largest angle
            uu = adj[np.argmax(angles)]

            # Append
            contour.append([lon[uu], lat[uu], depth[uu], uu])

        # Remove the last point because we already have it
        contour.pop()

        # All done
        return contour

       


#EOF
