'''
A parent class that deals with triangular patches fault

Written by Bryan Riel, Z. Duputel and R. Jolivet November 2013
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
from .Fault import Fault
from .geodeticplot import geodeticplot as geoplot


class TriangularPatches(Fault):

    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(TriangularPatches,self).__init__(name, utmzone, ellps)

        # Specify the type of patch
        self.patchType = 'triangle'

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

    def computeArea(self):
        '''
        Computes the area of all triangles.
        '''

        # Area
        self.area = []

        # Loop over patches
        for patch in self.patch:
            self.area.append(self.patchArea(patch))

        # all done
        return

    def patchArea(self, patch):
        '''
        Returns the area of one patch.
        Args:   
            * patch : one item of the patch list.
        '''

        # Get vertices of patch
        p1, p2, p3 = list(patch)

        # Compute side lengths
        a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)
        c = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)

        # Compute area using numerically stable Heron's formula
        c,b,a = np.sort([a, b, c])
        area = 0.25 * np.sqrt((a + (b + c)) * (c - (a - b))
                       * (c + (a - b)) * (a + (b - c)))

        # All Done
        return area

    def splitPatch(self, patch):
        '''
        Splits a patch into 4 patches, based on the mid-point of each side.
        Args:
            * patch : item of the patch list.
        '''

        # Get corners
        p1, p2, p3 = list(patch)
        if type(p1) is not list:
            p1 = p1.tolist()
        if type(p2) is not list:
            p2 = p2.tolist()
        if type(p3) is not list:
            p3 = p3.tolist()

        # Compute mid-points
        p12 = [p1[0] + (p2[0]-p1[0])/2., 
               p1[1] + (p2[1]-p1[1])/2.,
               p1[2] + (p2[2]-p1[2])/2.]
        p23 = [p2[0] + (p3[0]-p2[0])/2.,
               p2[1] + (p3[1]-p2[1])/2.,
               p2[2] + (p3[2]-p2[2])/2.]
        p31 = [p3[0] + (p1[0]-p3[0])/2.,
               p3[1] + (p1[1]-p3[1])/2.,
               p3[2] + (p1[2]-p3[2])/2.]

        # make 4 triangles
        t1 = [p1, p12, p31]
        t2 = [p12, p2, p23]
        t3 = [p31, p23, p3]
        t4 = [p31, p12, p23]

        # All done
        return t1, t2, t3, t4

    def selectPatches(self,minlon,maxlon,minlat,maxlat,mindep,maxdep):

        xmin,ymin = self.ll2xy(minlon,minlat)
        xmax,ymax = self.ll2xy(maxlon,maxlat)

        for p in range(len(self.patch)-1,-1,-1):
            x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(p)
            if x1<xmin or x1>xmax or x2<ymin or x2>ymax or x3<mindep or x3>maxdep:
                self.deletepatch(p)

        for i in range(len(self.xf)-1,-1,-1):
            x1 = self.xf[i]
            x2 = self.yf[i]
            if x1<xmin or x1>xmax or x2<ymin or x2>ymax:
                self.xf = np.delete(self.xf,i)
                self.yf = np.delete(self.yf,i)

    def vertices2ll(self):
        '''
        Converts all the vertices into lonlat coordinates.
        '''

        # Create a list
        vertices_ll = []

        # iterate
        for vertex in self.Vertices:
            lon, lat = self.xy2ll(vertex[0], vertex[1])
            vertices_ll.append([lon, lat, vertex[2]])

        # Save
        self.Vertices_ll = np.array(vertices_ll)

        # All done
        return

    def patches2triangles(self, fault):
        '''
        Takes a fault with rectangular patches and splits them into triangles to 
        initialize self.
        Args:
            * fault : instance of rectangular patches.
        '''

        # Initialize the lists of patches
        self.patch = []
        self.patchll = []

        # Initialize vertices and faces
        vertices = []
        faces = []

        # Each patch is being splitted in 4 triangles
        for patch in fault.patch:

            # Get the center
            center = np.array(fault.getpatchgeometry(patch, center=True)[:3])

            # Add vertices
            for vertex in patch.tolist():
                if vertex not in vertices:
                    vertices.append(vertex)
            vertices.append(list(center))

            # Find the vertices in the list
            i0 = np.flatnonzero(np.array([patch[0].tolist()==v for v in vertices]))[0]
            i1 = np.flatnonzero(np.array([patch[1].tolist()==v for v in vertices]))[0]
            i2 = np.flatnonzero(np.array([patch[2].tolist()==v for v in vertices]))[0]
            i3 = np.flatnonzero(np.array([patch[3].tolist()==v for v in vertices]))[0]
            ic = np.flatnonzero(np.array([center.tolist()==v for v in vertices]))[0]

            # Split in 4
            t1 = np.array([patch[0], patch[1], center])
            t2 = np.array([patch[1], patch[2], center])
            t3 = np.array([patch[2], patch[3], center])
            t4 = np.array([patch[3], patch[0], center])

            # faces
            faces.append([i0, i1, ic])
            faces.append([i1, i2, ic])
            faces.append([i2, i3, ic])
            faces.append([i3, i0, ic])
        
            # patches
            self.patch.append(t1)
            self.patch.append(t2)
            self.patch.append(t3)
            self.patch.append(t4)

        # Save
        self.Vertices = np.array(vertices)
        self.Faces = np.array(faces)

        # Convert
        self.vertices2ll()
        self.patch2ll()

        # Initialize slip
        self.initializeslip()

        # Set fault trace
        self.xf = fault.xf
        self.yf = fault.yf
        self.lon = fault.lon
        self.lat = fault.lat
        if hasattr(fault, 'xi'):
            self.xi = fault.xi
            self.yi = fault.yi
            self.loni = fault.loni
            self.lati = fault.lati

        # Set depth
        self.depth = np.max([v[2] for v in self.Vertices])
        self.z_patches = np.linspace(self.depth, 0.0, 5)

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
        # Initialize the lists of patches
        self.patch   = []
        self.patchll = []

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
            vids = np.array(vids,dtype=int)
            i0 = np.min(vids)
            vids = vids - i0
            i    = np.argsort(vids)
            vertices = np.array(vertices, dtype=float)[i,:]
            faces = np.array(faces, dtype=int) - i0

        # Resample vertices to UTM
        if utm:
            vx = vertices[:,0].copy()*factor_xy
            vy = vertices[:,1].copy()*factor_xy
            vertices[:,0],vertices[:,1] = self.xy2ll(vx,vy)
        else:
            vx, vy = self.ll2xy(vertices[:,0], vertices[:,1])
        vz = vertices[:,2]*factor_depth
        self.factor_depth = factor_depth
        self.Vertices = np.column_stack((vx, vy, vz))
        self.Vertices_ll = vertices
        self.Faces = faces
        if verbose:
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
            self.patch.append(np.array([p1, p2, p3]))
            self.patchll.append(np.array([pll1, pll2, pll3]))

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

    def writeGocadPatches(self, filename, utm=False):
        """
        Load a triangulated Gocad surface file. Vertices must be in geographical coordinates.
        """
        # Get the geographic vertices and connectivities from the Gocad file

        fid = open(filename, 'w')
        if utm:
            vertices = self.Vertices*1.0e3
        else:
            vertices = self.Vertices_ll
        for i in range(vertices.shape[0]):
            v = vertices[i]
            fid.write('VRTX {} {} {} {}\n'.format(i+1,v[0],v[1],v[2]))
        for i in range(self.Faces.shape[0]):
            vid = self.Faces[i,:]+1
            fid.write('TRGL {} {} {}\n'.format(vid[0],vid[1],vid[2]))
        fid.close()

        # All done
        return

    def getStrikes(self):
        '''
        Returns an array of strikes.
        '''

        # all done in one line
        return np.array([self.getpatchgeometry(p)[5] for p in self.patch])

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

            # Put the parameter number in the file as well if it exists
            parameter = ' '
            if hasattr(self,'index_parameter'):
                i = np.int(self.index_parameter[pIndex,0])
                j = np.int(self.index_parameter[pIndex,1])
                k = np.int(self.index_parameter[pIndex,2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            slipstring = ' # {} {} {} '.format(self.slip[pIndex,0],
                                               self.slip[pIndex,1], self.slip[pIndex,2])

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
                    fout.write('{} {} {} \n'.format(lon, lat, -1.*z))
            # Close file
            fout.close()

        # All done
        return


    def getEllipse(self, patch, ellipseCenter=None, Npoints=10, factor=1.0):
        '''
        Compute the ellipse error given Cm for a given patch
        args:
               (optional) center  : center of the ellipse
               (optional) Npoints : number of points on the ellipse
        '''

        # Get Cm
        Cm = np.diag(self.Cm[patch,:2])
        Cm[0,1] = Cm[1,0] = self.Cm[patch,2]

        # Get strike and dip
        xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(patch, center=True)
        dip *= np.pi/180.
        strike *= np.pi/180.
        if ellipseCenter!=None:
            xc,yc,zc = ellipseCenter

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
            self.N_slip = len(self.patch)
        for p in range(self.N_slip):
            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            # Get the slip vector
            slip = self.slip[p,:]
            rake = np.arctan2(slip[1],slip[0])

            # Compute the vector
            #x = np.sin(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake)
            #y = np.cos(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake)
            #z = -1.0*np.sin(dip)*np.sin(rake)
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
                self.ellipse.append(self.getEllipse(p,ellipseCenter=[xe, ye, ze],factor=factor))

            # Append slip direction
            self.slipdirection.append([[xc, yc, zc],[xe, ye, ze]])

        # All done
        return

    def deletepatch(self, patch, checkVertices=True):
        '''
        Deletes a patch.
        Args:
            * patch     : index of the patch to remove.
        '''
        # Save vertices
        vids = copy.deepcopy(self.Faces[patch])

        # Remove the patch
        del self.patch[patch]
        del self.patchll[patch]
        self.Faces = np.delete(self.Faces, patch, axis=0)
        
        # Check if vertices are to be removed
        v2del = []
        if checkVertices:
            for v in vids:
                if v not in self.Faces.flatten().tolist():
                    v2del.append(v)

        # Remove if needed
        if len(v2del)>0:
            self.deletevertices(v2del, checkPatch=False)

        # Clean slip vector
        if self.slip is not None:
            self.slip = np.delete(self.slip, patch, axis=0)
            self.N_slip = len(self.slip)
            self.numpatch -= 1

        # All done
        return

    def deletevertices(self, iVertices, checkPatch=True):
        '''
        Deletes some vertices.
        If some patches are composed of thexe vertices and checkPatch is True, 
            deletes the patches.
        Args:
            * iVertices     : List of vertices to delete.
            * checkPatch    : Check and delete if patches are concerned.
        ''' 

        # If some patches are concerned
        if checkPatch:
            # Check
            iPatch = []
            for iV in iVertices:
                i, j = np.where(self.Faces==iV)
                if len(i.tolist())>0:
                    iPatch.append(np.unique(i))
            iPatch = np.unique(np.concatenate(iPatch)).tolist()
            # Delete
            self.deletepatches(iPatch, checkVertices=False)

        # Modify the vertex numbers in Faces
        newFaces = copy.deepcopy(self.Faces)
        for v in iVertices:
            i,j = np.where(self.Faces>v)
            newFaces[i,j] -= 1
        self.Faces = newFaces
         
        # Do the deletion
        self.Vertices = np.delete(self.Vertices, iVertices, axis=0)
        self.Vertices_ll = np.delete(self.Vertices_ll, iVertices, axis=0)
        
        # All done
        return

    def deletevertex(self, iVertex, checkPatch=True):
        '''
        Delete a Vertex.
        If some patches are composed of this vertex and checkPatch is True, 
            deletes the patches.
        Args:
            * iVertex       : index of the vertex to delete
            * checkPatch    : Check and delete if patches are concerned.
        '''

        # Delete only one vertex
        self.deletevertices([iVertex], checkPatch)

        # All done
        return


    def deletepatches(self, tutu, checkVertices=True):
        '''
        Deletes a list of patches.
        '''

        while len(tutu)>0:

            # Get index to delete
            i = tutu.pop()

            # delete it
            self.deletepatch(i, checkVertices=checkVertices)

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
            * patch     : Geometry of the patch to add (km, not lon lat)
            * slip      : List of the strike, dip and tensile slip.
            * resetTents: Recompute the list of tents (default is True)
        '''

        # Check that patch is an array
        if type(patch) is list:
            patch = np.array(patch)
        assert type(patch) is np.ndarray, 'addPatch: Patch has to be a numpy array'

        # append the patch
        self.patch.append(patch)

        # modify the slip
        if self.N_slip!=None and self.N_slip==len(self.patch):
            sh = self.slip.shape
            nl = sh[0] + 1
            nc = 3
            tmp = np.zeros((nl, nc))
            if nl > 1:                      # Case where slip is empty
                tmp[:nl-1,:] = self.slip
            tmp[-1,:] = slip
            self.slip = tmp
        else:
            print('Slip has not been modified')

        # Check if vertices are already there
        vids = []
        for p in patch:
            ii = np.flatnonzero(np.array([(p.tolist()==v).all() for v in self.Vertices]))
            if len(ii)==0:
                self.Vertices = np.insert(self.Vertices, self.Vertices.shape[0],
                        p, axis=0)
                vids.append(self.Vertices.shape[0]-1)
            else:
                vids.append(ii[0])
        self.Faces = np.insert(self.Faces, self.Faces.shape[0], vids, axis=0)

        # Vertices2ll
        self.vertices2ll()

        # All done
        return

    def replacePatch(self, patch, iPatch):
        '''
        Replaces one patch by the given geometry.
        Args:
            * patch     : Patch geometry.
            * iPatch    : index of the patch to replace.
        '''

        # Replace
        if type(patch) is list:
            patch = np.array(patch)
        assert type(patch) is np.ndarray, 'replacePatch: Patch must be an array'
        self.patch[iPatch] = patch

        # All done
        return

    def getpatchgeometry(self, patch, center=False, retNormal=False, checkindex=True):
        '''
        Returns the patch geometry as needed for triangleDisp.
        Args:
            * patch         : index of the wanted patch or patch;
            * center        : if true, returns the coordinates of the center of the patch.
                              if False, returns the first corner
            * checkindex    : Checks the index of the patch
        '''

        # Get the patch
        u = None
        if patch.__class__ is int:
            u = patch
        else:
            if checkindex:
                u = self.getindex(patch)
        if u is not None:
            patch = self.patch[u]

        # Get the center of the patch
        x1, x2, x3 = self.getcenter(patch)

        # Get the vertices of the patch
        verts = copy.deepcopy(patch)
        p1, p2, p3 = [np.array([lst[1],lst[0],lst[2]]) for lst in verts]

        # Get a dummy width and height
        width = np.linalg.norm(p1 - p2)
        length = np.linalg.norm(p3 - p1)

        # Get the patch normal
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)
        # Enforce clockwise circulation
        if normal[2] > 0:
            normal *= -1.0
            p2, p3 = p3, p2

        # Get the strike vector and strike angle
        strike = np.arctan2(-normal[0], normal[1])
        if strike<0.:
            strike += 2*np.pi

        # Set the dip vector
        dip = np.arccos(-normal[2])

        if retNormal:
            return x1, x2, x3, width, length, strike, dip, normal
        else:
            return x1, x2, x3, width, length, strike, dip

    def distanceVertexToVertex(self, vertex1, vertex2, distance='center', lim=None):
        '''
        Measures the distance between two vertexes.
        Args:
            * patch1    : first patch or its index
            * patch2    : second patch or its index
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].
        '''

        if distance is 'center':

            # Get the centers
            x1, y1, z1 = vertex1
            x2, y2, z2 = vertex2

            # Compute the distance
            dis = np.sqrt((x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis > lim[0]:
                    dis = lim[1]

        else:
            raise NotImplementedError('only distance=center is implemented')

        # All done
        return dis

    def distanceMatrix(self, distance='center', lim=None):
        '''
        Returns a matrix of the distances between patches.
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
            p1 = self.patch[i]
            for j in range(self.N_slip):
                if j == i:
                    continue
                p2 = self.patch[j]
                Distances[i,j] = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)

        # All done
        return Distances

    def distancePatchToPatch(self, patch1, patch2, distance='center', lim=None):
        '''
        Measures the distance between two patches.
        Args:
            * patch1    : first patch or its index
            * patch2    : second patch or its index
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
            dis = np.sqrt((x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis > lim[0]:
                    dis = lim[1]

        else:
            raise NotImplementedError('only distance=center is implemented')

        # All done
        return dis

    def slip2dis(self, data, patch, slip=None):
        '''
        Computes the surface displacement for a given patch at the data location
        using a homogeneous half-space.

        Args:
            * data          : data object from gpsrates or insarrates.
            * patch         : number of the patch that slips
            * slip          : if a number is given, that is the amount of slip along strike.
                              if three numbers are given, that is the amount of slip along strike,
                              along dip and opening. if None, values from self.slip are taken.
        '''

        # Set the slip values
        if slip is None:
            SLP = [self.slip[patch,0], self.slip[patch,1], self.slip[patch,2]]
        elif slip.__class__ is float:
            SLP = [slip, 0.0, 0.0]
        elif slip.__class__ is list:
            SLP = slip

        # Get patch vertices
        vertices = list(self.patch[patch])

        # Get data position
        x = data.x
        y = data.y
        z = np.zeros_like(x)

        # Get strike slip displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, SLP[0], 0.0, 0.0)
        ss_dis = np.column_stack((ux, uy, uz))

        # Get dip slip displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, 0.0, SLP[1], 0.0)
        ds_dis = np.column_stack((ux, uy, uz))

        # Get opening displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, 0.0, 0.0, SLP[2])
        op_dis = np.column_stack((ux, uy, uz))

        # All done
        return ss_dis, ds_dis, op_dis

    def buildAdjacencyMap(self, verbose=True):
        """
        For each triangle, find the indices of the adjacent (edgewise) triangles.
        """
        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the adjacency map for all patches")

        self.adjacencyMap = []

        # Cache the vertices and faces arrays
        vertices, faces = self.Vertices, self.Faces

        # First find adjacent triangles for all triangles
        npatch = len(self.patch)
        for i in range(npatch):

            if verbose:
                sys.stdout.write('%i / %i\r' % (i, npatch))
                sys.stdout.flush()

            # Indices of Vertices of current patch
            refVertInds = faces[i,:]

            # Find triangles that share an edge
            adjacents = []
            for j in range(npatch):
                if j == i:
                    continue
                sharedVertices = np.intersect1d(refVertInds, faces[j,:])
                numSharedVertices = sharedVertices.size
                if numSharedVertices < 2:
                    continue
                adjacents.append(j)
                if len(adjacents) == 3:
                    break

            self.adjacencyMap.append(adjacents)

        if verbose:
            print('\n')
        return


    def buildLaplacian(self, extra_params=None, verbose=True):
        """
        Build a discrete Laplacian smoothing matrix.
        """
        
        if self.adjacencyMap is None or len(self.adjacencyMap) != len(self.patch):
            self.buildAdjacencyMap(verbose=verbose)

        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the Laplacian matrix")

        # Pre-compute patch centers
        centers = self.getcenters()

        # Cache the vertices and faces arrays
        vertices, faces = self.Vertices, self.Faces

        # Allocate array for Laplace operator
        npatch = len(self.patch)
        D = np.zeros((npatch,npatch))

        # Loop over patches
        for i in range(npatch):

            if verbose:
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

        if verbose:
            print('\n')
        D = D / np.max(np.abs(np.diag(D)))
        return D

    def getcenter(self, p):
        '''
        Get the center of one triangular patch.
        Args:
            * p    : Patch geometry.
        '''

        # Get center
        if type(p) is int:
            p1, p2, p3 = self.patch[p]
        else:
            p1, p2, p3 = p

        # Compute the center
        x = (p1[0] + p2[0] + p3[0]) / 3.0
        y = (p1[1] + p2[1] + p3[1]) / 3.0
        z = (p1[2] + p2[2] + p3[2]) / 3.0

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

        # Initialize a list
        center = []

        # loop over the patches
        for p in self.patch:
            x, y, z = self.getcenter(p)
            center.append(np.array([x, y, z]))

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
            if ((p[0,2]<=depth) and (p[2,2]>=depth)):
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
            dmin = np.min([p[i,2] for i in range(4)])
            dmax = np.max([p[i,2] for i in range(4)])

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
        depths = np.unique(np.array([[self.patch[i][u,2] for u in range(4)] for i in range(len(self.patch))]).flatten())
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
                fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Distance to origin (km) | x, y \n')
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
                    fout.write('{} {} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, area, dist, x, y))

        # Close file if done
        if filename is not None:
            fout.close()

        # All done
        return

    def plot(self, ref='utm', figure=134, add=False, value_to_plot='total', revmap=False):
        '''
        Plot the available elements of the fault.

        Args:
            * ref           : Referential for the plot ('utm' or 'lonlat').
            * figure        : Number of the figure.
        '''

        # Create a figure
        fig = geoplot(figure=figure, ref=ref)

        # Plot fault trace
        fig.faulttrace(self,  add=add)

        # Plot the fault 
        fig.faultpatches(self, slip=value_to_plot, revmap=revmap)

        # Show the thing
        fig.show(showFig=['fault'], triDaxis=None)

        # All done
        return


    def plotMayavi(self, neg_depth=True, value_to_plot='total', colormap='jet',
                   reverseSign=False):
        """
        Plot 3D representation of fault using MayaVi.

        Args:
            * neg_depth     : Flag to specify if patch depths are negative or positive
            * value_to_plot : What to plot on patches
            * colormap      : Colormap for patches
            * reverseSign   : Flag to reverse sign of value_to_plot
        """
        try:
            from mayavi import mlab
        except ImportError:
            print('mayavi module not installed. skipping plotting...')
            return

        # Sign factor for negative depths
        negFactor = -1.0
        if neg_depth:
            negFactor = 1.0

        # Sign for values
        valueSign = 1.0
        if reverseSign:
            valueSign = -1.0

        # Plot the wireframe
        x, y, z = self.Vertices[:,0], self.Vertices[:,1], self.Vertices[:,2]
        z *= negFactor
        mesh = mlab.triangular_mesh(x, y, z, self.Faces, representation='wireframe',
                                    opacity=0.6, color=(0.0,0.0,0.0))

        # Compute the scalar value to color the patches
        if value_to_plot == 'total':
            self.computetotalslip()
            plotval = self.totalslip
        elif value_to_plot == 'strikeslip':
            plotval = self.slip[:,0]
        elif value_to_plot == 'dipslip':
            plotval = self.slip[:,1]
        elif value_to_plot == 'tensile':
            plotval = self.slip[:,2]
        elif value_to_plot == 'index':
            plotval = np.linspace(0, len(self.patch)-1, len(self.patch))
        else:
            assert False, 'unsupported value_to_plot'

        # Assign the scalar data to a source dataset
        cell_data = mesh.mlab_source.dataset.cell_data
        cell_data.scalars = valueSign * plotval
        cell_data.scalars.name = 'Cell data'
        cell_data.update()

        # Make a new mesh with the scalar data applied to patches
        mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='Cell data')
        surface = mlab.pipeline.surface(mesh2, colormap=colormap)

        mlab.colorbar(surface)
        mlab.show()

        return


    def mapFault2Fault(self, Map, fault):
        '''
        User provides a Mapping function np.array((len(self.patch), len(fault.patch)))
        and a fault and the slip from the argument
        fault is mapped into self.slip.
        Function just does:
        self.slip[:,0] = np.dot(Map,fault.slip)
        ...
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

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

    def mapUnder2Above(self, deepfault):
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

#--------------------------------------------------------------
#--------------------------------------------------------------
# These routines have become useless. I comment them now.
# If someone is against it, let me know before August 2015
# Past August 1st, 2015, I will delete these
# RJ April 1st, 2015
#
#    def writeEDKSsubParams(self, data, edksfilename, amax=None, plot=False, w_file=True):
#        '''
#        Write the subParam file needed for the interpolation of the green's function in EDKS.
#        Francisco's program cuts the patches into small patches, interpolates the kernels to get the GFs at each point source,
#        then averages the GFs on the pacth. To decide the size of the minimum patch, it uses St Vernant's principle.
#        If amax is specified, the minimum size is fixed.
#        Args:
#            * data          : Data object from gpsrates or insarrates.
#            * edksfilename  : Name of the file containing the kernels
#            * amax          : Specifies the minimum size of the divided patch. If None, uses St Vernant's principle (default=None)
#            * plot          : Activates plotting (default=False)
#            * w_file        : if False, will not write the subParam fil (default=True)
#        Returns:
#            * filename         : Name of the subParams file created (only if w_file==True)
#            * TrianglePropFile : Name of the triangle properties file
#            * PointCoordFile   : Name of the Point coordinates file
#            * ReceiverFile     : Name of the receiver file
#            * method_par       : Dictionary including useful EDKS parameters
#        '''
#
#        # print
#        print ("---------------------------------")
#        print ("---------------------------------")
#        print ("Write the EDKS files for fault {} and data {}".format(self.name, data.name))
#
#        # Write the geometry to the EDKS file
#        fltname,TrianglePropFile, PointCoordFile = self.writeEDKSgeometry()
#
#        # Write the data to the EDKS file
#        datname,ReceiverFile = data.writeEDKSdata()
#
#        # Assign some EDKS parameters
#        if data.dtype is 'insarrates':
#            useRecvDir = True # True for InSAR, uses LOS information
#        else:
#            useRecvDir = False # False for GPS, uses ENU displacements
#        EDKSunits = 1000.0
#        EDKSfilename = '{}'.format(edksfilename)
#        prefix = 'edks_{}_{}'.format(fltname, datname)
#        plotGeometry = '{}'.format(plot)
#
#        # Build usefull outputs
#        parNames = ['useRecvDir', 'Amax', 'EDKSunits', 'EDKSfilename', 'prefix']
#        parValues = [ useRecvDir ,  amax ,  EDKSunits ,  EDKSfilename ,  prefix ]
#        method_par = dict(zip(parNames, parValues))
#
#        # Open the EDKSsubParams.py file
#        if w_file:
#            filename = 'EDKSParams_{}_{}.py'.format(fltname, datname)
#            fout = open(filename, 'w')
#
#            # Write in it
#            fout.write("# File with the triangle properties\n")
#            fout.write("TriPropFile = '{}'\n".format(TrianglePropFile))
#            fout.write("# File with the Triangles' Points (vertex) coordinates \n")
#            fout.write("TriPointsFile = '{}'\n".format(PointCoordFile))
#            fout.write("# File with id, E[km], N[km] coordinates of the receivers.\n")
#            fout.write("ReceiverFile = '{}'\n".format(ReceiverFile))
#            fout.write("# read receiver direction (# not yet implemented)\n")
#            fout.write("useRecvDir = {} # leace this to False for now\n".format(useRecvDir))
#            fout.write("# Maximum Area to subdivide triangles. If None, uses Saint-Venant's principle.\n")
#            if amax is None:
#                fout.write("Amax = None # None computes Amax automatically. \n")
#            else:
#                fout.write("Amax = {} # Minimum size for the patch division.\n".format(amax))
#            fout.write("EDKSunits = 1000.0 # to convert from kilometers to meters\n")
#            fout.write("EDKSfilename = '{}'\n".format(edksfilename))
#            fout.write("prefix = '{}'\n".format(prefix))
#            fout.write("plotGeometry = {} # set to False if you are running in a remote Workstation\n".format(plot))
#
#            # Close the file
#            fout.close()
#            return filename, TrianglePropFile, PointCoordFile, ReceiverFile, method_par
#        else:
#            return TrianglePropFile, PointCoordFile, ReceiverFile, method_par
#
#    def writeEDKSgeometry(self, ref=None):
#        '''
#        This routine spits out 2 files:
#        filename.TriangleProp: Patch ID | Lon (deg) | Lat | East (km) | North | Depth (km) | Strike (deg) | Dip  | Area (km^2) | Vertice ids
#        (coordinates are given for the center of the patch)
#        filename.PointCoord: Vertice ID | Lon (deg) | Lat | East (km) | North | Depth (km)
#
#        These files are to be used with edks/MPI_EDKS/calcGreenFunctions_EDKS_subTriangles.py
#        '''
#
#        # Filename
#        fltname = self.name.replace(' ','_')
#        filename = 'edks_{}'.format(fltname)
#        TrianglePropFile = filename+'.TriangleProp'
#        PointCoordFile   = filename+'.PointCoord'
#
#        # Open the output file and write headers
#        TriP = open(TrianglePropFile,'w')
#        h_format ='%-6s %10s %10s %10s %10s %10s %10s %10s %10s %6s %6s %6s\n'
#        h_tuple = ('%Tid','lon','lat','E[km]','N[km]','dep[km]','strike','dip',
#                  'Area[km2]','idP1','idP2','idP3')
#        TriP.write(h_format%h_tuple)
#        PoC = open(PointCoordFile,'w')
#        PoC.write('%-6s %10s %10s %10s %10s %10s\n'%('Pid','lon','lat','E[km]','N[km]','dep[km]'))
#
#        # Reference
#        if ref is not None:
#            refx, refy = self.putm(ref[0], ref[1])
#            refx /= 1000.
#            refy /= 1000.
#
#        # Loop over the patches
#        TriP_format = '%-6d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %6d %6d %6d\n'
#        for p in range(self.numpatch):
#            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
#            strike = strike*180./np.pi
#            dip = dip*180./np.pi
#            lonc, latc = self.xy2ll(xc,yc)
#            if ref is not None:
#                xc -= refx
#                yc -= refy
#            verts = copy.deepcopy(self.patch[p])
#            v1, v2, v3 = self.Faces[p,:]
#            TriP_tuple = (p,lonc,latc,xc,yc,zc,strike,dip,self.area[p],v1,v2,v3)
#            TriP.write(TriP_format%TriP_tuple)
#
#        PoC_format =  '%-6d %10.4f %10.4f %10.4f %10.4f %10.4f\n'
#        for v in range(self.Vertices.shape[0]):
#            xv,yv,zv = self.Vertices[v,:]
#            lonv,latv,zv2 = self.Vertices_ll[v,:]
#            assert zv == zv2*self.factor_depth, \
#                'inconsistent depth in Vertices and Vertices_ll'
#            if ref is not None:
#                xv -= refx
#                yv -= refy
#            PoC.write(PoC_format%(v,lonv,latv,xv,yv,zv))
#
#        # Close the files
#        TriP.close()
#        PoC.close()
#
#        # All done
#        return fltname,TrianglePropFile,PointCoordFile
#--------------------------------------------------------------
#--------------------------------------------------------------

#EOF
