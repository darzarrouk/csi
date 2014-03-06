'''
A parent class that deals with triangular patches fault

Written by R. Jolivet, Z. Duputel and Bryan Riel November 2013
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


    def computeArea(self):
        '''
        Computes the area of all triangles.
        '''

        # Area
        self.area = []

        # Loop over patches
        for patch in self.patch:

            # Get vertices of patch 
            p1, p2, p3 = patch[:3]

            # Compute side lengths
            a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
            b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)
            c = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)

            # Compute area using numerically stable Heron's formula
            c,b,a = np.sort([a, b, c])
            self.area.append(0.25 * np.sqrt((a + (b + c)) * (c - (a - b)) 
                           * (c + (a - b)) * (a + (b - c))))
            
        # all done
        return


    def readGocadPatches(self, filename, neg_depth=False, utm=False, factor_depth=1., set_trace=False):
        """
        Load a triangulated Gocad surface file. Vertices must be in geographical coordinates.
        """
        # Initialize the lists of patches
        self.patch   = []
        self.patchll = []

        # Factor to correct input negative depths (we want depths to be positive)
        if neg_depth:
            negFactor = -1.0
        else:
            negFactor = 1.0
        
        # Get the geographic vertices and connectivities from the Gocad file
        with open(filename, 'r') as fid:
            vertices = []
            faces = []
            for line in fid:
                if line.startswith('VRTX'):
                    name, vid, x, y, z = line.split()
                    vertices.append([float(x), float(y), negFactor*float(z)])
                elif line.startswith('TRGL'):
                    name, p1, p2, p3 = line.split()
                    faces.append([int(p1), int(p2), int(p3)])
            fid.close()
            vertices = np.array(vertices, dtype=float)
            faces = np.array(faces, dtype=int) - 1
        self.gocad_vertices_ll = vertices

        # Resample vertices to UTM
        if utm:
            vx = vertices[:,0].copy()*1.0e-3
            vy = vertices[:,1].copy()*1.0e-3
            vertices[:,0],vertices[:,1] = self.xy2ll(vx,vy)
        else:
            vx, vy = self.ll2xy(vertices[:,0], vertices[:,1])
        vz = vertices[:,2]*factor_depth
        self.gocad_vertices = np.column_stack((vx, vy, vz))
        self.gocad_faces = faces

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
            
        # Update the depth of the bottom of the fault
        if neg_depth:
            self.depth = np.min(vz)
        else:
            self.depth = np.max(vz)
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # Fault trace
        if set_trace:
            self.xf = []
            self.yf = []
            for p,pl in zip(self.patch,self.patchll):
                for v,vl in zip(p,pl):
                    if v[-1]!=0.:
                        continue
                    self.xf.append(v[0])
                    self.yf.append(v[1])
            self.xf = np.array(self.xf)
            self.yf = np.array(self.yf)
            i = np.argsort(self.yf)
            self.xf = self.xf[i]
            self.yf = self.yf[i]

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
            if not neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))

            # Get the end of the vector
            xc, yc, zc = p[1]
            lonc, latc = self.xy2ll(xc, yc)
            if not neg_depth:
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


    def getEllipse(self,patch,ellipseCenter=None,Npoints=10,factor=1.0):
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
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Create the array
        self.slipdirection = []
        
        # Check Cm if ellipse
        if ellipse:
            self.ellipse = []
            assert(self.Cm!=None), 'Provide Cm values'

        # Loop over the patches
        for p in range(len(self.patch)):  
            
            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True) 
            # Get the slip vector
            slip = self.getslip(self.patch[p]) 
            rake = np.arctan2(slip[1],slip[0])

            # Compute the vector
            x = np.sin(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake) 
            y = np.cos(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake)
            z = -1.0*np.sin(dip)*np.sin(rake)
        
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
                self.ellipse.append(self.getEllipse(p,ellipseCenter=[xe, ye, ze]))

            # Append slip direction
            self.slipdirection.append([[xc, yc, zc],[xe, ye, ze]])

        # All done
        return


    def deletepatch(self, patch):
        '''
        Deletes a patch.
        Args:   
            * patch     : index of the patch to remove.
        '''

        # Remove the patch
        del self.patch[patch]
        del self.patchll[patch]
        self.slip = np.delete(self.slip, patch, axis=0)

        # All done
        return


    def deletepatches(self, tutu):
        '''
        Deletes a list of patches.
        '''

        while len(tutu)>0:

            # Get index to delete
            i = tutu.pop()

            # delete it
            self.deletepatch(i)

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
            * patch     : Geometry of the patch to add
            * slip      : List of the strike, dip and tensile slip.
        '''

        # append the patch
        self.patch.append(patch)

        # modify the slip
        sh = self.slip.shape
        nl = sh[0] + 1
        nc = 3
        tmp = np.zeros((nl, nc))
        if nl > 1:                      # Case where slip is empty
            tmp[:nl-1,:] = self.slip
        tmp[-1,:] = slip
        self.slip = tmp

        # All done
        return

    def getpatchgeometry(self, patch, center=False, retNormal=False):
        '''
        Returns the patch geometry as needed for triangleDisp.
        Args:
            * patch         : index of the wanted patch or patch;
            * center        : if true, returns the coordinates of the center of the patch. 
                              if False, returns the first corner
        '''

        # Get the patch
        u = None
        if patch.__class__ is int:
            u = patch
        else:
            for i in range(len(self.patch)):
                if (self.patch[i]==patch).all():
                    u = i

        # Get the center of the patch
        x1, x2, x3 = self.getcenter(u)

        # Get the vertices of the patch 
        verts = copy.deepcopy(self.patch[u])
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

        # Set the dip vector
        dip = np.arccos(-normal[2])

        if retNormal:
            return x1, x2, x3, width, length, strike, dip, normal
        else:
            return x1, x2, x3, width, length, strike, dip


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
        vertices = self.patch[patch]

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

    def saveGFs(self, dtype='d', outputDir='.'):
        '''
        Saves the Green's functions in different files
        Args:
            dtype       : Format of the binary data saved.
            outputDir   : Directory to save binary data.
        '''

        # Print stuff
        print('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys(): 

            # Get the Green's function
            G = self.G[data]

            # StrikeSlip Component
            if 'strikeslip' in G.keys():
                gss = G['strikeslip'].flatten()
                filename = '{}_{}_SS.gf'.format(self.name, data)
                gss = gss.astype(dtype)
                gss.tofile(os.path.join(outputDir, filename))

            # DipSlip Component
            if 'dipslip' in G.keys():
                gds = G['dipslip'].flatten()
                filename = '{}_{}_DS.gf'.format(self.name, data)
                gds = gds.astype(dtype)
                gds.tofile(os.path.join(outputDir, filename))

            # Tensile
            if 'tensile' in G.keys():
                gts = G['tensile'].flatten()
                filename = '{}_{}_TS.gf'.format(self.name, data)
                gts = gts.astype(dtype)
                gts.tofile(os.path.join(outputDir, filename))

        # All done
        return

    def setGFsFromFile(self, data, strikeslip=None, dipslip=None, tensile=None, 
                       vertical=False, dtype='d'):
        '''
        Sets the Green's functions from binary files. Be carefull, these have to be in the 
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and 
        if insar, GF are projected already)
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : File containing the Green's functions for strikeslip displacements.
            * dipslip       : File containing the Green's functions for dipslip displacements.
            * tensile       : File containing the Green's functions for tensile displacements.
            * vertical      : Deal with the UP component (gps: default is false, 
                              insar: it will be true anyway).
            * dtype         : Type of binary data.
        '''

        print('---------------------------------')
        print('---------------------------------')
        print("Set up Green's functions for fault {} from files {}, {} and {}".format(self.name, strikeslip, dipslip, tensile))

        # Get the number of patches
        Npatchs = len(self.patch)

        # Read the files and reshape the GFs
        Gss = None; Gds = None; Gts = None
        if strikeslip is not None:
            Gss = np.fromfile(strikeslip, dtype=dtype)
            ndl = int(Gss.shape[0]/Npatchs)
            Gss = Gss.reshape((ndl, Npatchs))
        if dipslip is not None:
            Gds = np.fromfile(dipslip, dtype=dtype)
            ndl = int(Gds.shape[0]/Npatchs)
            Gds = Gds.reshape((ndl, Npatchs))
        if tensile is not None:
            Gts = np.fromfile(tensile, dtype=dtype)
            ndl = int(Gts.shape[0]/Npatchs)
            Gts = Gts.reshape((ndl, Npatchs))

        # Get the data type
        datatype = data.dtype

        # Cut the Matrices following what data do we have and set the GFs
        if datatype is 'gpsrates':
         
            # Initialize
            GssE = None; GdsE = None; GtsE = None
            GssN = None; GdsN = None; GtsN = None
            GssU = None; GdsU = None; GtsU = None

            # Get the values
            if strikeslip is not None:
                GssE = Gss[range(0,data.vel_enu.shape[0]),:]
                GssN = Gss[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GssU = Gss[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]
            if dipslip is not None:
                GdsE = Gds[range(0,data.vel_enu.shape[0]),:]
                GdsN = Gds[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GdsU = Gds[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]
            if tensile is not None:
                GtsE = Gts[range(0,data.vel_enu.shape[0]),:]
                GtsN = Gts[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GtsU = Gts[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]

            # set the GFs
            self.setGFs(data, strikeslip=[GssE, GssN, GssU], dipslip=[GdsE, GdsN, GdsU], 
                        tensile=[GtsE, GtsN, GtsU], vertical=vertical)

        elif datatype is 'insarrates':

            # Initialize
            GssLOS = None; GdsLOS = None; GtsLOS = None

            # Get the values
            if strikeslip is not None: 
                GssLOS = Gss
            if dipslip is not None:
                GdsLOS = Gds
            if tensile is not None:
                GtsLOS = Gts

            # set the GFs
            self.setGFs(data, strikeslip=[GssLOS], dipslip=[GdsLOS], tensile=[GtsLOS], 
                        vertical=True)

        elif datatype is 'cosicorrrates':
            # Don't need to do anything special here. The GF arrays we read in are
            # already in the right shape.
            self.setGFs(data, strikeslip=[Gss], dipslip=[Gds], tensile=[Gts], vertical=vertical)

        # all done
        return
        
    def setGFs(self, data, strikeslip=[None, None, None], dipslip=[None, None, None], 
               tensile=[None, None, None], vertical=False):
        '''
        Stores the input Green's functions matrices into the fault structure.
        Args:
            * data          : Data structure from gpsrates or insarrates.
            * strikeslip    : List of matrices of the Strikeslip Green's functions, ordered E, N, U
            * dipslip       : List of matrices of the dipslip Green's functions, ordered E, N, U
            * tensile       : List of matrices of the tensile Green's functions, ordered E, N, U
            If you provide InSAR GFs, these need to be projected onto the LOS direction already.
        '''

        # Get the number of data per point
        if data.dtype is 'insarrates':
            data.obs_per_station = 1
        elif data.dtype is 'gpsrates':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station = 3
        elif data.dtype is 'cosicorrrates':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if data.dtype is 'insarrates':
            self.d[data.name] = data.vel
            vertical = True # Always true for InSAR
        elif data.dtype is 'gpsrates':
            if vertical:
                self.d[data.name] = data.vel_enu.T.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
        elif data.dtype is 'cosicorrrates':
            self.d[data.name] = np.hstack((data.east.T.flatten(), data.north.T.flatten()))
            if vertical:
                self.d[data.name] = np.hstack((self.d[data.name],
                                               np.zeros_like(data.east.T.ravel())))

        # StrikeSlip
        if len(strikeslip) == 3:            # GPS case

            E_ss = strikeslip[0]
            N_ss = strikeslip[1]
            U_ss = strikeslip[2]
            ss = []
            nd = 0
            if (E_ss is not None) and (N_ss is not None): 
                d = E_ss.shape[0]
                m = E_ss.shape[1]
                ss.append(E_ss)
                ss.append(N_ss)
                nd += 2
            if (U_ss is not None):
                d = U_ss.shape[0]
                m = U_ss.shape[1]
                ss.append(U_ss)
                nd += 1
            if nd > 0:
                ss = np.array(ss)
                ss = ss.reshape((nd*d, m))
                G['strikeslip'] = ss

        elif len(strikeslip) == 1:          # InSAR case

            LOS_ss = strikeslip[0]
            if LOS_ss is not None:
                G['strikeslip'] = LOS_ss

        # DipSlip
        if len(dipslip) == 3:               # GPS case
            E_ds = dipslip[0]
            N_ds = dipslip[1]
            U_ds = dipslip[2]
            ds = []
            nd = 0
            if (E_ds is not None) and (N_ds is not None): 
                d = E_ds.shape[0]
                m = E_ds.shape[1]
                ds.append(E_ds)
                ds.append(N_ds)
                nd += 2
            if (U_ds is not None):
                d = U_ds.shape[0]
                m = U_ds.shape[1]
                ds.append(U_ds)
                nd += 1
            if nd > 0:
                ds = np.array(ds)
                ds = ds.reshape((nd*d, m))
                G['dipslip'] = ds
        
        elif len(dipslip) == 1:             # InSAR case

            LOS_ds = dipslip[0]
            if LOS_ds is not None:
                G['dipslip'] = LOS_ds

        # StrikeSlip
        if len(tensile) == 3:               # GPS case

            E_ts = tensile[0]
            N_ts = tensile[1]
            U_ts = tensile[2]
            ts = []
            nd = 0
            if (E_ts is not None) and (N_ts is not None): 
                d = E_ts.shape[0]
                m = E_ts.shape[1]
                ts.append(E_ts)
                ts.append(N_ts)
                nd += 2
            if (U_ts is not None):
                d = U_ts.shape[0]
                m = U_ts.shape[1]
                ts.append(U_ts)
                nd += 1
            if nd > 0:
                ts = np.array(ts)
                ts = ts.reshape((nd*d, m))
                G['tensile'] = ts

        elif len(tensile) == 1:             # InSAR Case

            LOS_ts = tensile[0]
            if LOS_ts is not None:
                G['dipslip'] = LOS_ds

        # All done
        return

    def assembleGFs(self, datas, polys=0, slipdir='sd', verbose=True):
        '''
        Assemble the Green's functions that have been built using build GFs.
        This routine spits out the General G and the corresponding data vector d.
        Args:
            * datas         : data sets to use as inputs (from gpsrates and insarrates).
            * polys         : 0 -> nothing additional is estimated
                              1 -> estimate a constant offset
                              3 -> estimate z = ax + by + c
                              4 -> estimate z = axy + bx + cy + d
                              'full' -> For GPS, estimates a rotation, translation and scaling 
                              with respect to the center of the network (Helmert transform).
                              'strain' -> For GPS, estimate the full strain tensor (Rotation 
                              + Translation + Internal strain)
            * slipdir       : directions of slip to include. can be any combination of s,d,t.
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print("Assembling G for fault {}".format(self.name))

        # Store the assembled slip directions
        self.slipdir = slipdir

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}

        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if polys.__class__ is not str:
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    if data.dtype is 'gpsrates':
                        self.poly[data.name] = polys
                    else:
                        print('Data type must be gpsrates to implement a Helmert transform')
                        return
        elif polys.__class__ is list:
            for d in range(len(datas)):
                if polys[d].__class__ is not str:
                    self.poly[datas[d].name] = polys[d]*datas[d].obs_per_station
                else:
                    if datas[d].dtype is 'gpsrates':
                        self.poly[datas[d].name] = polys[d]
                    else:
                        print('Data type must be gpsrates to implement a Helmert transform')
                        return

        # Get the number of parameters
        N = len(self.patch)
        Nps = N*len(slipdir)
        Npo = 0
        for data in datas :
            if self.poly[data.name] is 'full':
                if not hasattr(self, 'helmert'):
                    self.helmert = {}
                if data.obs_per_station==3:
                    Npo += 7                    # 3D Helmert transform is 7 parameters
                    self.helmert[data.name] = 7
                else:   
                    Npo += 4                    # 2D Helmert transform is 4 parameters
                    self.helmert[data.name] = 4
            elif self.poly[data.name] is 'strain':
                if not hasattr(self, 'strain'):
                    self.strain = {}
                if data.obs_per_station==2:
                    Npo += 6
                    self.strain[data.name] = 6
                else:
                    print('3d strain has not been implemented')
                    return
            else:
                Npo += (self.poly[data.name])
        Np = Nps + Npo

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Build the desired slip list
        sliplist = []
        if 's' in slipdir:
            sliplist.append('strikeslip')
        if 'd' in slipdir:
            sliplist.append('dipslip')
        if 't' in slipdir:
            sliplist.append('tensile')

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        polstart = Nps
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))
            
            # Fill Glocal
            ec = 0
            for sp in sliplist:
                Glocal[:,ec:ec+N] = self.G[data.name][sp]
                ec += N

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal

            # Build the polynomial function
            if self.poly[data.name].__class__ is not str:
                if self.poly[data.name] > 0:

                    if data.dtype is 'gpsrates':
                        orb = np.zeros((Ndlocal, self.poly[data.name]))
                        nn = Ndlocal/data.obs_per_station
                        orb[:nn, 0] = 1.0
                        orb[nn:2*nn, 1] = 1.0
                        if data.obs_per_station == 3:
                            orb[2*nn:3*nn, 2] = 1.0

                    elif data.dtype is 'insarrates':
                        orb = np.zeros((Ndlocal, self.poly[data.name]))
                        orb[:,0] = 1.0
                        if self.poly[data.name] >= 3:
                            # Compute normalizing factors
                            if not hasattr(self, 'OrbNormalizingFactor'):
                                self.OrbNormalizingFactor = {}
                            self.OrbNormalizingFactor[data.name] = {}
                            x0 = data.x[0]
                            y0 = data.y[0]
                            normX = np.abs(data.x - x0).max()
                            normY = np.abs(data.y - y0).max()
                            # Save them for later
                            self.OrbNormalizingFactor[data.name]['x'] = normX
                            self.OrbNormalizingFactor[data.name]['y'] = normY
                            self.OrbNormalizingFactor[data.name]['ref'] = [x0, y0]
                            # Fill in functionals
                            orb[:,1] = (data.x - x0) / normX
                            orb[:,2] = (data.y - y0) / normY
                        if self.poly[data.name] >= 4:
                            orb[:,3] = orb[:,1] * orb[:,2]
                        # Scale everything by the data factor
                        orb *= data.factor

                    elif data.dtype is 'cosicorrrates':

                        basePoly = self.poly[data.name] / data.obs_per_station
                        assert basePoly == 3 or basePoly == 6, """
                            only support 3rd or 4th order poly for cosicorr
                            """

                        # In case vertical is True, make sure we only include polynomials
                        # for horizontals
                        if basePoly == 3:
                            self.poly[data.name] = 6
                        else:
                            self.poly[data.name] = 12

                        # Compute normalizing factors
                        numPoints = Ndlocal // data.obs_per_station
                        if not hasattr(self, 'OrbNormalizingFactor'):
                            self.OrbNormalizingFactor = {}
                        x0 = data.x[0]
                        y0 = data.y[0]
                        normX = np.abs(data.x - x0).max()
                        normY = np.abs(data.y - y0).max()
                        # Save them for later
                        self.OrbNormalizingFactor[data.name] = {}
                        self.OrbNormalizingFactor[data.name]['ref'] = [x0, y0]
                        self.OrbNormalizingFactor[data.name]['x'] = normX
                        self.OrbNormalizingFactor[data.name]['y'] = normY

                        # Pre-compute position-dependent functional forms
                        f1 = data.factor * np.ones((numPoints,))
                        f2 = data.factor * (data.x - x0) / normX
                        f3 = data.factor * (data.y - y0) / normY
                        f4 = data.factor * (data.x - x0) * (data.y - y0) / (normX*normY)
                        f5 = data.factor * (data.x - x0)**2 / normX**2
                        f6 = data.factor * (data.y - y0)**2 / normY**2
                        polyFuncs = [f1, f2, f3, f4, f5, f6]

                        # Fill in orb matrix given an order
                        orb = np.zeros((numPoints, basePoly))
                        for ind in xrange(basePoly):
                            orb[:,ind] = polyFuncs[ind]

                        # Block diagonal for both components
                        orb = block_diag(orb, orb)

                        # Check to see if we're including verticals
                        if data.obs_per_station == 3:
                            orb = np.vstack((orb, np.zeros((numPoints, 2*basePoly))))

                    # Put it into G for as much observable per station we have
                    polend = polstart + self.poly[data.name]
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += self.poly[data.name]

            else:
                if self.poly[data.name] is 'full':
                    orb = self.getHelmertMatrix(data)
                    if data.obs_per_station==3:
                        nc = 7
                    elif data.obs_per_station==2:
                        nc = 4
                    # Put it into G for as much observable per station we have
                    polend = polstart + nc
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += nc
                if self.poly[data.name] is 'strain':
                    orb = self.get2DstrainEst(data)
                    if data.obs_per_station == 2:
                        nc = 6
                    polend = polstart + nc
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += nc
            # Update el to check where we are
            el = el + Ndlocal
            
        # Store G in self
        self.Gassembled = G

        # All done
        return


    def get2DstrainEst(self, data):
        '''
        Returns the matrix to estimate the full 2d strain tensor.
        '''

        # Check
        assert (data.dtype is 'gpsrates')

        # Get the number of gps stations
        ns = data.station.shape[0]

        # Get the data vector size
        nd = self.d[data.name].shape[0]

        # Get the number of parameters to look for
        if data.obs_per_station==2:
            nc = 6
        else:
            print('Not implemented')
            return

        # Check something
        assert data.obs_per_station*ns==nd

        # Get the center of the network
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)

        # Compute the baselines
        base_x = data.x - x0
        base_y = data.y - y0

        # Normalize the baselines 
        base_max = np.max([np.abs(base_x).max(), np.abs(base_y).max()])
        base_x /= base_max
        base_y /= base_max

        # Store the normalizing factor
        if not hasattr(self, 'StrainNormalizingFactor'):
            self.StrainNormalizingFactor = {}
        self.StrainNormalizingFactor[data.name] = base_max

        # Allocate a Base
        H = np.zeros((data.obs_per_station,nc))

        # Put the transaltion in the base
        H[:,:data.obs_per_station] = np.eye(data.obs_per_station)

        # Allocate the full matrix
        Hf = np.zeros((nd,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,data.obs_per_station:] = 0.0 

            # Get the values
            x1, y1 = base_x[i], base_y[i]

            # Store them
            H[0,2] = x1
            H[0,3] = 0.5*y1
            H[0,5] = 0.5*y1
            H[1,3] = 0.5*x1
            H[1,4] = y1
            H[1,5] = -0.5*y1

            # Put the lines where they should be
            Hf[i,:] = H[0,:]
            Hf[i+ns,:] = H[1,:]

        # All done
        return Hf

    def getHelmertMatrix(self, data):
        '''
        Returns a Helmert matrix for a gps data set.
        '''

        # Check
        assert (data.dtype is 'gpsrates')

        # Get the number of stations
        ns = data.station.shape[0]

        # Get the data vector size
        nd = self.d[data.name].shape[0]

        # Get the number of helmert transform parameters
        if data.obs_per_station==3:
            nc = 7
        else:
            nc = 4

        # Check something
        assert data.obs_per_station*ns==nd

        # Get the position of the center of the network
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)
        z0 = 0              # We do not deal with the altitude of the stations yet (later)

        # Compute the baselines
        base_x = data.x - x0
        base_y = data.y - y0
        base_z = 0

        # Normalize the baselines
        base_x_max = np.abs(base_x).max(); base_x /= base_x_max
        base_y_max = np.abs(base_y).max(); base_y /= base_y_max

        # Allocate a Helmert base
        H = np.zeros((data.obs_per_station,nc))
        
        # put the translation in it (that part never changes)
        H[:,:data.obs_per_station] = np.eye(data.obs_per_station)

        # Allocate the full matrix
        Hf = np.zeros((nd,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,data.obs_per_station:] = 0.0

            # Put the rotation components and the scale components
            x1, y1, z1 = base_x[i], base_y[i], base_z
            if nc==7:
                H[:,3:6] = np.array([[0.0, -z1, y1],
                                     [z1, 0.0, -x1],
                                     [-y1, x1, 0.0]])
                H[:,6] = np.array([x1, y1, z1])
            else:
                H[:,2] = np.array([y1, -x1])
                H[:,3] = np.array([x1, y1])

            # put the lines where they should be
            Hf[i,:] = H[0]
            Hf[i+ns,:] = H[1]
            if nc==7:
                Hf[i+2*ns,:] = H[2]

        # all done 
        return Hf

    def assembled(self, datas):
        ''' 
        Assembles the data vector corresponding to the stored green's functions.
        Args:
            * datas         : list of the data object involved (from gpsrates and insarrates).
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Assembling d vector")

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Create a data vector
        d = np.zeros((Nd,))

        # Loop over the datasets
        el = 0
        for data in datas:

                # print
                print("Dealing with data {}".format(data.name))

                # Get the local d
                dlocal = self.d[data.name]
                Ndlocal = dlocal.shape[0]

                # Store it in d
                d[el:el+Ndlocal] = dlocal

                # update el
                el += Ndlocal

        # Store d in self
        self.dassembled = d

        # All done
        return

    def assembleCd(self, datas, add_prediction=None):
        '''
        Assembles the data covariance matrixes that have been built by each data structure.
        add_prediction: Precentage of displacement to add to the Cd diagonal to simulate a Cp (prediction error).
        '''

        # Check if the Green's function are ready
        if self.Gassembled is None:
            print("You should assemble the Green's function matrix first")
            return

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            # Fill in Cd
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            # Add some Cp if asked
            if add_prediction is not None:
                Cd[st:se, st:se] += np.diag((self.d[data.name]*add_prediction/100.)**2)
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return


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
        vertices, faces = self.gocad_vertices, self.gocad_faces

        # First find adjacent triangles for all triangles
        npatch = len(self.patch)
        for i in range(npatch):
            
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


    def buildCmGaussian(self, sigma, extra_params=None):
        '''
        Builds a diagonal Cm with sigma values on the diagonal.
        sigma is a list of numbers, as long as you have components of slip (1, 2 or 3).
        extra_params is a list of extra parameters.
        '''
   
        # Get the number of slip directions
        slipdir = len(self.slipdir)
        numpatch = len(self.patch)

        # Number of parameters
        Np = numpatch * slipdir
        if extra_params is not None:
            Np += len(extra_params)

        # Create Cm
        Cm = np.zeros((Np, Np))

        # Loop over slip dir
        for i in range(slipdir):
            Cmt = np.diag(sigma[i] * np.ones(len(self.patch),))
            Cm[i*numpatch:(i+1)*numpatch,i*numpatch:(i+1)*numpatch] = Cmt

        # Put the extra parameter sigma values
        st = numpatch * slipdir
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Stores Cm
        self.Cm = Cm

        # all done
        return

    def buildCm(self, sigma, lam, lam0=None, extra_params=None, lim=None, verbose=True):
        '''
        Builds a model covariance matrix using the equation described in Radiguet et al 2010.
        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.
            * lam0          : Normalizing distance (if None, lam0=min(distance between patches)).
            * extra_params  : Add some extra values on the diagonal.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(lam))

        # Need the patch geometry
        if self.patch is None:
            print("You should build the patches and the Green's functions first.")
            return

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = ((np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())
                / (np.unique(self.centers[:,0]).size))
            yd = ((np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())
                / (np.unique(self.centers[:,1]).size))
            zd = ((np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())
                / (np.unique(self.centers[:,2]).size))
            lam0 = np.sqrt(xd**2 + yd**2 + zd**2)
        if verbose:
            print("Lambda0 = {}".format(lam0))
        C = (sigma * lam0 / lam)**2

        # Creates the principal Cm matrix
        Np = len(self.patch) * len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((len(self.patch), len(self.patch)))
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        npatch = len(self.patch)
        for i in range(npatch):
            distances = np.zeros((npatch,))
            p1 = self.patch[i]
            for j in range(npatch):
                if j == i:
                    continue
                p2 = self.patch[j]
                distances[j] = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)
            Cmt[i,:] = C * np.exp(-distances / lam)

        # Store that into Cm
        st = 0
        for i in range(len(slipdir)):
            se = st + len(self.patch)
            Cm[st:se, st:se] = Cmt
            st += len(self.patch)

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return


    def distancePatchToPatch(self, patch1, patch2, distance='center', lim=None):
        '''
        Measures the distance between two patches.
        Args:
            * patch1    : geometry of the first patch.
            * patch2    : geometry of the second patch.
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
            dis = np.sqrt( (x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis>lim[0]:
                    dis = lim[1]

        # All done
        return dis

    def writeEDKSsubParams(self, data, edksfilename, amax=None, plot=False):
        '''
        Write the subParam file needed for the interpolation of the green's function in EDKS.
        Francisco's program cuts the patches into small patches, interpolates the kernels to get the GFs at each point source, 
        then averages the GFs on the pacth. To decide the size of the minimum patch, it uses St Vernant's principle.
        If amax is specified, the minimum size is fixed.
        Args:
            * data          : Data object from gpsrates or insarrates.
            * edksfilename  : Name of the file containing the kernels.
            * amax          : Specifies the minimum size of the divided patch. If None, uses St Vernant's principle.
            * plot          : Activates plotting.
        Returns:
            * filename      : Name of the subParams file created.
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
        useRecvDir = False # False for GPS, uses ENU displacements
        EDKSunits = 1000.0
        EDKSfilename = '{}'.format(edksfilename)
        prefix = 'edks_{}_{}'.format(fltname, datname)
        plotGeometry = '{}'.format(plot)

        # Open the EDKSsubParams.py file
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

        # Build usefull outputs
        parNames = ['useRecvDir', 'Amax', 'EDKSunits', 'EDKSfilename', 'prefix']
        parValues = [ useRecvDir ,  amax ,  EDKSunits ,  EDKSfilename ,  prefix ]
        method_par = dict(zip(parNames, parValues))

        # All done
        return filename, TrianglePropFile, PointCoordFile, ReceiverFile, method_par

    def writeEDKSgeometry(self, ref=None):
        '''
        This routine spits out 2 files:
        filename.TriangleProp: Patch ID | Lon (deg) | Lat | Easti (km) | North | Depth (km) | Strike (deg) | Dip  | Area (km^2) | Vertice ids
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
        vcount = 1
        TriP_format = '%-6d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %6d %6d %6d\n'
        PoC_format =  '%-6d %10.4f %10.4f %10.4f %10.4f %10.4f\n'
        for p in range(len(self.patch)):
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            lonc, latc = self.xy2ll(xc,yc)
            if ref is not None:
                xc -= refx
                yc -= refy
            verts = copy.deepcopy(self.patch[p])
            vid   = []
            for xv,yv,zv in verts:
                lonv,latv = self.xy2ll(xv,yv)
                if ref is not None:
                    xv -= refx
                    yv -= refy
                PoC.write(PoC_format%(vcount,lonv,latv,xv,yv,zv))
                vid.append(vcount)
                vcount += 1
            TriP_tuple = (p,lonc,latc,xc,yc,zc,strike,dip,self.area[p],vid[0],vid[1],vid[2])
            TriP.write(TriP_format%TriP_tuple)

        # Close the files
        TriP.close()
        PoC.close()

        # All done
        return fltname,TrianglePropFile,PointCoordFile  


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
        depths = np.unique(np.array([[self.patch[i][u][2] for u in range(4)] for i in range(len(self.patch))]).flatten())
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
        x, y, z = self.gocad_vertices[:,0], self.gocad_vertices[:,1], self.gocad_vertices[:,2]
        z *= negFactor
        mesh = mlab.triangular_mesh(x, y, z, self.gocad_faces, representation='wireframe',
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
        User provides a Mapping function np.array((len(self.patch), len(fault.patch))) and a fault and the slip from the argument
        fault is mapped into self.slip.
        Function just does:
        self.slip[:,0] = np.dot(Map,fault.slip)
        ...
        '''

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

