'''
A class that deals with vertical faults.

Written by R. Jolivet, April 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import copy
import sys

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok
from .RectangularPatches import RectangularPatches

class verticalfault(RectangularPatches):

    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name          : Name of the fault.
        '''

        # Initialize base class
        super(self.__class__,self).__init__(name,utmzone,ellps)

        # Warning
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('             WARNING WARNING WARNING               ')
        print(' vertical fault class is not consistent with other ')
        print(' rectangular fault classes. The order of the patch ')
        print(' corner is different. We need to fix that asap...  ')
        print('             WARNING WARNING WARNING               ')
        print('-----------------------------------------------------')
        print('-----------------------------------------------------')


        # All done
        return

    def extrapolate(self, length_added=50, tol=2., fracstep=5., extrap='ud'):
        ''' 
        Extrapolates the surface trace. This is usefull when building deep patches for interseismic loading.
        Args:
            * length_added  : Length to add when extrapolating.
            * tol           : Tolerance to find the good length.
            * fracstep      : control each jump size.
            * extrap        : if u in extrap -> extrapolates at the end
                              if d in extrap -> extrapolates at the beginning
                              default is 'ud'
        '''

        # print 
        print ("Extrapolating the fault for {} km".format(length_added))

        # Check if the fault has been interpolated before
        if self.xi is None:
            print ("Run the discretize() routine first")
            return

        # Build the interpolation routine
        import scipy.interpolate as scint
        fi = scint.interp1d(self.xi, self.yi)

        # Build the extrapolation routine
        fx = self.extrap1d(fi)

        # make lists
        self.xi = self.xi.tolist()
        self.yi = self.yi.tolist()

        if 'd' in extrap:
        
            # First guess for first point
            xt = self.xi[0] - length_added/2.
            yt = fx(xt)
            d = np.sqrt( (xt-self.xi[0])**2 + (yt-self.yi[0])**2)

            # Loop to find the best distance
            while np.abs(d-length_added)>tol:
                # move up or down
                if (d-length_added)>0:
                    xt = xt + d/fracstep
                else:
                    xt = xt - d/fracstep
                # Get the corresponding yt
                yt = fx(xt)
                # New distance
                d = np.sqrt( (xt-self.xi[0])**2 + (yt-self.yi[0])**2) 
        
            # prepend the thing
            self.xi.reverse()
            self.xi.append(xt)
            self.xi.reverse()
            self.yi.reverse()
            self.yi.append(yt)
            self.yi.reverse()

        if 'u' in extrap:

            # First guess for the last point
            xt = self.xi[-1] + length_added/2.
            yt = fx(xt)
            d = np.sqrt( (xt-self.xi[-1])**2 + (yt-self.yi[-1])**2)

            # Loop to find the best distance
            while np.abs(d-length_added)>tol:
                # move up or down
                if (d-length_added)<0:
                    xt = xt + d/fracstep
                else:
                    xt = xt - d/fracstep
                # Get the corresponding yt
                yt = fx(xt)
                # New distance
                d = np.sqrt( (xt-self.xi[-1])**2 + (yt-self.yi[-1])**2)

            # Append the result
            self.xi.append(xt)
            self.yi.append(yt)

        # Make them array again
        self.xi = np.array(self.xi)
        self.yi = np.array(self.yi)

        # Build the corresponding lon lat arrays
        self.loni, self.lati = self.xy2ll(self.xi, self.yi)

        # All done
        return

    def extrap1d(self,interpolator):
        '''
        Linear extrapolation routine. Found on StackOverflow by sastanin.
        '''

        # import a bunch of stuff
        from scipy import arange, array, exp

        xs = interpolator.x
        ys = interpolator.y
        def pointwise(x):
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                return interpolator(x)
        def ufunclike(xs):
            return pointwise(xs) #array(map(pointwise, array(xs)))
        return ufunclike
    
    def setDepth(self, depth, top=0, num=5):
        '''
        Set the maximum depth of the fault patches.

        Args:
            * depth         : Depth of the fault patches.
            * num           : Number of fault patches at depth.
        '''

        # Set depth
        self.top = top
        self.depth = depth
        self.numz = num

        # All done
        return

    def build_patches(self):
        '''
        Builds rectangular patches from the discretized fault.
        A patch is a list of 4 corners.
        '''

        # If the maximum depth and the number of patches is not set
        if self.depth is None:
            print("Depth and number of patches are not set.")
            print("Please use setdepth to define maximum depth and number of patches")
            return

        print ("Build patches for fault {} between depths: {}, {}".format(self.name, self.top, self.depth))

        # Define the depth vector
        z = np.linspace(self.top, self.depth, num=self.numz+1)
        self.z_patches = z

        # If the discretization is not done
        if self.xi is None:
            self.discretize()

        # Define a patch list
        self.patch = []
        self.patchll = []
        self.slip = []

        # Iterate over the surface discretized fault points
        for i in range(len(self.xi)-1):
            # First corner
            x1 = self.xi[i]
            y1 = self.yi[i]
            lon1 = self.loni[i]
            lat1 = self.lati[i]
            # Second corner
            x2 = self.xi[i]
            y2 = self.yi[i]
            lon2 = self.loni[i]
            lat2 = self.lati[i]
            # Third corner
            x3 = self.xi[i+1]
            y3 = self.yi[i+1]
            lon3 = self.loni[i+1]
            lat3 = self.lati[i+1]
            # Fourth corner
            x4 = self.xi[i+1]
            y4 = self.yi[i+1]
            lon4 = self.loni[i+1]
            lat4 = self.lati[i+1]
            # iterate at depth
            for j in range(len(z)-1):
                p = np.zeros((4,3))
                pll = np.zeros((4,3))
                p[0,:] = [x1, y1, z[j]]
                pll[0,:] = [lon1, lat1, z[j]]
                p[3,:] = [x2, y2, z[j+1]]
                pll[3,:] = [lon2, lat2, z[j+1]]
                p[2,:] = [x3, y3, z[j+1]]
                pll[2,:] = [lon3, lat3, z[j+1]]
                p[1,:] = [x4, y4, z[j]]
                pll[1,:] = [lon4, lat4, z[j]]
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])

        # Translate slip to np.array
        self.slip = np.array(self.slip)

        # Compute the equivalent patches
        self.equivpatch = copy.deepcopy(self.patch)

        # All done
        return

    def BuildPatchesVarResolution(self, depths, Depthpoints, Resolpoints, interpolation='linear', minpatchsize=0.1, extrap=None):
        '''
        Patchizes the fault with a variable patch size at depth.
        The variable patch size is given by the respoints table.
        Depthpoints = [depth1, depth2, depth3, ...., depthN]
        Resolpoints = [Resol1, Resol2, Resol3, ...., ResolN]
        The final resolution is interpolated given the 'interpolation' method.
        Interpolation can be 'linear', 'cubic'.
        '''

        print('Build fault patches for fault {} between {} and {} km deep, with a variable resolution'.format(self.name, self.top, self.depth))

        # Define the depth vector
        z = np.array(depths)
        self.z_patches = z

        # Interpolate the resolution
        fint = sciint.interp1d(Depthpoints, Resolpoints, kind=interpolation)
        resol = fint(z)

        # build lists for storing things
        self.patch = []
        self.patchll = []
        self.slip = []

        # iterate over the depths 
        for j in range(len(z)-1):

            # discretize the fault at the desired resolution
            print('Discretizing at depth {}'.format(z[j]))
            self.discretize(every=np.floor(resol[j]), tol=resol[j]/20., fracstep=resol[j]/1000.)
            if extrap is not None:
                self.extrapolate(length_added=extrap[0], extrap=extrap[1])

            # iterate over the discretized fault
            for i in range(len(self.xi)-1):
                # First corner
                x1 = self.xi[i]
                y1 = self.yi[i]
                lon1 = self.loni[i]
                lat1 = self.lati[i]
                # Second corner
                x2 = self.xi[i]
                y2 = self.yi[i]
                lon2 = self.loni[i]
                lat2 = self.lati[i]
                # Third corner
                x3 = self.xi[i+1]
                y3 = self.yi[i+1]
                lon3 = self.loni[i+1]
                lat3 = self.lati[i+1]
                # Fourth corner
                x4 = self.xi[i+1]
                y4 = self.yi[i+1]
                lon4 = self.loni[i+1]
                lat4 = self.lati[i+1]
                # build patches
                p = np.zeros((4,3))
                pll = np.zeros((4,3))
                # fill them
                p[0,:] = [x1, y1, z[j]]
                pll[0,:] = [lon1, lat1, z[j]]
                p[1,:] = [x2, y2, z[j+1]]
                pll[1,:] = [lon2, lat2, z[j+1]]
                p[2,:] = [x3, y3, z[j+1]]
                pll[2,:] = [lon3, lat3, z[j+1]]
                p[3,:] = [x4, y4, z[j]]
                pll[3,:] = [lon4, lat4, z[j]]
                psize = np.sqrt( (x3-x2)**2 + (y3-y2)**2 )
                if psize>minpatchsize:
                    self.patch.append(p)
                    self.patchll.append(pll)
                    self.slip.append([0.0, 0.0, 0.0])
                else:           # Increase the size of the previous patch
                    self.patch[-1][2,:] = [x3, y3, z[j+1]]
                    self.patch[-1][3,:] = [x4, y4, z[j]]
                    self.patchll[-1][2,:] = [lon3, lat3, z[j+1]]
                    self.patchll[-1][3,:] = [lon4, lat4, z[j]]


        # Translate slip into a np.array
        self.slip = np.array(self.slip)

        # Compute the equivalent patches
        self.computeEquivRectangle()

        # all done
        return

    def rotationHoriz(self, center, angle):
        '''
        Rotates the geometry of the fault around center, of an angle.
        Args:
            * center    : [lon,lat]
            * angle     : degrees
        '''

        # Translate the center to x, y
        xc, yc = self.ll2xy(center[0], center[1])
        ref = np.array([xc, yc])

        # Create the rotation matrix
        angle = angle*np.pi/180.
        Rot = np.array( [ [np.cos(angle), -1.0*np.sin(angle)],
                          [np.sin(angle), np.cos(angle)] ] )

        # Loop on the patches
        for i in range(len(self.patch)):

            # Get patch
            p = self.patch[i]
            pll = self.patchll[i]

            for j in range(4):
                x, y = np.dot( Rot, p[j][:-1] - ref )
                p[j][0] = x + xc
                p[j][1] = y + yc
                lon, lat = self.xy2ll(p[j][0],p[j][1])
                pll[j][0] = lon
                pll[j][1] = lat

        # All done 
        return

    def translationHoriz(self, dx, dy):
        '''
        Translates the patches.
        Args:
            * dx    : Translation along x (km)
            * dy    : Translation along y (km)
        '''

        # Loop on the patches
        for i in range(len(self.patch)):

            # Get patch
            p = self.patch[i]
            pll = self.patchll[i]

            for j in range(4):
                p[j][0] += dx
                p[j][1] += dy
                lon, lat = self.xy2ll(p[j][0],p[j][1])
                pll[j][0] = lon
                pll[j][1] = lat

        # All done 
        return


    def mergePatches(self, p1, p2):
        '''
        Merges 2 patches that have common corners.
        Args:
            * p1    : index of the patch #1.
            * p2    : index of the patch #2.
        '''

        print('Merging patches {} and {} into patch {}'.format(p1,p2,p1))

        # Get the patches
        patch1 = self.patch[p1]
        patch2 = self.patch[p2]
        patch1ll = self.patchll[p1]
        patch2ll = self.patchll[p2]

        # Create the newpatches
        newpatch = np.zeros((4,3))
        newpatchll = np.zeros((4,3))

        # determine which corners are in common, needs at least two
        if ((list(patch1[0])==list(patch2[1])) and (list(patch1[3])==list(patch2[2]))):     # patch2 is above patch1
            newpatch[0,:] = patch2[0,:]; newpatchll[0,:] = patch2ll[0,:] 
            newpatch[1,:] = patch1[1,:]; newpatchll[1,:] = patch1ll[1,:]
            newpatch[2,:] = patch1[2,:]; newpatchll[2,:] = patch1ll[2,:]
            newpatch[3,:] = patch2[3,:]; newpatchll[3,:] = patch2ll[3,:]
        elif ((list(patch1[3])==list(patch2[0])) and (list(patch1[2])==list(patch2[1]))):   # patch2 is on the right of patch1
            newpatch[0,:] = patch1[0,:]; newpatchll[0,:] = patch1ll[0,:]
            newpatch[1,:] = patch1[1,:]; newpatchll[1,:] = patch1ll[1,:]
            newpatch[2,:] = patch2[2,:]; newpatchll[2,:] = patch2ll[2,:]
            newpatch[3,:] = patch2[3,:]; newpatchll[3,:] = patch2ll[3,:]
        elif ((list(patch1[1])==list(patch2[0])) and (list(patch1[2])==list(patch2[3]))):   # patch2 is under patch1
            newpatch[0,:] = patch1[0,:]; newpatchll[0,:] = patch1ll[0,:]
            newpatch[1,:] = patch2[1,:]; newpatchll[1,:] = patch2ll[1,:]
            newpatch[2,:] = patch2[2,:]; newpatchll[2,:] = patch2ll[2,:]
            newpatch[3,:] = patch1[3,:]; newpatchll[3,:] = patch1ll[3,:]
        elif ((list(patch1[0])==list(patch2[3])) and (list(patch1[1])==list(patch2[2]))):   # patch2 is on the left of patch1
            newpatch[0,:] = patch2[0,:]; newpatchll[0,:] = patch2ll[0,:]
            newpatch[1,:] = patch2[1,:]; newpatchll[1,:] = patch2ll[1,:]
            newpatch[2,:] = patch1[2,:]; newpatchll[2,:] = patch1ll[2,:]
            newpatch[3,:] = patch1[3,:]; newpatchll[3,:] = patch1ll[3,:]
        else:
            print('Patches do not have common corners...')
            return

        # Replace the patch 1 by the new patch
        self.patch[p1] = newpatch
        self.patchll[p1] = newpatchll

        # Delete the patch 2
        self.deletepatch(p2)

        # All done
        return

    def differentiateGFs(self, datas):
        '''
        Uses the Delaunay triangulation to prepare a differential Green's function matrix, data vector
        and data covariance matrix.
        Args:   
            * datas         : List of dataset concerned
        '''

        # Create temporary Green's function, data and Cd dictionaries to hold the new ones
        Gdiff = {}
        ddiff = {}

        # Loop over the datasets
        for data in datas:

            # Check something
            if data.dtype is not 'gpsrates':
                print('This has not been implemented for other data set than gpsrates')
                return

            # Get the GFs, the data and the data covariance
            G = self.G[data.name]
            d = self.d[data.name]
            Cd = data.Cd

            # Get some size informations
            nstation = data.station.shape[0]
            lengthd = d.shape[0]
            if (lengthd == 3*nstation):
               vertical = True
               ncomp = 3
            else:
               vertical = False
               ncomp = 2

            # Get the couples
            edges = data.triangle['Edges']

            # How many lines/columns ?
            Nd = edges.shape[0]
            k = G.keys()[0]
            Np = G[k].shape[1]

            # Create the spaces
            Gdiff[data.name] = {}
            for key in G.keys():
                Gdiff[data.name][key] = np.zeros((Nd*ncomp, Np))
            ddiff[data.name] = np.zeros((Nd*ncomp,))
            Cddiff = np.zeros((Nd*ncomp, Nd*ncomp))

            # Loop over the lines of Edges
            for i in range(Nd):

                # Get the couple
                m = edges[i][0]
                n = edges[i][1]

                # Deal with the GFs
                for key in G.keys():
                    # East component
                    Line1 = G[key][m,:]
                    Line2 = G[key][n,:]
                    Gdiff[data.name][key][i,:] = Line1 - Line2
                    # North Component
                    Line1 = G[key][m+nstation,:]
                    Line2 = G[key][n+nstation,:]
                    Gdiff[data.name][key][i+Nd,:] = Line1 - Line2
                    # Vertical
                    if vertical:
                        Line1 = G[key][m+2*nstation,:]
                        Line2 = G[key][n+2*nstation,:]
                        Gdiff[data.name][key][i+2*Nd,:] = Line1 - Line2

                # Deal with the data vector
                # East
                d1 = d[m]
                d2 = d[n]
                ddiff[data.name][i] = d1 - d2
                # North
                d1 = d[m+nstation]
                d2 = d[n+nstation]
                ddiff[data.name][i+Nd] = d1 - d2
                # Vertical
                if vertical:
                    d1 = d[m+2*nstation]
                    d2 = d[n+2*nstation]
                    ddiff[data.name][i+2*Nd] = d1 - d2

                # Deal with the Covariance (Only diagonal, for now)
                # East
                cd1 = Cd[m,m]
                cd2 = Cd[n,n]
                Cddiff[i,i] = cd1+cd2
                # North
                cd1 = Cd[m+nstation,m+nstation]
                cd2 = Cd[n+nstation,n+nstation]
                Cddiff[i+Nd,i+Nd] = cd1+cd2
                # Vertical
                if vertical:
                    cd1 = Cd[m+2*nstation,m+2*nstation]
                    cd2 = Cd[n+2*nstation,n+2*nstation]
                    Cddiff[i+2*Nd,i+2*Nd] = cd1+cd2

            # Once the data loop is done, store Cd
            data.Cd = Cddiff

        # Once it is all done, store G and d
        self.G = Gdiff
        self.d = ddiff

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

    def cutPatchesIn2Vertically(self, ipatches):
        '''
        Cut the Patches in 2 vertically.
        '''

        # Check something
        if ipatches.__class__ in (int, np.int):
            ipatches = [ipatches]
        elif ipatches.__class__ in (np.ndarray):
            ipatches = ipatches.tolist()
        else:
            print('Input must be of type int, np.int, list or array of int')
            return

        # Loop over the patches
        for ip in ipatches:

            # Get the patch
            patch = self.patch[ip]

            # Delete the patch 
            self.deletepatch(ip)

            # Find the middle of the two horizontal bars

        # All done
        return

    def associatePatch2PDFs(self, directory='.', prefix='step_001_param'):
        '''
        Associates a patch with a pdf called directory/prefix_{#}.dat.
        import AltarExplore....
        '''

        # Import necessary
        import AltarExplore as alt
        
        # Parameters index are in self.index_parameter
        istrikeslip = self.index_parameter[:,0]
        idipslip = self.index_parameter[:,1]
        itensile = self.index_parameter[:,2]

        # Create a list of slip pdfs
        self.slippdfs = []
        for i in range(self.slip.shape[0]):
            sys.stdout.write('\r Patch {}/{}'.format(i,self.slip.shape[0]))
            sys.stdout.flush()
            # integers are needed
            iss = np.int(istrikeslip[i])
            ids = np.int(idipslip[i])
            its = np.int(itensile[i])
            # Create the file names
            pss = None
            pds = None
            pts = None
            if istrikeslip[i]< 10000:
                pss = '{}/{}_{:03d}.dat'.format(directory, prefix, iss)
            if idipslip[i]<10000:
                pds = '{}/{}_{:03d}.dat'.format(directory, prefix, ids)
            if itensile[i]<10000:
                pts = '{}/{}_{:03d}.dat'.format(directory, prefix, its)
            # Create the parameters
            Pss = None; Pds = None; Pts = None
            if pss is not None:
                Pss = alt.parameter('{:03d}'.format(iss), pss)
            if pds is not None:
                Pds = alt.parameter('{:03d}'.format(ids), pds)
            if pts is not None:
                Pts = alt.parameter('{:03d}'.format(its), pts)
            # Store these
            self.slippdfs.append([Pss, Pds, Pts])

        sys.stdout.write('\n')
        sys.stdout.flush()

        # all done
        return

    def writePatches2File(self, filename, add_slip=None, scale=1.0, patch='normal',
                          stdh5=None, decim=1):
        '''
        Writes the patch corners in a file that can be used in psxyz.
        Args:
            * filename      : Name of the file.
            * add_slip      : Put the slip as a value for the color. Can be None, strikeslip, dipslip, total.
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
        '''

        # Write something
        print('Routine overwritten on top of base class')
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
        for p in range(nPatches):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip is 'strikeslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,p])
                    else:
                        slp = self.slip[p,0]*scale
                elif add_slip is 'dipslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,p+nPatches])
                    else:
                        slp = self.slip[p,1]*scale
                elif add_slip is 'total':
                    if stdh5 is not None:
                        slp = np.std(samples[:,p]**2 + samples[:,p+nPatches]**2)
                    else:
                        slp = np.sqrt(self.slip[p,0]**2 + self.slip[p,1]**2)*scale
                elif add_slip is 'normaltraction':
                    slp = self.Normal
                elif add_slip is 'strikesheartraction':
                    slp = self.ShearStrike
                elif add_slip is 'dipsheartraction':
                    slp = self.ShearDip
                # Make string
                string = '-Z{}'.format(slp)

            # Put the parameter number in the file as well if it exists
            parameter = ' ' 
            if hasattr(self,'index_parameter'):
                i = np.int(self.index_parameter[p,0])
                j = np.int(self.index_parameter[p,1])
                k = np.int(self.index_parameter[p,2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            slipstring = ' # {} {} {} '.format(self.slip[p,0], self.slip[p,1], self.slip[p,2])

            # Write the string to file
            fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Write the 4 patch corners (the order is to be GMT friendly)
            if patch in ('normal'):
                p = self.patchll[p]
            elif patch in ('equiv'):
                p = self.equivpatchll[p]
            pp=p[3]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[0]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[1]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[2]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))

        # Close th file
        fout.close()

        # Close h5 file if it is open
        if stdh5 is not None:
            h5fid.close()

        # All done 
        return
