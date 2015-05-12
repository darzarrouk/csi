'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy.spatial.distance as scidis
import copy
import sys

# Personals
from .SourceInv import SourceInv
from .geodeticplot import geodeticplot as geoplot

class insarrates(SourceInv):

    def __init__(self, name, utmzone='10', ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(insarrates,self).__init__(name,utmzone,ellps) 

        # Initialize the data set
        self.dtype = 'insarrates'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR data set {}".format(self.name))

        # Initialize some things
        self.vel = None
        self.synth = None
        self.err = None
        self.lon = None
        self.lat = None
        self.los = None
        self.corner = None
        self.xycorner = None
        self.Cd = None

        # All done
        return

    def checkNaNs(self):
        ''' 
        Checks and remove data points that have NaNs in vel, err, lon, lat or los.
        '''

        # Check 
        if self.vel is not None:
            uVel = np.flatnonzero(np.isnan(self.vel))
        else:
            uVel = np.array([])
        if self.err is not None:
            uErr = np.flatnonzero(np.isnan(self.err))
        else:
            uErr = np.array([])
        if self.lon is not None:
            uLon = np.flatnonzero(np.isnan(self.lon))
        else:
            uLon = np.array([])
        if self.lat is not None:
            uLat = np.flatnonzero(np.isnan(self.lat))
        else:
            uLat = np.array([])
        if self.los is not None:
            uLos, toto = np.where(np.isnan(self.los))
            uLos = np.unique(uLos.flatten())
        else:
            uLos = np.array([])

        # Concatenate all these guys
        uRemove = np.concatenate((uVel, uErr, uLon, uLat, uLos))

        # Reject pixels
        self.reject_pixel(uRemove)

        # All done
        return

    def read_from_ascii_simple(self, filename, factor=1.0, step=0.0, header=0, los=None):
        '''
        Read the InSAR data from an ascii file with 3 cols.
        Args:
            * filename      : Name of the input file. 
            Lon | Lat | los measurement 
            * factor        : Factor to multiply the LOS velocity.
            * step          : Add a value to the velocity.
            * header        : Size of the header.
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read it all
        Lines = fin.readlines()
        fin.close()

        # Initialize the business
        self.vel = []
        self.err = []
        self.lon = []
        self.lat = []

        # Loop over yje lines
        for i in range(len(Lines)):
            # Get values
            line = Lines[i].split()
            # Fill in the values
            self.lon.append(np.float(line[0]))
            self.lat.append(np.float(line[1]))
            self.vel.append(np.float(line[2]))
            self.err.append(np.float(line[3]))

        # Make arrays
        self.vel = (np.array(self.vel)+step)*factor
        self.err = np.array(self.err)*factor
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)

        # Compute lon lat to utm
        self.x, self.y = self.ll2xy(self.lon,self.lat)

        # store the factor
        self.factor = factor

        # LOS
        if los is not None:
            self.los = []
            fin = open(los, 'r')
            Lines = fin.readlines()
            fin.close()
            for line in Lines:
                line = line.split()
                self.los.append([float(line[0]), float(line[1]), float(line[2])])
        self.los = np.array(self.los)

        # All done
        return

    def read_from_ascii(self, filename, factor=1.0, step=0.0, header=0):
        '''
        Read the InSAR data from an ascii file.
        Args:
            * filename      : Name of the input file. Lon | Lat | los measurement | los uncertainty | los E | los N | los U.
            * factor        : Factor to multiply the LOS velocity.
            * step          : Add a value to the velocity.
            * header        : Size of the header.
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read it all
        Lines = fin.readlines()
        fin.close()

        # Initialize the business
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []
        self.corner = []

        # Loop over yje lines
        for i in range(len(Lines)):
            # Get values
            line = Lines[i].split()
            # Fill in the values
            self.lon.append(np.float(line[0]))
            self.lat.append(np.float(line[1]))
            self.vel.append(np.float(line[2]))
            self.err.append(np.float(line[3]))
            self.los.append([np.float(line[4]), np.float(line[5]), np.float(line[6])])

        # Make arrays
        self.vel = (np.array(self.vel)+step)*factor
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.err = np.array(self.err)*factor
        self.los = np.array(self.los)

        # Compute lon lat to utm
        self.x, self.y = self.ll2xy(self.lon,self.lat)

        # store the factor
        self.factor = factor

        # All done
        return

    def read_from_varres(self,filename, factor=1.0, step=0.0, header=2, cov=False):
        '''
        Read the InSAR LOS rates from the VarRes output.
        Args:
            * filename      : Name of the input file. Two files are opened filename.txt and filename.rsp.
            * factor        : Factor to multiply the LOS velocity.
            * step          : Add a value to the velocity.
            * header        : Size of the header.
            * cov           : Read an additional covariance file (binary float32, Nd*Nd elements).
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename+'.txt','r')
        fsp = open(filename+'.rsp','r')

        # Read it all
        A = fin.readlines()
        B = fsp.readlines()

        # Initialize the business
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []
        self.corner = []

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            tmp = A[i].split()
            self.vel.append(np.float(tmp[5]))
            self.lon.append(np.float(tmp[3]))
            self.lat.append(np.float(tmp[4]))
            self.err.append(np.float(tmp[6]))
            self.los.append([np.float(tmp[8]), np.float(tmp[9]), np.float(tmp[10])])
            tmp = B[i].split()
            self.corner.append([np.float(tmp[6]), np.float(tmp[7]), np.float(tmp[8]), np.float(tmp[9])])

        # Make arrays
        self.vel = (np.array(self.vel)+step)*factor
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.err = np.array(self.err)*factor
        self.los = np.array(self.los)
        self.corner = np.array(self.corner)

        # Close file
        fin.close()
        fsp.close()

        # Compute lon lat to utm
        self.x, self.y = self.ll2xy(self.lon,self.lat)

        # Compute corner to xy
        self.xycorner = np.zeros(self.corner.shape)
        x, y = self.ll2xy(self.corner[:,0], self.corner[:,1])
        self.xycorner[:,0] = x
        self.xycorner[:,1] = y
        x, y = self.ll2xy(self.corner[:,2], self.corner[:,3])
        self.xycorner[:,2] = x
        self.xycorner[:,3] = y

        # Read the covariance
        if cov:
            nd = self.vel.size
            self.Cd = np.fromfile(filename+'.cov', dtype=np.float32).reshape((nd, nd))*factor

        # Store the factor
        self.factor = factor

        # All done
        return

    def read_from_binary(self, data, lon, lat, err=None, factor=1.0, step=0.0, incidence=35.8, heading=-13.14, dtype=np.float32, remove_nan=True, downsample=1, remove_zeros=True):
        '''
        Read from binary file or from array.
        '''

        # Get the data
        if type(data) is str:
            vel = np.fromfile(data, dtype=dtype)[::downsample]*factor + step
        else:
            vel = data.flatten()[::downsample]*factor + step

        # Get the lon
        if type(lon) is str:
            lon = np.fromfile(lon, dtype=dtype)[::downsample]
        else:
            lon = lon[::downsample]

        # Get the lat
        if type(lat) is str:
            lat = np.fromfile(lat, dtype=dtype)[::downsample]
        else:
            lat = lat[::downsample]

        # Check sizes
        assert vel.shape==lon.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)
        assert vel.shape==lat.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)

        # Get the error
        if err is not None:
            if type(err) is str:
                err = np.fromfile(err, dtype=dtype)[::downsample]
            err = err * factor
            assert vel.shape==err.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)

        # If zeros
        if remove_zeros:
            vel[vel==0.] = np.nan

        # Check NaNs
        if remove_nan:
            iFinite = np.flatnonzero(np.isfinite(vel))
        else:
            iFinite = list(range(vel.shape[0]))

        # Set things in self
        self.vel = vel[iFinite]
        if err is not None:
            self.err = err[iFinite]
        else:
            self.err = None
        self.lon = lon[iFinite]
        self.lat = lat[iFinite]

        # Keep track of factor
        self.factor = factor

        # Compute the LOS
        if type(incidence) in (float, np.float, np.float32, np.ndarray):
            self.inchd2los(incidence, heading, origin='binaryfloat')
        elif type(incidence) is str:
            self.inchd2los(incidence, heading, origin='binary')
            self.los = self.los[::downsample,:]
            self.los = self.los[iFinite,:]
        else:
            self.los = None

        # compute x, y
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # All done
        return

    def read_from_mat(self, filename, factor=1.0, step=0.0, incidence=35.88, heading=-13.115):
        '''
        Reads velocity map from a mat file.
        Args:
            * filename  : Name of the input file
            * factor    : scale by a factor.
            * step      : add a step.
        '''

        # Initialize values
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []

        # Open the input file
        import scipy.io as scio
        A = scio.loadmat(filename)

        # Get the phase values
        self.vel = (A['velo'].flatten()+ step)*factor
        self.err = A['verr'].flatten()
        self.err[np.where(np.isnan(self.vel))] = np.nan
        self.vel[np.where(np.isnan(self.err))] = np.nan

        # Deal with lon/lat
        Lon = A['posx'].flatten()
        Lat = A['posy'].flatten()
        Lon,Lat = np.meshgrid(Lon,Lat)
        w,l = Lon.shape
        self.lon = Lon.reshape((w*l,)).flatten()
        self.lat = Lat.reshape((w*l,)).flatten()

        # Keep the non-nan pixels
        u = np.flatnonzero(np.isfinite(self.vel))
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.vel = self.vel[u]
        self.err = self.err[u]

        # Convert to utm
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # Deal with the LOS
        self.inchd2los(incidence, heading)

        # Store the factor
        self.factor = factor

        # All done
        return

    def inchd2los(self, incidence, heading, origin='onefloat'):
        '''
        From the incidence and the heading, defines the LOS vector.
        Args:
            * incidence : Incidence angle.
            * heading   : Heading angle.
            * origin    : What are these numbers onefloat: One number
                                                      grd: grd files
                                                   binary: Binary files
                                              binaryfloat: Arrays of float
        '''

        # Save values
        self.incidence = incidence
        self.heading = heading

        # Read the files if needed
        if origin in ('grd', 'GRD'):
            try:
                from netCDF4 import Dataset as netcdf
                fincidence = netcdf(incidence, 'r', format='NETCDF4')
                fheading = netcdf(heading, 'r', format='NETCDF4')
            except:
                import scipy.io.netcdf as netcdf
                fincidence = netcdf.netcdf_file(incidence)
                fheading = netcdf.netcdf_file(heading)
            incidence = np.array(fincidence.variables['z'][:]).flatten()
            heading = np.array(fheading.variables['z'][:]).flatten()
            self.origininchd = origin
        elif origin in ('binary', 'bin'):
            incidence = np.fromfile(incidence, dtype=np.float32)
            heading = np.fromfile(heading, dtype=np.float32)
            self.origininchd = origin
        elif origin in ('binaryfloat'):
            self.origininchd = origin

        self.Incidence = incidence
        self.Heading = heading

        # Convert angles
        alpha = (heading+90.)*np.pi/180.
        phi = incidence *np.pi/180.

        # Compute LOS
        Se = -1.0 * np.sin(alpha) * np.sin(phi)
        Sn = -1.0 * np.cos(alpha) * np.sin(phi)
        Su = np.cos(phi)

        # Store it
        if origin in ('grd', 'GRD', 'binary', 'bin'):
            self.los = np.ones((alpha.shape[0],3))
        else:
            self.los = np.ones((self.lon.shape[0],3))
        self.los[:,0] *= Se
        self.los[:,1] *= Sn
        self.los[:,2] *= Su

        # all done
        return

    def read_from_grd(self, filename, factor=1.0, step=0.0, incidence=None, heading=None,
                      los=None, keepnans=False):
        '''
        Reads velocity map from a grd file.
        Args:
            * filename  : Name of the input file
            * factor    : scale by a factor
            * step      : add a value.
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Initialize values
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []

        # Open the input file
        try:
            from netCDF4 import Dataset as netcdf
            fin = netcdf(filename, 'r', format='NETCDF4')
        except ImportError:
            import scipy.io.netcdf as netcdf
            fin = netcdf.netcdf_file(filename)

        # Get the values
        if len(fin.variables['z'].shape)==1:
            self.vel = (np.array(fin.variables['z'][:]) + step) * factor
        else:
            self.vel = (np.array(fin.variables['z'][:,:]).flatten() + step)*factor
        self.err = np.zeros((self.vel.shape)) * factor
        self.err[np.where(np.isnan(self.vel))] = np.nan
        self.vel[np.where(np.isnan(self.err))] = np.nan

        # Deal with lon/lat
        if 'x' in fin.variables.keys():
            Lon = fin.variables['x'][:]
            Lat = fin.variables['y'][:]
        elif 'lon' in fin.variables.keys():
            Lon = fin.variables['lon'][:]
            Lat = fin.variables['lat'][:]
        else:
            Nlon, Nlat = fin.variables['dimension'][:]
            Lon = np.linspace(fin.variables['x_range'][0], fin.variables['x_range'][1], Nlon)
            Lat = np.linspace(fin.variables['y_range'][1], fin.variables['y_range'][0], Nlat)
        self.lonarr = Lon.copy()
        self.latarr = Lat.copy()
        Lon, Lat = np.meshgrid(Lon,Lat)
        w, l = Lon.shape
        self.lon = Lon.reshape((w*l,)).flatten()
        self.lat = Lat.reshape((w*l,)).flatten()
        self.grd_shape = Lon.shape

        # Keep the non-nan pixels only
        if not keepnans:
            u = np.flatnonzero(np.isfinite(self.vel))
            self.lon = self.lon[u]
            self.lat = self.lat[u]
            self.vel = self.vel[u]
            self.err = self.err[u]

        # Convert to utm
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # Deal with the LOS
        self.los = np.ones((self.lon.shape[0],3))
        if heading is not None and incidence is not None and los is None:
            if type(heading) is str:
                ori = 'grd'
            else:
                ori = 'float'
            self.inchd2los(incidence, heading, origin=ori)
            if not keepnans:
                self.los = self.los[u,:] 
        elif los is not None:
            # If strings, they are meant to be grd files
            if type(los[0]) is str:
                if los[0][-4:] not in ('.grd'):
                    print('LOS input files do not seem to be grds as the displacement file')
                    print('There might be some issues...')
                    print('      Input files: {}, {} and {}'.format(los[0], los[1], los[2]))
                try:
                    from netCDF4 import Dataset
                    finx = Dataset(los[0], 'r', format='NETCDF4')
                    finy = Dataset(los[1], 'r', format='NETCDF4')
                    finz = Dataset(los[2], 'r', format='NETCDF4')
                except ImportError:
                    import sicpy.io.netcdf as netcdf
                    finx = netcdf.netcdf_file(los[0])
                    finy = netcdf.netcdf_file(los[1])
                    finz = netcdf.netcdf_file(los[2])
                losx = np.array(finx.variables['z']).flatten()
                losy = np.array(finy.variables['z']).flatten()
                losz = np.array(finz.variables['z']).flatten()
                # Remove NaNs?
                if not keepnans:
                    losx = losx[u]
                    losy = losy[u]
                    losz = losz[u]
                # Do as if binary
                los = [losx, losy, losz]

            # Store these guys
            self.los[:,0] *= los[0]
            self.los[:,1] *= los[1]
            self.los[:,2] *= los[2]
        else:
            print('Warning: not enough information to compute LOS')
            print('LOS will be set to 1,0,0')
            self.los[:,0] = 1.0
            self.los[:,1] = 0.0
            self.los[:,2] = 0.0

        # Store the factor
        self.factor = factor

        # All done
        return

    def ModelResolutionDownsampling(self, faults, threshold, damping, startingsize=10., minimumsize=0.5, tolerance=0.1, plot=False):
        '''
        Downsampling algorythm based on Lohman & Simons, 2005, G3.
        Args:
            faults          : List of faults, these need to have a buildGFs routine (ex: for RectangularPatches, it will be Okada).
            threshold       : Resolution threshold, if above threshold, keep dividing.
            damping         : Damping parameter. Damping is enforced through the addition of a identity matrix.
            startingsize    : Starting size of the downsampling boxes.
        '''

        # If needed
        from .imagedownsampling import imagedownsampling

        # Check if faults have patches and builGFs routine
        for fault in faults:
            assert (hasattr(fault, 'builGFs')), 'Fault object {} does not have a buildGFs attribute...'.format(fault.name)

        # Create the insar downsampling object
        downsampler = imagedownsampling('Downsampler {}'.format(self.name), self, faults)

        # Initialize the downsampling starting point
        downsampler.initialstate(startingsize, minimumsize, tolerance=tolerance)

        # Iterate until done
        downsampler.ResolutionBasedIterations(threshold, damping, plot=False)

        # Plot
        if plot:
            downsampler.plot()

        # Write outputs
        downsampler.writeDownsampled2File(self.name, rsp=True)

        # All done
        return

    def buildDiagCd(self):
        '''
        Builds a full Covariance matrix from the uncertainties.
        The Matrix is just a diagonal matrix.
        '''

        # Assert
        assert self.err is not None, 'Need some uncertainties on the LOS displacements...'

        # Get some size
        nd = self.vel.shape[0]

        # Fill Cd
        self.Cd = np.diag(self.err)

        # All done
        return

    def buildCd(self, sigma, lam, function='exp'):
        '''
        Builds the full Covariance matrix from values of sigma and lambda.

        If function='exp':

            Cov(i,j) = sigma*sigma * exp(-d[i,j] / lam)

        elif function='gauss':

            Cov(i,j) = sigma*sigma * exp(-(d[i,j]*d[i,j])/(2*lam))

        '''

        # Assert
        assert function in ('exp', 'gauss'), 'Unknown functional form for Covariance matrix'

        # Get some size
        nd = self.vel.shape[0]

        # Cleans the existing covariance
        self.Cd = np.zeros((nd, nd))

        # Loop over Cd
        for i in range(nd):
            for j in range(i,nd):

                # Get the distance
                d = self.distancePixel2Pixel(i,j)

                # Compute Cd
                if function is 'exp':
                    self.Cd[i,j] = sigma*sigma*np.exp(-1.0*d/lam)
                elif function is 'gauss':
                    self.Cd[i,j] = sigma*sigma*np.exp(-1.0*d*d/(2*lam))

                # Make it symmetric
                self.Cd[j,i] = self.Cd[i,j]

        # All done
        return

    def distancePixel2Pixel(self, i, j):
        '''
        Returns the distance in km between two pixels.
        '''

        # Get values
        x1 = self.x[i]
        y1 = self.y[i]
        x2 = self.x[j]
        y2 = self.y[j]

        # Compute the distance
        d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # All done
        return d

    def select_pixels(self, minlon, maxlon, minlat, maxlat):
        '''
        Select the pixels in a box defined by min and max, lat and lon.

        Args:
            * minlon        : Minimum longitude.
            * maxlon        : Maximum longitude.
            * minlat        : Minimum latitude.
            * maxlat        : Maximum latitude.
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.vel = self.vel[u]
        if self.err is not None:
            self.err = self.err[u]
        if self.los is not None:
            self.los = self.los[u]
        if self.synth is not None:
            self.synth = self.synth[u,:]
        if self.corner is not None:
            self.corner = self.corner[u,:]
            self.xycorner = self.xycorner[u,:]

        # Deal with the covariance matrix
        if self.Cd is not None:
            Cdt = self.Cd[u,:]
            self.Cd = Cdt[:,u]

        # All done
        return

    def setGFsInFault(self, fault, G, vertical=True):
        '''
        From a dictionary of Green's functions, sets these correctly into the fault 
        object fault for future computation.
        Args:
            * fault     : Instance of Fault
            * G         : Dictionary with 3 entries 'strikeslip', 'dipslip' and 'tensile'
                          These can be a matrix or None.
            * vertical  : Set here for consistency with other data objects, but will 
                          always be set to True, whatever you do.
        '''

        # Get the values
        try: 
            GssLOS = G['strikeslip']
        except:
            GssLOS = None
        try:
            GdsLOS = G['dipslip']
        except:
            GdsLOS = None
        try: 
            GtsLOS = G['tensile']
        except:
            GtsLOS = None
        try:
            GcpLOS = G['coupling']
        except:
            GcpLOS = None

        # set the GFs
        fault.setGFs(self, strikeslip=[GssLOS], dipslip=[GdsLOS], tensile=[GtsLOS],
                    coupling=[GcpLOS], vertical=True)

        # All done
        return

    def getPolyEstimator(self, ptype, x0=None, y0=None, normX=None, normY=None):
        '''
        Returns the Estimator for the polynomial form to estimate in the InSAR data.
        Args:
            * ptype : integer.
                if ptype==1:
                    constant offset to the data
                if ptype==3:
                    constant and linear function of x and y
                if ptype==4:
                    constant, linear term and cross term.
        '''

        # number of data points
        nd = self.vel.shape[0]

        # Create the Estimator
        orb = np.zeros((nd, ptype))
        orb[:,0] = 1.0

        if ptype >= 3:
            # Compute normalizing factors
            if not hasattr(self, 'OrbNormalizingFactor'):
                self.OrbNormalizingFactor = {}
            if x0 is None:
                x0 = self.x[0]
            if y0 is None:
                y0 = self.y[0]
            if normX is None:
                normX = np.abs(self.x - x0).max()
            if normY is None:
                normY = np.abs(self.y - y0).max()
            # Save them for later
            self.OrbNormalizingFactor['x'] = normX
            self.OrbNormalizingFactor['y'] = normY
            self.OrbNormalizingFactor['ref'] = [x0, y0]
            # Fill in functionals
            orb[:,1] = (self.x - x0) / normX
            orb[:,2] = (self.y - y0) / normY

        if ptype == 4:
            orb[:,3] = orb[:,1] * orb[:,2]

        # Scale everything by the data factor
        orb *= self.factor

        # All done
        return orb

    def computePoly(self, fault):
        '''
        Computes the orbital bias estimated in fault
        Args:
            * fault : Fault object that has a polysol structure.
        '''

        # Get the polynomial type
        ptype = fault.poly[self.name]

        # Get the parameters
        params = fault.polysol[self.name]

        # Get the estimator
        Horb = self.getPolyEstimator(ptype)

        # Compute the polynomial
        self.orbit = np.dot(Horb, params)

        # All done
        return

    def removePoly(self, fault, verbose=False):
        '''
        Removes a polynomial from the parameters that are in a fault.
        '''

        # compute the polynomial
        self.computePoly(fault)

        # Get the vector
        params = fault.polysol[self.name].tolist()

        # Print Something
        if verbose:
            print('Correcting insar rate {} from polynomial function: {}'.format(self.name, tuple(p for p in params)))

        # Correct
        self.vel -= self.orbit

        # All done
        return

    def removeTransformation(self, fault, verbose=False):
        '''
        Wrapper of removePoly to ensure consistency between data sets.
        '''

        self.removePoly(fault, verbose=verbose)

        # All done
        return

    def removeSynth(self, faults, direction='sd', poly=None, vertical=True):
        '''
        Removes the synthetics using the faults and the slip distributions that are in there.
        Args:
            * faults        : List of faults.
            * direction     : Direction of slip to use.
            * poly          : if a polynomial function has been estimated, build and/or include
            * vertical      : always True - used here for consistency among data types
        '''

        # Build synthetics
        self.buildsynth(faults, direction=direction, poly=poly)

        # Correct
        self.vel -= self.synth

        # All done
        return

    def buildsynth(self, faults, direction='sd', poly=None, vertical=True):
        '''
        Computes the synthetic data using the faults and the associated slip distributions.
        Args:
            * faults        : List of faults.
            * direction     : Direction of slip to use.
            * poly          : if a polynomial function has been estimated, build and/or include
            * vertical      : always True - used here for consistency among data types
        '''

        # Check list
        if type(faults) is not list:
            faults = [faults]

        # Number of data
        Nd = self.vel.shape[0]

        # Clean synth
        self.synth = np.zeros((self.vel.shape))

        # Loop on each fault
        for fault in faults:

            # Get the good part of G
            G = fault.G[self.name]

            if ('s' in direction) and ('strikeslip' in G.keys()):
                Gs = G['strikeslip']
                Ss = fault.slip[:,0]
                losss_synth = np.dot(Gs,Ss)
                self.synth += losss_synth
            if ('d' in direction) and ('dipslip' in G.keys()):
                Gd = G['dipslip']
                Sd = fault.slip[:,1]
                losds_synth = np.dot(Gd, Sd)
                self.synth += losds_synth
            if ('t' in direction) and ('tensile' in G.keys()):
                Gt = G['tensile']
                St = fault.slip[:,2]
                losop_synth = np.dot(Gt, St)
                self.synth += losop_synth
            if ('c' in direction) and ('coupling' in G.keys()):
                Gc = G['coupling']
                Sc = fault.coupling
                losdc_synth = np.dot(Gc,Sc)
                self.synth += losdc_synth

            if poly is not None:
                # Compute the polynomial 
                self.computePoly(fault)
                if poly is 'include':
                    self.removePoly(fault)
                else:
                    self.synth += self.orbit

        # All done
        return

    def writeEDKSdata(self):
        '''
        This routine prepares the data file as input for EDKS.
        '''

        # Get the x and y positions
        x = self.x
        y = self.y

        # Get LOS informations
        los = self.los

        # Open the file
        datname = self.name.replace(' ','_')
        filename = 'edks_{}.idEN'.format(datname)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N E_los N_los U_los\n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '{:5d} {} {} {} {} {} \n'.format(i, x[i], y[i], los[i,0], los[i,1], los[i,2])
            fout.write(string)

        # Close the file
        fout.close()

        # All done
        return datname,filename

    def reject_pixel(self, u):
        '''
        Reject pixels.
        Args:
            * u         : Index of the pixel to reject.
        '''

        self.lon = np.delete(self.lon, u)
        self.lat = np.delete(self.lat, u)
        self.x = np.delete(self.x, u)
        self.y = np.delete(self.y, u)
        if self.err is not None:
            self.err = np.delete(self.err, u)
        self.los = np.delete(self.los, u, axis=0)
        self.vel = np.delete(self.vel, u)

        if self.Cd is not None:
            self.Cd = np.delete(self.Cd, u, axis=0)
            self.Cd = np.delete(self.Cd, u, axis=1)

        if self.corner is not None:
            self.corner = np.delete(self.corner, u, axis=0)
            self.xycorner = np.delete(self.xycorner, u, axis=0)

        if self.synth is not None:
            self.synth = np.delete(self.synth, u, axis=0)

        # All done
        return

    def reject_pixels_fault(self, dis, faults):
        '''
        Rejects the pixels that are dis km close to the fault.
        Args:
            * dis       : Threshold distance.
            * faults    : list of fault objects.
        '''

        # Variables to trim are  self.corner,
        # self.xycorner, self.Cd, (self.synth)

        # Check something
        if faults.__class__ is not list:
            faults = [faults]

        # Build a line object with the faults
        fl = []
        for flt in faults:
            f = [[x, y] for x,y in np.vstack((flt.xf, flt.yf)).T.tolist()]
            fl = fl + f

        # Get all the positions
        pp = [[x, y] for x,y in zip(self.x, self.y)]

        # Get distances
        D = scidis.cdist(pp, fl)

        # Get minimums
        d = np.min(D, axis=1)
        del D

        # Find the close ones
        if dis>0.:
            u = np.where(d<=dis)[0]
        else:
            u = np.where(d>=(-1.0*dis))[0]

        # Reject
        self.reject_pixel(u)

        # All done
        return u

    def getprofile(self, name, loncenter, latcenter, length, azimuth, width):
        '''
        Project the SAR velocities onto a profile.
        Works on the lat/lon coordinates system.
        Args:
            * name              : Name of the profile.
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees.
            * width             : Width of the profile.
        '''

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        Dalong, vel, err, Dacros, boxll, xe1, ye1, xe2, ye2, synth, los, lon, lat = self.coord2prof(
                xc, yc, length, azimuth, width)

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['Lon'] = lon
        dic['Lat'] = lat
        dic['LOS Velocity'] = vel
        dic['LOS Synthetics'] = synth
        dic['LOS Error'] = err
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.putm(xe1*1000., ye1*1000., inverse=True)
        lone2, late2 = self.putm(xe2*1000., ye2*1000., inverse=True)
        dic['EndPointsLL'] = [[lone1, late1],
                              [lone2, late2]]
        dic['LOS vector'] = los

        # All done
        return

    def getprofileAlongCurve(self, name, lon, lat, width, widthDir):
        '''
        Project the SAR velocities onto a profile.
        Works on the lat/lon coordinates system.
        Args:
            * name              : Name of the profile.
            * lon               : Longitude of the Line around which we do the profile
            * lat               : Latitude of the Line around which we do the profile
            * width             : Width of the zone around the line.
            * widthDir          : Direction to of the width.
        '''

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # lonlat2xy
        xl = []
        yl = []
        for i in range(len(lon)):
            x, y = self.ll2xy(lon[i], lat[i])
            xl.append(x)
            yl.append(y)

        # Get the profile
        Dalong, vel, err, Dacros, boxll, xc, yc, xe1, ye1, xe2, ye2, length = self.curve2prof(xl, yl, width, widthDir)

        # get lon lat center
        loncenter, latcenter = self.xy2ll(xc, yc)

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['LOS Velocity'] = vel
        dic['LOS Error'] = err
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.putm(xe1*1000., ye1*1000., inverse=True)
        lone2, late2 = self.putm(xe2*1000., ye2*1000., inverse=True)
        dic['EndPointsLL'] = [[lone1, late1],
                              [lone2, late2]]

        # All done
        return
    def referenceProfile(self, name, xmin, xmax):
        '''
        Removes the mean value of points between xmin and xmax.
        '''

        # Get the profile
        profile = self.profiles[name]

        # Get the indexes
        ii = self._getindexXlimProfile(name, xmin, xmax)

        # Get average value
        average = profile['LOS Velocity'][ii].mean()

        # Set the reference
        profile['LOS Velocity'][:] -= average

        # all done
        return

    def cleanProfile(self, name, xlim=None, zlim=None):
        '''
        Cleans a specified profile.
        '''

        # Get profile
        profile = self.profiles[name]

        # Distance cleanup
        if xlim is not None:
            ii = self._getindexXlimProfile(name, xlim[0], xlim[1])
            profile['Distance'] = profile['Distance'][ii]
            profile['LOS Velocity'] = profile['LOS Velocity'][ii]
            profile['Normal Distance'] = profile['Normal Distance'][ii]
            if profile['LOS Error'] is not None:
                profile['LOS Error'] = profile['LOS Error'][ii]

        # Amplitude cleanup
        if zlim is not None:
            ii = self._getindexZlimProfile(name, zlim[0], zlim[1])
            profile['Distance'] = profile['Distance'][ii]
            profile['LOS Velocity'] = profile['LOS Velocity'][ii]
            profile['Normal Distance'] = profile['Normal Distance'][ii]
            if profile['LOS Error'] is not None:
                profile['LOS Error'] = profile['LOS Error'][ii]

        return

    def smoothProfile(self, name, window, method='mean'):
        '''
        Computes smoothed  profile.
        '''

        # Get profile
        dis = self.profiles[name]['Distance']
        vel = self.profiles[name]['LOS Velocity']
        los = self.profiles[name]['LOS vector']

        # Create the bins
        bins = np.arange(dis.min(), dis.max(), window)
        indexes = np.digitize(dis, bins)
        
        # Create Lists
        outvel = []
        outerr = []
        outdis = []
        if los is not None:
            outlos = []

        # Run a runing average on it
        for i in range(len(bins)-1):

            # Find the guys inside this bin
            uu = np.flatnonzero(indexes==i)

            # If there is points in this bin
            if len(uu)>0:
                
                # Get the mean
                if method in ('mean'):
                    m = vel[uu].mean()
                elif method in ('median'):
                    m = np.median(vel[uu])

                # Get the mean distance
                d = dis[uu].mean()

                # Get the error
                e = vel[uu].std()

                # Set it
                outvel.append(m)
                outerr.append(e)
                outdis.append(d)

                # Get the LOS
                if los is not None:
                    l = los[uu,:].mean(axis=0)
                    outlos.append(l)

        # Copy the old profile and modify it
        newName = 'Smoothed {}'.format(name)
        self.profiles[newName] = copy.deepcopy(self.profiles[name])
        self.profiles[newName]['LOS Velocity'] = np.array(outvel)
        self.profiles[newName]['LOS Error'] = np.array(outerr)
        self.profiles[newName]['Distance'] = np.array(outdis)
        if los is not None:
            self.profiles[newName]['LOS vector'] = np.array(outlos)

        # All done
        return

    def _getindexXlimProfile(self, name, xmin, xmax):
        '''
        Returns the index of the points that are in between xmin & xmax.
        '''

        # Get the distance array
        distance = self.profiles[name]['Distance']

        # Get the indexes
        ii = np.flatnonzero(distance>=xmin)
        jj = np.flatnonzero(distance<=xmax)
        uu = np.intersect1d(ii,jj)

        # All done
        return uu

    def _getindexZlimProfile(self, name, zmin, zmax):
        '''
        Returns the index of the points that are in between zmin & zmax.
        '''

        # Get the velocity
        velocity = self.profiles[name]['LOS Velocity']

        # Get the indexes
        ii = np.flatnonzero(velocity>=zmin)
        jj = np.flatnonzero(velocity<=zmax)
        uu = np.intersect1d(ii,jj)

        # All done
        return uu

    def coord2prof(self, xc, yc, length, azimuth, width, plot=False):
        '''
        Routine returning the profile
        Args:
            * xc                : X pos of center
            * yc                : Y pos of center
            * length            : length of the profile.
            * azimuth           : azimuth of the profile.
            * width             : width of the profile.
            * plot              : if true, makes a small plot
        Returns:
            dis                 : Distance from the center
            vel                 : values
            err                 : errors
            norm                : distance perpendicular to profile
            boxll               : lon lat coordinates of the profile box used
            xe1, ye1            : coordinates (UTM) of the profile endpoint
            xe2, ye2            : coordinates (UTM) of the profile endpoint
            los                 : List of los vectors for each point of the profile
        '''

        # Azimuth into radians
        alpha = azimuth*np.pi/180.

        # Copmute the across points of the profile
        xa1 = xc - (width/2.)*np.cos(alpha)
        ya1 = yc + (width/2.)*np.sin(alpha)
        xa2 = xc + (width/2.)*np.cos(alpha)
        ya2 = yc - (width/2.)*np.sin(alpha)

        # Compute the endpoints of the profile
        xe1 = xc + (length/2.)*np.sin(alpha)
        ye1 = yc + (length/2.)*np.cos(alpha)
        xe2 = xc - (length/2.)*np.sin(alpha)
        ye2 = yc - (length/2.)*np.cos(alpha)

        # Convert the endpoints
        elon1, elat1 = self.xy2ll(xe1, ye1)
        elon2, elat2 = self.xy2ll(xe2, ye2)

        # Design a box in the UTM coordinate system.
        x1 = xe1 - (width/2.)*np.cos(alpha)
        y1 = ye1 + (width/2.)*np.sin(alpha)
        x2 = xe1 + (width/2.)*np.cos(alpha)
        y2 = ye1 - (width/2.)*np.sin(alpha)
        x3 = xe2 + (width/2.)*np.cos(alpha)
        y3 = ye2 - (width/2.)*np.sin(alpha)
        x4 = xe2 - (width/2.)*np.cos(alpha)
        y4 = ye2 + (width/2.)*np.sin(alpha)

        # Convert the box into lon/lat for further things
        lon1, lat1 = self.xy2ll(x1, y1)
        lon2, lat2 = self.xy2ll(x2, y2)
        lon3, lat3 = self.xy2ll(x3, y3)
        lon4, lat4 = self.xy2ll(x4, y4)

        # make the box
        box = []
        box.append([x1, y1])
        box.append([x2, y2])
        box.append([x3, y3])
        box.append([x4, y4])

        # make latlon box
        boxll = []
        boxll.append([lon1, lat1])
        boxll.append([lon2, lat2])
        boxll.append([lon3, lat3])
        boxll.append([lon4, lat4])

        # Get the InSAR points in this box.
        # 1. import shapely and nxutils
        import matplotlib.path as path
        import shapely.geometry as geom

        # 2. Create an array with the InSAR positions
        SARXY = np.vstack((self.x, self.y)).T

        # 3. Create a box
        rect = path.Path(box, closed=False)

        # 4. Find those who are inside
        Bol = rect.contains_points(SARXY)

        # 4. Get these values
        xg = self.x[Bol]
        yg = self.y[Bol]
        lon = self.lon[Bol]
        lat = self.lat[Bol]
        vel = self.vel[Bol]
        if self.synth is not None:
            synth = self.synth[Bol]
        else:
            synth = None
        if self.err is not None:
            err = self.err[Bol]
        else:
            err = None
        if self.los is not None:
            los = self.los[Bol]
        else:
            los = None

        # Check if lengths are ok
        if len(xg) > 5:

            # 5. Get the sign of the scalar product between the line and the point
            vec = np.array([xe1-xc, ye1-yc])
            sarxy = np.vstack((xg-xc, yg-yc)).T
            sign = np.sign(np.dot(sarxy, vec))

            # 6. Compute the distance (along, across profile) and get the velocity
            # Create the list that will hold these values
            Dacros = []; Dalong = []; V = []; E = []
            # Build lines of the profile
            Lalong = geom.LineString([[xe1, ye1], [xe2, ye2]])
            Lacros = geom.LineString([[xa1, ya1], [xa2, ya2]])
            # Build a multipoint
            PP = geom.MultiPoint(np.vstack((xg,yg)).T.tolist())
            # Loop on the points
            for p in range(len(PP.geoms)):
                Dalong.append(Lacros.distance(PP.geoms[p])*sign[p])
                Dacros.append(Lalong.distance(PP.geoms[p]))

        else:
            Dalong = vel
            Dacros = vel

        Dalong = np.array(Dalong)
        Dacros = np.array(Dacros)

        # Toss out nans
        jj = np.flatnonzero(np.isfinite(vel)).tolist()
        vel = vel[jj]
        lon = lon[jj]
        lat = lat[jj]
        Dalong = Dalong[jj]
        Dacros = Dacros[jj]
        if los is not None:
            los = los[jj]
        if err is not None:
            err = err[jj]
        if synth is not None:
            synth = synth[jj]

        if plot:
            plt.figure(1234)
            plt.clf()
            plt.subplot(121)
            import matplotlib.cm as cmx
            import matplotlib.colors as colors
            cmap = plt.get_cmap('jet')
            cNorm = colors.Normalize(vmin=self.vel.min(), vmax=self.vel.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
            scalarMap.set_array(self.vel)
            plt.scatter(self.x, self.y, s=10, c=self.vel, cmap=cmap, linewidths=0.0)
            xb = [box[i][0] for i in range(4)]
            yb = [box[i][1] for i in range(4)]
            plt.plot(xb,yb,'.k')
            xb.append(xb[0])
            yb.append(yb[0])
            plt.plot(xb, yb, '-k')
            plt.colorbar(orientation='horizontal', shrink=0.6)
            plt.subplot(122)
            plt.plot(Dalong, vel, '.b')
            plt.show()

        # All done
        return Dalong, vel, err, Dacros, boxll, xe1, ye1, xe2, ye2, synth, los, lon, lat

    def curve2prof(self, xl, yl, width, widthDir):
        '''
        Routine returning the profile along a curve.
        Args:
            * xl                : List of the x coordinates of the line.
            * yl                : List of the y coordinates of the line.
            * width             : Width of the zone around the line.
            * widthDir          : Direction to of the width.
        '''

        # If not list
        if type(xl) is not list:
            xl = xl.tolist()
            yl = yl.tolist()

        # Get the widthDir into radians
        alpha = widthDir*np.pi/180.

        # Get the endpoints
        xe1 = xl[0]
        ye1 = yl[0]
        xe2 = xl[-1]
        ye2 = yl[-1]

        # Convert the endpoints
        elon1, elat1 = self.xy2ll(xe1, ye1)
        elon2, elat2 = self.xy2ll(xe2, ye2)

        # Translate the line into the withDir direction on both sides
        transx = np.sin(alpha)*width/2.
        transy = np.cos(alpha)*width/2.

        # Make a box with that
        box = []
        pts = zip(xl, yl)
        for x, y in pts:
            box.append([x+transx, y+transy])
        box.append([xe2, ye2])
        pts.reverse()
        for x, y in pts:
            box.append([x-transx, y-transy])
        box.append([xe1, ye1])

        # Convert the box into lon lat to save it for further purpose
        boxll = []
        for b in box:
            boxll.append(self.xy2ll(b[0], b[1]))

        # vector perpendicular to the curve
        vec = np.array([xe1-box[-2][0], ye1-box[-2][1]])

        # Get the InSAR points inside this box
        SARXY = np.vstack((self.x, self.y)).T
        rect = path.Path(box, closed=False)
        Bol = rect.contains_points(SARXY)
        xg = self.x[Bol]
        yg = self.y[Bol]
        vel = self.vel[Bol]
        if self.err is not None:
            err = self.err[Bol]
        else:
            err = None

        # Compute the cumulative distance along the line
        dis = np.zeros((len(xl),))
        for i in range(1, len(xl)):
            d = np.sqrt((xl[i] - xl[i-1])**2 + (yl[i] - yl[i-1])**2)
            dis[i] = dis[i-1] + d

        # Sign of the position across
        sarxy = np.vstack((np.array(xg-xe1), np.array(yg-ye1))).T
        sign = np.sign(np.dot(sarxy, vec))

        # Get their position along and across the line
        Dalong = []
        Dacross = []
        for x, y, s in zip(xg.tolist(), yg.tolist(), sign.tolist()):
            d = scidis.cdist([[x, y]], [[xli, yli] for xli, yli in zip(xl, yl)])[0]
            imin1 = d.argmin()
            dmin1 = d[imin1]
            d[imin1] = 99999999.
            imin2 = d.argmin()
            dmin2 = d[imin2]
            # Put it along the fault
            dtot = dmin1+dmin2
            xcd = (xl[imin1]*dmin1 + xl[imin2]*dmin2)/dtot
            ycd = (yl[imin1]*dmin1 + yl[imin2]*dmin2)/dtot
            # Distance
            if dmin1<dmin2:
                jm = imin1
            else:
                jm = imin2
            # Append
            Dalong.append(dis[jm] + np.sqrt( (xcd-xl[jm])**2 + (ycd-yl[jm])**2) )
            Dacross.append(s*np.sqrt( (xcd-x)**2 + (ycd-y)**2 )) 

        # Remove NaNs
        jj = np.flatnonzero(np.isfinite(vel)).tolist()
        vel = vel[jj]
        Dalong = np.array(Dalong)[jj]
        Dacross = np.array(Dacross)[jj]
        if err is not None:
            err = err[jj]

        # Length
        length = dis[-1]

        # Center
        uu = np.argmin(np.abs(dis-length/2.))
        xc = xl[uu]
        yc = yl[uu]

        # All done
        return Dalong, vel, err, Dacross, boxll, xc, yc, xe1, ye1, xe2, ye2, length

    def getAlongStrikeOffset(self, name, fault, interpolation=None, width=1.0,
            length=10.0, faultwidth=1.0, tolerance=0.2, azimuthpad=2.0):

        '''
        Runs along a fault to determine variations of the phase offset in the
        along strike direction.
        Args:
            * name              : name of the results stored in AlongStrikeOffsets
            * fault             : a fault object.
            * interpolation     : interpolation distance
            * width             : width of the profiles used
            * length            : length of the profiles used
            * faultwidth        : width of the fault zone.
        '''

        # the Along strike measurements are in a dictionary
        if not hasattr(self, 'AlongStrikeOffsets'):
            self.AlongStrikeOffsets = {}

        # Interpolate the fault object if asked
        if interpolation is not None:
            fault.discretize(every=interpolation, tol=tolerance)
            xf = fault.xi
            yf = fault.yi
        else:
            xf = fault.xf
            yf = fault.yf

        # Initialize some lists
        ASprof = []
        ASx = []
        ASy = []
        ASazi = []

        # Loop
        for i in range(len(xf)):

            # Write something
            sys.stdout.write('\r Fault point {}/{}'.format(i,len(xf)))
            sys.stdout.flush()

            # Get coordinates
            xp = xf[i]
            yp = yf[i]

            # get the local profile and fault azimuth
            Az, pAz = self._getazimuth(xf, yf, i, pad=azimuthpad)

            # If there is something
            if np.isfinite(Az):

                # Get the profile
                dis, vel, err, norm = self.coord2prof(xp, yp, length, pAz,
                        width, plot=False)[0:4]

                # Keep only the non NaN values
                pts = np.flatnonzero(np.isfinite(vel))
                dis = np.array(dis)[pts]
                ptspos = np.flatnonzero(dis>0.0)
                ptsneg = np.flatnonzero(dis<0.0)

                # If there is enough points, on both sides, get the offset value
                if (len(pts)>20 and len(ptspos)>10 and len(ptsneg)>10):

                    # Select the points
                    vel = vel[pts]
                    err = err[pts]
                    norm = np.array(norm)[pts]

                    # Symmetrize the profile
                    mindis = np.min(dis)
                    maxdis = np.max(dis)
                    if np.abs(mindis)>np.abs(maxdis):
                       pts = np.flatnonzero(dis>-1.0*maxdis)
                    else:
                        pts = np.flatnonzero(dis<=-1.0*mindis)

                    # Get the points
                    dis = dis[pts]
                    ptsneg = np.flatnonzero(dis>0.0)
                    ptspos = np.flatnonzero(dis<0.0)

                    # If we still have enough points on both sides
                    if (len(pts)>20 and len(ptspos)>10 and len(ptsneg)>10 and np.abs(mindis)>(10*faultwidth/2)):

                        # Get the values
                        vel = vel[pts]
                        err = err[pts]
                        norm = norm[pts]

                        # Get offset
                        off = self._getoffset(dis, vel, faultwidth, plot=False)

                        # Store things in the lists
                        ASprof.append(off)
                        ASx.append(xp)
                        ASy.append(yp)
                        ASazi.append(Az)

                    else:

                        # Store some NaNs
                        ASprof.append(np.nan)
                        ASx.append(xp)
                        ASy.append(yp)
                        ASazi.append(Az)

                else:

                    # Store some NaNs
                    ASprof.append(np.nan)
                    ASx.append(xp)
                    ASy.append(yp)
                    ASazi.append(Az)
            else:

                # Store some NaNs
                ASprof.append(np.nan)
                ASx.append(xp)
                ASy.append(yp)
                ASazi.append(Az)

        ASprof = np.array(ASprof)
        ASx = np.array(ASx)
        ASy = np.array(ASy)
        ASazi = np.array(ASazi)

        # Store things
        self.AlongStrikeOffsets[name] = {}
        dic = self.AlongStrikeOffsets[name]
        dic['xpos'] = ASx
        dic['ypos'] = ASy
        lon, lat = self.xy2ll(ASx, ASy)
        dic['lon'] = lon
        dic['lat'] = lat
        dic['offset'] = ASprof
        dic['azimuth'] = ASazi

        # Compute along strike cumulative distance
        if interpolation is not None:
            disc = True
        dic['distance'] = fault.cumdistance(discretized=disc)

        # Clean screen
        sys.stdout.write('\n')
        sys.stdout.flush()

        # all done
        return

    def writeAlongStrikeOffsets2File(self, name, filename):
        '''
        Write the variations of the offset along strike in a file.
        '''

        # Open a file
        fout = open(filename, 'w')

        # Write the header
        fout.write('# Distance (km) || Offset || Azimuth (rad) || Lon || Lat \n')

        # Get the values from the dictionary
        x = self.AlongStrikeOffsets[name]['distance']
        y = self.AlongStrikeOffsets[name]['offset']
        azi = self.AlongStrikeOffsets[name]['azimuth']
        lon = self.AlongStrikeOffsets[name]['lon']
        lat = self.AlongStrikeOffsets[name]['lat']

        # Write to file
        for i in range(len(x)):
            fout.write('{} {} {} {} {} \n'.format(x[i], y[i], azi[i], lon[i], lat[i]))

        # Close file
        fout.close()

    def writeProfile2File(self, name, filename, fault=None):
        '''
        Writes the profile named 'name' to the ascii file filename.
        '''

        # open a file
        fout = open(filename, 'w')

        # Get the dictionary
        dic = self.profiles[name]

        # Write the header
        fout.write('#---------------------------------------------------\n')
        fout.write('# Profile Generated with CSI\n')
        fout.write('# Center: {} {} \n'.format(dic['Center'][0], dic['Center'][1]))
        fout.write('# Endpoints: \n')
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][0][0], dic['EndPointsLL'][0][1]))
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][1][0], dic['EndPointsLL'][1][1]))
        fout.write('# Box Points: \n')
        fout.write('#           {} {} \n'.format(dic['Box'][0][0],dic['Box'][0][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][1][0],dic['Box'][1][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][2][0],dic['Box'][2][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][3][0],dic['Box'][3][1]))

        # Place faults in the header
        if fault is not None:
            if fault.__class__ is not list:
                fault = [fault]
            fout.write('# Fault Positions: \n')
            for f in fault:
                d = self.intersectProfileFault(name, f)
                fout.write('# {}           {} \n'.format(f.name, d))

        fout.write('#---------------------------------------------------\n')

        # Write the values
        for i in range(len(dic['Distance'])):
            d = dic['Distance'][i]
            Vp = dic['LOS Velocity'][i]
            if dic['LOS Error'] is not None:
                Ep = dic['LOS Error'][i]
            else:
                Ep = None
            Lon = dic['Lon'][i]
            Lat = dic['Lat'][i]
            if np.isfinite(Vp):
                fout.write('{} {} {} {} {} \n'.format(d, Vp, Ep, Lon, Lat))

        # Close the file
        fout.close()

        # all done
        return

    def plotprofile(self, name, legendscale=10., fault=None, norm=None, ref='utm', synth=False):
        '''
        Plot profile.
        Args:
            * name      : Name of the profile.
            * legendscale: Length of the legend arrow.
            * fault     : Fault object
            * norm      : Colorscale limits
            * ref       : utm or lonlat
            * synth     : Plot synthetics (True/False).
        '''

        # Plot the insar
        self.plot(faults=fault, norm=norm, show=False)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((len(b)+1, 2))
        for i in range(len(b)):
            x = b[i,0]
            if x<0.:
                x += 360.
            bb[i,0] = x
            bb[i,1] = b[i,1]
        bb[-1,0] = bb[0,0]
        bb[-1,1] = bb[0,1]
        self.fig.carte.plot(bb[:,0], bb[:,1], '-k', zorder=40)

        # open a figure
        fig = plt.figure()
        prof = fig.add_subplot(111)

        # plot the profile
        x = self.profiles[name]['Distance']
        y = self.profiles[name]['LOS Velocity']
        ey = self.profiles[name]['LOS Error']
        p = prof.errorbar(x, y, yerr=ey, label='los velocity', marker='.', linestyle='')
        if synth:
            sy = self.profiles[name]['LOS Synthetics']
            s = prof.plot(x, sy, '-r', label='synthetics')

        # If a fault is here, plot it
        if fault is not None:
            # If there is only one fault
            if fault.__class__ is not list:
                fault = [fault]
            # Loop on the faults
            for f in fault:
                # Get the distance
                d = self.intersectProfileFault(name, f)
                if d is not None:
                    ymin, ymax = prof.get_ylim()
                    prof.plot([d, d], [ymin, ymax], '--', label=f.name)

        # plot the legend
        prof.legend()

        # Show to screen
        self.fig.show(showFig=['map'])

        # All done
        return

    def intersectProfileFault(self, name, fault):
        '''
        Gets the distance between the fault/profile intersection and the profile center.
        Args:
            * name      : name of the profile.
            * fault     : fault object from verticalfault.
        '''

        # Import shapely
        import shapely.geometry as geom

        # Grab the fault trace
        xf = fault.xf
        yf = fault.yf

        # Grab the profile
        prof = self.profiles[name]

        # import shapely
        import shapely.geometry as geom

        # Build a linestring with the profile center
        Lp = geom.LineString(prof['EndPoints'])

        # Build a linestring with the fault
        ff = []
        for i in range(len(xf)):
            ff.append([xf[i], yf[i]])
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            if type(Pi) is geom.point.Point:
                p = Pi.coords[0]
            else:
                return None
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        xc, yc = self.ll2xy(lonc, latc)

        # Get the sign
        xa,ya = prof['EndPoints'][0]
        vec1 = [xa-xc, ya-yc]
        vec2 = [p[0]-xc, p[1]-yc]
        sign = np.sign(np.dot(vec1, vec2))

        # Compute the distance to the center
        d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

        # All done
        return d

    def getRMS(self):
        '''
        Computes the RMS of the data and if synthetics are computed, the RMS of the residuals
        '''

        # Get the number of points
        N = self.vel.shape[0]

        # RMS of the data
        dataRMS = np.sqrt( 1./N * sum(self.vel**2) )

        # Synthetics
        if self.synth is not None:
            synthRMS = np.sqrt( 1./N *sum( (self.vel - self.synth)**2 ) )
            return dataRMS, synthRMS
        else:
            return dataRMS, 0.

        # All done

    def getVariance(self):
        '''
        Computes the Variance of the data and if synthetics are computed, the RMS of the residuals
        '''

        # Get the number of points
        N = self.vel.shape[0]

        # Varianceof the data
        dmean = self.vel.mean()
        dataVariance = ( 1./N * sum((self.vel-dmean)**2) )

        # Synthetics
        if self.synth is not None:
            rmean = (self.vel - self.synth).mean()
            synthVariance = ( 1./N *sum( (self.vel - self.synth - rmean)**2 ) )
            return dataVariance, synthVariance
        else:
            return dataVariance, 0.

        # All done

    def getMisfit(self):
        '''
        Computes the Summed Misfit of the data and if synthetics are computed, the RMS of the residuals
        '''

        # Misfit of the data
        dataMisfit = sum((self.vel))

        # Synthetics
        if self.synth is not None:
            synthMisfit =  sum( (self.vel - self.synth) )
            return dataMisfit, synthMisfit
        else:
            return dataMisfit, 0.

        # All done

    def plot(self, faults=None, figure=None, gps=None, decim=False, norm=None, data='data', show=True, drawCoastlines=True, expand=0.2):
        '''
        Plot the data set, together with a fault, if asked.

        Args:
            * ref       : utm or lonlat.
            * faults    : list of fault objects.
            * figure    : number of the figure.
            * gps       : list of gps objects.
            * decim     : plot the insar following the decimation process of varres.
        '''

        # Get lons lats
        lonmin = self.lon.min()-expand
        if lonmin<0.:
            lonmin += 360.
        lonmax = self.lon.max()+expand
        if lonmax<0.:
            lonmax += 360.
        latmin = self.lat.min()-expand
        latmax = self.lat.max()+expand

        # Create a figure
        fig = geoplot(figure=figure, lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)

        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(drawLand=True, parallels=5, meridians=5, drawOnFault=True)

        # Plot the gps data if asked
        if gps is not None:
            if type(gps) is not list:
                gps = [gps]
            for g in gps:
                fig.gpsvelocities(g)

        # Plot the decimation process, if asked
        if decim:
            fig.insar(self, norm=norm, colorbar=True, data=data, plotType='decimate')

        # Plot the insar
        if not decim:
            fig.insar(self, norm=norm, colorbar=True, data=data, plotType='scatter')

        # Plot the fault trace if asked
        if faults is not None:
            if type(faults) is not list:
                faults = [faults]
            for fault in faults:
                fig.faulttrace(fault)

        # Show
        if show:
            fig.show(showFig=['map'])
        else:
            del fig.faille

        # All done
        return

    def write2grd(self, fname, oversample=1, data='data', interp=100, cmd='surface', tension=None):
        '''
        Uses surface to write the output to a grd file.
        Args:
            * fname     : Filename
            * oversample: Oversampling factor.
            * data      : can be 'data' or 'synth'.
            * interp    : Number of points along lon and lat.
            * cmd       : command used for the conversion( i.e., surface or xyz2grd)
        '''

        # Get variables
        x = self.lon
        y = self.lat
        if data == 'data':
            z = self.vel
        elif data == 'synth':
            z = self.synth
        elif data == 'poly':
            z = self.orb
        elif data == 'res':
            z = self.vel - self.synth

        # Write these to a dummy file
        fout = open('xyz.xyz', 'w')
        for i in range(x.shape[0]):
            fout.write('{} {} {} \n'.format(x[i], y[i], z[i]))
        fout.close()

        # Import subprocess
        import subprocess as subp

        # Get Rmin/Rmax/Rmin/Rmax
        lonmin = x.min()
        lonmax = x.max()
        latmin = y.min()
        latmax = y.max()
        R = '-R{}/{}/{}/{}'.format(lonmin, lonmax, latmin, latmax)

        # Create the -I string
        if type(interp)!=list:
            Nlon = int(interp)*int(oversample)
            Nlat = Nlon
        else:
            Nlon = int(interp[0])
            Nlat = int(interp[1])
        I = '-I{}+/{}+'.format(Nlon,Nlat)

        # Create the G string
        G = '-G'+fname

        # Create the command
        com = [cmd, R, I, G]

        # Add tension
        if tension is not None and cmd in ('surface'):
            T = '-T{}'.format(tension)

        # open stdin and stdout
        fin = open('xyz.xyz', 'r')

        # Execute command
        subp.call(com, stdin=fin, shell=True)

        # CLose the files
        fin.close()

        # All done
        return


    def write2ascii(self, fname, data='data'):
        '''
        Uses surface to write the output to a grd file.
        Args:
            * fname     : Filename
            * data      : can be 'data' or 'synth'.
        '''

        # Get variables
        x = self.lon
        y = self.lat
        if data is 'data':
            z = self.vel
        elif data is 'synth':
            z = self.synth
        elif data is 'poly':
            z = self.orb

        # Write these to a dummy file
        fout = open(fname, 'w')
        for i in range(x.shape[0]):
            fout.write('{} {} {} \n'.format(x[i], y[i], z[i]))
        fout.close()

        return


    def _getazimuth(self, x, y, i, pad=2):
        '''
        Get the azimuth of a line.
        Args:
            * x,y       : x,y values of the line.
            * i         : index of the position of interest.
            * pad       : number of points to take into account.
        '''
        # Compute distances along trace
        dis = np.sqrt((x-x[i])**2 + (y-y[i])**2)
        # Get points that are close than pad/2.
        pts = np.where(dis<=pad/2)
        # Get the azimuth if there is more than 2 points
        if len(pts[0])>=2:
                d = y[pts]
                G = np.vstack((np.ones(d.shape),x[pts])).T
                m,res,rank,s = np.linalg.lstsq(G,d)
                Az = np.arctan(m[1])
                pAz= Az+np.pi/2
        else:
                Az = np.nan
                pAz = np.nan
        # All done
        return Az*180./np.pi,pAz*180./np.pi

    def _getoffset(self, x, y, w, plot=True):
        '''
        Computes the offset around zero along a profile.
        Args:
            * x         : X-axis of the profile
            * y         : Y-axis of the profile
            * w         : Width of the zero zone.
        '''

        # Initialize plot
        if plot:
            plt.figure(1213)
            plt.clf()
            plt.plot(x,y,'.k')

        # Define function
        G = np.vstack((np.ones(y.shape),x)).T

        # fit a function on the negative side
        pts = np.where(x<=-1.*w/2.)
        dneg = y[pts]
        Gneg = np.squeeze(G[pts,:])
        mneg,res,rank,s = np.linalg.lstsq(Gneg,dneg)
        if plot:
            plt.plot(x,np.dot(G,mneg),'-r')

        # fit a function on the positive side
        pts = np.where(x>=w/2)
        dpos = y[pts]
        Gpos = np.squeeze(G[pts,:])
        mpos,res,rank,s = np.linalg.lstsq(Gpos,dpos)
        if plot:
            plt.plot(x,np.dot(G,mpos),'-g')

        # Offset
        off = mpos[0] - mneg[0]

        # plot
        if plot:
            print('Measured offset: {}'.format(off))
            plt.show()

        # all done
        return off

    def checkLOS(self, figure=1, factor=100., decim=1):
        '''
        Plots the LOS vectors in a 3D plot.
        Args:
            * figure:   Figure number.
            * factor:   Increases the size of the vectors.
        '''

        # Display
        print('Checks the LOS orientation')       

        # Create a figure
        fig = plt.figure(figure)

        # Create an axis instance
        ax = fig.add_subplot(111, projection='3d')

        # Loop over the LOS
        for i in range(0,self.vel.shape[0],decim):
            x = [self.x[i], self.x[i]+self.los[i,0]*factor]
            y = [self.y[i], self.y[i]+self.los[i,1]*factor]
            z = [0, self.los[i,2]*factor]
            ax.plot3D(x, y, z, '-k')

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_zlabel('Up')

        # Show it
        plt.show()

        # All done
        return

#EOF
