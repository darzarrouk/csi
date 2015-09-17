'''
A class that deals with COSI-Corr data, after decimation using VarRes.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import scipy.spatial.distance as scidis

# Personals
from .SourceInv import SourceInv
from .geodeticplot import geodeticplot as geoplot
from . import csiutils as utils

class opticorr(SourceInv):

    def __init__(self, name, utmzone='10', ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(opticorr,self).__init__(name,utmzone,ellps) 

        # Initialize the data set 
        self.dtype = 'opticorr'

        self.verbose = verbose
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Cosicorr data set {}".format(self.name))

        # Initialize some things
        self.east = None
        self.north = None
        self.east_synth = None
        self.north_synth = None
        self.up_synth = None
        self.err_east = None
        self.err_north = None
        self.lon = None
        self.lat = None
        self.corner = None
        self.xycorner = None
        self.Cd = None

        # All done
        return

    def lonlat2xy(self, lon, lat):
        '''
        Uses the transformation in self to convert  lon/lat vector to x/y utm.
        Args:
            * lon           : Longitude array.
            * lat           : Latitude array.
        '''

        x, y = self.putm(lon,lat)
        x /= 1000.
        y /= 1000.

        return x, y

    def xy2lonlat(self, x, y):
        '''
        Uses the transformation in self to convert x.y vectors to lon/lat.
        Args:
            * x             : Xarray
            * y             : Yarray
        '''

        lon, lat = self.putm(x*1000., y*1000., inverse=True)
        return lon, lat

    def read_from_varres(self,filename, factor=1.0, step=0.0, header=2, cov=False):
        '''
        Read the COSI-Corr east-north offsets from the VarRes output.
        Args:
            * filename      : Name of the input file. Two files are opened filename.txt and filename.rsp.
            * factor        : Factor to multiply the east-north offsets.
            * step          : Add a value to the velocity.
            * header        : Size of the header.
            * cov           : Read an additional covariance file (binary float32, Nd*Nd elements).
        '''
        if self.verbose:
            print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename+'.txt', 'r')
        fsp = open(filename+'.rsp', 'r')

        # Read it all
        A = fin.readlines()
        B = fsp.readlines()

        # Initialize the business
        self.lon = []
        self.lat = []
        self.east = []
        self.north = []
        self.err_east = []
        self.err_north = []
        self.corner = []

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            tmp = A[i].split()
            self.lon.append(np.float(tmp[1]))
            self.lat.append(np.float(tmp[2]))
            self.east.append(np.float(tmp[3]))
            self.north.append(np.float(tmp[4]))
            self.err_east.append(np.float(tmp[5]))
            self.err_north.append(np.float(tmp[6]))
            tmp = B[i].split()
            self.corner.append([np.float(tmp[6]), np.float(tmp[7]), 
                                np.float(tmp[8]), np.float(tmp[9])])


        # Make arrays
        self.east = factor * (np.array(self.east) + step)
        self.north = factor * (np.array(self.north) + step)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.err_east = np.array(self.err_east) * factor
        self.err_north = np.array(self.err_north) * factor
        self.corner = np.array(self.corner)

        # Close file
        fin.close()
        fsp.close()

        # Compute lon lat to utm
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # Compute corner to xy
        self.xycorner = np.zeros(self.corner.shape)
        x, y = self.putm(self.corner[:,0], self.corner[:,1])
        self.xycorner[:,0] = x/1000.
        self.xycorner[:,1] = y/1000.
        x, y = self.putm(self.corner[:,2], self.corner[:,3])
        self.xycorner[:,2] = x/1000.
        self.xycorner[:,3] = y/1000.

        # Read the covariance
        if cov:
            nd = self.east.size + self.north.size
            self.Cd = np.fromfile(filename + '.cov', dtype=np.float32).reshape((nd,nd))
            self.Cd *= factor

        # Store the factor
        self.factor = factor
    
        # Save number of observations per station
        self.obs_per_station = 2

        # All done
        return

    def read_from_xyz(self, filename, factor=1.0, step=0.0, header=0):
        '''
        Reads the maps from a xyz file
        lon lat east north east_err north_err
        Args:
            * filename  : name of the input file.
            * factor    : scale by a factor.
            * step      : add a value.
        '''

        # Initialize values
        self.east = []
        self.north = []
        self.lon = []
        self.lat = []
        self.err_east = []
        self.err_north = []

        # Open the file and read
        fin = open(filename, 'r')
        A = fin.readlines()
        fin.close()

        # remove the header lines
        A = A[header:]

        # Loop 
        for line in A:
            l = line.split()
            self.lon.append(np.float(l[0]))
            self.lat.append(np.float(l[1]))
            self.east.append(np.float(l[2]))
            self.north.append(np.float(l[3]))
            self.err_east.append(np.float(l[4]))
            self.err_north.append(np.float(l[5]))
        
        # Make arrays
        self.east = factor * (np.array(self.east) + step)
        self.north = factor * (np.array(self.north) + step)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.err_east = np.array(self.err_east) * factor
        self.err_north = np.array(self.err_north) * factor

        # Compute lon lat to utm
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # Store the factor
        self.factor = factor
   
        # Save number of observations per station
        self.obs_per_station = 2

        # All done
        return       

    def read_from_binary(self, east, north, lon, lat, err_east=None, err_north=None, factor=1.0, step=0.0, dtype=np.float32, remove_nan=True):
        '''
        Read from a set of binary files or from a set of arrays.
        Args:
            * east      : array or filename of the east displacement 
            * north     : array or filename of the north displacement
            * lon       : array or filename of the longitude
            * lat       : array or filename of the latitude
            * err_east  : uncertainties on the east displacement (file or array)
            * err_north : uncertainties on the north displacememt (file or array)
            * factor    : multiplication factor
            * step      : offset
            * dtype     : type of binary file
        '''

        # Get east
        if type(east) is str:
            east = np.fromfile(east, dtype=dtype)
        east = np.array(east).flatten()

        # Get north
        if type(north) is str:
            north = np.fromfile(north, dtype=dtype)
        north = np.array(north).flatten()

        # Get lon
        if type(lon) is str:
            lon = np.fromfile(lon, dtype=dtype)
        lon = np.array(lon).flatten()

        # Get Lat 
        if type(lat) is str:
            lat = np.fromfile(lat, dtype=dtype)
        lat = np.array(lat).flatten()

        # Errors
        if err_east is not None:
            if type(err_east) is str:
                err_east = np.fromfile(err_east, dtype=dtype)
            err_east = np.array(err_east).flatten()
        else:
            err_east = np.zeros(east.shape)
        if err_north is not None:
            if type(err_north) is str:
                err_north = np.fromfile(err_north, dtype=dtype)
            err_north = np.array(err_north).flatten()
        else:
            err_north = np.zeros(north.shape)

        # Check NaNs
        if remove_nan:
            eFinite = np.flatnonzero(np.isfinite(east))
            nFinite = np.flatnonzero(np.isfinite(north))
            iFinite = np.intersect1d(eFinite, nFinite).tolist()
        else:
            iFinite = range(east.shape[0])

        # Set things in there
        self.east = factor * (east[iFinite] + step)
        self.north = factor * (north[iFinite] + step)
        self.lon = lon[iFinite]
        self.lat = lat[iFinite]
        self.err_east = err_east[iFinite] * factor
        self.err_north = err_north[iFinite] * factor

        # Compute lon lat to utm
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # Store the factor
        self.factor = factor
   
        # Save number of observations per station
        self.obs_per_station = 2

        # All done
        return

    def read_from_envi(self, filename, component='EW', remove_nan=True):
        '''
        Reads displacement map from an ENVI file.
        Args:
            * filename  : Name of the input file
            * component : 'EW' or 'NS'
        '''
        
        assert component=='EW' or component=='NS', 'component must be EW or NS'

        # Read header
        hdr = open(filename+'.hdr','r').readlines()
        for l in hdr:
            items = l.strip().split('=')
            if items[0].strip()=='data type':
                assert float(items[1])==4,'data type is not float32'
            if items[0].strip()=='samples':
                self.samples = int(items[1])
            if items[0].strip()=='lines':
                self.lines   = int(items[1])
            if items[0].strip()=='map info':
                map_items = l.strip().split('{')[1].strip('}').split(',')
                assert map_items[0].strip()=='UTM', 'Map is not UTM {}'.format(map_items[0])
                x0 = float(map_items[3])
                y0 = float(map_items[4])
                dx = float(map_items[5])
                dy = float(map_items[6])
                assert int(map_items[7])==self.utmzone, 'UTM   zone does not match'
                assert map_items[9].strip().replace('-','')==self.ellps, 'ellps zone does not match'
        
        # Coordinates
        x = x0 + np.arange(self.samples) * dx
        y = y0 - np.arange(self.lines)   * dy
        xg,yg = np.meshgrid(x,y)
        self.x = xg.flatten()/1000.
        self.y = yg.flatten()/1000.
        self.lon, self.lat = self.xy2ll(self.x,self.y)
        
        # Data        
        if component=='EW':
            self.east = np.fromfile(filename,dtype='float32')
            print('read length',len(self.east))
            if remove_nan:
                u = np.flatnonzero(np.isfinite(self.east))
                self.east = self.east[u]
            self.err_east=np.zeros(self.east.shape)  # set to zero error for now
            print('after mask',len(self.east))
        elif component=='NS':
            self.north = np.fromfile(filename,dtype='float32')
            if remove_nan:
                u = np.flatnonzero(np.isfinite(self.north))
                self.north = self.north[u]
            self.err_north=np.zeros(self.north.shape)  # set to zero error for now
        if remove_nan:
            self.lon = self.lon[u]; self.lat = self.lat[u]
            self.x   = self.x[u]  ; self.y   = self.y[u]

        # Check size of arrays
        if self.north!=None and self.east!=None:
            assert len(self.north) == len(self.east), 'inconsistent data size'
            assert len(self.lon)==len(self.lat),   'inconsistent lat/lon size'
            assert len(self.lon)==len(self.north), 'inconsistent lon/data size'            
        
        self.factor = 1.0
        self.obs_per_station = 2

        # All done
        return

    def splitFromShapefile(self, shapefile, remove_nan=True):
        '''
        Uses the paths defined in a Shapefile to select and return particular domains of self.
        Args:   
            * shapefile : Input file (shapefile format).
        '''

        # Import necessary library
        import shapefile as shp
        import matplotlib.path as path

        # Build a list of points
        AllXY = np.vstack((self.x, self.y)).T

        # Read the shapefile
        shape = shp.Reader(shapefile)

        # Create the list of new objects
        OutCosi = []

        # Iterate over the shape records
        for record, iR in zip(shape.shapeRecords(), range(len(shape.shapeRecords()))):
            
            # Get x, y
            x = np.array([record.shape.points[i][0] for i in range(len(record.shape.points))])/1000.
            y = np.array([record.shape.points[i][1] for i in range(len(record.shape.points))])/1000.
            xyroi = np.vstack((x, y)).T

            # Build a path with that
            roi = path.Path(xyroi, closed=False)

            # Get the ones inside
            check = roi.contains_points(AllXY)

            # Get the values
            east = self.east[check]
            north = self.north[check]
            lon = self.lon[check]
            lat = self.lat[check]
            if self.err_east is not None:
                err_east = self.err_east[check]
            else:
                err_east = None
            if self.err_north is not None:
                err_north = self.err_north[check]
            else:
                err_north = None

            # Create a new cosicorr object
            cosi = opticorr('{} #{}'.format(self.name, iR), utmzone=self.utmzone)

            # Put the values in there
            cosi.read_from_binary(east, north, lon, lat, err_east=err_east, err_north=err_north, remove_nan=True)

            # Store this object
            OutCosi.append(cosi)

        # All done
        return OutCosi
        
    def read_from_grd(self, filename, factor=1.0, step=0.0, cov=False, flip=False, keepnans=False):
        '''
        Reads velocity map from a grd file.
        Args:
            * filename  : Name of the input file 
                    As we are reading two files, the files are:
                    filename_east.grd and filename_north.grd
            * factor    : scale by a factor
            * step      : add a value.
        '''

        if self.verbose:
            print ("Read from file {} into data set {}".format(filename, self.name))

        # Initialize values
        self.east = []
        self.north = []
        self.lon = []
        self.lat = []
        self.err_east = []
        self.err_north = []

        # Open the input file
        try:
            import scipy.io.netcdf as netcdf
            feast = netcdf.netcdf_file(filename+'_east.grd')
            fnorth = netcdf.netcdf_file(filename+'_north.grd')
        except:
            from netCDF4 import Dataset as netcdf
            feast = netcdf(filename+'_east.grd', format='NETCDF4')
            fnorth = netcdf(filename+'_north.grd', format='NETCDF4')
        
        # Shape
        self.grd_shape = feast.variables['z'][:].shape

        # Get the values
        self.east = (feast.variables['z'][:].flatten() + step)*factor
        self.north = (fnorth.variables['z'][:].flatten() + step)*factor
        self.err_east = np.ones((self.east.shape)) * factor
        self.err_north = np.ones((self.north.shape)) * factor
        self.err_east[np.where(np.isnan(self.east))] = np.nan
        self.err_north[np.where(np.isnan(self.north))] = np.nan

        # Deal with lon/lat
        if 'x' in feast.variables.keys():
            Lon = feast.variables['x'][:]
            Lat = feast.variables['y'][:]
        elif 'x_range' in feast.variables.keys():
            LonS, LonE = feast.variables['x_range'][:]
            LatS, LatE = feast.variables['y_range'][:]
            nLon, nLat = feast.variables['dimension'][:]
            Lon = np.linspace(LonS, LonE, num=nLon)
            Lat = np.linspace(LatS, LatE, num=nLat)
        self.lonarr = Lon.copy()
        self.latarr = Lat.copy()
        Lon, Lat = np.meshgrid(Lon,Lat)

        # Flip if necessary
        if flip:
            Lat = np.flipud(Lat)
        w, l = Lon.shape
        self.lon = Lon.reshape((w*l,)).flatten()
        self.lat = Lat.reshape((w*l,)).flatten()

        # Keep the non-nan pixels only
        if not keepnans:
            u = np.flatnonzero(np.isfinite(self.east))
            self.lon = self.lon[u]
            self.lat = self.lat[u]
            self.east = self.east[u]
            self.north = self.north[u]
            self.err_east = self.err_east[u]
            self.err_north = self.err_north[u]

        # Convert to utm
        self.x, self.y = self.lonlat2xy(self.lon, self.lat) 

        # Store the factor and step
        self.factor = factor
        self.step = step
    
        # Save number of observations per station
        self.obs_per_station = 2

        # All done
        return
    
    def read_with_reader(self, readerFunc, filePrefix, factor=1.0, cov=False):
        '''
        Read data from a *.txt file using a user provided reading function. Assume the user
        knows what they are doing and are returning the correct values.
        '''

        lon,lat,east,north,east_err,north_err = readerFunc(filePrefix + '.txt')
        self.lon = lon
        self.lat = lat
        self.east = factor * east
        self.north = factor * north
        self.err_east = factor * east_err
        self.err_north = factor * north_err

        # Convert to utm
        self.x, self.y = self.lonlat2xy(self.lon, self.lat) 

        # Read the covariance 
        if cov:
            nd = self.east.size + self.north.size
            self.Cd = np.fromfile(filePrefix + '.cov', dtype=np.float32).reshape((nd,nd))

        # Store the factor
        self.factor = factor

        # Save number of observations per station
        self.obs_per_station = 2

        # All done
        return

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
        npts = self.lon.size
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.east = self.east[u]
        self.north = self.north[u]
        self.err_east = self.err_east[u]
        self.err_north = self.err_north[u]
        if self.east_synth is not None:
            self.east_synth = self.east_synth[u]
            self.north_synth = self.north_synth[u]
        if self.corner is not None:
            self.corner = self.corner[u,:]
            self.xycorner = self.xycorner[u,:]

        # Deal with the covariance matrix
        if self.Cd is not None:
            indCd = np.hstack((u, u+npts))
            Cdt = self.Cd[indCd,:]
            self.Cd = Cdt[:,indCd]
        
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
            * vertical  : Do we use vertical predictions? Default is True
        '''
        
        # Get values
        try:
            Gss = G['strikeslip']
        except:
            Gss = None
        try:
            Gds = G['dipslip']
        except:
            Gds = None
        try:
            Gts = G['tensile']
        except:
            Gts = None
        try: 
            Gcp = G['coupling']
        except:
            Gcp = None

        # Set these values
        fault.setGFs(self, strikeslip=[Gss], dipslip=[Gds], tensile=[Gts], coupling=[Gcp], vertical=vertical)

        # All done
        return

    def getPolyEstimator(self, ptype):
        '''
        Returns the Estimator for the polynomial form to estimate in the optical correlation data.
        Args:
            * ptype : integer.
                if ptype==1:
                    constant offset to the data
                if ptype==3:
                    constant and linear function of x and y
                if ptype==4:
                    constant, linear term and cross term.
        Watch out: If vertical is True, you should only estimate polynomials for the horizontals.
        '''

        # Get the basic size of the polynomial
        basePoly = ptype / self.obs_per_station
        assert basePoly >= 3 or basePoly == 1, """
            only support 1st, 3rd or 4th order poly for cosicorr
            """

        # Get number of points
        nd = self.east.shape[0]

	if ptype >=3: 
		# Compute normalizing factors
		x0 = self.x[0]
		y0 = self.y[0]
		normX = np.abs(self.x - x0).max()
		normY = np.abs(self.y - y0).max()

		# Save them for later
		self.OrbNormalizingFactor = {}
		self.OrbNormalizingFactor['ref'] = [x0, y0]
		self.OrbNormalizingFactor['x'] = normX
		self.OrbNormalizingFactor['y'] = normY

		# Pre-compute position-dependent functional forms
		f1 = self.factor * np.ones((nd,))
		f2 = self.factor * (self.x - x0) / normX
		f3 = self.factor * (self.y - y0) / normY
		f4 = self.factor * (self.x - x0) * (self.y - y0) / (normX*normY)
		f5 = self.factor * (self.x - x0)**2 / normX**2
		f6 = self.factor * (self.y - y0)**2 / normY**2
        	polyFuncs = [f1, f2, f3, f4, f5, f6]

 	       # Fill in orb matrix given an order
		orb = np.zeros((nd, basePoly))
		for ind in range(basePoly):
		    orb[:,ind] = polyFuncs[ind]

		# Block diagonal for both components
		orb = block_diag(orb, orb)

		# Check to see if we're including verticals
		if self.obs_per_station == 3:
		    orb = np.vstack((orb, np.zeros((nd, 2*basePoly))))

	elif basePoly==1:
		orb = np.zeros((2*nd,1*basePoly))
		orb[:,0] = self.factor	


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
        tmporbit = np.dot(Horb, params)

        # Store them
        nd = self.east.shape[0]
        self.east_orbit = tmporbit[:nd]
        self.north_orbit = tmporbit[nd:2*nd]

        # All done
        return

    def removePoly(self, fault):
        '''
        Removes a polynomial from the parameters that are in a fault.
        '''

        # Compute the polynomial
        self.computePoly(fault)

        # Correct data
        self.east -= self.east_orbit
        self.north -= self.north_orbit

        # All done
        return

    def removeRamp(self, order=3, maskPoly=None):
        '''
        Pre-remove a ramp from the data that fall outside of mask. If no mask is provided,
        we use all the points to fit a mask.
        '''

        assert order == 1 or order == 3, 'unsupported order for ramp removal'
        
        # Define normalized coordinates
        x0, y0 = self.x[0], self.y[0]
        xd = self.x - x0
        yd = self.y - y0
        normX = np.abs(xd).max()
        normY = np.abs(yd).max()
        
        # Find points that fall outside of mask
        east = self.east.copy()
        north = self.north.copy()
        if maskPoly is not None:
            path = Path(maskPoly)
            mask = path.contains_points(zip(self.x, self.y))
            badIndices = mask.nonzero()[0]
            xd = xd[~badIndices]
            yd = yd[~badIndices]
            east = east[~badIndices]
            north = north[~badIndices]
            
        # Construct ramp design matrix
        nPoints = east.shape[0]
        nDat = 2 * nPoints
        nPar = 2 * order
        Gramp = np.zeros((nDat,nPar))
        if order == 1:
            Gramp[:nPoints,0] = 1.0
            Gramp[nPoints:,1] = 1.0
        else:
            Gramp[:nPoints,0] = 1.0
            Gramp[:nPoints,1] = xd / normX
            Gramp[:nPoints,2] = yd / normY
            Gramp[nPoints:,3] = 1.0
            Gramp[nPoints:,4] = xd / normX
            Gramp[nPoints:,5] = yd / normY

        # Estimate ramp parameters
        d = np.hstack((east,north))
        m_ramp = np.linalg.lstsq(Gramp, d)[0]
        


    def removeSynth(self, faults, direction='sd', poly=None, vertical=False, custom=False):
        '''
        Removes the synthetics using the faults and the slip distributions that are in there.
        Args:
            * faults        : List of faults.
            * direction     : Direction of slip to use.
            * include_poly  : if a polynomial function has been estimated, include it.
            * custom        : if True, uses the fault.custom and fault.G[data.name]['custom'] to correct
        '''

        # Build synthetics
        self.buildsynth(faults, direction=direction, poly=poly, custom=False)

        # Correct
        self.east -= self.east_synth
        self.north -= self.north_synth
	
        # All done
        return

    def buildsynth(self, faults, direction='sd', poly=None, vertical=False, custom=False):
        '''
        Computes the synthetic data using the faults and the associated slip distributions.
        Args:
            * faults        : List of faults.
            * direction     : Direction of slip to use.
            * include_poly  : if a polynomial function has been estimated, include it.
            * custom        : if True, uses the fault.custom and fault.G[data.name]['custom'] to correct
        '''

        # Number of data points
        Nd = self.lon.shape[0]

        # Clean synth
        self.east_synth = np.zeros((Nd,))
        self.north_synth = np.zeros((Nd,))
        self.up_synth = np.zeros((Nd,))

        # Loop on each fault
        for fault in faults:

            # Get the good part of G
            G = fault.G[self.name]

            if ('s' in direction) and ('strikeslip' in G.keys()):
                Gs = G['strikeslip']
                Ss = fault.slip[:,0]
                ss_synth = np.dot(Gs, Ss)
                self.east_synth += ss_synth[:Nd]
                self.north_synth += ss_synth[Nd:2*Nd]
                if vertical:
                    self.up_synth += ss_synth[2*Nd:]
            if ('d' in direction) and ('dipslip' in G.keys()):
                Gd = G['dipslip']
                Sd = fault.slip[:,1]
                sd_synth = np.dot(Gd, Sd)
                self.east_synth += sd_synth[:Nd]
                self.north_synth += sd_synth[Nd:2*Nd]
                if vertical:
                    self.up_synth += sd_synth[2*Nd:]
            if ('t' in direction) and ('tensile' in G.keys()):
                Gt = G['tensile']
                St = fault.slip[:,2]
                st_synth = np.dot(Gt, St)
                self.east_synth += st_synth[:Nd]
                self.north_synth += st_synth[Nd:2*Nd]
                if vertical:
                    self.up_synth += st_synth[2*Nd:]
            if ('c' in direction) and ('coupling' in G.keys()):
                Gc = G['coupling']
                Sc = fault.coupling
                dc_synth = np.dot(Gc, Sc)
                self.east_synth += dc_synth[:Nd]
                self.north_synth += dc_synth[Nd:2*Nd]
                if vertical:
                    self.up_synth += dc_synth[2*Nd:]

            if custom:
                Gc = G['custom']
                Sc = fault.custom
                dc_synth = np.dot(Gc, Sc)
                self.east_synth += dc_synth[:Nd]
                self.north_synth += dc_synth[Nd:2*Nd]
                if vertical:
                    self.up_synth += dc_synth[2*Nd:]

            if poly is not None:
                self.computePoly(fault)
                if poly == 'include':
                    self.east_synth += self.east_orbit
                    self.north_synth += self.east_north

        # All done
        return

    def writeEDKSdata(self):
        '''
        This routine prepares the data file as input for EDKS.
        '''

        # Get the x and y positions
        x = self.x
        y = self.y

        # Open the file
        datname = self.name.replace(' ','_')
        filename = 'edks_{}.idEN'.format(datname)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N \n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '{:5d} {} {} \n'.format(i, x[i], y[i])
            fout.write(string)

        # Close the file
        fout.close()

        # All done
        return datname, filename

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
        self.east = np.delete(self.east, u)
        self.north = np.delete(self.north, u)
        self.err_east = np.delete(self.err_east, u)
        self.err_north = np.delete(self.err_north, u)

        if self.Cd is not None:
            nd = self.east.shape[0]
            Cd1 = np.delete(self.Cd[:nd, :nd], u, axis=0)
            Cd1 = np.delete(Cd1, u, axis=1)
            Cd2 = np.delete(self.Cd[nd:,nd:], u, axis=0)
            Cd2 = np.delete(Cd2, u, axis=1)
            Cd = np.vstack( (np.hstack((Cd1, np.zeros((nd,nd)))), 
                np.hstack((np.zeros((nd,nd)),Cd2))) )
            self.Cd = Cd

        if self.corner is not None:
            self.corner = np.delete(self.corner, u, axis=0)
            self.xycorner = np.delete(self.xycorner, u, axis=0)

        if self.east_synth is not None:
            self.east_synth = np.delete(self.east_synth, u, axis=0)

        if self.north_synth is not None:
            self.north_synth = np.delete(self.north_synth, u, axis=0)

        # All done
        return

    def reject_pixels_fault(self, dis, faults):
        ''' 
        Rejects the pixels that are dis km close to the fault.
        Args:
            * dis       : Threshold distance.
                          If the distance is negative, rejects the pixels that are
                          more than -1.0*distance away from the fault.
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
            
        # Delete 
        self.reject_pixel(u)

        # All done
        return u

    def getprofile(self, name, loncenter, latcenter, length, azimuth, width):
        '''
        Project the GPS velocities onto a profile. 
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

        # Lon lat to x y
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, lon, lat = \
                utils.coord2prof(self, xc, yc, length, azimuth, width)

        # Get these values
        east = self.east[Bol]
        north = self.north[Bol]

        # Get some numbers
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

        # 8. Compute the fault Normal/Parallel displacements
        Vec1 = np.array([x2-xe1, y2-y1])
        Vec1 = Vec1/np.sqrt( Vec1[0]**2 + Vec1[1]**2 )
        FPar = np.dot([[east[i], north[i]] for i in range(east.shape[0])], Vec1)
        Vec2 = np.array([x4-x1, y4-y1])
        Vec2 = Vec2/np.sqrt( Vec2[0]**2 + Vec2[1]**2 )
        FNor = np.dot([[east[i], north[i]] for i in range(east.shape[0])], Vec2)

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['East'] = east
        dic['North'] = north
        dic['Fault Normal'] = FNor
        dic['Fault Parallel'] = FPar
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.putm(xe1*1000., ye1*1000., inverse=True)
        lone2, late2 = self.putm(xe2*1000., ye2*1000., inverse=True)
        dic['EndPointsLL'] = [[lone1, late1], 
                            [lone2, late2]]

        # All done
        return

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
        fout.write('# Profile Generated with StaticInv\n')
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
            Ep = dic['East'][i]
            Np = dic['North'][i]
            Fn = dic['Fault Normal'][i]
            Fp = dic['Fault Parallel'][i]
            if np.isfinite(Ep):
                fout.write('{} {} {} {} {} \n'.format(d, Ep, Np, Fn, Fp))

        # Close the file
        fout.close()

        # all done
        return

    def plotprofile(self, name, legendscale=5., fault=None):
        '''
        Plot profile.
        Args:
            * name      : Name of the profile.
            * legendscale: Length of the legend arrow.
        '''

        # Check the profile
        xd = self.profiles[name]['Distance']
        yn = self.profiles[name]['Fault Normal']
        yp = self.profiles[name]['Fault Parallel']
        assert len(xd)>5, 'There is less than 5 points in your profile...'

        # Plot each data set and the box on top
        self.fig = []
        self.plot(data='dataEast', faults=fault, show=False)
        self.fig[0].close(fig2close='fault')
        self.fig[0].titlemap('East disp.')
        self.plot(data='dataNorth', faults=fault, show=False)
        self.fig[1].titlemap('North disp.')

        # Plot the box
        for fig in self.fig:
            b = self.profiles[name]['Box']
            bb = np.zeros((len(b)+1,2))
            for i in range(len(b)):
                x = b[i,0]
                if x<0.:
                    x += 360.
                bb[i,0] = x
                bb[i,1] = b[i,1]
            bb[-1,0] = b[0,0]
            bb[-1,1] = b[0,1]
            fig.carte.plot(bb[:,0], bb[:,1], '-k', zorder=40)

        # open a figure
        fig = plt.figure()
        prof = fig.add_subplot(111)

        # Plot profiles
        pe = prof.plot(xd, yn, label='Fault Normal displacement', 
                marker='.', color='r', linestyle='')
        pn = prof.plot(xd, yp, label='Fault Par. displacement', 
                marker='.', color='b', linestyle='')

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
        legend = prof.legend()
        legend.draggable = True

        # Show
        self.fig[1].show(showFig=['map'])

        # All done
        return

    def intersectProfileFault(self, name, fault):
        '''
        Gets the distance between the fault/profile intersection and the profile center.
        Args:
            * name      : name of the profile.
            * fault     : fault object from verticalfault.
        '''

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
        ff = [[xf[i], yf[i]] for i in range(xf.shape[0])]
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            p = Pi.coords[0]
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        xc, yc = self.lonlat2xy(lonc, latc)

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

        raise NotImplementedError('do it later')
        return        

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
        N = self.east.shape[0]
        
        # Varianceof the data 
        emean = self.east.mean()
        nmean = self.north.mean()
        eVariance = ( 1./N * sum((self.east-emean)**2) )
        nVariance = ( 1./N * sum((self.north-nmean)**2) )
        dataVariance = eVariance + nVariance

        # Synthetics
        if self.east_synth is not None:
            emean = (self.east - self.east_synth).mean()
            synthEastVariance = ( 1./N *sum( (self.east - self.east_synth - emean)**2 ) )
            synthNorthVariance = ( 1./N *sum( (self.north - self.north_synth - nmean)**2 ) )
            synthVariance = synthEastVariance + synthNorthVariance
            return dataVariance, synthVariance
        else:
            return dataVariance, 0.

        # All done  

    def getMisfit(self):
        '''                                                                                                      
        Computes the Summed Misfit of the data and if synthetics are computed, the RMS of the residuals                    
        '''

        raise NotImplementedError('do it later')
        return

        # Misfit of the data                                                                                        
        dataMisfit = sum((self.vel))

        # Synthetics
        if self.synth is not None:
            synthMisfit =  sum( (self.vel - self.synth) )
            return dataMisfit, synthMisfit
        else:
            return dataMisfit, 0.

        # All done  

    def plot(self, faults=None, figure=None, gps=None, decim=False, axis='equal', norm=None, data='data', show=True, drawCoastlines=True, expand=0.2):
        '''
        Plot the data set, together with a fault, if asked.

        Args:
            * ref       : utm or lonlat.
            * faults    : list of fault object.
            * figure    : number of the figure.
            * gps       : superpose a GPS dataset.
            * decim     : plot the insar following the decimation process of varres.
            * data      : plot either 'dataEast', 'dataNorth', 'synthNorth', 'synthEast',
                          'resEast', 'resNorth', 'data', 'synth' or 'res'
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

        # Plot the fault trace if asked
        if faults is not None:
            if type(faults) is not list:
                faults = [faults]
            for fault in faults:
                fig.faulttrace(fault)

        # Plot the gps data if asked
        if gps is not None:
            if type(gps) is not list:
                gps = [gps]
            for g in gps:
                fig.gps(g)

        # Plot the decimation process, if asked
        if decim:
            fig.cosicorr(self, norm=norm, colorbar=False, data=data, plotType='decimate')

        # Plot the data
        fig.cosicorr(self, norm=norm, colorbar=True, data=data, plotType='scatter')

        # Show
        if show:
            fig.show(showFig=['map'])
    
        # Keep Figure
        if not hasattr(self, 'fig'):
            self.fig = []
        self.fig.append(fig)

        # All done
        return

    def write2binary(self, prefix, dtype=np.float):
        '''
        Writes the records in a binary file. The files will be called
        prefix_north.dat    : North displacement
        prefix_east.dat     : East displacement
        prefix_lon.dat      : Longitude
        prefix_lat.dat      : Latitude
        '''
        
        if self.verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Write in binary format to files {}_east.dat and {}_north.dat'.format(prefix, prefix))

        # North 
        fname = '{}_north.dat'.format(prefix)
        data = self.north.astype(dtype)
        data.tofile(fname)

        # East 
        fname = '{}_east.dat'.format(prefix)
        data = self.east.astype(dtype)
        data.tofile(fname)

        # Longitude
        fname = '{}_lon.dat'.format(prefix)
        data = self.lon.astype(dtype)
        data.tofile(fname)

        # Latitude
        fname = '{}_lat.dat'.format(prefix)
        data = self.lat.astype(dtype)
        data.tofile(fname)

        # All done 
        return

    def write2grd(self, fname, oversample=1, data='data', interp=100, cmd='surface', useGMT=False):
        '''
        Uses surface to write the output to a grd file.
        Args:
            * fname     : Filename
            * oversample: Oversampling factor.
            * data      : can be 'data' or 'synth'.
            * interp    : Number of points along lon and lat (can be a list).
            * cmd       : command used for the conversion( i.e., surface or xyz2gmt)
        '''

        if self.verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Write in grd format to files {}_east.grd and {}_north.grd'.format(fname, fname))


        # Get variables
        x = self.lon
        y = self.lat
        if data is 'data':
            e = self.east
            n = self.north
        elif data is 'synth':
            e = self.synth_east
            n = self.synth_north

        if not useGMT:

            # Write
            utils.write2netCDF('{}_east.grd'.format(fname), x, y, e, nSamples=interp, title='East Displacements')
            utils.write2netCDF('{}_north.grd'.format(fname), x, y, n, nSamples=interp, title='North Displacements')

        else:

            # Write these to a dummy file
            foute = open('east.xyz', 'w')
            foutn = open('north.xyz', 'w')
            for i in range(x.shape[0]):
                foute.write('{} {} {} \n'.format(x[i], y[i], e[i]))
                foutn.write('{} {} {} \n'.format(x[i], y[i], n[i]))
            foute.close()
            foutn.close()

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
                Nlon = int(interp[0])*int(oversample)
                Nlat = int(interp[1])*int(oversample)
            I = '-I{}+/{}+'.format(Nlon,Nlat)        

            # Create the G string
            Ge = '-G'+fname+'_east.grd'
            Gn = '-G'+fname+'_north.grd'

            # Create the command
            come = [cmd, R, I, Ge]
            comn = [cmd, R, I, Gn]

            # open stdin and stdout
            fine = open('east.xyz', 'r')
            finn = open('north.xyz', 'r')

            # Execute command
            subp.call(come, stdin=fine)
            subp.call(comn, stdin=finn)

            # CLose the files
            fine.close()
            finn.close()

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

#EOF