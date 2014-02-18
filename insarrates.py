'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import sys

# Personals
from .SourceInv import SourceInv

class insarrates(SourceInv):

    def __init__(self, name, utmzone='10', ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(self.__class__,self).__init__(name,utmzone,ellps) 

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

    def inchd2los(self, incidence, heading):
        '''
        From the incidence and the heading, defines the LOS vector.
        Args:
            * incidence : Incidence angle.
            * heading   : Heading angle.
        '''

        # Save values
        self.incidence = incidence
        self.heading = heading

        # Convert angles
        alpha = (heading+90.)*np.pi/180.
        phi = incidence *np.pi/180.

        # Compute LOS
        Se = -1.0 * np.sin(alpha) * np.sin(phi)
        Sn = -1.0 * np.cos(alpha) * np.sin(phi)
        Su = np.cos(phi)

        # Store it
        self.los = np.ones((self.lon.shape[0],3))
        self.los[:,0] *= Se
        self.los[:,1] *= Sn
        self.los[:,2] *= Su

        # all done
        return

    def read_from_grd(self, filename, factor=1.0, step=0.0, incidence=None, heading=None,
                      los=None):
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
        import scipy.io.netcdf as netcdf
        fin = netcdf.netcdf_file(filename)

        # Get the values
        self.vel = (fin.variables['z'][:,:].flatten() + step)*factor
        self.err = np.ones((self.vel.shape)) * factor
        self.err[np.where(np.isnan(self.vel))] = np.nan
        self.vel[np.where(np.isnan(self.err))] = np.nan

        # Deal with lon/lat
        Lon = fin.variables['x'][:]
        Lat = fin.variables['y'][:]
        self.lonarr = Lon.copy()
        self.latarr = Lat.copy()
        Lon, Lat = np.meshgrid(Lon,Lat)
        w, l = Lon.shape
        self.lon = Lon.reshape((w*l,)).flatten()
        self.lat = Lat.reshape((w*l,)).flatten()
        self.grd_shape = Lon.shape

        # Keep the non-nan pixels only
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
            self.inchd2los(incidence, heading)
        elif los is not None:
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
        from .insardownsampling import insardownsampling
        
        # Check if faults have patches and builGFs routine
        for fault in faults:
            assert (hasattr(fault, 'builGFs')), 'Fault object {} does not have a buildGFs attribute...'.format(fault.name)

        # Create the insar downsampling object
        downsampler = insardownsampling('Downsampler {}'.format(self.name), self, faults)

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
        self.err = self.err[u]
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

    def removePoly(self, fault):
        '''
        Removes a polynomial from the parameters that are in a fault.
        '''

        # Get the number
        Oo = fault.polysol[self.name].shape[0]
        assert ( (Oo==1) or (Oo==3) or (Oo==4) ), \
            'Number of polynomial parameters can be 1, 3 or 4.'

        # Get the parameters
        Op = fault.polysol[self.name]

        # Create the transfer matrix
        Nd = self.vel.shape[0]
        orb = np.zeros((Nd, Oo))

        # Print Something
        print('Correcting insar rate {} from polynomial function: {}'.format(self.name, tuple(Op[i] for i in range(Oo))))

        # Fill in the first columns
        orb[:,0] = 1.0

        # If more columns
        if Oo >= 3:
            normX = fault.OrbNormalizingFactor[self.name]['x']
            normY = fault.OrbNormalizingFactor[self.name]['y']
            x0, y0 = fault.OrbNormalizingFactor[self.name]['ref']
            orb[:,1] = (self.x - x0) / normX
            orb[:,2] = (self.y - y0) / normY
        if Oo >= 4:
            orb[:,3] = orb[:,1] * orb[:,2]
        # Scale everything by the data factor
        orb *= self.factor

        # Get the correction
        self.orb = np.dot(orb, Op)

        # Correct
        self.vel -= self.orb

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

            if poly == 'build' or poly == 'include':
                if (self.name in fault.polysol.keys()):
                    # Get the orbital parameters
                    sarorb = fault.polysol[self.name]
                    # Get reference point and normalizing factors
                    x0, y0 = fault.OrbNormalizingFactor[self.name]['ref']
                    normX = fault.OrbNormalizingFactor[self.name]['x']
                    normY = fault.OrbNormalizingFactor[self.name]['y']
                    if sarorb is not None:
                        polyModel = sarorb[0]
                        if sarorb.size >= 3:
                            polyModel += sarorb[1] * (self.x - x0) / normX
                            polyModel += sarorb[2] * (self.y - y0) / normY
                        if sarorb.size >= 4:
                            polyModel += sarorb[3] * (self.x-x0)*(self.y-y0)/(normX*normY)

                if poly == 'include':
                    self.synth += polyModel

        # All done
        if poly == 'build':
            return polyModel
        else:
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
        if len(self.name.split())>1:
            datname = self.name.split()[0]
            for s in self.name.split()[1:]:
                datname = datname+'_'+s
        else:
            datname = self.name
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
        return

    def reject_pixel(self, u):
        '''
        Reject one pixel.
        Args:
            * u         : Index of the pixel to reject.
        '''

        self.lon = np.delete(self.lon, u)
        self.lat = np.delete(self.lat, u)
        self.x = np.delete(self.x, u)
        self.y = np.delete(self.y, u)
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
        
        # Import shapely
        import shapely.geometry as geom
        
        # Variables to trim are  self.corner,
        # self.xycorner, self.Cd, (self.synth)

        # Check something 
        if faults.__class__ is not list:
            faults = [faults]

        # Build a line object with the fault
        mll = []
        for f in faults:
            xf = f.xf
            yf = f.yf
            mll.append(np.vstack((xf,yf)).T.tolist())
        Ml = geom.MultiLineString(mll)

        # Build the distance map
        d = []
        for i in range(len(self.x)):
            p = [self.x[i], self.y[i]]
            PP = geom.Point(p)
            d.append(Ml.distance(PP))
        d = np.array(d)

        # Find the close ones
        u = np.where(d<=dis)[0].tolist()
            
        while len(u)>0:
            ind = u.pop()
            self.reject_pixel(ind)

        # All done
        return

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
        Dalong, vel, err, Dacros, boxll, xe1, ye1, xe2, ye2 = self.coord2prof(
                xc, yc, length, azimuth, width)

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
        vel = self.vel[Bol]
        err = self.err[Bol]

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
        return Dalong, vel, err, Dacros, boxll, xe1, ye1, xe2, ye2

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
        fout.write('# Profile Generated with StaticInv\n')
        fout.write('# Center: {} {} \n'.format(dic['Center'][0], dic['Center'][1]))
        fout.write('# Endpoints: \n')
        fout.write('#           {} {} \n'.format(dic['EndPoints'][0][0], dic['EndPoints'][0][1]))
        fout.write('#           {} {} \n'.format(dic['EndPoints'][1][0], dic['EndPoints'][1][1]))
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
            Ep = dic['LOS Error'][i]
            if np.isfinite(Vp):
                fout.write('{} {} {} \n'.format(d, Vp, Ep))

        # Close the file
        fout.close()

        # all done
        return

    def plotprofile(self, name, legendscale=10., fault=None):
        '''
        Plot profile.
        Args:
            * name      : Name of the profile.
            * legendscale: Length of the legend arrow.
        '''

        # open a figure
        fig = plt.figure()
        carte = fig.add_subplot(121)
        prof = fig.add_subplot(122)

        # Prepare a color map for insar
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=self.vel.min(), vmax=self.vel.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # plot the InSAR Points on the Map
        carte.scatter(self.x, self.y, s=10, c=self.vel, cmap=cmap, vmin=self.vel.min(), vmax=self.vel.max(), linewidths=0.0)
        scalarMap.set_array(self.vel) 
        plt.colorbar(scalarMap)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((5, 2))
        for i in range(4):
            x, y = self.ll2xy(b[i,0], b[i,1])
            bb[i,0] = x
            bb[i,1] = y
        bb[4,0] = bb[0,0]
        bb[4,1] = bb[0,1]
        carte.plot(bb[:,0], bb[:,1], '.k')
        carte.plot(bb[:,0], bb[:,1], '-k')

        # plot the selected stations on the map
        # Later

        # plot the profile
        x = self.profiles[name]['Distance']
        y = self.profiles[name]['LOS Velocity']
        ey = self.profiles[name]['LOS Error']
        p = prof.errorbar(x, y, yerr=ey, label='los velocity', marker='.', linestyle='')

        # If a fault is here, plot it
        if fault is not None:
            # If there is only one fault
            if fault.__class__ is not list:
                fault = [fault]
            # Loop on the faults
            for f in fault:
                carte.plot(f.xf, f.yf, '-')
                # Get the distance
                d = self.intersectProfileFault(name, f)
                if d is not None:
                    ymin, ymax = prof.get_ylim()
                    prof.plot([d, d], [ymin, ymax], '--', label=f.name)

        # plot the legend
        prof.legend()

        # axis of the map
        carte.axis('equal')

        # Show to screen 
        plt.show()

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
            p = Pi.coords[0]
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

    def writeProfile2File(self, name, outfile):
        '''
        Write the profile you asked for into a ascii file.
        Args:
                * name          : Name of the profile
                * outfile       : Name of the output file
        '''

        return

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

    def plot(self, ref='utm', faults=None, figure=133, gps=None, decim=False, axis='equal', norm=None, data='data'):
        '''
        Plot the data set, together with a fault, if asked.

        Args:
            * ref       : utm or lonlat.
            * faults    : list of fault object.
            * figure    : number of the figure.
            * gps       : superpose a GPS dataset.
            * decim     : plot the insar following the decimation process of varres.
        '''

        # select data to plt
        if data is 'data':
            z = self.vel
        elif data is 'synth':
            z = self.synth

        # Create the figure
        fig = plt.figure(figure)
        ax = fig.add_subplot(111)

        # Set the axes
        if ref is 'utm':
            ax.set_xlabel('Easting (km)')
            ax.set_ylabel('Northing (km)')
        else:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        # Plot the surface fault trace if asked
        if faults is not None:
            if faults.__class__ is not list:
                faults = [faults] 
            for fault in faults:
                if ref is 'utm':
                    ax.plot(fault.xf, fault.yf, '-b')
                else:
                    ax.plot(fault.lon, fault.lat, '-b')
        
        # Plot the gps if asked
        if gps is not None:
            for g in gps:
                if ref is 'utm':
                        ax.quiver(g.x, g.y, g.vel_enu[:,0], g.vel_enu[:,1])
                else:
                        ax.quiver(g.lon, g.lat, g.vel_enu[:,0], g.vel_enu[:,1])

        # Norm 
        if norm is None:
            vmin = z.min()
            vmax = z.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # prepare a color map for insar
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # Plot the decimation process
        if decim and (ref is 'utm'):
            import matplotlib.collections as colls
            for i in range(self.xycorner.shape[0]):
                x = []
                y = []
                # upper left
                x.append(self.xycorner[i,0])
                y.append(self.xycorner[i,1])
                # upper right
                x.append(self.xycorner[i,2])
                y.append(self.xycorner[i,1])
                # down right
                x.append(self.xycorner[i,2])
                y.append(self.xycorner[i,3])
                # down left
                x.append(self.xycorner[i,0])
                y.append(self.xycorner[i,3])
                verts = [zip(x, y)]
                rect = colls.PolyCollection(verts)
                rect.set_color(scalarMap.to_rgba(z[i]))
                rect.set_edgecolors('k')
                ax.add_collection(rect)

        # Plot the insar
        if ref is 'utm':
            ax.scatter(self.x, self.y, s=10, c=z, cmap=cmap, vmin=z.min(), vmax=z.max(), linewidths=0.)
        else:
            ax.scatter(self.lon, self.lat, s=10, c=z, cmap=cmap, vmin=z.min(), vmax=z.max(), linewidths=0.)

        # Colorbar
        scalarMap.set_array(z)
        plt.colorbar(scalarMap)

        # Axis
        plt.axis(axis)

        # Show
        plt.show()

        # All done
        return

    def write2grd(self, fname, oversample=1, data='data', interp=100):
        '''
        Uses surface to write the output to a grd file.
        Args:
            * fname     : Filename
            * oversample: Oversampling factor.
            * data      : can be 'data' or 'synth'.
            * interp    : Number of points along lon and lat.
        '''

        # Get variables
        x = self.lon
        y = self.lat
        if data is 'data':
            z = self.vel
        elif data is 'synth':
            z = self.synth

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
        Nlon = int(interp)*int(oversample)
        Nlat = Nlon
        I = '-I{}+/{}+'.format(Nlon,Nlat)

        # Create the G string
        G = '-G'+fname

        # Create the command
        com = ['surface', R, I, G]

        # open stdin and stdout
        fin = open('xyz.xyz', 'r')

        # Execute command
        subp.call(com, stdin=fin)

        # CLose the files
        fin.close()

        # All done
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
