'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import shapely.geometry as geom
import matplotlib.pyplot as plt

class insarrates(object):

    def __init__(self, name, utmzone='10'):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * utmzone       : UTM zone. Default is 10 (Western US).
        '''

        # Initialize the data set 
        self.name = name
        self.utmzone = utmzone
        self.dtype = 'insarrates'

        print ("---------------------------------")
        print ("---------------------------------")
        print (" Initialize InSAR data set {}".format(self.name))

        # Initialize the UTM transformation
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

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
        self.x, self.y = self.lonlat2xy(self.lon,self.lat)

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
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # Deal with the LOS
        alpha = (heading+90.0)*np.pi/180.0
        phi = incidence*np.pi/180.0
        Se = -1.0 * np.sin(alpha) * np.sin(phi)
        Sn = -1.0 * np.cos(alpha) * np.sin(phi)
        Su = np.cos(phi)
        self.los = np.ones((self.lon.shape[0],3))
        self.los[:,0] *= Se
        self.los[:,1] *= Sn
        self.los[:,2] *= Su

        # Store the factor
        self.factor = factor

        # All done
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

        # Keep the non-nan pixels only
        u = np.flatnonzero(np.isfinite(self.vel))
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.vel = self.vel[u]
        self.err = self.err[u]

        # Convert to utm
        self.x, self.y = self.lonlat2xy(self.lon, self.lat) 

        # Deal with the LOS
        self.los = np.ones((self.lon.shape[0],3))
        if heading is not None and incidence is not None and los is None:
            alpha = (heading+90.0)*np.pi/180.0
            phi = incidence*np.pi/180.0 
            Se = -1.0*np.sin(alpha) * np.sin(phi) 
            Sn = -1.0*np.cos(alpha) * np.sin(phi) 
            Su = np.cos(phi) 
            self.los[:,0] *= Se
            self.los[:,1] *= Sn
            self.los[:,2] *= Su
        elif los is not None:
            self.los[:,0] *= los[0]
            self.los[:,1] *= los[1]
            self.los[:,2] *= los[2]
        else:
            assert False, 'not enough information to compute LOS'

        # Store the factor
        self.factor = factor

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
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.vel = self.vel[u,:]
        self.err = self.err[u,:]
        self.los = self.los[u,:]
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

    def removeSynth(self, faults, direction='sd', include_poly=False):
        '''
        Removes the synthetics using the faults and the slip distributions that are in there.
        Args:
            * faults        : List of faults.
            * direction     : Direction of slip to use.
            * include_poly  : if a polynomial function has been estimated, include it.
        '''

        # Build synthetics
        self.buildsynth(faults, direction=direction, include_poly=include_poly)

        # Correct
        self.vel -= self.synth

        # All done
        return

    def buildsynth(self, faults, direction='sd', include_poly=False):
        '''
        Computes the synthetic data using the faults and the associated slip distributions.
        Args:
            * faults        : List of faults.
            * direction     : Direction of slip to use.
            * include_poly  : if a polynomial function has been estimated, include it.
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

            if include_poly:
                if (self.name in fault.polysol.keys()):
                    # Get the orbital parameters
                    sarorb = fault.polysol[self.name]
                    # Get reference point and normalizing factors
                    x0, y0 = fault.OrbNormalizingFactor[self.name]['ref']
                    normX = fault.OrbNormalizingFactor[self.name]['x']
                    normY = fault.OrbNormalizingFactor[self.name]['y']
                    if sarorb is not None:
                        self.synth += sarorb[0]
                        if sarorb.size >= 3:
                            self.synth += sarorb[1] * (self.x - x0) / normX
                            self.synth += sarorb[2] * (self.y - y0) / normY
                        if sarorb.size >= 4:
                            self.synth += sarorb[3] * (self.x-x0)*(self.y-y0)/(normX*normY)

        # All done
        return

    def ll2xy(self):
        '''
        Converts the lat lon positions into utm coordinates.
        '''

        x, y = self.putm(self.lon, self.lat)
        self.x = x/1000.
        self.y = y/1000.

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

        # Azimuth into radians
        alpha = azimuth*np.pi/180.

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.lonlat2xy(loncenter, latcenter)

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
        elon1, elat1 = self.xy2lonlat(xe1, ye1)
        elon2, elat2 = self.xy2lonlat(xe2, ye2)

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
        lon1, lat1 = self.xy2lonlat(x1, y1)
        lon2, lat2 = self.xy2lonlat(x2, y2)     
        lon3, lat3 = self.xy2lonlat(x3, y3)
        lon4, lat4 = self.xy2lonlat(x4, y4)

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
        import shapely.geometry as geom
        import matplotlib.nxutils as mnu

        # 2. Create an array with the InSAR positions
        SARXY = np.vstack((self.x, self.y)).T

        # 3. Find those who are inside
        Bol = mnu.points_inside_poly(SARXY, box)

        # 4. Get these values
        xg = self.x[Bol]
        yg = self.y[Bol]
        vel = self.vel[Bol]
        err = self.err[Bol]

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
            x, y = self.lonlat2xy(b[i,0], b[i,1])
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
        

