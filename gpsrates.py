''' 
A class that deals with gps rates.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt

# Personals
from .SourceInv import SourceInv

class gpsrates(SourceInv):

    def __init__(self, name, utmzone='10', ellps='WGS84'):
        '''
        Args:
            * name      : Name of the dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(self.__class__,self).__init__(name,utmzone,ellps) 
        
        # Set things
        self.dtype = 'gpsrates'
 
        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize GPS array {}".format(self.name))

        # Initialize things
        self.vel_enu = None
        self.err_enu = None
        self.rot_enu = None
        self.synth = None

        # All done
        return

    def getvelo(self, station):
        '''
        Gets the velocities enu for the station.
        Args:
            * station   : name of the station.
        '''

        # Get the index
        u = np.     latnonzero(self.station == station)

        # return the values
        return self.vel_enu[u,0], self.vel_enu[u,1], self.vel_enu[u,2]

    def geterr(self, station):
        '''
        Gets the errors enu for the station.
        Args:
            * station   : name of the station.
        '''

        # Get the index
        u = np.flatnonzero(self.station == station)

        # return the values
        return self.err_enu[u,0], self.err_enu[u,1], self.err_enu[u,2]

    def scale_errors(self, scale):
        '''
        Multiplies the errors by scale.
        '''

        # Multiplyt
        self.err_enu[:,:] *= scale

        # all done
        return

    def buildCd(self, direction='en'):
        '''
        Builds a diagonal data covariance matrix using the formal uncertainties in the GPS data.
        Args:
            * direction : Direction to take into account. Can be any combination of e, n and u.
        '''

        # get the size of the total thing
        Nd = self.vel_enu.shape[0]
        Ndt = Nd*len(direction)

        # Initialize Cd
        Cd = np.zeros((Ndt, Ndt))

        # Store that diagonal matrix
        st = 0
        if 'e' in direction:
            se = st + Nd
            Cd[st:se, st:se] = np.diag(self.err_enu[:,0])
            st += Nd
        if 'n' in direction:
            se = st + Nd
            Cd[st:se, st:se] = np.diag(self.err_enu[:,1])
            st += Nd
        if 'u' in direction:
            se = st + Nd
            Cd[st:se, st:se] = np.diag(self.err_enu[:,2])

        # Store Cd
        self.Cd = Cd

        # All done
        return

    def scale(self, factor):
        '''
        Scales the gps velocities by a factor.
        Args:
            * factor    : multiplication factor.
        '''

        self.err_enu = self.err_enu*factor
        self.vel_enu = self.vel_enu*factor
        if self.rot_enu is not None:
            self.rot_enu = self.rot_enu*factor
        if self.synth is not None:
            self.synth = self.synth*factor

        # All done
        return

    def getprofile(self, name, loncenter, latcenter, length, azimuth, width, data='data'):
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
            * data              : Do the profile through the 'data' or the 'synth'etics.
        '''

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # What data do we want
        if data is 'data':
            values = self.vel_enu
        elif data is 'synth':
            values = self.synth

        # Azimuth into radians
        alpha = azimuth*np.pi/180.

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

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

        # Get the GPSs in this box.
        # Import shapely and path
        import shapely.geometry as geom
        import matplotlib.path as path
        
        # 1. Create an array with the GPS positions
        GPSXY = np.vstack((self.x, self.y)).T

        # 2. Create a box
        rect = path.Path(box, closed=False)
        
        # 3. Find those who are inside
        Bol = rect.contains_points(GPSXY)

        # 4. Get these GPS
        xg = self.x[Bol]
        yg = self.y[Bol]
        vel = self.vel_enu[Bol,:]
        err = self.err_enu[Bol,:]
        names = self.station[Bol]

        # 5. Get the sign of the scalar product between the line and the point
        vec = np.array([xe1-xc, ye1-yc])
        gpsxy = np.vstack((xg-xc, yg-yc)).T
        sign = np.sign(np.dot(gpsxy, vec))

        # 6. Compute the distance (along, across profile) and the velocity (normal/along profile)
        # Create the list that will hold these values
        Dacros = []; Dalong = []; Vacros = []; Valong = []; Vup = []; Eacros = []; Ealong = []; Eup = []
        # Build a line object
        Lalong = geom.LineString([[xe1, ye1], [xe2, ye2]])
        Lacros = geom.LineString([[xa1, ya1], [xa2, ya2]])
        # Build a multipoint
        PP = geom.MultiPoint(np.vstack((xg,yg)).T.tolist())
        # Create vectors
        vec1 = vec/np.sqrt(vec[0]**2 + vec[1]**2)
        vec2 = np.array([xa1-xc, ya1-yc]); vec2 /= np.sqrt(vec2[0]**2 + vec2[1]**2)
        # Loop on the points
        for p in range(len(PP.geoms)):
            Dalong.append(Lacros.distance(PP.geoms[p])*sign[p])
            Dacros.append(Lalong.distance(PP.geoms[p]))
            Vacros.append(np.dot(vec2,vel[p,0:2]))
            Valong.append(np.dot(vec1,vel[p,0:2]))
            Eacros.append(np.dot(vec2,err[p,0:2]))
            Ealong.append(np.dot(vec1,err[p,0:2]))
            Vup.append(vel[p,2])
            Eup.append(err[p,2])
            
        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name] 
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['Parallel Velocity'] = np.array(Vacros)
        dic['Parallel Error'] = np.array(Eacros)
        dic['Normal Velocity'] = np.array(Valong)
        dic['Normal Error'] = np.array(Ealong)
        dic['Vertical Velocity'] = np.array(Vup)
        dic['Vertical Error'] = np.array(Eup)
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['Stations'] = names
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.putm(xe1*1000., ye1*1000., inverse=True)
        lone2, late2 = self.putm(xe2*1000., ye2*1000., inverse=True)
        dic['EndPointsLL'] = [[lone1, late1],
                              [lone2, late2]]
    
        # all done
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
                fout.write('# {}          {} \n'.format(f.name, d))
        
        fout.write('#---------------------------------------------------\n')

        # Write the values
        for i in range(len(dic['Distance'])):
            d = dic['Distance'][i]
            Vp = dic['Parallel Velocity'][i]
            Ep = dic['Parallel Error'][i]
            Vn = dic['Normal Velocity'][i]
            En = dic['Normal Error'][i]
            Vu = dic['Vertical Velocity'][i]
            Eu = dic['Vertical Error'][i]
            fout.write('{} {} {} {} {} {} {} \n'.format(d, Vp, Ep, Vn, En, Vu, Eu))

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

        # plot the GPS stations on the map
        p = carte.quiver(self.x, self.y, self.vel_enu[:,0], self.vel_enu[:,1], width=0.0025, color='k')
        carte.quiverkey(p, 0.1, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color='k')

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

        # plot the profile
        x = self.profiles[name]['Distance']
        y = self.profiles[name]['Parallel Velocity']
        ey = self.profiles[name]['Parallel Error']
        p = prof.errorbar(x, y, yerr=ey, label='Fault parallel velocity', marker='.', linestyle='')
        y = self.profiles[name]['Normal Velocity']
        ey = self.profiles[name]['Normal Error']
        q = prof.errorbar(x, y, yerr=ey, label='Fault normal velocity', marker='.', linestyle='')

        # Plot the center of the profile
        lonc, latc = self.profiles[name]['Center']
        xc, yc = self.putm(lonc, latc)
        xc /= 1000.; yc /= 1000.
        carte.plot(xc, yc, '.r', markersize=20)

        # Plot the central line of the profile
        xe1, ye1 = self.profiles[name]['EndPoints'][0]
        xe2, ye2 = self.profiles[name]['EndPoints'][1]
        carte.plot([xe1, xe2], [ye1, ye2], '--k')

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

    def read_from_enu(self, velfile, factor=1., minerr=1., header=0):
        '''
        Reading velocities from a enu file:
        StationName | Lon | Lat | e_vel | n_vel | u_vel | e_err | n_err | u_err
        Args:
            * velfile   : File containing the velocities.
            * factor    : multiplication factor for velocities
            * minerr    : if err=0, then err=minerr.
        '''

        print ("Read data from file {} into data set {}".format(velfile, self.name))

        # Keep the file
        self.velfile = velfile

        # open the file
        fvel = open(self.velfile, 'r')

        # read it 
        Vel = fvel.readlines()

        # Initialize things
        self.lon = []           # Longitude list
        self.lat = []           # Latitude list
        self.vel_enu = []       # ENU velocities list
        self.err_enu = []       # ENU errors list
        self.station = []       # Name of the stations

        for i in range(header,len(Vel)):

            A = Vel[i].split()
            if 'nan' not in A:

                self.station.append(A[0])
                self.lon.append(np.float(A[1]))
                self.lat.append(np.float(A[2]))

                east = np.float(A[3])
                north = np.float(A[4])
                up = np.float(A[5])
                self.vel_enu.append([east, north, up])

                east = np.float(A[6])
                north = np.float(A[7])
                up = np.float(A[8])
                if east == 0.:
                    east = minerr
                if north == 0.:
                    north = minerr
                if up == 0:
                    up = minerr
                self.err_enu.append([east, north, up])

        # Make np array with that
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.vel_enu = np.array(self.vel_enu)*factor
        self.err_enu = np.array(self.err_enu)*factor
        self.station = np.array(self.station)
        self.factor = factor

        # Pass to xy 
        self.lonlat2xy()

        # All done
        return

    def read_from_unavco(self, velfile, factor=1., minerr=1., header=37):
        '''
        Reading velocities from a unavco file
        '''

        print ("Read data from file {} into data set {}".format(velfile, self.name))

        # Keep the file
        self.velfile = velfile

        # open the file
        fvel = open(self.velfile, 'r')

        # read it 
        Vel = fvel.readlines()

        # Initialize things
        self.lon = []           # Longitude list
        self.lat = []           # Latitude list
        self.vel_enu = []       # ENU velocities list
        self.err_enu = []       # ENU errors list
        self.station = []       # Name of the stations

        for i in range(header,len(Vel)):

            A = Vel[i].split()
            if 'nan' not in A:

                self.station.append(A[0])
                self.lon.append(np.float(A[8]))
                self.lat.append(np.float(A[7]))

                east = np.float(A[20])
                north = np.float(A[19])
                up = np.float(A[21])
                self.vel_enu.append([east, north, up])

                east = np.float(A[23])
                north = np.float(A[22])
                up = np.float(A[24])
                if east == 0.:
                    east = minerr
                if north == 0.:
                    north = minerr
                if up == 0:
                    up = minerr
                self.err_enu.append([east, north, up])

        # Make np array with that
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.vel_enu = np.array(self.vel_enu)*factor
        self.err_enu = np.array(self.err_enu)*factor
        self.station = np.array(self.station)
        self.factor = factor

        # Pass to xy 
        self.lonlat2xy()

        # All done
        return

    def read_from_sopac(self,velfile, coordfile, factor=1., minerr=1.):
        '''
        Reading velocities from Sopac file and converting to mm/yr.
        Args:
            * velfile   : File containing the velocities.
            * coordfile : File containing the coordinates.
        '''

        print ("Read data from file {} into data set {}".format(velfile, self.name))

        # Keep the files, to remember
        self.velfile = velfile+'.vel'
        self.coordfile = coordfile+'.cor'
        self.factor = factor

        # open the files
        fvel = open(self.velfile, 'r')
        fcor = open(self.coordfile, 'r')

        # read them
        Vel = fvel.readlines()
        Cor = fcor.readlines()

        # Get both names
        vnames = []
        for i in range(len(Vel)):
            vnames.append(Vel[i].split()[0])
        vnames = np.array(vnames)
        cnames = []
        for i in range(len(Cor)):
            cnames.append(Cor[i].split()[0])
        cnames = np.array(cnames)

        # Initialize things
        self.lon = []           # Longitude list
        self.lat = []           # Latitude list
        self.vel_enu = []       # ENU velocities list
        self.err_enu = []       # ENU errors list
        self.station = []       # Name of the stations

        # Loop
        for i in range(len(Vel)):

            # Check if we have the position
            c = np.flatnonzero(cnames==vnames[i])

            if len(c)>0:
                self.station.append(Vel[i].split()[0])
                self.lon.append(np.float(Cor[c].split()[9]))
                self.lat.append(np.float(Cor[c].split()[8]))
                east = np.float(Vel[i].split()[8])
                north = np.float(Vel[i].split()[7])
                up = np.float(Vel[i].split()[9])
                self.vel_enu.append([east, north, up])
                east = np.float(Vel[i].split()[11])
                north = np.float(Vel[i].split()[10])
                up = np.float(Vel[i].split()[12])
                if east == 0.:
                    east = minerr
                if north == 0.:
                    north = minerr
                if up == 0.:
                    up = minerr
                self.err_enu.append([east, north, up])

        # Make np array with that
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.vel_enu = np.array(self.vel_enu)*factor
        self.err_enu = np.array(self.err_enu)*factor
        self.station = np.array(self.station)

        # Pass to xy 
        self.lonlat2xy()

        # All done
        return


    def lonlat2xy(self):
        '''
        Pass the position of the stations into the utm coordinate system.
        '''
        
        # Transform
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # All done
        return 

    def xy2lonlat(self):
        '''
        Convert all stations x, y to lon lat using the utm transform.
        '''

        self.lon, self.lat = self.xy2ll(self.x, self.y)

        # all done
        return 

    def select_stations(self, minlon, maxlon, minlat, maxlat):
        ''' 
        Select the stations in a box defined by min and max, lat and lon.
        
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
        self.station = self.station[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.vel_enu = self.vel_enu[u,:]
        self.err_enu = self.err_enu[u,:]
        if self.rot_enu is not None:
            self.rot_enu = self.rot_enu[u,:]

        # All done
        return

    def project2InSAR(self, los):
        '''
        Projects the GPS data into the InSAR Line-Of-Sight provided.
        Args:
            * los       : list of three components of the line-of-sight vector.
        '''

        # Create a variable for the projected gps rates
        self.vel_los = np.zeros((self.vel_enu.shape[0]))

        # Convert los to numpy array
        los = np.array(los)
        self.los = los

        # Loop over 
        for i in range(self.vel_enu.shape[0]):
            self.vel_los[i] = np.dot( self.vel_enu[i,:], self.los )

        # All done 
        return

    def keep_stations(self, stations):
        '''
        Keeps only the stations on the arg list.
        Args:
            * stations  : list of stations to keep.
        '''

        # Get the total list of stations
        allsta = self.station

        # remove the stations from that list
        for sta in stations:
            u = np.flatnonzero(allsta==sta)
            allsta = np.delete(allsta,u)

        # Rejection list
        rejsta = allsta.tolist()

        # Reject 
        self.reject_stations(rejsta)

        # All done
        return

    def reject_stations_fault(self, dis, faults):
        ''' 
        Rejects the pixels that are dis km close to the fault.
        Args:
            * dis       : Threshold distance.
            * faults    : list of fault objects.
        '''

        # Import stuff
        import shapely.geometry as geom

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
        
        # reject them
        self.reject_stations(self.station[u].tolist())

        # All done
        return

    def reject_stations(self, station):
        '''
        Reject the stations named in stations.
        Args:
            * station   : name or list of names of station.
        '''

        if station.__class__ is str:

            # Get the concerned station
            u = np.flatnonzero(self.station == station)

            if u.size > 0:

                # Delete
                self.station = np.delete(self.station, u, axis=0)
                self.lon = np.delete(self.lon, u, axis=0)
                self.lat = np.delete(self.lat, u, axis=0)
                self.vel_enu = np.delete(self.vel_enu, u, axis=0)
                self.err_enu = np.delete(self.err_enu, u, axis=0)
                if self.rot_enu is not None:
                    self.rot_enu = np.delete(self.rot_enu, u, axis=0)

        elif station.__class__ is list:

            for sta in station:

                u = np.flatnonzero(self.station == sta)

                if u.size > 0:

                    self.station = np.delete(self.station, u, axis=0)
                    self.lon = np.delete(self.lon, u, axis=0)
                    self.lat = np.delete(self.lat, u, axis=0)
                    self.vel_enu = np.delete(self.vel_enu, u, axis=0)
                    self.err_enu = np.delete(self.err_enu, u, axis=0)
                    if self.rot_enu is not None:
                        self.rot_enu = np.delete(self.rot_enu, u, axis=0)

        # Update x and y
        self.lonlat2xy()

        # All done
        return

    def reference(self, station):
        '''
        References the velocities to a single station.
        Args:
            * station   : name of the station or list of station names.
        '''
    
        if station.__class__ is str:

            # Get the concerned station
            u = np.flatnonzero(self.station == station)
           
            # Case station missing
            if len(u) == 0:
                print("This station is not part of your network")
                return

            # Reference
            self.vel_enu = self.vel_enu - self.vel_enu[u,:]

        elif station.__class__ is list:

            # Get the concerned stations
            u = []
            for sta in station:
                u.append(np.flatnonzero(self.station == sta))

            # Get the mean velocities
            mve = np.mean(self.vel_enu[u,0])
            mvn = np.mean(self.vel_enu[u,1])
            mvu = np.mean(self.vel_enu[u,2])

            # Reference
            self.vel_enu[:,0] = self.vel_enu[:,0] - mve
            self.vel_enu[:,1] = self.vel_enu[:,1] - mvn
            self.vel_enu[:,2] = self.vel_enu[:,2] - mvu

        # All done
        return
    
    def removePoly(self, fault):
        '''
        Removes the polynomial form inverted.
        '''

        # Get the parameters of the polynomial
        Ns = fault.polysol[self.name].shape[0]
        Cx = 0.0;
        Cy = 0.0;
        Cz = 0.0;
        if Ns==2:
            Cx, Cy = fault.polysol[self.name]
        elif Ns==3:
            Cx, Cy, Cz = fault.polysol[self.name]

        # Add that to the values
        self.vel_enu[:,0] -= Cx
        self.vel_enu[:,1] -= Cy
        self.vel_enu[:,2] -= Cz

        # all done
        return
    
    def compute2Dstrain(self, fault):
        '''
        Computes the 2D strain tensor stored in the fault given as an argument.
        '''

        # Get the size of the strain tensor
        assert fault.strain[self.name]
        Nh = fault.strain[self.name]

        # Get the number of obs per station
        if not hasattr(self, 'obs_per_station'):
            if Nh == 6:
                self.obs_per_station = 2
            else:
                print('Strain estimation for 3d not implemented')
        No = self.obs_per_station

        # Get the position of the center of the network
        x0 = np.mean(self.x)
        y0 = np.mean(self.y)

        # Compute the baseline and normalize these
        base_x = self.x - x0
        base_y = self.y - y0

        # Normalize the baselines
        base_max = fault.StrainNormalizingFactor[self.name]
        base_x /= base_max
        base_y /= base_max

        # Allocate a Strain base
        H = np.zeros((No,Nh)) 

        # Fill in the part that does not change
        H[:,:No] = np.eye(No) 

        # Store the transform here
        self.Strain = np.zeros(self.vel_enu.shape) 

        # Get the parameters for this data set
        Svec = fault.polysol[self.name]
        self.StrainTensor = Svec
        print('Removing the estimated Strain Tensor from the gpsrates {}'.format(self.name)) 
        print('Parameters: ')
        print('  X Translation :    {} mm/yr'.format(Svec[0]))
        print('  Y Translation :    {} mm/yr'.format(Svec[1]))
        print('     Strain xx  :    {} mm/yr/km'.format(Svec[2]/base_max))
        print('     Strain xy  :    {} mm/yr/km'.format(Svec[3]/base_max))
        print('     Strain yy  :    {} mm/yr/km'.format(Svec[4]/base_max))
        print('     Rotation   :    {}'.format(Svec[5]))

        # Loop over the station
        for i in range(self.station.shape[0]):

            # Clean the part that changes
            H[:,No:] = 0.0

            # Get the values
            x1, y1 = base_x[i], base_y[i]

            # Store the rest
            H[0,2] = x1 
            H[0,3] = 0.5*y1 
            H[0,5] = 0.5*y1 
            H[1,3] = 0.5*x1
            H[1,4] = y1
            H[1,5] = -0.5*y1  

            # Do the transform
            newv = np.dot(H, Svec)
            self.Strain[i,:No] = newv

        # All done
        return
         
    def remove2Dstrain(self, fault):
        '''
        Computess the 2D strain and removes it.
        '''

        # Computes the strain
        self.compute2Dstrain(fault)

        # Correct 
        self.vel_enu = self.vel_enu - self.Strain

        # All done
        return

    def computeHelmertTransform(self, fault):
        '''
        Removes the Helmert Transform stored in the fault given as argument.
        '''

        # Get the size of the transform
        assert fault.helmert[self.name]
        Nh = fault.helmert[self.name]

        # Get the number of observation per station
        if not hasattr(self, 'obs_per_station'):
            if Nh == 4:
                self.obs_per_station = 2
            else:
                self.obs_per_station = 3
        No = self.obs_per_station

        # Get the position of the center of the network
        x0 = np.mean(self.x)
        y0 = np.mean(self.y)
        z0 = 0                                              # No 3D for now, later

        # Compute the baselines
        base_x = self.x - x0
        base_y = self.y - y0
        base_z = 0

        # Normalize the baselines
        base_x_max = np.abs(base_x).max(); base_x /= base_x_max
        base_y_max = np.abs(base_y).max(); base_y /= base_y_max

        # Allocate a Helmert base
        H = np.zeros((No,Nh))

        # Fill in the part that does not change
        H[:,:No] = np.eye(No)

        # Store the transform here
        self.HelmTransform = np.zeros(self.vel_enu.shape)

        # Get the parameters for this data set
        Hvec = fault.polysol[self.name]
        self.HelmertParameters = Hvec
        print('Removing a {} parameters Helmert Tranform from the gpsrates {}'.format(Nh, self.name))
        print('Parameters: {}'.format(tuple(Hvec[i] for i in range(Nh))))

        # Loop over the station
        for i in range(self.station.shape[0]):

            # Clean the part that changes
            H[:,No:] = 0.0

            # Put the rotation components and the scale components
            x1, y1, z1 = base_x[i], base_y[i], base_z
            if Nh==7:
                H[:,3:6] = np.array([[0.0, -z1, y1],
                                     [z1, 0.0, -x1],
                                     [-y1, x1, 0.0]])
                H[:,7] = np.array([x1, y1, z1])
            else:
                H[:,2] = np.array([y1, -x1])
                H[:,3] = np.array([x1, y1])

            # Do the transform
            newv = np.dot(H,Hvec)
            self.HelmTransform[i,:No] = newv

            # Correct the data
            self.vel_enu[i,:No] -= newv

        # All done
        return

    def removeHelmertTransform(self, fault):
        '''
        Computess the Helmert and removes it.
        '''

        # Computes the strain
        self.computeHelmertTransform(fault)

        # Correct 
        self.vel_enu = self.vel_enu - self.HelmTransform

        # All done
        return

    def remove_euler_rotation(self, eradius=6378137.0):
        '''
        Removes the best fit Euler rotation from the network.
        Args:
            * eradius   : Radius of the earth (should not change that much :-)).
        '''

        print ("Remove the best fot euler rotation in GPS data set {}".format(self.name))

        import eulerPoleUtils as eu

        # Estimate the roation pole coordinates and the velocity
        self.elat,self.elon,self.omega = eu.gps2euler(self.lat*np.pi/180., self.lon*np.pi/180., np.zeros(self.lon.shape), self.vel_enu[:,0]/self.factor, self.vel_enu[:,1]/self.factor)

        # Remove the rotation
        self.remove_rotation(self.elon, self.elat, self.omega)

        # All done
        return

    def remove_rotation(self, elon, elat, omega):
        '''
        Removes a rotation from the lon, lat and velocity of a rotation pole.
        Args:
            * elon   : Longitude of the rotation pole
            * elat   : Latitude of the rotation pole
            * omega : Amplitude of the rotation.
        '''

        import eulerPoleUtils as eu

        # Convert pole parameters to Cartesian
        evec_xyz = eu.llh2xyz(elat, elon, 0.0)
        self.epole = omega * evec_xyz / np.linalg.norm(evec_xyz)
        
        # Predicted station velocities
        Pxyz = eu.llh2xyz(self.lat*np.pi/180., self.lon*np.pi/180., np.zeros(self.lon.shape))
        self.rot_enu = eu.euler2gps(self.epole, Pxyz.T)*self.factor

        # Correct the velocities from the prediction
        self.vel_enu = self.vel_enu - self.rot_enu

        # All done
        return 

    def makeDelaunay(self, plot=False):
        '''
        Builds a Delaunay triangulation of the GPS network.
        Args:   
            * plot          : True/False(default).
        '''

        # import needed matplotlib
        import matplotlib.delaunay as triangle

        # Do the triangulation
        Cense, Edges, Triangles, Neighbors = triangle.delaunay(self.x, self.y)

        # plot
        if plot:
            plt.figure()
            for ed in Edges:
                plt.plot([self.x[ed[0]], self.x[ed[1]]], [self.y[ed[0]], self.y[ed[1]]], '-')
            plt.plot(self.x, self.y, '.k')
            plt.show()

        # Store the triangulation scheme
        self.triangle = {}
        self.triangle['CircumCenters'] = Cense
        self.triangle['Edges'] = Edges
        self.triangle['Triangles'] = Triangles
        self.triangle['Neighbours'] = Neighbors

        # All done
        return

    def removeSynth(self, faults, direction='sd', poly=None):
        '''
        Removes the synthetics from a slip model.
        Args:
            * faults        : list of faults to include.
            * direction     : list of directions to use. Can be any combination of 's', 'd' and 't'.
            * include_poly  : if a polynomial function has been estimated, include it.
        '''

        # build the synthetics
        self.buildsynth(faults, direction=direction, poly=poly)

        # Correct the data from the synthetics
        self.vel_enu -= self.synth

        # All done
        return

    def buildsynth(self, faults, direction='sd', poly=None):
        '''
        Takes the slip model in each of the faults and builds the synthetic displacement using the Green's functions.
        Args:
            * faults        : list of faults to include.
            * direction     : list of directions to use. Can be any combination of 's', 'd' and 't'.
            * include_poly  : if a polynomial function has been estimated, include it.
        '''

        # Number of data
        Nd = self.vel_enu.shape[0]

        # Clean synth
        self.synth = np.zeros((self.vel_enu.shape))

        # Loop on each fault
        for fault in faults:

            # Get the good part of G
            G = fault.G[self.name]

            if ('s' in direction) and ('strikeslip' in G.keys()):
                Gs = G['strikeslip']
                Ss = fault.slip[:,0]
                ss_synth = np.dot(Gs,Ss)
                self.synth[:,0] += ss_synth[0:Nd]
                self.synth[:,1] += ss_synth[Nd:2*Nd]
                if ss_synth.size > 2*Nd:
                    self.synth[:,2] += ss_synth[2*Nd:3*Nd]
            if ('d' in direction) and ('dipslip' in G.keys()):
                Gd = G['dipslip']
                Sd = fault.slip[:,1]
                ds_synth = np.dot(Gd, Sd)
                self.synth[:,0] += ds_synth[0:Nd]
                self.synth[:,1] += ds_synth[Nd:2*Nd]
                if ds_synth.size >2*Nd:
                    self.synth[:,2] += ds_synth[2*Nd:3*Nd]
            if ('t' in direction) and ('tensile' in G.keys()):
                Gt = G['tensile']
                St = fault.slip[:,2]
                op_synth = np.dot(Gt, St)
                self.synth[:,0] += op_synth[0:Nd]
                self.synth[:,1] += op_synth[Nd:2*Nd]
                if op_synth.size >2*Nd:
                    self.synth[:,2] += op_synth[2*Nd:3*Nd]

            if poly == 'build' or poly == 'include':
                if (self.name in fault.poly.keys()):
                    gpsref = fault.poly[self.name]
                    if type(gpsref) is str:
                        if gpsref is 'strain':
                            self.compute2Dstrain(fault)
                            self.synth = self.synth + self.Strain
                        elif gpsref is 'full':
                            self.computeHelmertTransform(fault)
                            self.synth = self.synth + self.HelmTransform
                    elif type(gpsref) is float:
                        self.synth[:,0] += gpsref[0]
                        self.synth[:,1] += gpsref[1]
                        if len(gpsref)==3:
                            self.synth += gpsref[2]

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
        if len(self.name.split())>1:
            datname = self.name.split()[0]
            for s in self.name.split()[1:]:
                datname = datname+'_'+s
        else:
            datname = self.name
        filename = 'edks_{}.idEN'.format(datname)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N\n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '{:5d} {} {} \n'.format(i, x[i], y[i])
            fout.write(string)

        # Close the file
        fout.close()

        # All done
        return

    def write2file(self, namefile=None, data='data', outDir='./'):
        '''
        Args:
            * namefile  : Name of the output file.
            * data      : data, synth, strain.

        '''

        # Determine file name
        if namefile is None:
            filename = ''
            for a in self.name.split():
                filename = filename+a+'_'
            filename = outDir+filename+data+'.dat'
        else: 
            filename = outDir+namefile

        print ("Write {} set {} to file {}".format(data, self.name, filename))

        # open the file
        fout = open(filename,'w')

        # write a header
        fout.write('# Name lon lat v_east v_north v_up e_east e_north e_up \n')

        # Get the data 
        if data is 'data':
            z = self.vel_enu
        elif data is 'synth':
            z = self.synth
        elif data is 'strain':
            z = self.Strain
        else:
            print('Unknown data type to write...')
            return

        # Loop over stations
        for i in range(len(self.station)):
            fout.write('{} {} {} {} {} {} {} {} {} \n'.format(self.station[i], self.lon[i], self.lat[i], 
                                                        z[i,0], z[i,1], z[i,2],
                                                        self.err_enu[i,0], self.err_enu[i,1], self.err_enu[i,2]))
        
        # Close file
        fout.close()

        # All done 
        return

    def getRMS(self):
        '''
        Computes the RMS of the data and if synthetics are computed, the RMS of the residuals
        '''

        # Get the number of points
        N = self.vel_enu.shape[0] * 3.

        # RMS of the data
        dataRMS = np.sqrt( 1./N * sum(self.vel_enu.flatten()**2) )

        # Synthetics
        if self.synth is not None:
            synthRMS = np.sqrt( 1./N *sum( (self.vel_enu.flatten() - self.synth.flatten())**2 ) )
            return dataRMS, synthRMS
        else:
            return dataRMS, 0.

        # All done

    def getVariance(self):
        '''                                                                                                      
        Computes the Variance of the data and if synthetics are computed, the RMS of the residuals                    
        '''
        
        # Get the number of points                                                                               
        N = self.vel_enu.shape[0] * 3.                                                                           
        
        # Varianceof the data                                                                                        
        dmean = self.vel_enu.flatten().mean()
        dataVariance = ( 1./N * sum((self.vel_enu.flatten()-dmean)**2) ) 
        
        # Synthetics
        if self.synth is not None:           
            rmean = (self.vel_enu.flatten() - self.synth.flatten()).mean()
            synthVariance = ( 1./N *sum( (self.vel_enu.flatten() - self.synth.flatten() - rmean)**2 ) )                
            return dataVariance, synthVariance
        else:
            return dataVariance, 0.                                                                                   
        
        # All done       
        
    def getMisfit(self):
        '''                                                                                                      
        Computes the Summed Misfit of the data and if synthetics are computed, the RMS of the residuals                    
        '''

        # Misfit of the data                                                                                        
        dataMisfit = sum((self.vel_enu.flatten()))

        # Synthetics
        if self.synth is not None:
            synthMisfit = sum( (self.vel_enu.flatten() - self.synth.flatten()) )
            return dataMisfit, synthMisfit
        else:
            return dataMisfit, 0.

        # All done

    def plot(self, ref='utm', faults=None, figure=135, name=False, legendscale=10., color='b', scale=150, plot_los=False):
        '''
        Args:
            * ref       : can be 'utm' or 'lonlat'.
            * figure    : number of the figure.
            * faults    : List of fault objects to plot the surface trace of a fault object (see verticalfault.py).
            * plot_los  : Plot the los projected gps as scatter points
        '''

        # Import some things
        import matplotlib.pyplot as plt

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
            for fault in faults:
                if ref is 'utm':
                    ax.plot(fault.xf, fault.yf, '-r', label=fault.name)
                else:
                    ax.plot(fault.lon, fault.lat, '-r', label=fault.name)

        # Plot the gps projected
        if plot_los:
            if not hasattr(self,'los'):
                print('Need to project the GPS first, cannot plot projected...')
            else:
                if ref is 'utm':
                    ax.scatter(self.x, self.y, 100, self.vel_los, linewidth=1)
                else:
                    ax.scatter(self.lon, self.lat, 100, self.vel_los, linewidth=1)

        # Plot the GPS velocities
        if ref is 'utm':
            p = ax.quiver(self.x, self.y, self.vel_enu[:,0], self.vel_enu[:,1], label='data', color=color, scale=scale)
            q = ax.quiverkey(p, 0.04, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color=color)
        else:
            p = ax.quiver(self.lon, self.lat, self.vel_enu[:,0], self.vel_enu[:,1], label='data', scale=scale)
            q = ax.quiverkey(p, 0.04, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color=color)

        # if some synthetics exist
        if self.synth is not None:
            if ref is 'utm':
                s = ax.quiver(self.x, self.y, self.synth[:,0], self.synth[:,1], label='synth', color='r', scale=scale)
                q = ax.quiverkey(s, 0.04, 0.8, legendscale, "{}".format(legendscale), coordinates='axes', color='r')
            else:
                s = ax.quiver(self.lon, self.lat, self.synth[:,0], self.synth[:,1], label='synth', color='r', scale=scale)
                q = ax.quiverkey(s, 0.04, 0.8, legendscale, "{}".format(legendscale), coordinates='axes', color='r')

        # If the Helmert transform has been estimated
        if hasattr(self, 'HelmTransform'):
            if ref is 'utm':
                s = ax.quiver(self.x, self.y, self.HelmTransform[:,0], self.HelmTransform[:,1], label='Helmert Tranform', color='b', scale=scale)
                q = ax.quiverkey(s, 0.04, 0.05, legendscale, "{}".format(legendscale), coordinates='axes', color='b')
            else:
                s = ax.quiver(self.lon, self.lat, self.HelmTransform[:,0], self.HelmTransform[:,1], label='Helmert Tranform', color='b', scale=scale)
                q = ax.quiverkey(s, 0.04, 0.05, legendscale, "{}".format(legendscale), coordinates='axes', color='b')

        # Plot the name of the stations if asked
        if name:
            if ref is 'utm':
                for i in range(len(self.x)):
                    ax.text(self.x[i], self.y[i], self.station[i], fontsize=12)
            else:
                for i in range(len(self.lon)):
                    ax.text(self.lon[i], self.lat[i], self.station[i], fontsize=12)

        # Plot the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        # Axis
        ax.axis('equal')

        # Do the plot
        plt.show()

        # All done
        return

