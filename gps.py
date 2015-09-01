''' 
A class that deals with gps rates.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import os
import copy
import sys

# Personals
from .SourceInv import SourceInv
from .gpstimeseries import gpstimeseries
from .geodeticplot import geodeticplot as geoplot
from . import csiutils as utils

class gps(SourceInv):

    def __init__(self, name, utmzone='10', ellps='WGS84', verbose=True):
        '''
        Args:
            * name      : Name of the dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(gps,self).__init__(name,utmzone,ellps) 
        
        # Set things
        self.dtype = 'gps'
 
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize GPS array {}".format(self.name))
        self.verbose = verbose

        # Initialize things
        self.vel_enu = None
        self.err_enu = None
        self.rot_enu = None
        self.synth = None

        # All done
        return

    def setStat(self,sta_name,x,y,loc_format='LL', initVel=False):
        '''
        Set station names and locations attributes
        Args:
            * sta_name: station names
            * x: x coordinate (longitude or UTM) 
            * y: y coordinate (latitude or UTM)
            * loc_format: location format ('LL' for lon/lat or 'XY' for UTM)
        '''

        # Check input parameters
        assert len(sta_name)==len(x)==len(y), 'sta_name, x and y must have the same length'
        assert loc_format=='LL' or loc_format=='XY', 'loc_format can be LL or XY'        
        if type(x)==list:
            x = np.array(x)
        if type(y)==list:
            y = np.array(y)

        # Assign input parameters to station attributes
        self.station = copy.deepcopy(sta_name)
        self.lon = np.array([],dtype='float64')
        self.lat = np.array([],dtype='float64')
        self.x   = np.array([],dtype='float64')
        self.y   = np.array([],dtype='float64')
        if loc_format=='LL':            
            self.lon = np.append(self.lon,x)
            self.lat = np.append(self.lat,y)
            self.x, self.y = self.ll2xy(self.lon,self.lat)
        else:
            self.x = np.append(self.x,x)
            self.y = np.append(self.y,y)
            self.lon, self.lat = self.xy2ll(self.x,self.y)            
        
        # Initialize the vel_enu array
        if initVel:
            self.vel_enu = np.zeros((len(sta_name), 3))
            self.err_enu = np.zeros((len(sta_name), 3))

        # convert to arrays
        if type(self.station) is list:
            self.station = np.array(self.station)

        # All done
        return
    
    def combineNetworks(self, gpsdata, newNetworkName='Combined Network'):
        '''
        Combine networks into the current network.
        Args:
            * gpsdata           : List of gps instances.
            * newNetworkName    : Name of the returned network
        '''

        # Create lists
        lon = []
        lat = []
        name = []
        vel = []
        err = []

        # Iterate to get name, lon, lat
        for gp in gpsdata:
            lon += gp.lon.tolist()
            lat += gp.lat.tolist()
            name += gp.station.tolist()
            vel += gp.vel_enu.tolist()
            err += gp.err_enu.tolist()

        # Create a new instance
        gp = gps(newNetworkName, utmzone=self.utmzone, verbose=self.verbose)

        # Fill it
        gp.setStat(np.array(name), np.array(lon), np.array(lat))

        # Set displacements and errors
        gp.vel_enu = np.array(vel)
        gp.err_enu = np.array(err)

        # All done
        return gp

    def readStat(self,station_file,loc_format='LL'):
        '''
        Read simple station ascii file and populate station attributes
        Args:
            * station_file: station filename including station coordinates
            * loc_format:  station file format (default= 'LL')
        file format:
        STNAME  X_COORD Y_COORD (if loc_format='XY')
        STNAME  LON LAT (if loc_format='LL')
        '''
        
        # Assert if station file exists
        assert os.path.exists(station_file), 'Cannot read %s (no such file)'%(station_file)

        # Assert file format
        assert loc_format=='LL' or loc_format=='XY', 'loc_format can be either LL or XY'
        
        # Read the file 
        X = []
        Y = []
        sta_name = []
        for l in open(station_file):
            if (l.strip()[0]=='#'):
                continue
            items = l.strip().split()
            sta_name.append(items[0].strip())
            X.append(float(items[1]))
            Y.append(float(items[2]))

        # Set station attributes
        self.setStat(sta_name,X,Y,loc_format)

        # Initialize the vel_enu array
        self.vel_enu = np.zeros((len(sta_name), 3))
        self.err_enu = np.zeros((len(sta_name), 3))

        # All done
        return    

    def getvelo(self, station, data='data'):
        '''
        Gets the velocities enu for the station.
        Args:
            * station   : name of the station.
            * data      : which velocity do you want ('data' or 'synth')
        '''

        # Get the index
        u = np.flatnonzero(np.array(self.station) == station)

        # return the values
        if data in ('data'):
            return self.vel_enu[u,0], self.vel_enu[u,1], self.vel_enu[u,2]
        elif data in ('synth'):
            return self.synth[u,0], self.synth[u,1], self.synth[u,2]

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

    def getSubNetwork(self, name, stations):
        '''
        Given a list of station names, returns the corresponding gps object.
        Args:
            * name      : Name of the returned gps object
            * stations  : List of station names.
        '''
        
        # initialize lists
        Lon = []
        Lat = []
        Vel = []
        Err = []

        # Get lon lat velocities and errors values for each stations
        for station in stations:
            assert station in self.station, 'Site {} not in {} GPS object'.format(station,
                    self.name)
            Lon.append(self.lon[station==self.station])
            Lat.append(self.lat[station==self.station])
            Vel.append(self.getvelo(station))
            Err.append(self.geterr(station))

        # Create the object
        gpsNew = gps(name, utmzone=self.utmzone, verbose=self.verbose)

        # Set Stations
        gpsNew.setStat(stations, Lon, Lat)

        # Set Velocity and Error
        gpsNew.vel_enu = np.array(Vel)
        gpsNew.err_enu = np.array(Err)

        # all done
        return gpsNew

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
            Cd[st:se, st:se] = np.diag(self.err_enu[:,0]*self.err_enu[:,0])
            st += Nd
        if 'n' in direction:
            se = st + Nd
            Cd[st:se, st:se] = np.diag(self.err_enu[:,1]*self.err_enu[:,1])
            st += Nd
        if 'u' in direction:
            se = st + Nd
            Cd[st:se, st:se] = np.diag(self.err_enu[:,2]*self.err_enu[:,2])

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
        self.profiles[name] = {}

        # What data do we want
        if data is 'data':
            values = self.vel_enu
            self.profiles[name]['data type'] = 'data'
        elif data is 'synth':
            values = self.synth
            self.profiles[name]['data type'] = 'synth'

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, lon, lat = \
                utils.coord2prof(self, xc, yc, length, azimuth, width, minNum=1)

        # 4. Get these GPS
        vel = values[Bol,:]
        err = self.err_enu[Bol,:]
        names = self.station[Bol]
        if hasattr(self, 'vel_los'):
            vel_los = self.vel_los[Bol]

        # Create the lists that will hold these values
        Vacros = []; Valong = []; Vup = []; Eacros = []; Ealong = []; Eup = []

        # Get some numbers
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

        # Create vectors
        vec1 = np.array([x2-xe1, y2-y1])
        vec1 = vec1/np.sqrt( vec1[0]**2 + vec1[1]**2 )
        vec2 = np.array([x4-x1, y4-y1])
        vec2 = vec2/np.sqrt( vec2[0]**2 + vec2[1]**2 )
        ang1 = np.arctan2(vec1[1], vec1[0])
        ang2 = np.arctan2(vec2[1], vec2[0])

        # Loop on the points
        for p in range(vel.shape[0]):

            # Project velocities
            Vacros.append(np.dot(vec2,vel[p,0:2]))
            Valong.append(np.dot(vec1,vel[p,0:2]))
            # Get errors (along the ellipse)
            x = err[p,0]*err[p,1] \
                    * np.sqrt( 1. / (err[p,1]**2 + (err[p,0]*np.tan(ang2))**2) )
            y = x * np.tan(ang2)
            Eacros.append(np.sqrt(x**2 + y**2))
            x = err[p,0]*err[p,1] \
                    * np.sqrt( 1. / (err[p,1]**2 + (err[p,0]*np.tan(ang1))**2) )
            y = x * np.tan(ang1)
            Ealong.append(np.sqrt(x**2 + y**2))
            # Up direction
            Vup.append(vel[p,2])
            Eup.append(err[p,2])
            
        # Store it in the profile list
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
        dic['Vectors'] = [vec1, vec2]
        if hasattr(self, 'vel_los'):
            dic['LOS Velocity'] = vel_los
    
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

        # Plo the map
        self.plot(faults=fault, figure=None, show=False, legendscale=legendscale)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((5, 2))
        for i in range(4):
            x, y = b[i,:]
            if x<0.:
                x += 360.
            bb[i,0] = x
            bb[i,1] = y
        bb[4,0] = bb[0,0]
        bb[4,1] = bb[0,1]
        self.fig.carte.plot(bb[:,0], bb[:,1], '.k', zorder=40)
        self.fig.carte.plot(bb[:,0], bb[:,1], '-k', zorder=40)

        # open a figure
        fig = plt.figure()
        prof = fig.add_subplot(111)

        # plot the profile
        x = self.profiles[name]['Distance']
        y = self.profiles[name]['Parallel Velocity']
        ey = self.profiles[name]['Parallel Error']
        p = prof.errorbar(x, y, yerr=ey, 
                label='Fault parallel velocity', marker='.', linestyle='')
        y = self.profiles[name]['Normal Velocity']
        ey = self.profiles[name]['Normal Error']
        q = prof.errorbar(x, y, yerr=ey, 
                label='Fault normal velocity', marker='.', linestyle='')

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

    def read_from_en(self, velfile, factor=1., minerr=1., header=0):
        '''
        Reading velocities from a enu file:
        StationName | Lon | Lat | e_vel | n_vel | e_err | n_err 
        Args:
            * velfile   : File containing the velocities.
            * factor    : multiplication factor for velocities
            * minerr    : if err=0, then err=minerr.
        '''

        if self.verbose:
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
                self.vel_enu.append([east, north, 0.0])

                east = np.float(A[5])
                north = np.float(A[6])
                up = 0.0
                if east == 0.:
                    east = minerr
                if north == 0.:
                    north = minerr
                if up == 0:
                    up = minerr
                self.err_enu.append([east, north, up])

        # Make np array with that
        self.lon = np.array(self.lon).squeeze()
        self.lat = np.array(self.lat).squeeze()
        self.vel_enu = np.array(self.vel_enu).squeeze()*factor
        self.err_enu = np.array(self.err_enu).squeeze()*factor
        self.station = np.array(self.station).squeeze()
        self.factor = factor

        # Pass to xy 
        self.lonlat2xy()

        # All done
        return

    def read_from_enu(self, velfile, factor=1., minerr=1., header=0):
        '''
        Reading velocities from a enu file:
        StationName | Lon | Lat | e_vel | n_vel | u_vel | e_err | n_err | u_err
        Args:
            * velfile   : File containing the velocities.
            * factor    : multiplication factor for velocities
            * minerr    : if err=0, then err=minerr.
        '''
        if self.verbose:
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
        self.lon = np.array(self.lon).squeeze()
        self.lat = np.array(self.lat).squeeze()
        self.vel_enu = np.array(self.vel_enu).squeeze()*factor
        self.err_enu = np.array(self.err_enu).squeeze()*factor
        self.station = np.array(self.station).squeeze()
        self.factor = factor

        # Pass to xy 
        self.lonlat2xy()

        # All done
        return

    def read_from_ICM(self, velfile, factor=1., header=1):
        '''
        Reading velocities from an ICM (F. Ortega's format) file:
        Args:
            * velfile   : File containing the velocities.
            * factor    : multiplication factor for velocities
        '''
        if self.verbose:
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

        # Loop over lines
        for i in range(header,len(Vel)):
            
            # Get the line
            A = Vel[i].split()

            # If no NaN in the line
            if 'nan' not in A:

                # Get the direction array
                direction = np.array([np.int(A[3]), np.int(A[4]), np.int(A[5])])

                # Which direction are we looking at?
                d = np.flatnonzero(direction==1.)
                
                if A[9] not in self.station:    # If we do not know that station yet
                    
                    # Store the name
                    self.station.append(A[9])

                    # Store the lon lat
                    self.lon.append(np.float(A[10]))
                    self.lat.append(np.float(A[11]))

                    # Create a velocity array
                    vel = [0.,0.,0.]
                    err = [0.,0.,0.]

                    # Put velocities and errors there
                    vel[d] = np.float(A[1])
                    err[d] = np.float(A[2])

                    # Append velocities and errors
                    self.vel_enu.append(vel)
                    self.err_enu.append(err)

                else:                           # If we know that station

                    # Get the station index
                    i = [u for u in range(len(self.station)) if self.station[u] in (A[9])][0]

                    # store the velocity
                    self.vel_enu[i][d] = np.float(A[1])
                    self.err_enu[i][d] = np.float(A[2])

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

        if self.verbose:
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
        if self.verbose:
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
        self.station = np.array(self.station)[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.vel_enu = self.vel_enu[u,:]
        if self.err_enu is not None:
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

    def reference(self, station, refSynth=False):
        '''
        References the velocities to a single station.
        Args:
            * station   : name of the station or list of station names.
            * refSynth  : Apply referencing to synthetics as well (default=False)
        '''
    
        if station.__class__ is str:

            # Get the concerned station
            u = np.flatnonzero(self.station == station)
           
            # Case station missing
            assert len(u)>0, 'This station is not part of your network'

            # synth?
            if refSynth:
                self.synth = self.synth - self.vel_enu[u,:]

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

            # Synth?
            if refSynth:
                self.synth[:,0] -= mve
                self.synth[:,1] -= mvn
                self.synth[:,2] -= mvu

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
            * vertical  : Do we set the vertical GFs? default is True
        '''

        # Initialize the variables
        GssE = None; GdsE = None; GtsE = None; GcpE = None
        GssN = None; GdsN = None; GtsN = None; GcpN = None
        GssU = None; GdsU = None; GtsU = None; GcpU = None

        # Check components
        east  = False
        north = False
        if not np.isnan(self.vel_enu[:,0]).any():
            east = True
        if not np.isnan(self.vel_enu[:,1]).any():
            north = True
        if vertical and np.isnan(self.vel_enu[:,2]).any():
            raise ValueError('vertical can only be true if all stations have vertical components')

        # Get the 3 components
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

        # Get the values
        if Gss is not None:
            N = 0
            if east:
                GssE = Gss[range(0,self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if north:
                GssN = Gss[range(N,N+self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if vertical:
                GssU = Gss[range(N,N+self.vel_enu.shape[0]),:]
        if Gds is not None:
            N = 0
            if east:
                GdsE = Gds[range(0,self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if north:
                GdsN = Gds[range(N,N+self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if vertical:
                GdsU = Gds[range(N,N+self.vel_enu.shape[0]),:]
        if Gts is not None:
            N = 0
            if east:
                GtsE = Gts[range(0,self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if north:
                GtsN = Gts[range(N,N+self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if vertical:
                GtsU = Gts[range(N,N+self.vel_enu.shape[0]),:]
        if Gcp is not None:
            N = 0
            if east:
                GcpE = Gcp[range(0,self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if north:
                GcpN = Gcp[range(N,N+self.vel_enu.shape[0]),:]
                N += self.vel_enu.shape[0]
            if vertical:
                GcpU = Gcp[range(N,N+self.vel_enu.shape[0]),:]

        # set the GFs
        fault.setGFs(self, strikeslip=[GssE, GssN, GssU], dipslip=[GdsE, GdsN, GdsU],
                    tensile=[GtsE, GtsN, GtsU], coupling=[GcpE, GcpN, GcpU], vertical=vertical)

        # All done
        return

    def getNumberOfTransformParameters(self, transformation):
        '''
        Returns the number of transform parameters for the given transformation.
        Strain is only computed as an aerial strain (2D). If verticals are included, it just estimates 
        a vertical translation for the network.
        Args:
            * transformation : String. Can be
                        'strain', 'full', 'strainnorotation', 'strainnotranslation', 'strainonly'
        '''

        # Helmert Transform
        if transformation is 'full':
            if self.obs_per_station==3:
                Npo = 7                    # 3D Helmert transform is 7 parameters
            else:
                Npo = 4                    # 2D Helmert transform is 4 parameters
        # Full Strain (Translation + Strain + Rotation)
        elif transformation is 'strain':
            Npo = 6
        # Strain without rotation (Translation + Strain)
        elif transformation is 'strainnorotation':
            Npo = 5
        # Strain Only (Strain)
        elif transformation is 'strainonly':
            Npo = 3
        # Strain without translation (Strain + Rotation)
        elif transformation is 'strainnotranslation':
            Npo = 4
        # Translation
        elif transformation is 'translation':
            Npo = 2
        # Translation and Rotation
        elif transformation is 'translationrotation':
            Npo = 3
        # Uknown
        else:
            return 0

        # If verticals
        if not np.isnan(self.vel_enu[:,2]).any() and transformation is not 'full':
            Npo += 1
            
        # All done
        return Npo

    def getTransformEstimator(self, transformation):
        '''
        Returns the estimator for the transform.
        Args:
            * transformation : String. Can be
                        'strain', 'full', 'strainnorotation', 'strainnotranslation', 'strainonly'
        '''
        
        # Helmert Transform
        if transformation is 'full':
            orb = self.getHelmertMatrix(components=self.obs_per_station)
        # Strain + Rotation + Translation
        elif transformation is 'strain':
            orb = self.get2DstrainEst()
        # Strain + Translation
        elif transformation is 'strainnorotation':
            orb = self.get2DstrainEst(rotation=False)
        # Strain
        elif transformation is 'strainonly':
            orb = self.get2DstrainEst(rotation=False, translation=False)
        # Strain + Rotation
        elif transformation is 'strainnotranslation':
            orb = self.get2DstrainEst(translation=False)
        # Translation
        elif transformation is 'translation':
            orb = self.get2DstrainEst(strain=False, rotation=False)
        # Translation and Rotation
        elif transformation is 'translationrotation': 
            orb = self.get2DstrainEst(strain=False)
        # Unknown case
        else:
            print('No Transformation asked for object {}'.format(self.name))
            return None
        
        # All done
        return orb

    def computeTransformation(self, fault, verbose=False, custom=False):
        '''
        Computes the transformation that is stored with a particular fault.
        Stores it in transformation.
        '''

        # Get the transformation 
        transformation = fault.poly[self.name]
        if type(transformation) is list:
            transformation = transformation[0]
        if verbose and self.verbose:
            print('Computing transformation of type {} on data set {}'.format(transformation, self.name))

        # Get the estimator
        orb = self.getTransformEstimator(transformation)

        # make the array
        self.transformation = np.zeros(self.vel_enu.shape)

        # Check
        if orb is None:
            return

        # Get the corresponding values
        vec = fault.polysol[self.name]

        # Compute the synthetics
        tmpsynth = np.dot(orb, vec)

        # Fill it
        no = self.vel_enu.shape[0]
        self.transformation[:,0] = tmpsynth[:no]
        self.transformation[:,1] = tmpsynth[no:2*no]
        if self.obs_per_station==3:
            self.transformation[:,2] = tmpsynth[2*no:]

        # Compute custom
        if custom:
            self.computeCustom(fault)
            self.transformation[:,0] += self.custompred[:,0]
            self.transformation[:,1] += self.custompred[:,1]
            if self.obs_per_station==3:
                self.transformation[:,2] += self.custompred[:,2]

        # All done
        return

    def computeCustom(self, fault):
        '''
        Computes the displacements associated with the custom green's functions.
        '''

        # Get GFs and parameters
        G = fault.G[self.name]['custom']
        custom = fault.custom

        # Compute
        self.custompred = np.dot(G,custom)
        
        # Reshape
        if self.obs_per_station==3:
            self.custompred = self.custompred.reshape((self.vel_enu.shape))
        else: 
            self.custompred = self.custompred.reshape((self.vel_enu.shape[0], 2))

        # All done
        return

    def removeTransformation(self, fault, custom=False):
        '''
        Removes the transformation that is stored in a fault.
        '''

        # Compute the transformation
        self.computeTransformation(fault, custom=custom)

        # Do the correction
        self.vel_enu -= self.transformation

        # All done
        return

    def get2DstrainEst(self, strain=True, rotation=True, translation=True):
        '''
        Returns the matrix to estimate the full 2d strain tensor.
        Positive is clockwise for the rotation.
        Only works with 2D.
        '''

        # Get the number of gps stations
        ns = self.station.shape[0]

        # Parameter size
        nc = 6

        # Get the center of the network
        x0 = np.mean(self.x)
        y0 = np.mean(self.y)

        # Compute the baselines
        base_x = self.x - x0
        base_y = self.y - y0

        # Normalize the baselines
        base_max = np.max([np.abs(base_x).max(), np.abs(base_y).max()])
        base_x /= base_max
        base_y /= base_max

        # Store the normalizing factor
        self.StrainNormalizingFactor = base_max

        # Allocate a Base
        H = np.zeros((2,nc))

        # Put the transaltion in the base
        H[:,:2] = np.eye(2)

        # Allocate the full matrix
        Hf = np.zeros((ns*2,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,2:] = 0.0

            # Get the values
            x1, y1 = base_x[i], base_y[i]

            # Store them
            H[0,2] = x1
            H[0,3] = 0.5*y1
            H[1,3] = 0.5*x1
            H[1,4] = y1
            H[0,5] = 0.5*y1
            H[1,5] = -0.5*x1

            # Put the lines where they should be
            Hf[i,:] = H[0,:]
            Hf[i+ns,:] = H[1,:]

        # Select what we send back
        columns = []
        if translation:
            columns.append(0)
            columns.append(1)
        if strain:
            columns.append(2)
            columns.append(3)
            columns.append(4)
        if rotation:
            columns.append(5)

        # Get the right values
        Hout = Hf[:,columns]

        # Add a translation for the verticals
        if not np.isnan(self.vel_enu[:,2]).any():
            Hout = np.vstack((np.hstack((Hout, np.zeros((Hout.shape[0], 1)))), np.zeros((ns,len(columns)+1))))
            Hout[-ns:,-1] = 1.

        # All done
        return Hout

    def getHelmertMatrix(self, components=2):
        '''
        Returns a Helmert matrix for a gps data set.
        '''

        # Get the number of stations
        ns = self.station.shape[0]

        # Get the data vector size
        nd = ns*components

        # Get the number of helmert transform parameters
        if components==3:
            nc = 7
        else:
            nc = 4

        # Get the position of the center of the network
        x0 = np.mean(self.x)
        y0 = np.mean(self.y)
        z0 = 0              # We do not deal with the altitude of the stations yet (later)

        # Compute the baselines
        base_x = self.x - x0
        base_y = self.y - y0
        base_z = 0

        # Normalize the baselines
        base_x_max = np.abs(base_x).max()
        base_y_max = np.abs(base_y).max()
        meanbase = (base_x_max + base_y_max)/2.
        base_x /= meanbase
        base_y /= meanbase

        # Store 
        self.HelmertNormalizingFactor = meanbase

        # Allocate a Helmert base
        H = np.zeros((components,nc))
        
        # put the translation in it (that part never changes)
        H[:,:components] = np.eye(components)

        # Allocate the full matrix
        Hf = np.zeros((nd,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,components:] = 0.0

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
            Hf[i,:] = H[0,:]
            Hf[i+ns,:] = H[1,:]
            if nc==7:
                Hf[i+2*ns,:] = H[2,:]

        # all done 
        return Hf

    def compute2Dstrain(self, fault, write2file=False, verbose=False):
        '''
        Computes the 2D strain tensor stored in the fault given as an argument.
        '''

        # Compute the transformation
        self.computeTransformation(fault)

        # Store the transform here
        self.Strain = self.transformation
        self.StrainTensor = fault.polysol[self.name]

        # Get some info
        if write2file or verbose:
            transformation = fault.poly[self.name]
            if transformation is 'strain':
                strain = True
                translation = True
                rotation = True
            elif transformation is 'strainnorotation':
                strain = True
                translation = True
                rotation = False
            elif transformation is 'strainnotranslation':
                strain = True
                translation = False
                rotation = True
            elif transformation is 'translation':
                strain = False
                rotation = False
                translation = True
            elif transformation is 'translationrotation':
                strain = False
                rotation = True
                translation = True
            elif transformation is 'strainonly':
                strain = True
                rotation = False
                translation = False

        # Verbose?
        if verbose:
            
            # Get things to write 
            Svec = self.StrainTensor
            base_max = self.StrainNormalizingFactor

            # Print stuff
            print('--------------------------------------------------')
            print('--------------------------------------------------')
            print('Removing the estimated Strain Tensor from the gps {}'.format(self.name)) 
            print('Note: Extension is negative...')
            print('Note: ClockWise Rotation is positive...')
            print('Note: There might be a scaling factor to apply to get the things right.')
            print('         Example: if data is mm and distances in km ==> factor = 10e-6')
            print('Parameters: ')
            

            # write stuff to the screen
            iP = 0
            if translation:
                print('  X Translation :    {} '.format(Svec[iP]))
                print('  Y Translation :    {} '.format(Svec[iP+1]))
                iP += 2
            if strain:
                print('     Strain xx  :    {} '.format(Svec[iP]/base_max))
                print('     Strain xy  :    {} '.format(Svec[iP+1]/base_max))
                print('     Strain yy  :    {} '.format(Svec[iP+2]/base_max))
                iP += 3
            if rotation:
                print('     Rotation   :    {}'.format(Svec[iP]))

        # Write 2 a file
        if write2file:

            # Get things to write 
            Svec = self.StrainTensor
            base_max = self.StrainNormalizingFactor

            # Filename
            filename = '{}_{}_strain.dat'.format(self.name.replace(" ",""),fault.name.replace(" ",""))
            fout = open(filename, 'w')
            fout.write('# Strain Estimated on the data set {}\n'.format(self.name))
            fout.write('# Be carefull. There might be a corrective factor to apply to be in strain units\n')
            iP = 0
            if translation:
                fout.write('# Translation terms:\n')
                fout.write('# X-axis | Y-axis \n')
                fout.write('   {}       {} \n'.format(Svec[iP],Svec[iP+1]))
                iP += 2
            if strain:
                fout.write('# Strain Tensor: \n')
                fout.write(' {}  {}  \n'.format(Svec[iP]/base_max, Svec[iP+1]/base_max))
                fout.write(' {}  {}  \n'.format(Svec[iP+1]/base_max, Svec[iP+2]/base_max))
                iP += 3
            if rotation:
                fout.write('# Rotation term: \n')
                fout.write(' {} \n'.format(Svec[iP]))
            fout.close()

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

    def computeHelmertTransform(self, fault, verbose=False):
        '''
        Removes the Helmert Transform stored in the fault given as argument.
        '''

        # Compute transofmration
        self.computeTransformation(fault)

        # Store the transform here
        self.HelmTransform = self.transformation

        # Get the parameters for this data set
        if verbose:
            Hvec = fault.polysol[self.name]
            Nh = Hvec.shape[0]
            self.HelmertParameters = Hvec
            print('Removing a {} parameters Helmert Tranform from the gps {}'.format(Nh, self.name))
            print('Parameters: {}'.format(tuple(Hvec[i] for i in range(Nh))))

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

    def remove_euler_rotation(self, eradius=6378137.0, stations=None, verbose=True):
        '''
        Removes the best fit Euler rotation from the network.
        Args:
            * eradius   : Radius of the earth (should not change that much :-)).
            * stations  : List of stations on which rotation is estimated. If None, 
                          uses all the stations.
        '''
        if verbose:
            print('--------------------------------------------------')
            print('--------------------------------------------------')
            print ("Remove the best fot euler rotation in GPS data set {}".format(self.name))

        import eulerPoleUtils as eu

        # Get the list of stations
        if stations is not None:
            lon = []
            lat = []
            vel = []
            for station in stations:
                u = np.flatnonzero(self.station==station)
                if len(u)>0:
                    lon.append(self.lon[u[0]]*np.pi/180.)
                    lat.append(self.lat[u[0]]*np.pi/180.)
                    vel.append(self.vel_enu[u[0],:2]/self.factor)
            lon = np.array(lon)
            lat = np.array(lat)
            vel = np.array(vel)
        else:
            lon = self.lon*np.pi/180.
            lat = self.lat*np.pi/180.
            vel = self.vel_enu[:,:2]/self.factor

        # Estimate the roation pole coordinates and the velocity
        self.elat,self.elon,self.omega = eu.gps2euler(lat, lon, np.zeros(lon.shape), vel[:,0], vel[:,1])
        
        # In degrees
        self.elat *= 180./np.pi
        self.elon *= 180./np.pi
        self.omega *= 180./np.pi

        # Remove the rotation
        self.remove_rotation(self.elon*np.pi/180., self.elat*np.pi/180., self.omega*np.pi/180.)

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

    def removeSynth(self, faults, direction='sd', poly=None, custom=False):
        '''
        Removes the synthetics from a slip model.
        Args:
            * faults        : list of faults to include.
            * direction     : list of directions to use. Can be any combination of 's', 'd' and 't'.
            * poly          : if a polynomial function has been estimated, include it.
            * custom        : if some custom green's function was used, include it.
        '''

        # build the synthetics
        self.buildsynth(faults, direction=direction, poly=poly, custom=custom)

        # Correct the data from the synthetics
        self.vel_enu -= self.synth

        # All done
        return

    def buildsynth(self, faults, direction='sd', poly=None, vertical=True, custom=False):
        '''
        Takes the slip model in each of the faults and builds the synthetic displacement using the Green's functions.
        Args:
            * faults        : list of faults to include.
            * direction     : list of directions to use. Can be any combination of 's', 'd' and 't'.
            * include_poly  : if a polynomial function has been estimated, include it.
            * custom        : if some custom green's function was used, include it.
        '''

        # Check list
        if type(faults) is not list:
            faults = [faults]

        # Number of data
        Nd = self.x.shape[0]

        # Check components
        east     = False
        north    = False
        if not np.isnan(self.vel_enu[:,0]).any():
            east = True
        if not np.isnan(self.vel_enu[:,1]).any():
            north = True
        if not np.isnan(self.vel_enu[:,2]).any() and vertical:
            vertical = True

        # Clean synth
        self.synth = np.zeros((Nd,3))

        # Loop on each fault
        for fault in faults:

            # Get the good part of G
            G = fault.G[self.name]

            if ('s' in direction) and ('strikeslip' in G.keys()):
                Gs = G['strikeslip']
                Ss = fault.slip[:,0]
                ss_synth = np.dot(Gs,Ss)
                N = 0
                if east:
                    self.synth[:,0] += ss_synth[0:Nd]
                    N += Nd
                if north:
                    self.synth[:,1] += ss_synth[N:N+Nd]
                    N += Nd
                if vertical:
                    # if ss_synth.size > 2*Nd and east and north:
                    self.synth[:,2] += ss_synth[N:N+Nd]
            if ('d' in direction) and ('dipslip' in G.keys()):
                Gd = G['dipslip']
                Sd = fault.slip[:,1]
                ds_synth = np.dot(Gd, Sd)
                N = 0
                if east:
                    self.synth[:,0] += ds_synth[0:Nd]
                    N += Nd
                if north:
                    self.synth[:,1] += ds_synth[N:N+Nd]
                    N += Nd
                if vertical:
                    #if ds_synth.size > 2*Nd and east and north:
                    self.synth[:,2] += ds_synth[N:N+Nd]
            if ('t' in direction) and ('tensile' in G.keys()):
                Gt = G['tensile']
                St = fault.slip[:,2]
                op_synth = np.dot(Gt, St)
                N = 0
                if east:                
                    self.synth[:,0] += op_synth[0:Nd]
                    N += Nd
                if north:
                    self.synth[:,1] += op_synth[N:N+Nd]
                    N += Nd
                if vertical:
                    #if op_synth.size > 2*Nd and east and north:
                    self.synth[:,2] += op_synth[N:N+Nd]
            if ('c' in direction) and ('coupling' in G.keys()):
                Gc = G['coupling']
                Sc = fault.coupling
                dc_synth = np.dot(Gc,Sc)
                N = 0
                if east:
                    self.synth[:,0] += dc_synth[N:Nd]
                    N += Nd
                if north:
                    self.synth[:,1] += dc_synth[N:N+Nd]
                    N += Nd
                if vertical:
                    #if dc_synth.size > 2*Nd and east and north:
                    self.synth[:,2] += dc_synth[N:N+Nd]

            if custom:
                Gc = G['custom']
                Sc = fault.custom
                cu_synth = np.dot(G, Sc)
                N = 0
                if east:
                    self.synth[:,0] += cu_synth[N:Nd]
                    N += Nd
                if north:
                    self.synth[:,1] += cu_synth[N:N+Nd]
                    N += Nd
                if vertical:
                    self.synth[:,2] += cu_synth[N:N+Nd]

            if poly == 'build' or poly == 'include':
                if (self.name in fault.poly.keys()):
                    gpsref = fault.poly[self.name]
                    if type(gpsref) is str:
                        if gpsref in ('strain', 'strainonly', 'strainnorotation', 'strainnotranslation'):
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
        datname = self.name.replace(' ','_')
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
        return datname,filename

    def write2file(self, namefile=None, data='data', outDir='./'):
        '''
        Args:
            * namefile  : Name of the output file.
            * data      : data, synth, strain, transformation.

        '''

        # Determine file name
        if namefile is None:
            filename = ''
            for a in self.name.split():
                filename = filename+a+'_'
            filename = outDir+'/'+filename+data+'.dat'
        else: 
            filename = outDir+'/'+namefile
        
        if self.verbose:        
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
        elif data is 'res':
            z = self.vel_enu - self.synth
        elif data is 'strain':
            z = self.Strain
        elif data is 'transformation':
            z = self.transformation
        else:
            print('Unknown data type to write...')
            return

        z = z.squeeze()
        self.err_enu = self.err_enu.squeeze()

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

    def initializeTimeSeries(self, start, end, interval=1, verbose=False):
        '''
        Initializes a time series for all the stations.
        Args:
            * start     : Starting date
            * end       : Ending date
            * interval  : in days (default=1).
        '''

        # Create a list
        self.timeseries = {}

        # Loop over the stations
        for station in self.station:
            
            self.timeseries[station] = gpstimeseries(station, utmzone=self.utmzone, verbose=verbose)
            self.timeseries[station].initializeTimeSeries(start, end, interval=interval)

        # All done
        return

    def simulateTimeSeriesFromCMT(self, sismo, scale=1., verbose=True, elasticstructure='okada'):
        '''
        Takes a seismolocation object with CMT informations and computes the time series from these.
        Args:
            * sismo     : seismiclocation object (needs to have CMTinfo object and the corresponding faults list of dislocations).
            * scale     : Scales the results (default is 1.).
        '''

        # Check sismo
        assert hasattr(sismo, 'CMTinfo'), '{} object (seismiclocation class) needs a CMTinfo dictionary...'.format(sismo.name)
        assert hasattr(sismo, 'faults'), '{} object (seismiclocation class) needs a list of faults. Please run Cmt2Dislocation...'.format(sismo.name)
            
        # Check self
        assert hasattr(self, 'timeseries'), '{} object (gps class) needs a timeseries list. Please run initializeTimeSeries...'.format(self.name)

        # Re-set the time series
        for station in self.station:
            self.timeseries[station].east[:] = 0.0
            self.timeseries[station].north[:] = 0.0
            self.timeseries[station].up[:] = 0.0

        # Loop over the earthquakes
        for i in range(len(sismo.CMTinfo)):

            # Get the time of the earthquake
            eqTime = sismo.time[i]

            # Get the fault
            fault = sismo.faults[i]

            # Verbose
            if verbose:
                name = sismo.CMTinfo[i]['event name']
                mag = sismo.mag[i]
                strike = sismo.CMTinfo[i]['strike']
                dip = sismo.CMTinfo[i]['dip']
                rake = sismo.CMTinfo[i]['rake']
                depth = sismo.CMTinfo[i]['depth']
                print('Running for event {}:'.format(name))
                print('                 Mw : {}'.format(mag))
                print('               Time : {}'.format(eqTime.isoformat()))
                print('             strike : {}'.format(strike*180./np.pi))
                print('                dip : {}'.format(dip*180./np.pi))
                print('               rake : {}'.format(rake*180./np.pi))
                print('              depth : {}'.format(depth))
                print('        slip vector : {}'.format(fault.slip))
                
            # Compute the Green's functions
            if elasticstructure in ('okada'):
                fault.buildGFs(self, verbose=verbose, method='okada')
            else:
                fault.kernelsEDKS = elasticstructure
                fault.sourceSpacing = 0.1
                fault.buildGFs(self, verbose=verbose, method='edks')

            # Compute the synthetics
            self.buildsynth([fault])

            # Loop over the stations to add the step
            for station in self.station:

                # Get some informations
                TStime = np.array(self.timeseries[station].time)
                TSlength = len(TStime)
            
                # Create the time vector
                step = np.zeros(TSlength)

                # Put ones where needed
                step[TStime>eqTime] = 1.0

                # Add step to the time series
                e, n, u = self.getvelo(station, data='synth')
                e = step*e*scale
                n = step*n*scale
                u = step*u*scale
                self.timeseries[station].east += e
                self.timeseries[station].north += n
                self.timeseries[station].up += u

        # All done
        return

    def extractTimeSeriesOffsets(self, date1, date2, destination='data'):
        '''
        Puts the offset from the time series between date1 and date2 into an instance of self.
        Args:
            * date1         : datetime object.
            * date2         : datetime object.
            * destination   : if 'data', results are in vel_enu
                              if 'synth', results are in synth
        '''

        # Initialize
        vel = np.zeros((len(self.station), 3))
        err = np.zeros((len(self.station), 3))

        # Loop
        for i in range(len(self.lon)):
            station = self.station[i]
            e, n, u = self.timeseries[station].getOffset(date1, date2, data='data')
            es, ns, us = self.timeseries[station].getOffset(date1, date2, data='std')
            vel[i,0] = e
            vel[i,1] = n
            vel[i,2] = u
            err[i,0] = es
            err[i,1] = ns
            err[i,2] = us

        # Get destination
        if destination in ('data'):
            self.vel_enu = vel
        elif destination in ('synth'):
            self.synth = vel
        self.err_enu = err

        # All done
        return

    def simulateTimeSeriesFromCMTWithRandomPerturbation(self, sismo, N, xstd=10., ystd=10., depthstd=10., Moperc=30., scale=1., verbose=True, plot='jhasgc', elasticstructure='okada', relative_location_is_ok=False):
        '''
        Runs N simulations of time series from CMT.
        xtsd, ystd, depthstd are the standard deviation of the Gaussian used to perturbe the location
        The moment will be perturbed by a fraction of the moment (Moperc).
        if relative_location_is_ok is True, then all the mechanisms are moved by a common translation.
        '''

        if verbose:
            print ('Running {} perturbed earthquakes to derive the Synthetic GPS Time Series'.format(N))

        # Loop and Generate new time series
        # the time series are going to be stored in 'station name id'
        # Example: 'atjn 0001', 'atjn 0002', atjn 0003', etc and the mean will be 'atjn'
        for n in range(N):
        
            if verbose:
                sys.stdout.write('\r {}/{:03d} models'.format(n, N))
                sys.stdout.flush()

            # Copy the sismo object
            earthquakes = copy.deepcopy(sismo)

            # Check 
            if relative_location_is_ok:
                DX = np.random.randn()*xstd
                DY = np.random.randn()*ystd
                DZ = np.random.randn()*depthstd
                DMo = np.random.randn()*Moperc
    
            # Pertubate
            for fault in earthquakes.faults:

                # Move the fault
                if not relative_location_is_ok:
                    dx = np.random.randn()*xstd
                    dy = np.random.randn()*ystd
                    dz = np.random.randn()*depthstd
                    dmo = np.random.randn()*Moperc
                else:
                    dx = DX
                    dy = DY
                    dz = DZ
                    dmo = DMo
                fault.moveFault(dx, dy, dz)

                # Scale the slip by Moperc %
                fault.slip[:,0] = fault.slip[:,0] + fault.slip[:,0]*dmo/100.
                fault.slip[:,1] = fault.slip[:,1] + fault.slip[:,1]*dmo/100.
                fault.slip[:,2] = fault.slip[:,2] + fault.slip[:,2]*dmo/100.

            # Compute the simulation for that set of faults
            self.simulateTimeSeriesFromCMT(earthquakes, scale=scale, elasticstructure=elasticstructure, verbose=False)

            # Take these simulation and copy them
            for station in self.station:
                self.timeseries['{}_{:03d}'.format(station,n)] = copy.deepcopy(self.timeseries['{}'.format(station)])

        if verbose:
            print('Done')

        # Compute the average and the standard deviation
        for station in self.station:
            
            # Verbose
            if verbose:
                sys.stdout.write('\r {}'.format(station))
                sys.stdout.flush()

            # Re-set the time series
            self.timeseries[station].east[:] = 0.0
            self.timeseries[station].north[:] = 0.0
            self.timeseries[station].up[:] = 0.0
            self.timeseries[station].std_east[:] = 0.0
            self.timeseries[station].std_north[:] = 0.0
            self.timeseries[station].std_up[:] = 0.0

            # Loop over the samples to get the mean
            for n in range(N):
                self.timeseries[station].east += self.timeseries['{}_{:03d}'.format(station,n)].east
                self.timeseries[station].north += self.timeseries['{}_{:03d}'.format(station,n)].north
                self.timeseries[station].up += self.timeseries['{}_{:03d}'.format(station,n)].up
            # Mean
            self.timeseries[station].east /= np.float(N)
            self.timeseries[station].north /= np.float(N)
            self.timeseries[station].up /= np.float(N)

            # Loop over the samples to get the std
            for n in range(N):
                self.timeseries[station].std_east += (self.timeseries['{}_{:03d}'.format(station,n)].east - self.timeseries[station].east)**2
                self.timeseries[station].std_north += (self.timeseries['{}_{:03d}'.format(station,n)].north - self.timeseries[station].north)**2
                self.timeseries[station].std_up += (self.timeseries['{}_{:03d}'.format(station,n)].up - self.timeseries[station].up)**2
            # Samples
            self.timeseries[station].std_east /= np.float(N)
            self.timeseries[station].std_north /= np.float(N)
            self.timeseries[station].std_up /= np.float(N)
            # Std
            self.timeseries[station].std_east = np.sqrt(self.timeseries[station].std_east)
            self.timeseries[station].std_north = np.sqrt(self.timeseries[station].std_north)
            self.timeseries[station].std_up = np.sqrt(self.timeseries[station].std_up)

            # plot
            if plot in (station):
                for n in range(N):
                    self.timeseries['{}_{:03d}'.format(station,n)].plot(styles=['-k'], show=False)
                self.timeseries[station].plot(styles=['-r'])

        # Clean screen
        print ('Done')

        # all done
        return

    def plot(self, faults=None, figure=135, name=False, legendscale=10., scale=None, 
            plot_los=False, drawCoastlines=True, expand=0.2, show=True, 
            data=['data'], color=['k']):
        '''
        Args:
            * ref       : can be 'utm' or 'lonlat'.
            * figure    : number of the figure.
            * faults    : List of fault objects to plot the surface trace of a fault object (see verticalfault.py).
            * plot_los  : Plot the los projected gps as scatter points
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

        # plot GPS along the LOS
        if plot_los:
            fig.gps_projected(self, colorbar=True)

        # Plot GPS velocities
        fig.gps(self, data=data, name=name, legendscale=legendscale, scale=scale, color=color)

        # Save fig
        self.fig = fig

        # Show
        if show:
            fig.show(showFig=['map'])

        # All done
        return

#EOF
