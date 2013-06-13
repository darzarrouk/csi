''' 
A class that deals with gps rates.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp

class gpsrates(object):

    def __init__(self, name, utmzone='10'):
        '''
        Args:
            * name      : Name of the dataset.
            * datatype  : can be 'gps' or 'insar' for now.
            * utmzone   : UTM zone. Default is 10 (Western US).
        '''

        # Set things
        self.name = name
        self.dtype = 'gpsrates'
        self.utmzone = utmzone
 
        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize GPS array %s"%self.name)

        # Create a utm transformation
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

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
        u = np.flatnonzero(self.station == station)

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

    def read_from_enu(self, velfile, factor=1., minerr=1., header=0):
        '''
        Reading velocities from a enu file:
        StationName | Lon | Lat | e_vel | n_vel | u_vel | e_err | n_err | u_err
        Args:
            * velfile   : File containing the velocities.
            * factor    : multiplication factor for velocities
            * minerr    : if err=0, then err=minerr.
        '''

        print ("Read data from file %s into data set %s"%(velfile, self.name))

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
        self.ll2xy()

        # All done
        return

    def read_from_sopac(self,velfile, coordfile, factor=1., minerr=1.):
        '''
        Reading velocities from Sopac file and converting to mm/yr.
        Args:
            * velfile   : File containing the velocities.
            * coordfile : File containing the coordinates.
        '''

        print ("Read data from file %s into data set %s"%(velfile, self.name))

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
        self.ll2xy()

        # All done
        return

    def ll2xy(self):
        '''
        Pass the position into the utm coordinate system.
        '''
        
        # Do the transformation
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # All done
        return

    def lonlat2xy(self, lo, la):
        '''
        Pass the position into the utm coordinate system.
        '''

        x, y = self.putm(lo, la)
        x = x/1000.
        y = y/1000.

        # All done
        return x, y

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
        self.ll2xy()

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
    
    def remove_euler_rotation(self, eradius=6378137.0):
        '''
        Removes the best fit Euler rotation from the network.
        Args:
            * eradius   : Radius of the earth (should not change that much :-)).
        '''

        print ("Remove the best fot euler rotation in GPS data set %s"%self.name)

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

    def buildsynth(self, faults, direction='sd', include_poly=False):
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

            if (self.name in fault.polysol.keys()) and (include_poly):
                gpsref = fault.polysol[self.name]
                if gpsref is not None:
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
        filename = 'edks_%s.idEN'%(self.name)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N\n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '%5i %f %f \n'%(i, x[i], y[i])
            fout.write(string)

        # Close the file
        fout.close()

        # All done
        return

    def writetofile(self, filename):
        '''
        Args:
            * filename  : Name of the output file.
        '''

        print ("Write data set %s to file %s"%(self.name, filename))

        # open the file
        fout = open(filename,'w')

        # write a header
        fout.write('# Name lon lat v_east v_north v_up e_east e_north e_up \n')

        # Loop over stations
        for i in range(len(self.station)):
            fout.write('%s %f %f %f %f %f %f %f %f \n'%(self.station[i], self.lon[i], self.lat[i], 
                                                        self.vel_enu[i,0], self.vel_enu[i,1], self.vel_enu[i,2],
                                                        self.err_enu[i,0], self.err_enu[i,1], self.err_enu[i,2]))
        
        # Close file
        fout.close()

        # All done 
        return

    def plot(self, ref='utm', faults=None, figure=135):
        '''
        Args:
            * ref       : can be 'utm' or 'lonlat'.
            * figure    : number of the figure.
            * faults    : List of fault objects to plot the surface trace of a fault object (see verticalfault.py).
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
                    ax.plot(fault.xf, fault.yf, '-b')
                else:
                    ax.plot(fault.lon, fault.lat, '-b')

        # Plot the GPS velocities
        if ref is 'utm':
            ax.quiver(self.x, self.y, self.vel_enu[:,0], self.vel_enu[:,1])
        else:
            ax.quiver(self.lon, self.lat, self.vel_enu[:,0], self.vel_enu[:,1])

        # if some synthetics exist
        if self.synth is not None:
            if ref is 'utm':
                ax.quiver(self.x, self.y, self.synth[:,0], self.synth[:,1], 'b')
            else:
                ax.quiver(self.lon, self.lat, self.synth[:,0], self.synth[:,1], 'b')

        # Do the plot
        plt.show()

        # All done
        return

