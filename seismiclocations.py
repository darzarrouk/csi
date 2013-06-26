'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt

class seismiclocations(object):

    def __init__(self, name, utmzone='10'):
        '''
        Args:
            * name          : Name of the Seismic dataset.
            * utmzone       : UTM zone. Default is 10 (Western US).
        '''

        # Initialize the data set 
        self.name = name
        self.utmzone = utmzone
        self.dtype = 'seismiclocations'

        print ("---------------------------------")
        print ("---------------------------------")
        print (" Initialize Seismicity data set %s"%self.name)

        # Initialize the UTM transformation
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

        # Initialize some things
        self.time = None
        self.lon = None
        self.lat = None
        self.depth = None
        self.date = None
        self.mag = None

        # All done
        return

    def read_from_NCSN(self,filename, header=65):
        '''
        Read the Seismic catalog from the NCSN networks (Template from F. Waldhauser).
        Args:
            * filename      : Name of the input file. 
            * header        : Size of the header.
        '''

        print ("Read from file %s into data set %s"%(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            # Split the string line 
            tmp = A[i].split()

            # Get the values
            yr = np.int(tmp[0])
            mo = np.int(tmp[1])
            da = np.int(tmp[2])
            hr = np.int(tmp[3])
            mi = np.int(tmp[4])
            lat = np.float(tmp[6])
            lon = np.float(tmp[7])
            depth = np.float(tmp[8])
            mag = np.float(tmp[13])

            # Create the time object
            d = dt.datetime(yr, mo, da, hr, mi)
            
            # Store things in self 
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.ll2xy()

        # All done
        return

    def selectbox(self, minlon, maxlon, minlat, maxlat, depth=100000.):
        ''' 
        Select the earthquakes in a box defined by min and max, lat and lon.
        
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
        print( "Selecting the earthquakes in the box Lon: %f to %f and Lat: %f to %f"%(minlon, maxlon, minlat, maxlat))
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon) & (self.depth < depth))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.time = self.time[u]
        self.depth = self.depth[u]
        self.mag = self.mag[u]

        # All done
        return

    def selecttime(self, start=[2001, 1, 1], end=[2001, 1, 1]):
        '''
        Selects the earthquake in between two dates. Dates can be datetime.datetime or lists.
        Args:
            * start     : Beginning of the period.
            * end       : End of the period.
        '''

        # check start and end
        if (start.__class__ is float) or (start.__class__ is int) :
            st = dt.datetime(start, 1, 1)
        if (start.__class__ is list):
            if len(start) == 1:
                st = dt.datetime(start[0], 1, 1)
            elif len(start) == 2:
                st = dt.datetime(start[0], start[1], 1)
            elif len(start) == 3:
                st = dt.datetime(start[0], start[1], start[2])
            elif len(start) == 4:
                st = dt.datetime(start[0], start[1], start[2], start[3])
            elif len(start) == 5:
                st = dt.datetime(start[0], start[1], start[2], start[3], start[4])
            elif len(start) == 6:
                st = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5])
        if start.__class__ is dt.datetime:
            st = start

        if (end.__class__ is float) or (end.__class__ is int) :
            ed = dt.datetime(np.int(end), 1, 1)
        if (end.__class__ is list):
            if len(end) == 1:
                ed = dt.datetime(end[0], 1, 1)
            elif len(end) == 2:
                ed = dt.datetime(end[0], end[1], 1)
            elif len(end) == 3:
                ed = dt.datetime(end[0], end[1], end[2])
            elif len(end) == 4:
                ed = dt.datetime(end[0], end[1], end[2], end[3])
            elif len(end) == 5:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4])
            elif len(end) == 6:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5])
        if end.__class__ is dt.datetime:
            ed = end

        # Get values
        print ("Selecting earthquake between %s and %s"%(st.isoformat(),ed.isoformat()))
        u = np.flatnonzero((self.time > st) & (self.time < ed))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.time = self.time[u]
        self.depth = self.depth[u]
        self.mag = self.mag[u] 
                
        # All done
        return  

    def selectmagnitude(self, minimum, maximum=10):
        '''
        Selects the earthquakes between two magnitudes.
        Args:
            * minimum   : Minimum earthquake magnitude wanted.
            * maximum   : Maximum earthquake magnitude wanted.
        '''
        
        # Get the magnitude
        mag = self.mag

        # get indexes
        print ("Selecting earthquake between magnitudes %f and %f"%(minimum, maximum))
        u = np.flatnonzero((self.mag > minimum) & (self.mag < maximum))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.time = self.time[u]
        self.depth = self.depth[u]
        self.mag = self.mag[u] 
                
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

    def GRplot(self, ion=False):
        ''' 
        Plots the Gutemberg-Richter distribution.
        Args:
            * ion       : Turns on the plt.ion().
        '''

        # Get the magnitude
        mag = self.mag

        # ion
        if ion:
            plt.ion()

        # Create a figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Get the histogram
        h, x = np.histogram(self.mag, bins=20)
        x = (x[1:] + x[:-1])/2.

        # Store that somewhere
        self.Histogram = [x, h]

        # plot the values
        ax.semilogy(x, h, '.r', markersize=10, linewidth=1)

        # show to the screen
        plt.show()

        # All done
        return
        
    def distance2trace(self, faults, distance=5.):
        '''
        Selects the earthquakes that are located less than 'distance' km away from a given surface fault trace.
        Args:
            * faults    : list of structures created from verticalfault.
            * distance  : threshold distance.
        '''

        # Import necessary things
        import shapely.geometry as sg

        # Create a list with the earthquakes locations
        LL = np.vstack((self.x, self.y)).T.tolist()

        # Create a MultiPoint object 
        PP = sg.MultiPoint(LL)

        # Loop over faults
        u = []
        for fault in faults:
            dis = []
            # Build a line object
            FF = np.vstack((fault.xf, fault.yf)).T.tolist()
            trace = sg.LineString(FF)
            # Get the distance between each point and this line
            for uu in PP.geoms:
                dis.append(trace.distance(uu))
            dis = np.array(dis)
            # Get the indexes of the ones that are close to the fault
            ut = np.flatnonzero( dis < distance )
            # Fill in u
            for i in ut:
                u.append(i)

        # make u an array
        u = np.array(u)
        u = np.unique(u)

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.time = self.time[u]
        self.depth = self.depth[u]
        self.mag = self.mag[u]
            
        # All done
        return  

    def write2file(self, filename):
        '''
        Write the earthquakes to a file.
        Args:
            * filename      : Name of the output file.
        '''

        # open the file
        fout = open(filename, 'w')

        # Write a header
        fout.write('# Lon | Lat | Depth (km) | Mw \n')

        # Loop over the earthquakes
        for u in range(len(self.lon)):
            fout.write('%f %f %f %f \n'%(self.lon[u], self.lat[u], self.depth[u], self.mag[u]))
        
        # Close the file
        fout.close()

        # all done
        return

