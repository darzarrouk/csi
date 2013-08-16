'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt

class creepmeters(object):

    def __init__(self, name, utmzone='10'):
        '''
        Args:
            * name          : Name of the Seismic dataset.
            * utmzone       : UTM zone. Default is 10 (Western US).
        '''

        # Initialize the data set 
        self.name = name
        self.utmzone = utmzone
        self.dtype = 'creepmeters'

        print ("---------------------------------")
        print ("---------------------------------")
        print (" Initialize Creepmeters data set {}".format(self.name))

        # Initialize the UTM transformation
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

        # Initialize some things
        self.data = {}

        # All done
        return

    def readStationList(self, filename):
        '''
        Reads the list of Stations.
        Args:
            filename        : Input file.
        '''

        # open the file
        fin = open(filename, 'r')

        # Read all lines
        Text = fin.readlines()
        fin.close()

        # Create lists
        self.station = []
        self.lon = []
        self.lat = []

        # Loop
        for t in Text:
            tex = t.split()
            self.station.append(tex[0])
            self.lon.append(np.float(tex[1]))
            self.lat.append(np.float(tex[2]))

        # translate to array
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)

        # Convert to utm
        self.ll2xy()

        # All done
        return

    def readStationData(self, station, directory='.'):
        '''
        From the name of a station, reads what is in station.day.
        '''

        # Create the storage
        self.data[station] = {}
        self.data[station]['Time'] = []
        self.data[station]['Offset'] = []
        t = self.data[station]['Time']
        o = self.data[station]['Offset']

        # open the file
        filename = '{}/{}.day'.format(directory,station)
        fin = open(filename, 'r')

        # Read everything in it
        Text = fin.readlines()
        fin.close()

        # Loop 
        for text in Text:

            # Get values
            yr = np.int(text.split()[0])
            da = np.int(text.split()[1])
            of = np.float(text.split()[2])

            # Compute the time 
            time = dt.datetime.fromordinal(dt.datetime(yr, 1, 1).toordinal() + da)

            # Append
            t.append(time)
            o.append(of)

        # Arrays
        self.data[station]['Time'] = np.array(self.data[station]['Time'])
        self.data[station]['Offset'] = np.array(self.data[station]['Offset'])

        # All done
        return

    def selectbox(self, minlon, maxlon, minlat, maxlat):
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
        print( "Selecting the earthquakes in the box Lon: {} to {} and Lat: {} to {}".format(minlon, maxlon, minlat, maxlat))
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.station = sellf.station[u]

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

    def fitLinear(self, station, period=None, directory='.'):
        '''
        Fits a linear trend onto the offsets for the station 'station'.
        Can specify a period=[startdate, enddate].
        ex:     fitLinear('xva1', period=[(2006,01,23), (2007,12,31)])
        '''

        # Check if the station has been read before
        if not (station in self.data.keys()):
            self.readStationData(station, directory=directory)

        # Creates a storage
        self.data[station]['Linear'] = {}
        store = self.data[station]['Linear']

        # Create the dates
        if period is None:
            date1 = self.data[station]['Time'][0]
            date2 = self.data[station]['Time'][1]
        else:
            date1 = dt.datetime(period[0][0], period[0][1], period[0][2])
            date2 = dt.datetime(period[1][0], period[1][1], period[1][2])

        # Keep the period
        store['Period'] = []
        store['Period'].append(date1)
        store['Period'].append(date2)

        # Get the data we want
        time = self.data[station]['Time']
        offset = self.data[station]['Offset']

        # Get the dates we want
        u = np.flatnonzero(time>=date1)
        v = np.flatnonzero(time<=date2)
        w = np.intersect1d(u,v)
        ti = time[w]
        of = offset[w]

        # pass the dates into real numbers
        tr = np.array([self.date2real(ti[i]) for i in range(ti.shape[0])])

        # Make an array
        A = np.ones((tr.shape[0], 2))
        A[:,0] = tr

        # invert
        m = np.dot( np.dot( np.linalg.inv(np.dot(A.T, A)), A.T ), of)

        # Stores the results
        store['Fit'] = m

        # all done
        return

    def plotStation(self, station, figure=100, save=None):
        '''
        Plots one station evolution through time.
        '''

        # Check if the station has been read
        if not (station in self.data.keys()):
            print('Read the data first...')
            return

        # Create figure
        fig = plt.figure(figure)
        ax = fig.add_subplot(111)

        # Title
        ax.set_title(station)

        # plot data
        t = self.data[station]['Time']
        o = self.data[station]['Offset']
        ax.plot(t, o, '.k')

        # plot fit
        if 'Linear' in self.data[station].keys():
            v = self.data[station]['Linear']['Fit'][0]
            c = self.data[station]['Linear']['Fit'][1]
            date1 = self.data[station]['Linear']['Period'][0]
            date2 = self.data[station]['Linear']['Period'][1]
            dr1 = self.date2real(date1)
            dr2 = self.date2real(date2)
            plt.plot([date1, date2],[c+dr1*v, c+dr2*v], '-r')

        # save
        if save is not None:
            plt.savefig(save)

        # Show
        plt.show()

        # All done 
        return


    def date2real(self, date):
        '''
        Pass from a datetime to a real number.
        '''

        yr = date.year
        yrordi = dt.datetime(yr, 1, 1).toordinal()
        ordi = date.toordinal()
        days = ordi - yrordi
        return yr + days/365.25

