''' 
A class that deals with gps time series.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import sys

class gpstimeseries:

    def __init__(self, name, utmzone='10', verbose=True):
        '''
        Args:
            * name      : Name of the station.
            * datatype  : can be 'gps' or 'insar' for now.
            * utmzone   : UTM zone. Default is 10 (Western US).
            * verbose   : Speak to me (default=True)
        '''

        # Set things
        self.name = name
        self.dtype = 'gpstimeseries'
        self.utmzone = utmzone
 
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize GPS Time Series {}".format(self.name))

        # Create a utm transformation
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')

        # All done
        return

    def lonlat2xy(self):
        '''
        Pass the position into the utm coordinate system.
        '''

        x, y = self.putm(self.lon, self.lat)
        self.x = x/1000.
        self.y = y/1000.

        # All done
        return

    def xy2lonlat(self):
        '''
        Pass the position from utm to lonlat.
        '''

        lon, lat = self.putm(x*1000., y*1000.)
        self.lon = lon
        self.lat = lat

        # all done
        return

    def initializeTimeSeries(self, start, end, interval=1):
        '''
        Initializes the time series by creating whatever is necessary.
        Args:
            * starttime:        Begining of the time series.
            * endtime:          End of the time series.
            * interval:         In days.
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

        # Initialize a time vector
        delta = ed - st
        delta_sec = np.int(np.floor(delta.days * 24 * 60 * 60 + delta.seconds))
        time_step = np.int(np.floor(interval * 24 * 60 * 60))
        self.time = [st + dt.timedelta(0, t) for t in range(0, delta_sec, time_step)]

        # Initialize position vectors
        self.north = np.zeros(len(self.time))
        self.east = np.zeros(len(self.time))
        self.up = np.zeros(len(self.time))

        # Initialize uncertainties
        self.std_north = np.zeros(len(self.time))
        self.std_east = np.zeros(len(self.time))
        self.std_up = np.zeros(len(self.time))

        # All done
        return

    def addPointInTime(self, time, east=0.0, north=0.0, up = 0.0, std_east=0.0, std_north=0.0, std_up=0.0):
        '''
        Augments the time series by one point.
        time is a datetime object.
        if east, north and up values are not provided, 0.0 is used.
        '''

        # Find the index
        u = 0
        t = self.time[u]
        while t<time:
            u += 1
            t = self.time[u]

        # insert
        self.time.insert(u, time)
        self.east = np.insert(self.east, u, east)
        self.north = np.insert(self.north, u, north)
        self.up = np.insert(self.up, u, up)
        self.std_east = np.insert(self.std_east, u, std_east)
        self.std_north = np.insert(self.std_north, u, std_north)
        self.std_up = np.insert(self.std_up, u, std_up)
        
        # All done
        return

    def getOffset(self, date1, date2, nodate=np.nan, data='data'):
        '''
        Get the offset between date1 and date2.
        If the 2 dates are not available, returns NaN.
        Args:
            date1       : datetime object
            date2       : datetime object
            data        : can be 'data' or 'std'
        '''

        # Get the indexes
        u1 = np.flatnonzero(np.array(self.time)==date1)
        u2 = np.flatnonzero(np.array(self.time)==date2)

        # Check
        if len(u1)==0:
            return nodate, nodate, nodate
        if len(u2)==0:
            return nodate, nodate, nodate

        # Select 
        if data in ('data'):
            east = self.east
            north = self.north
            up = self.up
        elif data in ('std'):
            east = self.std_east
            north = self.std_north
            up = self.std_up

        # all done
        return east[u2]-east[u1], north[u2]-north[u1], up[u2]-up[u1]

    def write2file(self, outfile, steplike=False):
        '''
        Writes the time series to a file.
        Args:   
            * outfile   : output file.
            * steplike  : doubles the output each time so that the plot looks like steps.
        '''

        # Open the file
        fout = open(outfile, 'w')
        fout.write('# Time | east | north | up | east std | north std | up std \n')

        # Loop over the dates
        for i in range(len(self.time)-1):
            t = self.time[i].isoformat()
            e = self.east[i]
            n = self.north[i]
            u = self.up[i]
            es = self.std_east[i]
            ns = self.std_north[i]
            us = self.std_up[i]
            fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))
            if steplike:
                e = self.east[i+1]
                n = self.north[i+1]
                u = self.up[i+1]
                es = self.std_east[i+1]
                ns = self.std_north[i+1]
                us = self.std_up[i+1]
                fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))

        t = self.time[i].isoformat()
        e = self.east[i]
        n = self.north[i]
        u = self.up[i]
        es = self.std_east[i]
        ns = self.std_north[i]
        us = self.std_up[i]
        fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))

        # Done 
        fout.close()

        # All done
        return

    def plot(self, figure=1, styles=['.r'], show=True):
        '''
        Plots the time series.
        Args:
            figure  :   Figure id number (default=1)
            styles  :   List of styles (default=['.r'])
            show    :   Show to me (default=True)
        '''

        # Create a figure
        fig = plt.figure(figure)

        # Create axes
        axnorth = fig.add_subplot(311)
        axeast = fig.add_subplot(312)
        axup = fig.add_subplot(313)

        # Plot ts
        for style in styles:
            axnorth.plot(self.time, self.north, style)
            axeast.plot(self.time, self.east, style)
            axup.plot(self.time, self.up, style)

        # show
        if show:
            plt.show()

        # All done
        return


#EOF
