''' 
A class that deals with gps time series.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import sys, os

# Personal
from .timeseries import timeseries
from .SourceInv import SourceInv

class gpstimeseries(SourceInv):

    def __init__(self, name, utmzone=None, verbose=True, lon0=None, lat0=None, ellps='WGS84'):
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
 
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize GPS Time Series {}".format(self.name))
        self.verbose = verbose

        # Base class init
        super(gpstimeseries,self).__init__(name,
                                           utmzone=utmzone,
                                           lon0=lon0,
                                           lat0=lat0, 
                                           ellps=ellps)

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

    def read_from_JPL(self, filename):
        '''
        Reads the time series from a file which has been sent from JPL.
        Format is a bit awkward...
        '''

        # Open, read, close file
        fin = open(filename, 'r')
        Lines = fin.readlines() 
        fin.close()

        # Create values
        time = []
        east = []; north = []; up = []
        stdeast = []; stdnorth = []; stdup = []

        # Read these
        for line in Lines:
            values = line.split()
            time.append(dt.datetime(int(values[11]), 
                                    int(values[12]),
                                    int(values[13]),
                                    int(values[14]),
                                    int(values[15]),
                                    int(values[16])))
            east.append(float(values[1]))
            north.append(float(values[2]))
            up.append(float(values[3]))
            stdeast.append(float(values[4]))
            stdnorth.append(float(values[5]))
            stdup.append(float(values[6]))

        # Initiate some timeseries
        self.north = timeseries('North', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.east = timeseries('East', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.up = timeseries('Up', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

        # Set time
        self.time = np.array(time)
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Set values
        self.north.value = np.array(north)
        self.north.error = np.array(stdnorth)
        self.east.value = np.array(east)
        self.east.error = np.array(stdeast)
        self.up.value = np.array(up)
        self.up.error = np.array(stdup)

        # All done
        return

    def read_from_sql(self, filename, 
                      tables={'e': 'east', 'n': 'north', 'u': 'up'},
                      sigma={'e': 'sigma_east', 'n': 'sigma_north', 'u': 'sigma_up'}):
        '''
        Reads the East, North and Up components of the station in a sql file.
        This follows the organization of M. Simons' group at Caltech.
        Args:
            * filename  : Name of the sql file
        '''

        # Import necessary bits
        try:
            import pandas
            from sqlalchemy import create_engine
        except:
            assert False, 'Could not import pandas or sqlalchemy...'

        # Open the file
        assert os.path.exists(filename), 'File cannot be found'
        engine = create_engine('sqlite:///{}'.format(filename))
        east = pandas.read_sql_table(tables['e'], engine)
        north = pandas.read_sql_table(tables['n'], engine)
        up = pandas.read_sql_table(tables['u'], engine)
        sigmaeast = pandas.read_sql_table(sigma['e'], engine)
        sigmanorth = pandas.read_sql_table(sigma['n'], engine)
        sigmaup = pandas.read_sql_table(sigma['u'], engine)

        # Find the time
        assert (east['DATE'].values==north['DATE'].values).all(), \
                'There is something weird with the timeline of your station'
        ns = 1e-9 # Number of nanoseconds in a second
        self.time = np.array([dt.datetime.utcfromtimestamp(t.astype(int)*ns) \
                              for t in east['DATE'].values])

        # Initiate some timeseries
        self.north = timeseries('North', utmzone=self.utmzone, verbose=self.verbose,
                                lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.east = timeseries('East', utmzone=self.utmzone, verbose=self.verbose,
                               lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.up = timeseries('Up', utmzone=self.utmzone, verbose=self.verbose,
                             lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

        # set time
        self.east.time = self.time
        self.north.time = self.time
        self.up.time = self.time

        # Set the values
        self.north.value = north[self.name].values
        self.north.error = sigmanorth[self.name].values
        self.east.value = east[self.name].values
        self.east.error = sigmaeast[self.name].values
        self.up.value = up[self.name].values
        self.up.error = sigmaup[self.name].values
        
        # All done
        return

    def read_from_caltech(self, filename):
        '''
        Reads the data from a time series file from CalTech (Avouac's group).
        Time is in decimal year...
        '''

        # Open, read, close file
        fin = open(filename, 'r')
        Lines = fin.readlines() 
        fin.close()

        # Create values
        time = []
        east = []; north = []; up = []
        stdeast = []; stdnorth = []; stdup = []

        # Read these
        for line in Lines:
            values = line.split()
            year = np.floor(float(values[0]))
            doy = np.floor((float(values[0])-year)*365.24).astype(int)
            time.append(dt.datetime.fromordinal(dt.datetime(year.astype(int), 1, 1).toordinal() + doy))
            east.append(float(values[1]))
            north.append(float(values[2]))
            up.append(float(values[3]))
            stdeast.append(float(values[4]))
            stdnorth.append(float(values[5]))
            stdup.append(float(values[6]))

        # Initiate some timeseries
        self.north = timeseries('North', utmzone=self.utmzone, 
                                lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.east = timeseries('East', utmzone=self.utmzone, 
                               lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.up = timeseries('Up', utmzone=self.utmzone, 
                             lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

        # Set time
        self.time = np.array(time)
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Set values
        self.north.value = np.array(north)
        self.north.error = np.array(stdnorth)
        self.east.value = np.array(east)
        self.east.error = np.array(stdeast)
        self.up.value = np.array(up)
        self.up.error = np.array(stdup)

        # All done
        return
    
    def removeNaNs(self):
        '''
        Remove NaNs in the time series
        '''

        # Get the indexes
        east = self.east.checkNaNs()
        north = self.north.checkNaNs()
        up = self.north.checkNaNs()

        # check
        enu = np.union1d(east, north)
        enu = np.union1d(enu, up)

        # Remove these guys
        self.east.removePoints(enu)
        self.north.removePoints(enu)
        self.up.removePoints(enu)

        # All done
        return

    def initializeTimeSeries(self, time=None, start=None, end=None, interval=1):
        '''
        Initializes the time series by creating whatever is necessary.
        Args:
            * time              Time vector
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
        if end is not None:
            delta = ed - st
            delta_sec = np.int(np.floor(delta.days * 24 * 60 * 60 + delta.seconds))
            time_step = np.int(np.floor(interval * 24 * 60 * 60))
            self.time = [st + dt.timedelta(0, t) for t in range(0, delta_sec, time_step)]
        if time is not None:
            self.time = time

        # Initialize timeseries instances
        self.north = timeseries('North', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.east = timeseries('East', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.up = timeseries('Up', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

        # Time
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Values
        self.north.value = np.zeros(self.time.shape)
        self.east.value = np.zeros(self.time.shape)
        self.up.value = np.zeros(self.time.shape)

        # Initialize uncertainties
        self.north.error = np.zeros(len(self.time))
        self.east.error = np.zeros(len(self.time))
        self.up.error = np.zeros(len(self.time))

        # All done
        return

    def trimTime(self, start, end=dt.datetime(2100,1,1)):
        '''
        Keeps the epochs between start and end
        '''
        
        # Trim
        self.north.trimTime(start, end=end)
        self.east.trimTime(start, end=end)
        self.up.trimTime(start, end=end)

        # Fix time
        self.time = self.up.time

        # All done
        return

    def addPointInTime(self, time, east=0.0, north=0.0, up = 0.0, std_east=0.0, std_north=0.0, std_up=0.0):
        '''
        Augments the time series by one point.
        time is a datetime object.
        if east, north and up values are not provided, 0.0 is used.
        '''

        # insert
        self.east.addPointInTime(time, value=east, std=std_east)
        self.north.addPointInTime(time, value=north, std=std_north)
        self.up.addPointInTime(time, value=up, std=std_up)
 
        # Time vector
        self.time = self.up.time

        # All done
        return

    def fitFunction(self, function, m0, solver='L-BFGS-B', iteration=1000, tol=1e-8):
        '''
        Fits a function to the timeseries
        Args:
            * function  : Prediction function, 
            * m0        : Initial model
            * solver    : Solver type (see list of solver in scipy.optimize.minimize)
            * iteration : Number of iteration for the solver
            * tol       : Tolerance
        '''

        # Do it for the three components
        self.east.fitFunction(function, m0, solver=solver, iteration=iteration, tol=tol)
        self.north.fitFunction(function, m0, solver=solver, iteration=iteration, tol=tol)
        self.up.fitFunction(function, m0, solver=solver, iteration=iteration, tol=tol)

        # All done
        return

    def fitTidalConstituents(self, steps=None, linear=False, tZero=dt.datetime(2000, 1, 1), 
            chunks=None, cossin=False, constituents='all'):
        '''
        Fits tidal constituents on the time series.
        Args:
            * steps     : list of datetime instances to add step functions in the estimation process.
            * linear    : estimate a linear trend.
            * tZero     : origin time (datetime instance).
            * chunks    : List [ [start1, end1], [start2, end2]] where the fit is performed.
        '''

        # Do it for each time series
        self.north.fitTidalConstituents(steps=steps, linear=linear, tZero=tZero, 
                chunks=chunks, cossin=cossin, constituents=constituents)
        self.east.fitTidalConstituents(steps=steps, linear=linear, tZero=tZero, 
                chunks=chunks, cossin=cossin, constituents=constituents)
        self.up.fitTidalConstituents(steps=steps, linear=linear, tZero=tZero, 
                chunks=chunks, cossin=cossin, constituents=constituents)

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

        # Get offsets
        east = self.east.getOffset(date1, date2, nodate=nodate, data=data)
        north = self.north.getOffset(date1, date2, nodate=nodate, data=data)
        up = self.up.getOffset(date1, date2, nodate=nodate, data=data)

        # all done
        return east, north, up

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
            e = self.east.value[i]
            n = self.north.value[i]
            u = self.up.value[i]
            es = self.east.value[i]
            ns = self.north.value[i]
            us = self.up.value[i]
            fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))
            if steplike:
                e = self.east.value[i+1]
                n = self.north.value[i+1]
                u = self.up.value[i+1]
                es = self.east.error[i+1]
                ns = self.north.error[i+1]
                us = self.up.error[i+1]
                fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))

        t = self.time[i].isoformat()
        e = self.east.value[i]
        n = self.north.value[i]
        u = self.up.value[i]
        es = self.east.error[i]
        ns = self.north.error[i]
        us = self.up.error[i]
        fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))

        # Done 
        fout.close()

        # All done
        return

    def plot(self, figure=1, styles=['.r'], show=True, data='data'):
        '''
        Plots the time series.
        Args:
            figure  :   Figure id number (default=1)
            styles  :   List of styles (default=['.r'])
            show    :   Show to me (default=True)
            data    :   What do you show (data, synth)
        '''

        # list 
        if type(data) is not list:
            data = [data]

        # Create a figure
        fig = plt.figure(figure)

        # Create axes
        axnorth = fig.add_subplot(311)
        axeast = fig.add_subplot(312)
        axup = fig.add_subplot(313)

        # Plot
        self.north.plot(figure=fig, subplot=axnorth, styles=styles, data=data, show=False)
        self.east.plot(figure=fig, subplot=axeast, styles=styles, data=data, show=False)
        self.up.plot(figure=fig, subplot=axup, styles=styles, data=data, show=False)

        # show
        if show:
            plt.show()

        # All done
        return



#EOF
