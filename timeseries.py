''' 
A class that deals with time series of one variable.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import sys

# Personal
from .functionfit import functionfit
from .tidalfit import tidalfit
from .SourceInv import SourceInv

class timeseries(SourceInv):

    def __init__(self, name, utmzone=None, verbose=True, lon0=None, lat0=None, ellps='WGS84'):
        '''
        Args:
            * name      : Name of the station.
            * datatype  : can be 'gps' or 'insar' for now.
            * utmzone   : UTM zone. Default is 10 (Western US).
            * verbose   : Speak to me (default=True)
        '''

        # base class ini
        super(timeseries, self).__init__(name, 
                                         utmzone=utmzone,
                                         lon0=lon0, lat0=lat0,
                                         ellps=ellps)

        # Set things
        self.name = name
        self.dtype = 'timeseries'
 
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Time Series {}".format(self.name))
        self.verbose = verbose
    
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

    def readAscii(self, infile, header=0):
        '''
        Reads from an ascii file.
        Format:
        yr mo da hr mi sd value (err)
        '''

        # Read file
        fin = open(infile, 'r')
        Lines = fin.readlines()
        fin.close()

        # Initialize things
        time = []
        value = []
        error = []

        # Loop 
        for i in range(header, len(Lines)):
            tmp = Lines[i].split()
            yr = np.int(tmp[0])
            mo = np.int(tmp[1])
            da = np.int(tmp[2])
            hr = np.int(tmp[3])
            mi = np.int(tmp[4])
            sd = np.int(tmp[5])
            time.append(dt.datetime(yr, mo, da, hr, mi, sd))
            value.append(np.float(tmp[6]))
            if len(tmp)>7:
                error.append(np.float(tmp[7]))
            else:
                error.append(0.0)

        # arrays
        self.time = np.array(time)
        self.value = np.array(value)
        self.error = np.array(error)

        # Sort 
        self.SortInTime()

        # All done
        return

    def checkNaNs(self):
        '''
        Returns the index of the NaNs
        '''

        # All done
        return np.flatnonzero(np.isnan(self.value))

    def removePoints(self, indexes):
        '''
        Removes the points from the time series
        Args:
            * indexes:  Indexes of the poitns to remove
        '''

        self.value = np.delete(self.value, indexes)
        self.error = np.delete(self.error, indexes)
        self.time = np.delete(self.time, indexes)

        # All done
        return

    def SortInTime(self):
        '''
        Sort ascending in time.
        '''

        # argsort
        u = np.argsort(self.time)

        # Sort
        self.time = self.time[u]
        self.value = self.value[u]
        self.error = self.error[u]

        # All done
        return

    def trimTime(self, start, end=dt.datetime(2100, 1, 1)):
        '''
        Keeps the station between start and end.
        start and end are 2 datetime.datetime objects.
        '''

        # Assert
        assert type(start) is dt.datetime, 'Starting date must be datetime.datetime instance'
        assert type(end) is dt.datetime, 'Ending date must be datetime.datetime instance'

        # Get indexes
        u1 = np.flatnonzero(self.time>=start)
        u2 = np.flatnonzero(self.time<=end)
        u = np.intersect1d(u1, u2)

        # Keep'em
        self._keepDates(u)

        # All done
        return

    def addPointInTime(self, time, value=0.0, std=0.0):
        '''
        Augments the time series by one point.
        time is a datetime object.
        if east, north and up values are not provided, 0.0 is used.
        '''

        # Find the index
        u = 0
        t = self.time[u]
        while t<time and u<self.time.shape[0]:
            t = self.time[u]
            u += 1

        # insert
        self.time = np.insert(self.time, u, time)
        self.value = np.insert(self.value, u, value)
        self.error = np.insert(self.error, u, std)
        
        # All done
        return

    def computeDoubleDifference(self):
        '''
        Compute the derivative of the TS with a central difference scheme.
        '''

        # Get arrays
        up = self.value[2:]
        do = self.value[:-2]
        tup = self.time[2:].tolist()
        tdo = self.time[:-2].tolist()

        # Compute
        self.derivative = np.zeros((self.time.shape[0],))
        timedelta = np.array([(tu-td).total_seconds() for tu,td in zip(tup, tdo)])
        self.derivative[1:-1] = (up - do)/timedelta

        # First and last
        self.derivative[0] = (self.value[1] - self.value[0])/(self.time[1] - self.time[0]).total_seconds()
        self.derivative[-1] = (self.value[-2] - self.value[-1])/(self.time[-2] - self.time[-1]).total_seconds()

        # All Done
        return

    def smoothGlitches(self, biggerThan=999999., smallerThan=-999999., interpNum=5, interpolation='linear'):
        '''
        Removes the glitches and replace them by a value interpolated on interpNum points.
        Args:
            * biggerThan    : Values higher than biggerThan are glitches.
            * smallerThan   : Values smaller than smallerThan are glitches.
            * interpNum     : Number of points to take before and after the glicth to predict its values.
            * interpolation : Interpolation method.
        '''

        # Find glitches
        u = np.flatnonzero(self.value>biggerThan)
        d = np.flatnonzero(self.value<smallerThan)
        g = np.union1d(u,d).tolist()

        # Loop on glitches
        while len(g)>0:
            
            # Get index
            iG = g.pop()

            # List
            iGs = [iG]

            # Check next ones
            go = False
            if len(g)>0:
                if (iG-g[-1]<interpNum):
                    go = True
            while go:
                iG = g.pop()
                iGs.append(iG)
                go = False
                if len(g)>0:
                    if (iG-g[-1]<interpNum):
                        go = True

            # Sort
            iGs.sort()

            # Make a list of index to use for interpolation
            iMin = max(0, iGs[0]-interpNum)
            iMax = min(iGs[-1]+interpNum+1, self.value.shape[0])
            iIntTmp = range(iMin, iMax)
            iInt = []
            for i in iIntTmp:
                if i not in iGs:
                    iInt.append(i)
            iInt.sort()
            
            # Build the interpolator
            time = np.array([(self.time[t]-self.time[iInt[0]]).total_seconds() for t in iInt])
            value = np.array([self.value[t] for t in iInt])
            interp = sciint.interp1d(time, self.value[iInt], kind=interpolation)

            # Interpolate
            self.value[iGs] = np.array([interp((self.time[t]-self.time[iInt[0]]).total_seconds()) for t in iGs])

        # All done
        return

    def removeMean(self, start=None, end=None):
        '''
        Removes the mean between start and end.
        start and end are two instance of datetime.datetime.
        '''

        # Start end
        if start is None:
            start = self.time[0]
        if end is None:
            end = self.time[-1]

        # Get index
        u1 = np.flatnonzero(self.time>=start)
        u2 = np.flatnonzero(self.time<=end)
        u = np.intersect1d(u1, u2)

        # Get Mean
        mean = np.nanmean(self.value[u])

        # Correct
        self.value -= mean

        # All Done
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

        # Do the fit
        fit = functionfit(function, verbose=self.verbose)
        fit.doFit(self, m0, solver=solver, iteration=iteration, tol=tol)

        # Do the prediction
        fit.predict(self)

        # Save
        self.m = fit.m

        # All done
        return

    def fitTidalConstituents(self, steps=None, linear=False, tZero=dt.datetime(2000, 1, 1), chunks=None, cossin=False, constituents='all'):
        '''
        Fits tidal constituents on the time series.
        Args:
            * steps     : list of datetime instances to add step functions in the estimation process.
            * linear    : estimate a linear trend.
            * tZero     : origin time (datetime instance).
            * chunks    : List [ [start1, end1], [start2, end2]] where the fit is performed.
        '''

        # Initialize a tidalfit
        tf = tidalfit(constituents=constituents, linear=linear, steps=steps, cossin=cossin)

        # Fit the constituents
        tf.doFit(self, tZero=tZero, chunks=chunks)

        # Predict the time series
        if steps is not None:
            sT = True
        else:
            sT = False
        tf.predict(self,constituents=constituents, linear=linear, steps=sT, cossin=cossin)

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
            value = self.value
        elif data in ('std'):
            value = self.error

        # all done
        return value[u2] - value[u1]

    def write2file(self, outfile, steplike=False):
        '''
        Writes the time series to a file.
        Args:   
            * outfile   : output file.
            * steplike  : doubles the output each time so that the plot looks like steps.
        '''

        # Open the file
        fout = open(outfile, 'w')
        fout.write('# Time | value | std \n')

        # Loop over the dates
        for i in range(len(self.time)-1):
            t = self.time[i].isoformat()
            e = self.value[i]
            es = self.std[i]
            fout.write('{} {} {} \n'.format(t, e, es))
            if steplike:
                e = self.value[i+1]
                es = self.std[i+1]
                fout.write('{} {} {} \n'.format(t, e, es))

        t = self.time[i].isoformat()
        e = self.value[i]
        es = self.std[i]
        fout.write('{} {} {} \n'.format(t, e, es))

        # Done 
        fout.close()

        # All done
        return

    def findZeroIntersect(self, data='data'):
        '''
        Returns all the points just before the function crosses 0.
        Args:
            * data      : Can be 'data', 'synth' or 'derivative'.
        '''

        # Get the good data
        if data is 'data':
            v = self.value
        elif data is 'synth':
            v = self.synth
        elif data is 'derivative':
            v = self.derivative

        # List 
        indexes = []

        # Loop
        for i in xrange(len(v)-1):
            if (v[i]>0. and v[i+1]<0.) or (v[i]<0. and v[i+1]>0.):
                indexes.append(i)

        # All done
        return indexes

    def plot(self, figure=1, styles=['.r'], show=True, data='data', subplot=None):
        '''
        Plots the time series.
        Args:
            figure  :   Figure id number (default=1)
            styles  :   List of styles (default=['.r'])
            show    :   Show to me (default=True)
            data    :   can be 'data', 'derivative', 'synth'
                        or a list of those
        '''

        # Get values
        if type(data) is str:
            data = [data]

        # iterate
        values = []
        for d in data:
            if d in ('data'):
                v = self.value
            elif d in ('derivative'):
                v = self.derivative
            elif d in ('synth'):
                v = self.synth
            elif d in ('res'):
                v = self.value-self.synth
            else:
                print('Unknown component to plot')
                return
            values.append(v)

        # Create a figure
        if (figure=='new') or type(figure) is int:
            fig = plt.figure(figure)
        else:
            fig = figure

        # Create axes
        if subplot is not None:
            ax = subplot
        else:
            ax = fig.add_subplot(111)

        # Plot ts
        for v,style in zip(values, styles):
            u = np.argsort(self.time)
            ax.plot(self.time[u], v[u], style)

        # show
        if show:
            plt.show()

        # All done
        return

    def _keepDates(self, u):
        '''
        Keeps the dates corresponding to index u.
        '''

        self.time = self.time[u]
        self.value = self.value[u]
        self.error = self.error[u]
        if hasattr(self, 'synth'):
            self.synth = self.synth[u]

        # All done
        return

#EOF
