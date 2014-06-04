
''' 
A class that offers a fit on a time series.

Written by R. Jolivet, June 2014.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import sys


class tidalfit(object):

    def __init__(self, constituents='all', linear=False, steps=None):
        '''
        Initialize a tidalfit object.
        Args:
            * constituents  : List of constituents to use. Can be 'all'.
            * linear        : Include a linear trend (default is False).
            * steps         : List of datetime instances to add step functions in the fit.
        '''
        
        # Tidal Periods in hours (source: wikipedia)
        self.tidePeriodDict = {'M2' : 12.4206012,       # Principal Lunar
                               'S2' : 12.0,             # Principal Solar
                               'N2' : 12.65834751,      # Lunar elliptic
                               'v2' : 12.62600509,      # Larger Lunar evectional 
                               'MU2': 12.8717576,       # Variational
                               '2N2': 12.90537297,      # Lunar elliptical semidiurnal 2nd order
                               'Lambda2': 12.22177348,  # Smaller lunar evectional
                               'T2' : 12.01644934,      # Larger Solar elliptic
                               'R2' : 11.98359564,      # Smaller Solar elliptic
                               'L2' : 12.19162085,      # Smalle lunar elliptic semidiurnal
                               'K2' : 11.96723606,      # Lunisolar
                               'K1' : 23.93447213,      # Lunisolar
                               'O1' : 25.81933871,      # Principal Lunar
                               'OO1': 22.30608083,      # Lunar diurnal
                               'S1' : 24.00000000,      # Solar diurnal
                               'M1' : 24.84120241,      # Smaller lunar elliptic diurnal
                               'J1' : 23.09848146,      # Smaller lunar elliptic diurnal
                               'Rho': 26.72305326,      # Larger lunar evectional diurnal
                               'P1' : 24.06588766,      # Principal Solar
                               'Q1' : 26.868350,        # Elliptic Lunar
                               '2Q1': 28.00621204,      # Larger elliptic diurnal
                               'Mf' : 327.8599387,      # Fortnightly
                               'Msf': 354.3670666,      # Lunisolar synodic fortnightly
                               'Mm' : 661.3111655,      # Monthly
                               'Ssa': 4383.076325,      # Solar semiannual
                               'Sa' : 8766.15265 }      # Solar annual

        # What periods do we want
        if type(constituents) is str:
            if constituents in ('All', 'all', 'ALL'):
                constituents = self.tidePeriodDict.keys()
        self.constituents = constituents

        # How many parameters?
        nStep = 0; nLin = 0
        self.steps = steps
        self.linear = linear
        if steps is not None:
            nStep = len(steps)
        if linear:
            nLin = 1
        nTid = len(periods)
        nParam = nStep + nLin + nTid + 1  # +1 is for the constant
        self.nParam = nParam 

        # All done
        return

    def doFit(self, tZero=dt.datetime(2000, 01, 01), chunks=None):
        '''
        Performs the fit on the chunks of data specified in chunks.
        Args:
            * timeseries: Timeseries instance.
            * tZero     : Sets the origin time (datetime instance).
            * chunks    : if not None, provide a list: [[start1, end1], [start2, end2], ...[startn, endn]].
                          if None, takes all the data.
        '''

        # Sets tZero
        self.tZero = tZero

        # Build G and data
        self.buildG(timeseries, chunks=chunks, linear=self.linear, steps=self.steps, constituents=self.constituents)

        # Solve
        m, res, rank, s = np.linalg.lstsq(self.G, self.data) 

        # Save Linear and Steps
        self.Offset = m[0]
        iP = 1
        if self.linear:
            self.Linear = m[1]
            iP += 1
        if self.steps is not None:
            self.Steps = []
            for step in self.steps:
                self.Steps.append(m[iP])
                iP += 1

        # Save Constituents
        self.Constituents = {}
        for constituent in self.constituents:
            self.Constituents[constituent] = m[iP:iP+2]
            iP += 2

        # All done
        return

    def buildG(timeseries, chunks=None, linear=False, steps=None, constituents='all'):
        ''' 
        Builds the G matrix we will invert.
        Args:
            * timeseries    : Timeseries instance.
            * chunks        : List of chunks of dates [[start1, end1], [start2, end2], ...[startn, endn]]. 
            * linear        : True/False.
            * steps         : List of datetime to add steps.
            * constituents  : List of consituents.
        '''
        
        # Get things
        time = timeseries.time
        value = timeseries.value
        
        # What periods do we want
        if type(constituents) is str:
            if constituents in ('All', 'all', 'ALL'):
                constituents = self.tidePeriodDict.keys()

        # Get the data indexes
        if chunks is None:
            u = range(time.shape[0])
        else:
            u = []
            for chunk in chunks:
                u1 = np.flatnonzero(time>=chunk[0])
                u2 = np.flatnonzero(time<=chunk[1])
                uu = np.interesect1d(u1, u2)
                u.append(uu)
            u = np.array(u).flatten().tolist()

        # How many data
        nData = len(u)
        self.nData = nData

        # Initialize G
        G = np.zeros((self.nData, nParm))

        # Build time and data vectors
        Tvec = np.array([(time[i]-tZero).total_seconds()/(60.*60.*24.) for i in u])  # In Days
        self.data = value[u]

        # Constant term
        G[:,0] = 1.0
        iP = 1

        # Linear?
        if linear:
            G[:,1] = time
            iP += 1

        # Steps?
        if steps is not None:
            for step in steps:
                sline = np.zeros((data.shape[0],))
                u = np.flatnonzero(time>=(step-tZero).total_seconds()/(60.*60.*24.))
                sline[u] = 1.0
                G[:,iP] = sline
                iP += 1

        # Constituents
        periods = []
        for constituent in constituents:
            period = self.tidePeriodDict[constituent]/24.
            G[:,iP] = np.cos(2*np.pi*Tvec/period)
            G[:,iP+1] = np.sin(2*np.pi*Tvec/period)
            periods.append(period)
            iP += 2
        self.periods = periods

        # Save G
        self.G = G

        # All done
        return

    def predict(self, 
