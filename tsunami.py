'''
A class that deals with seismic or high-rate GPS data (not finished)

Written by Z. Duputel, April 2013.
'''

# Externals
import os
import sys
import copy
import shutil
import numpy  as np
import pyproj as pp
import matplotlib.pyplot as plt


# Personals
#xfrom WaveMod    import sac
from .SourceInv import SourceInv

class tsunami(SourceInv):

    def __init__(self,name,dtype='tsunami',utmzone=None,ellps='WGS84', lon0=None, lat0=None):
        '''
        Args:
            * name      : Name of the dataset.
            * dtype     : data type (optional, default='seismic')
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(tsunami,self).__init__(name,utmzone=utmzone,ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the data set
        self.dtype = dtype

        # Data
        self.d   = []
        self.Cd  = None
        self.sta = None

        self.G = None

        # All done
        return

    def readFromTxtFile(self,filename,factor=1.0):
        '''
        Read d, Cd from files filename.d filename.Cd
        '''

        self.Cd = np.loadtxt(filename+'.Cd')*factor*factor
        self.d  = np.loadtxt(filename+'.data')*factor
        self.sta = open(filename+'.id').readlines()

        # All done
        return

    def getGF(self,filename,fault,factor=1.0):
        '''
        Read GF from file filename.gf
        returns GF_SS and GF_DS
        '''
        GF = np.loadtxt(filename+'.gf')*factor
        n  = GF.shape[1]/2
        assert n == len(fault.slip), 'Incompatible tsunami GF size'
        GF_SS = GF[:,:n]
        GF_DS = GF[:,n:]

        #  All done
        return GF_SS, GF_DS

    def buildsynth(self, faults, direction='sd'):
        '''
        Takes the slip model in each of the faults and builds the synthetic displacement using the Green's functions.
        Args:
            * faults        : list of faults to include.
            * direction     : list of directions to use. Can be any combination of 's', 'd' and 't'.
        '''

        Nd = len(self.d)

        # Clean synth
        self.synth = np.zeros(self.d.shape)

        for fault in faults:

            # Get the good part of G
            G = fault.G[self.name]

            if ('s' in direction) and ('strikeslip' in G.keys()):
                Gs = G['strikeslip']
                Ss = fault.slip[:,0]
                self.synth += np.dot(Gs,Ss)
            if ('d' in direction) and ('dipslip' in G.keys()):
                Gd = G['dipslip']
                Sd = fault.slip[:,1]
                self.synth += np.dot(Gd, Sd)

        # All done
        return


    def plot(self, nobs_per_trace, plot_synth=False):
        '''
        Plot tsunami traces
        '''
        fig = plt.figure(figsize=(13,10))
        nsamp = nobs_per_trace
        nstat = len(self.d)/nobs_per_trace
        for i in range(nstat): 
            data  = self.d[i*nsamp:nsamp*i+nsamp]
            synth = self.synth[i*nsamp:nsamp*i+nsamp]
            plt.subplot(2,nstat/2,i+1)
            plt.plot(data,'k')
            plt.plot(synth,'r')
            plt.title(self.sta[i])
            if not i%2:
                plt.ylabel('Water height, m')
            if i>=nstat/2:
                plt.xlabel('Time since arrival, min')
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # All done
        return


    def write2file(self, namefile, data='synth'):
        '''
        Plot tsunami traces
        '''
        if data == 'synth':
            np.savetxt(namefile, self.synth.T)
        elif data == 'data':
            np.savetxt(namefile, self.d.T)

        # All done
        return
#EOF
