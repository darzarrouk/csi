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

    def __init__(self,name,dtype='tsunami',utmzone=None,ellps='WGS84'):
        '''
        Args:
            * name      : Name of the dataset.
            * dtype     : data type (optional, default='seismic')
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        super(self.__class__,self).__init__(name,utmzone,ellps)

        # Initialize the data set
        self.dtype = dtype

        # Data
        self.d   = []
        self.Cd  = None
        self.sta = None
        self.lat = None
        self.lon = None
        self.t0  = None
        self.G = None

        # All done
        return

    def readFromTxtFile(self,filename,factor=1.0,fileinfo=None):
        '''
        Read d, Cd from files filename.d filename.Cd
        '''

        self.Cd = np.loadtxt(filename+'.Cd')*factor*factor
        self.d  = np.loadtxt(filename+'.data')*factor
        self.sta = open(filename+'.id').readlines()
        if fileinfo is not None:
            f = open(fileinfo,'rt')
            self.lon = []            
            self.lat = []
            self.t0  = []
            for l in f:
                items = list(map(float,l.strip().split()[1:]))
                self.lon.append(items[0])
                self.lat.append(items[1])
                self.t0.append(items[2])
            f.close()
            assert len(self.t0)==len(self.sta)
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


    def plot(self, nobs_per_trace, plot_synth=False,alpha=1.,figsize=(13,10),left=0.07,bottom=0.1,
             right=0.99,top=0.9,wspace=0.31,hspace=0.47,scale=100.,ylim=None,yticks=None):
        '''
        Plot tsunami traces
        '''
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(bottom=bottom,top=top,left=left,right=right,wspace=wspace,hspace=hspace)
        nsamp = nobs_per_trace
        nstat = len(self.d)/nobs_per_trace
        print(nstat)
        for i in range(nstat): 
            data  = self.d[i*nsamp:nsamp*i+nsamp]
            if len(self.synth.shape)==2:
                synth = self.synth[i*nsamp:nsamp*i+nsamp,:]
            else:
                synth = self.synth[i*nsamp:nsamp*i+nsamp]
            plt.subplot(2,np.ceil(nstat/2.),i+1)
            t = np.arange(len(synth))
            if self.t0 is not None:
                t += self.t0[i]
            plt.plot(t,synth*scale,'r',alpha=alpha)            
            plt.plot(t,data*scale,'k')
            #plt.grid()
            plt.title(self.sta[i])
            if not i%np.ceil(nstat/2.):
                plt.ylabel('Water height, cm')
            if i>=nstat/2:
                if self.t0 is not None:
                    plt.xlabel('Time, min')
                else:
                    plt.xlabel('Time since arrival, min')
            if ylim is not None:
                plt.ylim(ylim[0],ylim[1])
            if yticks is not None:
                plt.yticks(yticks)    
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
