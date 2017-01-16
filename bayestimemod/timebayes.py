# Main class for the interpolation problem

# imports
import numpy as np
import sys, gc, os
import copy
from .resample import resample

class timebayes(object):

    def __init__(self, data, time, sigma, dt, bounds, nsamples=1000):
        '''
        Initialization of a timebayes instance.
        This class solves the interpolation problem 
        in a Bayesian framework iteratively.

        Args:
            * data      : vector of data to fit
            * time      : time vector
            * sigma     : noise (1 value, std)
            * dt        : delta-time between the interpolating functions
            * bounds    : bounds of the uniform prior PDF
            * nsamples  : number of samples
        '''

        # Init
        self.data = data
        self.time = time
        self.sigma = sigma
        self.dt = dt
        self.bounds = bounds
        self.nsamples = nsamples
        
        # Create the mpi framework
        import mpi4py
        from mpi4py import MPI

        # Store
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.me = MPI.COMM_WORLD.Get_rank()

        # All done
        return

    def finalize(self):
        '''
        Kill MPI workers
        '''
        self.MPI.Finalize()
        # All done
        return
        
    def generateInitialSample(self):
        '''
        Generate the initial set of samples for the first step, 
        given a uniform prior (just samples the prior).
        '''

        return np.random.rand(self.nsamples)*(self.bounds[1]-self.bounds[0]) + self.bounds[0]


    def initializePredFunction(self,triTimes,times,h=None):
        '''
        Initialize the prediction function.

        Args:
             * triTimes: triangle center times
             * triTimes: observation times
        '''

        # Check h
        if h is None:
            h = triTimes[1]-triTimes[0]

        # Do stuff
        def predict(alphas):
            '''
            Linear interpolation using triangular base functions
            Args:
                - alphas: amplitude of each triangle
            '''
            # Linear interpolation
            p = np.zeros(times.shape)
            for i in range(len(triTimes)):
                dt = np.abs(times - triTimes[i])
                j = np.where(dt<h)
                p[j] += alphas[i] * (1-dt[j]/h)
            # All done
            return p

        # Save the prediction function
        self.fpred = predict

        # All done
        return

    def initializeTriangleMap(self):
        '''
        Initialize a map linking data time steps to the index of the preceeding triangle
        '''
        self.triangleMap = np.array(list(map(int,self.time/self.dt)))

        # All done
        return


    def oneTimeStep(self, step):
        '''
        Does the posterior sampling for a new time step.

        Args:
            * step      : index of the time step.
        '''

        import time as pouet

        # Identify the data
        triIndex  = self.triangleMap[step] # Index of the previous triangle
        ti = triIndex*self.dt                     # Time of the previous triangle
        di = np.where(np.logical_and(self.time>=ti-self.dt,self.time<=self.time[step]))
        data = self.data[di]
        time = self.time[di]
        
        # Identify the samples to update/create and the fixed ones
        fixed   = np.zeros((self.nsamples,))
        samples = np.zeros((self.nsamples,2))
        if self.me==0:
            if step == 0:
                fixed = np.zeros((self.nsamples,))
                samples[:,0] = self.generateInitialSample()
                samples[:,1] = self.generateInitialSample()
            else:
                fixed = np.array(self.samples[triIndex-1])
                samples[:,0] = self.samples[triIndex]
                if triIndex > self.triangleMap[step-1]:
                    samples[:,1] = self.generateInitialSample()
                else:
                    samples[:,1] = self.samples[triIndex+1]
 
        # Create a prediction function
        triTimes = np.arange(triIndex-1,triIndex+2)*self.dt        
        self.initializePredFunction(triTimes,time,h=triTimes[1]-triTimes[0])
        
        # Split
        splitSamples = _split_seq(samples, self.comm.Get_size())
        splitFixed = _split_seq(fixed, self.comm.Get_size())

        # Send to each worker
        if self.me==0:
            for worker in range(self.comm.Get_size()):
                self.comm.Send(splitSamples[worker], dest=worker, tag=2*worker)
                self.comm.Send(splitFixed[worker], dest=worker, tag=2*worker+1)

        # Create holders of the right size
        subsamples = np.zeros((splitSamples[self.me].shape[0],2))
        subfixed = np.zeros((splitFixed[self.me].shape[0],))

        # Wait for everybody
        self.comm.Barrier()

        # Receive
        self.comm.Recv(subsamples, source=0, tag=2*self.me)
        self.comm.Recv(subfixed, source=0, tag=2*self.me+1)

        # Clean up
        del samples, fixed
        gc.collect()

        # Walk the chains in each worker
        sampler = resample(data, self.sigma, time, 
                           subsamples, subfixed, self.bounds, 
                           self.fpred, self.comm, niter=self.chainlength)
        subsamples = sampler.sample()

        # Send to master
        self.comm.send(subsamples, dest=0, tag=self.me+20)
        del subsamples
        gc.collect()

        # Update/Append samples
        if self.me==0:
            alpha1 = []; alpha2 = []
            for worker in range(self.comm.Get_size()):
                newsamples = self.comm.recv(source=worker, tag=worker+20)
                alpha1 += newsamples[0]
                alpha2 += newsamples[1]
            del newsamples
            gc.collect()
        else:
            alpha1   = None
            alpha2   = None
        
        # All done
        return alpha1, alpha2, triIndex

    def walkWithTime(self, chainLength=1000):
        '''
        Advance throught time.
        '''

        # Model sample set
        self.samples = []

        # Create data <-> triangle map
        self.initializeTriangleMap()

        # Chain length 
        self.chainlength = chainLength

        # Iterate over the data
        for step in range(self.data.size):
            # Print stuff
            if self.me==0:
                sys.stdout.write('\r Time Step {} / {}'.format(step, 
                                    self.data.size))
                sys.stdout.flush()
            # Walk one step
            alpha1,alpha2,triIndex = self.oneTimeStep(step)
            # Update sample set
            if alpha1 is not None:
                if triIndex>len(self.samples)-1:
                    self.samples.append(alpha1)
                else:
                    self.samples[triIndex] = copy.deepcopy(alpha1)
                if triIndex+1>len(self.samples)-1:
                    self.samples.append(alpha2)
                else:
                    self.samples[triIndex+1] = copy.deepcopy(alpha2)
            # Collect the garbage
            gc.collect()
        
        print('All done')

        # All done
        return
            
    def plot(self):
        '''
        Plot data and results if there is some
        '''
        if not self.me:
            import matplotlib.pyplot as plt
            # Triangle center times
            triTimes = np.arange(len(self.samples))*self.dt
            # Initialize prediction function
            self.initializePredFunction(triTimes,self.time,h=triTimes[1]-triTimes[0])
            # Get stochastic predictions
            preds = []
            for amplitudes in np.array(self.samples).T:
                preds.append(self.fpred(amplitudes))
            # Plot them all
            plt.plot(self.time,np.array(preds).T,'0.75')
            plt.plot(self.time,self.data,'ko-')
            plt.show()
        # All done
        return
                
        
        
# List splitter
def _split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq
