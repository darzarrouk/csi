# Main class for the interpolation problem

# imports
import numpy as np

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
        self.me = MPI.COMM_WORL.Get_rank()

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

        # Do stuff
        def predict(alphas):
            '''
            Linear interpolation using triangular base functions
            Args:
            - triTimes: time at the center of each triangle
            - alphas: amplitude of each triangle
            - times: observation times
            - h: triangle half-width (if not specified, we assume h=triTimes[1]-triTimes[0])
            '''
            # Triangle half-width
            if h==None:
                h = triTimes[1]-triTimes[0]
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

        # Identify the data
        triIndex  = self.triangleMap[step] # Index of the previous triangle
        ti = triIndex*self.dt                     # Time of the previous triangle
        di = np.where(np.logical_and(self.time>=ti-self.dt,self.time<=self.time[step]))
        data = self.data[di]
        time = self.time[di]
        
        # Identify the samples to update/create and the fixed ones
        fixed   = None
        samples = None
        if not self.me:
            samples = np.zeros((self.nsamples,2))            
            if step == 0:
                fixed = np.zeros((self.nsamples,))
                samples[:,0] = self.generateInitialSample()
                samples[:,1] = self.generateInitialSample()
            else:
                fixed = self.samples[triIndex-1].copy()
                samples[:,0] = self.samples[:,triIndex].copy()
                if triIndex > self.triangleMap[step-1]:
                    samples[:,1] = self.generateInitialSample()
                else:
                    samples[:,1] = self.samples[:,triIndex+1].copy()
                
        # Create a prediction function
        triTimes = np.arange(triIndex-1,triIndex+2)*self.dt        
        self.initializePredFunction(triTimes,time,h=triTimes[1]-triTimes[0])
        
        # Split the samples in as many workers 
        if self.me==0:
            
            # Split
            splitSamples = _split_seq(samples, self.comm.Get_size())
            splitFixed = _split_seq(fixed, self.comm.Get_size())

            # Iterate over the workers
            for worker in range(self.comm.Get_size()):

                # Send the packages
                self.comm.send([splitSamples[worker], splitFixed[worker]], 
                               dest=worker, tag=worker+10)

        # Wait for everybody
        self.comm.Barrier()

        # Receive
        subsamples,subfixed= self.comm.recv(source=0, tag=self.me+10)

        # Walk the chains in each worker
        sampler = resample(data, self.sigma, time, 
                           subsamples, subfixed, self.bounds, 
                           self.fpred, self.comm)
        subsamples = sampler.sample()

        # Send to master
        self.comm.send(subsamples, dest=0, tag=self.me+20)
        
        # Update/Append samples
        if self.me==0:
            alpha1 = []; alpha2 = []
            for worker in range(self.comm.Get_size()):
                newsamples = self.comm.recv(source=worker, tag=worker+20)
                alpha1 += newsamples[0]
                alpha2 += newsamples[1]
        else:
            alpha1   = None
            alpha2   = None
                
        # All done
        return alpha1, alpha2, triIndex

    def walkWithTime(self):
        '''
        Advance throught time.
        '''

        # Model sample set
        self.samples = []

        # Create data <-> triangle map
        self.initializeTriangleMap()

        # Iterate over the data
        for step in range(self.data.size):
            # Walk one step
            alpha1,alpha2, triIndex = self.oneTimeStep(step)
            # Update sample set
            if alpha1 is not None:
                if triIndex>len(self.samples)-1:
                    self.samples.append(alpha1)
                else:
                    self.samples[triIndex] = alpha1.copy()
                if triIndex+1>len(self.samples)-1:
                    self.samples.append(alpha2)
                else:
                    self.samples[triIndex+1] = alpha2.copy()
        
        # All done
        return
            
    def plot(self):
        '''
        Plot data and results if there is some
        '''
        if not self.me:
            # Triangle center times
            triTimes = np.arange(self.samples.shape[1])*self.dt
            # Initialize prediction function
            self.initializePredFunction(triTimes,self.time,h=triTimes[1]-triTimes[0])
            # Get stochastic predictions
            preds = []
            for i in range(self.samples.shape[0]):
                preds.append(self.fpred(self.samples[i,:]))
            # Plot them all
            plt.plot(self.time,preds,'0.75')
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
