# Main class for the interpolation problem

# imports
import numpy as np

class timebayes(object):

    def __init__(self, data, time, sigma, dt):
        '''
        Initialization of a timebayes instance.
        This class solves the interpolation problem 
        in a Bayesian framework iteratively.

        Args:
            * data      : vector of data to fit
            * time      : time vector
            * sigma     : noise (1 value, std)
            * dt        : delta-time between the interpolating functions
        '''

        # Init
        self.data = data
        self.time = time
        self.sigma = sigma
        self.dt = dt
        
        # Create the mpi framework
        import mpi4py
        from mpi4py import MPI

        # Store
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.me = MPI.COMM_WORL.Get_rank()

        # All done
        return

    def generateInitialSample(self, bounds, nSamples=1000):
        '''
        Generate the initial set of samples for the first step, 
        given a uniform prior (just samples the prior).

        Args:
            * bounds    : Bounds of the uniform pdf
            * nSamples  : Number of samples to generate (default: 1000)
        '''

        # Generate
        if self.me == 0:
            samples = np.random.rand(nSamples)*(bounds[1]-bounds[0]) - bounds[0]
        else:
            samples = None

        # Save the samples
        self.samples = [samples]

        # All done
        return

   def initializePredFunction(self):
       '''
       Initialize the prediction function.

       Args:
            * 
            * 
        '''

        # Do stuff

        # Save the prediction function
        self.fpred = predictionFunction

        # All done
        return

    def oneTimeStep(self, step):
        '''
        Does the posterior sampling for a new time step.

        Args:
            * step      : index of the time step.
        '''

        # Identify the data

        # Identify the samples to update/create and the fixed ones

        fixed = 
        samples = # un truc de dimension (2, nSamples)

        # Create a prediction function

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
                           prediction, self.comm)
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

        # All done
        return alpha1, alpha2

    def walkWithTime(self):
        '''
        Advance throught time.
        '''

        # Iterate over the data


# List splitter
def _split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq
