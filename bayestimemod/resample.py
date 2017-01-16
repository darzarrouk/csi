# Class to sample the posterior when adding data from a prior set of samples

# Import stuff
import numpy as np
import pymc

class resample(object):
    '''
    Class to sample the posterior probability function of 2 interpolating variables
    given a 1D set of data.
    2 functions are estimated, while the third one, covering the data before the 2nd 
    triangle kicks in, stays fixed.
    '''

    def __init__(self, data, sigma, time, 
                       samples, fixedSamples, bounds, 
                       fpred, comm, niter=1000):
        '''
        Initialization of an instance.

        Args:
            * data          : Set of data 
            * sigma         : standard dev of data covariance
            * time          : Time of each data
            * Samples       : Set of samples to be taken as initial state
            * fixedSample   : Sample set of the third triangle
            * bounds        : Prior pdf bounds (priors are uniform)
                         ex : bounds = (0., 10.)
            * fpred         : Prediction function
            * comm          : MPI communicator
            * niter         : number of iterations per chain
        '''

        # Save the communicator
        self.comm = comm
        self.me = comm.Get_rank()

        # save the samples
        self.samples = samples
        self.fixedsamples = fixedSamples

        # save the bounds
        self.bounds = bounds

        # save the data and time
        self.data = data
        self.sigma = sigma
        self.time = time

        # save the prediction function
        self.fpred = fpred

        # save the iteration number
        self.niter = niter

        # All done 
        return

    def walkOneChain(self, startingPoints, fixedPoint):
        '''
        Do a metropolis walk starting from a sample.

        Args:
            * startingPoints    : Sample to start from (ex: [0.2342, 1.345])
            * fixedPoint        : Value of the preceding triangle.
        '''

        # Create the priors
        alphaOne = pymc.Uniform('Alpha 1', self.bounds[0], self.bounds[1], 
                                value=startingPoints[0])
        alphaTwo = pymc.Uniform('Alpha 2', self.bounds[0], self.bounds[1],
                                value=startingPoints[1])
        Priors = [alphaOne, alphaTwo]

        # Data prediction function
        @pymc.deterministic(plot=False)
        def forward(theta=Priors):
            alpha1, alpha2 = theta
            return self.fpred([fixedPoint, alpha1, alpha2])

        # Create a multivariate normal likelihood (the pdf is gaussian so far, so 
        # a diagonal covariance matrix will do it)
        likelihood = pymc.Normal('Data likelihood', mu=forward, 
                                 tau=1./self.sigma**2, 
                                 value=self.data, observed=True)
        
        # Create a sampler
        sampler = pymc.MCMC(Priors+[likelihood])
        for p in Priors:
            sampler.use_step_method(pymc.Metropolis, p)

        # Sample
        sampler.sample(iter=self.niter, burn=self.niter-1, progress_bar=False)

        # All done -- return the last sample
        return sampler.trace('Alpha 1')[-1], sampler.trace('Alpha 2')[-1]

    def sample(self):
        '''
        Samples the posterior PDF, starting from the samples in self.samples
        '''

        # New list of samples
        nextSample1 = []
        nextSample2 = []

        # Iterate over the samples
        for alpha1, alpha2, fixed in zip(self.samples[:,0], 
                                         self.samples[:,1], 
                                         self.fixedsamples):
            newAlpha1, newAlpha2 = self.walkOneChain((alpha1, alpha2), fixed)
            nextSample1.append(newAlpha1)
            nextSample2.append(newAlpha2)

        # All done
        return np.array(nextSample1), np.array(nextSample2)

#EOF
