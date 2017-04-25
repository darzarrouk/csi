'''
A class that searches for the best fault to fit some geodetic data.
This class is made for a simple planar fault geometry.
It is close to what R. Grandin has implemented but with a MCMC approach
Grandin's approach will be coded in another class.

Author:
R. Jolivet 2017
'''

# Externals
import sys, os, copy
import numpy as np
import matplotlib.pyplot as plt

# PyMC
import pymc

# Personals
from .SourceInv import SourceInv
from .faultwithdip import faultwithdip 

# Class explorefault
class explorefault(SourceInv):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        '''
        Creates an object that will solve for the best fault details.

        Args:
            * name          : Name of the object

        KwArgs:
            * utmzone       : UTM zone number
            * ellps         : Ellipsoid
            * lon0/lat0     : Refernece of the zone
        '''

        # Initialize the fault
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault exploration {}".format(self.name))
        self.verbose = verbose

        # Base class init
        super(explorefault,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # All done
        return

    def setPriors(self, bounds):
        '''
        Initializes the prior likelihood functions.

        Args:
            * bounds        : Bounds is a dictionary that holds the following keys
                    'lon'        -- Longitude (tuple or float)
                    'lat'        -- Latitude (tuple or float)
                    'depth'      -- Depth in km of the top of the fault (tuple or float)
                    'dip'        -- Dip in degree (tuple or float)
                    'width'      -- Along-dip size in km (tuple or float)
                    'length'     -- Along-strike length in km (tuple or float)
                    'strike'     -- Azimuth of the strike (tuple or float)
                    'dipslip'    -- Dip slip (tuple or float)
                    'strikeslip' -- Strike Slip (tuple or float)

            If the specified bound is a tuple of 2 floats, the prior will be uniform
            If the specified bound is a float, this parameter will not be searched for
        '''

        # Keys to look for
        self.keys = ['lon', 'lat', 'depth', 'dip', 
                     'width', 'length', 'strike', 
                     'dipslip', 'strikeslip']

        # Make a list of priors
        if not hasattr(self, 'Priors'):
            self.Priors = []

        # Iterate over the keys
        for key in self.keys:

            # Check the key has been provided
            assert key in bounds, '{} not defined in the input dictionary'

            # Get the values
            bound = bounds[key]

            # Check
            if type(bound) is tuple:
                # Get the upper/lower bound and build uniform prior
                lower, upper = bound
                prior = pymc.Uniform(key, lower, upper)
            elif type(bound) is float:
                # Create a degenerate prior
                prior = pymc.Degenerate(key, bound)

            # Save it
            self.Priors.append(prior)

        # All done
        return

    def setLikelihood(self, datas, bounds=None, vertical=True):
        '''
        Builds the data likelihood object from the list of geodetic data in datas.

        Args:   
            * datas         : csi geodetic data object (gps or insar) or 
                              list of csi geodetic objects. 
                              TODO: Add other types of data (opticorr)

        KwArgs:
            * vertical      : Use the verticals for GPS?
            * bounds        : dictionary with keys identical to data names
                              if tuple, it will set a Uniform prior for a reference
                              if float, it will set a Degenerate prior as a reference
        '''

        # Initialize the object
        if type(datas) is not list:
            self.datas = list(datas)

        # List of likelihoods
        self.Likelihoods = []
        if not hasattr(self, 'Priors'):
            self.Priors = []

        # Create a likelihood function for each of the data set
        for data in self.datas:

            # Get the data type
            if data.dtype=='gps':
                # Get data
                if vertical:
                    value = np.flatten(data.vel_enu)
                else:
                    value = np.flatten(data.vel_enu[:,:-1])
            elif data.dtype='insar':
                # Get data
                value = data.vel

            # Make sure Cd exists
            assert hasattr(data, 'Cd'), 'No data covariance for data set {}'.format(data.name)
            Cd = data.Cd

            # Create the forward method
            @pymc.deterministic(plot=False)
            def forward(theta=self.Priors):
                return self.fpred(theta, data)
            data.forward = forward

            # Build likelihood function
            likelihood = pymc.MvNormalCov('Data Likelihood: {}'.format(data.name), 
                                          mu=data.forward, 
                                          C=Cd, 
                                          value=value, 
                                          observed=True)
            
            # Save the likelihood function
            self.Likelihoods.append(likelihood)

            # Create a prior for the data set reference term
            if bounds is not None:
                # Get it
                assert data.name in bounds, 'No bounds provided for prior for data {}'.format(data.name)
                bound = bounds[data.name]
                # Check
                if type(bound) is tuple:
                    prior = pymc.Uniform('Reference {}'.format(data.name), 
                                         bound[0], bound[1])
                elif type(bound) is float:
                    prior = pymc.Degenerate('Reference {}'.format(data.name), 
                                            bound)
                # Store it
                self.Priors.append(prior)
                self.keys.append('Reference {}'.format(data.name))

        # save the method 
        self.fpred = predict

        # All done 
        return

    def buildPredictionMethod(self, vertical=True):
        '''
        Builds the dictionary of prediction methods.

        Args:   
            * datas             : List of datasets
            * vertical          : True/False
        '''

        # Define a function
        def predict(theta, data):

            # Take the values in theta and distribute
            lon, lat, depth, dip, width, length, strike, dipslip, strikeslip = theta

            # Build a planar fault
            fault = planarfault('mcmc fault', utmzone=self.umtzone, 
                                              lon0=self.lon0, 
                                              lat0=self.lat0,
                                              ellps=self.ellps, 
                                              verbose=False)
            fault.buildPatches(lon, lat, depth, strike, dip, 
                           length, width, 1, 1, verbose=False)

            # Build the green's functions
            fault.buildGFs(data, vertical=vertical, slipdir='sd', verbose=False)

            # Build the synthetics
            data.buildsynth(fault)

            # check data type 
            if data.dtype=='gps':
                if vertical: 
                    return np.flatte(data.synth)
                else:
                    return np.flatten(data.synth[:,:-1])
            elif data.dtype=='insar':
                return np.flatten(data.synth)

        # All done
        return

    def walk(self, niter=10000, nburn=5000):
        '''
        March the MCMC.

        Args:
            * niter             : Number of steps to walk
            * nburn             : Numbero of steps to burn
        '''

        # Create a sampler
        sampler = pymc.MCMC(self.Priors+self.Likelihoods)

        # Make sure step method is Metropolis
        for prior in self.Priors:
            sampler.use_step_method(pymc.Metropolis, prior)

        # Sample
        sampler.sampler(iter=niter, burn=nburn)

        # Save the sampler
        self.sampler = sampler
        self.nsamples = niter - nburn

        # All done
        return

    def returnModel(self, model='mean'):
        '''
        Returns a fault corresponding to the desired model.

        Kwargs:
            * model             : Can be 'mean', 'median', 
                                  'rand' or an integer
        '''

        # Create a dictionary
        specs = {}

        # Iterate over the keys
        for key in self.keys:

            # Get it 
            if model=='mean':
                value = self.sampler.trace(key).mean()
            elif model=='median':
                value = self.sampler.trace(key).median()
            elif model=='std':
                value = self.sampler.trace(key).std()
            else: 
                assert type(model) is int, 'Model type unknown: {}'.format(model)
                value = self.sampler.trace(key)[model]

            # Set it
            specs[key] = value

        # Create a fault
        fault = planarfault('{} model'.format(model), 
                            utmzone=self.utmzone, 
                            lon0=self.lon0, 
                            lat0=self.lat0,
                            ellps=self.ellps, 
                            verbose=False)
        fault.buildPatches(specs['lon'], specs['lat'], 
                           specs['depth'], specs['strike'],
                           specs['dip'], specs['length'],
                           specs['width'], 1, 1, verbose=False)

        # Save the desired model 
        self.model = specs

        # All done
        return fault
    
    def plot(self, model='mean'):
        '''
        Plots the PDFs and the desired model predictions and residuals.
        '''

        # Plot the pymc stuff
        pymc.Matplot.plot(self.sampler)

        # Get the model
        fault = self.returnModel(model=model)

        # Build predictions
        for data in self.datas:

            # Build the green's functions
            fault.buildGFs(data, slipdir='sd', verbose=False)

            # Buld the synthetics
            data.buildsynth(fault)

    def save2h5(self, filename):
        '''
        Save the results to a h5 file.

        Args:
            * filename          : Name of the input file
        '''

        # Open an h5file
        fout = h5py.File(filename, 'w')

        # Create the data sets for the keys
        for key in self.keys:
            fout.create_dataset(key, data=self.sampler.trace(key)[:])

        # Close file
        fout.close()

        # All done
        return

