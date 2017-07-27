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
import h5py
import numpy as np
import matplotlib.pyplot as plt

# PyMC
import pymc

# Personals
from .SourceInv import SourceInv
from .planarfault import planarfault

# Class explorefault
class explorefault(SourceInv):

    def __init__(self, name, utmzone=None, 
                 ellps='WGS84', lon0=None, lat0=None, 
                 verbose=True):

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
            print ("Initializing fault exploration {}".format(name))
        self.verbose = verbose

        # Base class init
        super(explorefault,self).__init__(name, utmzone=utmzone, 
                                          ellps=ellps, 
                                          lon0=lon0, lat0=lat0)

        # Keys to look for
        self.keys = ['lon', 'lat', 'depth', 'dip', 
                     'width', 'length', 'strike', 
                     'strikeslip', 'dipslip']

        # Initialize the fault object
        self.fault = planarfault('mcmc fault', utmzone=self.utmzone, 
                                               lon0=self.lon0, 
                                               lat0=self.lat0,
                                               ellps=self.ellps, 
                                               verbose=False)

        # All done
        return

    def setPriors(self, bounds, datas=None):
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
                    'strikeslip' -- Strike Slip (tuple or float)
                    'dipslip'    -- Dip slip (tuple or float)

            If the specified bound is a tuple of 2 floats, the prior will be uniform
            If the specified bound is a float, this parameter will not be searched for

            * datas         : Data sets that will be used. This is in case bounds has
                              tuples or floats for reference of an InSAR data set
        '''

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
            else:
                assert False, 'Unknown bound type'

            # Save it
            self.Priors.append(prior)

        # Create a prior for the data set reference term
        # Works only for InSAR data yet
        if datas is not None:

            # Check 
            if type(datas) is not list:
                datas = [datas]
                
            # Iterate over the data
            for data in datas:
                
                # Get it
                assert data.name in bounds, \
                    'No bounds provided for prior for data {}'.format(data.name)
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
                data.refnumber = len(self.Priors)-1

        # All done
        return

    def setLikelihood(self, datas, vertical=True):
        '''
        Builds the data likelihood object from the list of geodetic data in datas.

        Args:   
            * datas         : csi geodetic data object (gps or insar) or 
                              list of csi geodetic objects. 
                              TODO: Add other types of data (opticorr)

        KwArgs:
            * vertical      : Use the verticals for GPS?
        '''

        # Build the prediction method
        # Initialize the object
        if type(datas) is not list:
            self.datas = [datas]

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
                    value = data.vel_enu.flatten()
                else:
                    value = data.vel_enu[:,:-1].flatten()
            elif data.dtype=='insar':
                # Get data
                value = data.vel

            # Make sure Cd exists
            assert hasattr(data, 'Cd'), \
                    'No data covariance for data set {}'.format(data.name)
            Cd = data.Cd

            # Create the forward method
            @pymc.deterministic(plot=False)
            def forward(theta=self.Priors):
                return self.Predict(theta, data, vertical=vertical)
            data.forward = forward

            # Build likelihood function
            likelihood = pymc.MvNormalCov('Data Likelihood: {}'.format(data.name), 
                                          mu=data.forward, 
                                          C=Cd, 
                                          value=value, 
                                          observed=True)
            
            # Save the likelihood function
            self.Likelihoods.append(likelihood)

        # All done 
        return

    # Define a function
    def Predict(self, theta, data, vertical=True):
        '''
        Calculates a prediction of the measurement from the theta vector
        theta = [lon, lat, depth, dip, width, length, strike, strikeslip, dipslip]
        '''

        # Take the values in theta and distribute
        lon, lat, depth, dip, width, length, strike, strikeslip, dipslip = theta[:9]
        if hasattr(data, 'refnumber'):
            reference = theta[data.refnumber]
        else:
            reference = 0.

        # Get the fault
        fault = self.fault

        # Build a planar fault
        fault.buildPatches(lon, lat, depth, strike, dip, 
                       length, width, 1, 1, verbose=False)

        # Build the green's functions
        fault.buildGFs(data, vertical=vertical, slipdir='sd', verbose=False)

        # Set slip 
        fault.slip[:,0] = strikeslip
        fault.slip[:,1] = dipslip

        # Build the synthetics
        data.buildsynth(fault)

        # check data type 
        if data.dtype=='gps':
            if vertical: 
                return data.synth.flatten()
            else:
                return data.synth[:,:-1].flatten()
        elif data.dtype=='insar':
            return data.synth.flatten()+reference

        # All done
        return

    def walk(self, niter=10000, nburn=5000, method='AdaptiveMetropolis'):
        '''
        March the MCMC.

        Args:
            * niter             : Number of steps to walk
            * nburn             : Numbero of steps to burn
        '''

        # Create a sampler
        sampler = pymc.MCMC(self.Priors+self.Likelihoods)

        # Make sure step method is what is asked for
        for prior in self.Priors:
            sampler.use_step_method(getattr(pymc, method), prior)

        # Sample
        sampler.sample(iter=niter, burn=nburn)

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
                                  'rand', an integer or a dictionary
                                  with the appropriate keys
        '''

        # Create a dictionary
        specs = {}

        # Iterate over the keys
        for key in self.keys:

            # Get it 
            if model=='mean':
                value = self.sampler.trace(key)[:].mean()
            elif model=='median':
                value = self.sampler.trace(key)[:].median()
            elif model=='std':
                value = self.sampler.trace(key)[:].std()
            else: 
                if type(model) is int:
                    assert type(model) is int, 'Model type unknown: {}'.format(model)
                    value = self.sampler.trace(key)[model]
                elif type(model) is dict:
                    value = model[key]

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
        
        # Set slip values
        fault.slip[:,0] = specs['strikeslip']
        fault.slip[:,1] = specs['dipslip']

        # Save the desired model 
        self.model = specs

        # All done
        return fault
    
    def plot(self, model='mean', show=True):
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

            # Build the synthetics
            data.buildsynth(fault)

            # Check ref
            if 'Reference {}'.format(data.name) in self.keys:
                data.synth += self.sampler.trace('Reference {}'.format(data.name))[:].mean()

            # Plot the data and synthetics
            cmin = np.min(data.vel)
            cmax = np.max(data.vel)
            data.plot(data='data',  show=False, norm=[cmin, cmax])
            data.plot(data='synth', show=False, norm=[cmin, cmax])
        
        # Plot
        if show:
            plt.show()

        # All done
        return

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

#EOF
