'''
A class that allows to compute, fit and display the
empirical covariances in a function.

Written by R. Jolivet, July 2014.
'''

# Externals
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import sys
import copy

# Personals
from .insarrates import insarrates
from .cosicorrrates import cosicorrrates

# Some Usefull functions    
def exp_fn(t,sil,sig,lam):
    return sil - (sig**2)*np.exp(-t/lam)

def gauss_fn(t, sil, sig, lam):
    return sil - (sig**2)*np.exp(-(t**2)/(2*(lam)**2))

def covariance(t,sig,lam, covfn='exp'):
    if covfn in ('exp'):
        return (sig**2)*np.exp(-t/lam)
    elif covfn in ('gauss'):
        return (sig**2)*np.exp(-(t**2)/(2*(lam)**2))

def ramp_fn(t,a,b,c):
    v = np.array([a,b,c])
    return np.dot(t, v)

# Main class
class imagecovariance(object):

    def __init__(self, name, image, verbose=True):
        '''
        Args:
            * name      : Name of the downsampler.
            * image     : InSAR or Cosicorr data set to be downsampled.
            * faults    : List of faults.
        '''

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR covariance tools {}".format(name))

        # Save it
        self.verbose = verbose

        # Set the name
        self.name = name
        self.datatype = image.dtype

        # Set the transformation
        self.utmzone = image.utmzone
        self.putm = image.putm
        self.ll2xy = image.ll2xy
        self.xy2ll = image.xy2ll

        # Save the image
        self.image = image

        # Iterate and save the datasets to consider
        self.datasets = {}
        if self.datatype is 'insarrates':
            dname = '{}'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.vel}
        elif self.datatype is 'cosicorrrates':
            dname = '{} East'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.east}
            dname = '{} North'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.north}
        else:
            print('Data type unknown or not recognized by imagecovariance type...')
            sys.exit()

        # All done
        return

    def maskOut(self, box):
        '''
        Picks out some points in order to mask them before computing the covariance.
        Args:
            * box   : List of min and max lon and lat coordinates.
                      Can be a list of lists to specify multiple regions.
                      ex: [[ -120, -119, 34, 35], [-122, -121.7, 34.2, 34.3]]
        '''

        # Check how many zones do we have to remove
        self.maskedZones = []
        if type(box[0]) in (int, float):
            self.maskedZones.append(box)
        else:
            for b in box:
                self.maskedZones.append(b)

        # Iterate over the data sets
        for dname in self.datasets:
            if self.verbose:
                print('Masking data set {}'.format(dname))
            # Iterate over the boxes
            for box in self.maskedZones:
                if self.verbose:
                    print('     Mask: {} <= Lon <= {} || {} <= Lat <= {}'.format(box[0], box[1], box[2], box[3]))
                # Get lon lat
                lon = self.datasets[dname]['lon']
                lat = self.datasets[dname]['lat']
                # Find out the points
                ii = np.flatnonzero(np.logical_and(lon>=box[0], lon<=box[1]))
                jj = np.flatnonzero(np.logical_and(lat>=box[2], lat<=box[3]))
                # intersection
                uu = np.intersect1d(ii,jj)
                # Take them out
                self.datasets[dname]['x'] = np.delete(self.datasets[dname]['x'], uu)
                self.datasets[dname]['y'] = np.delete(self.datasets[dname]['y'], uu)
                self.datasets[dname]['lon'] = np.delete(self.datasets[dname]['lon'], uu)
                self.datasets[dname]['lat'] = np.delete(self.datasets[dname]['lat'], uu)
                self.datasets[dname]['data'] = np.delete(self.datasets[dname]['data'], uu)

        # All done
        return

    def maskIn(self, box):
        '''
        Select Boxes on which to compute the covariance.
        Args:
        * box: List of min and max lon and lat coordinates.
               Can be a list of lists to specify multiple regions.
               ex: [[ -120, -119, 34, 35], [-122, -121.7, 34.2, 34.3]]
        '''

        # Check how many zones do we have to keep
        self.selectedZones = []
        if type(box[0]) in (int, float):
            self.selectedZones.append(box)
        else:
            for b in box:
                self.selectedZones.append(b)

        # Iterate over the data sets
        for dname in self.datasets:
            if self.verbose:
                print('Dealing with data set {}'.format(dname))
            # Create a new data set
            self.datasets['New One'] = {'x': np.empty(0), 
                                        'y': np.empty(0),
                                        'lon': np.empty(0), 
                                        'lat': np.empty(0), 
                                        'data': np.empty(0)}
            # Iterate over the boxes
            for box in self.selectedZones:
                if self.verbose:
                    print('     Zone of Interest: {} <= Lon <= {} || {} <= Lat <= {}'.format(box[0], 
                        box[1], box[2], box[3]))
                # Get lon lat
                lon = self.datasets[dname]['lon']
                lat = self.datasets[dname]['lat']
                # Find out the points
                ii = np.flatnonzero(np.logical_and(lon>=box[0], lon<=box[1]))
                jj = np.flatnonzero(np.logical_and(lat>=box[2], lat<=box[3]))
                # intersection
                uu = np.intersect1d(ii,jj)
                # Take them in
                x = self.datasets[dname]['x'][uu]
                y = self.datasets[dname]['y'][uu]
                lon = self.datasets[dname]['lon'][uu]
                lat = self.datasets[dname]['lat'][uu]
                data = self.datasets[dname]['data'][uu]
                # Put them in the new data set
                self.datasets['New One']['x'] = np.hstack((self.datasets['New One']['x'], x))
                self.datasets['New One']['y'] = np.hstack((self.datasets['New One']['y'], y))
                self.datasets['New One']['lon'] = np.hstack((self.datasets['New One']['lon'], lon))
                self.datasets['New One']['lat'] = np.hstack((self.datasets['New One']['lat'], lat))
                self.datasets['New One']['data'] = np.hstack((self.datasets['New One']['data'], data))

            # Replace the data set by the New One
            self.datasets[dname] = copy.deepcopy(self.datasets['New One'])
            del self.datasets['New One']

        # All done
        return

    def empiricalSemivariograms(self, frac=0.4, every=1., distmax=50., rampEst=True):
        '''
        Computes the empirical Semivariogram as a function of distance.
        Args:
            * frac      : Size of the fraction of the dataset to take (0 to 1)
                          frac can be an integer, then it is going to be the number of 
                          pixels used to compute the covariance
            * distmax   : Truncate the covariance function.
            * every     : Binning of the covariance function.
        '''

        # Iterate over the datasets
        for dname in self.datasets:

            # print
            if self.verbose:
                print('Computing 1-D empirical semivariogram function for data set {}'.format(dname))

            # Get data set
            data = self.datasets[dname]

            # Get values
            x = data['x']
            y = data['y']
            d = data['data']

            # How many samples do we use
            if type(frac) is int:
                Nsamp = frac
                if Nsamp>d.shape[0]:
                    Nsamp = d.shape[0]
            else:
                Nsamp = np.int(np.floor(frac*x.size))
            if self.verbose: 
                print('Selecting {} random samples to estimate the covariance function'.format(Nsamp))

            # Create a vector
            regular = np.vstack((x.squeeze(),y.squeeze(),d.squeeze())).T
            
            # Take a random permutation
            randomized = np.random.permutation(regular)

            # Take the first frac of it
            x = randomized[:Nsamp,0]
            y = randomized[:Nsamp,1]
            d = randomized[:Nsamp,2]

            # Remove a ramp
            if rampEst:
                G = np.zeros((Nsamp,4))
                G[:,3] = x*y
                G[:,0] = x
                G[:,1] = y
                G[:,2] = 1.
                pars = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),d) 
                a = pars[0]; b = pars[1]; c = pars[2]; w = pars[3]
                d = d - (a*x + b*y + c + w*x*y)
                if self.verbose:
                    print('Estimated Orbital Plane: {}xy + {}x + {}y + {}'.format(w,a,b,c))  
                # Save it
                data['Ramp'] = [a, b, c, w]       

            # Build all the permutations
            if self.verbose:
                print('Build the permutations')
            ii, jj = np.meshgrid(range(Nsamp), range(Nsamp))
            ii = ii.flatten()
            jj = jj.flatten()
            uu = np.flatnonzero(ii>jj)
            ii = ii[uu]
            jj = jj[uu]

            # Compute the distances
            dx = x[ii] - x[jj]
            dy = y[ii] - y[jj]
            dis = np.sqrt(dx*dx + dy*dy)

            # Compute the semivariogram
            dv = (d[ii] - d[jj])**2

            # Digitize
            if self.verbose:
                print('Digitize the histogram')
            bins = np.arange(0., distmax, every)
            inds = np.digitize(dis, bins)

            # Average
            distance = []
            semivariogram = []
            for i in range(len(bins)-1):
                uu = np.flatnonzero(inds==i)
                if len(uu)>0:
                    distance.append(bins[i] + (bins[i+1] - bins[i])/2.)
                    semivariogram.append(np.mean(dv[uu]))

            # Store these guys
            data['Distance'] = np.array(distance)
            data['Semivariogram'] = np.array(semivariogram)

        # All done
        return

    def computeCovariance(self, function='exp', frac=0.4, every=1., distmax=50., rampEst=True):
        '''
        Computes the covariance functions.
        Args:
            * function  : Type of function to fit. Can be 'exp'or 'gauss'.
            * frac      : Size of the fraction of the dataset to take.
            * distmax   : Truncate the covariance function.
            * every     : Binning of the covariance function.
            * rampEst   : estimate a ramp (default True).
        '''

        # Compute the semivariograms
        if self.verbose:
            print('Computing semivariograms')
        self.empiricalSemivariograms(frac=frac, every=every, distmax=distmax, rampEst=rampEst)

        # Fit the semivariograms
        if self.verbose:
            print('Fitting Covariance functions')
        for dname in self.datasets:
            
            # Get the dataset
            data = self.datasets[dname]

            # Get the data
            semivar = data['Semivariogram']
            distance = data['Distance']

            # Save the type of function
            data['function'] = function

            # Fit that 
            if function is 'exp':
                try:
                    pars, cova = sp.curve_fit(exp_fn, distance, semivar)
                except:
                    try: 
                        pars, cova = sp.curve_fit(exp_fn, distance, semivar, ftol=1e-5)
                    except:
                        print('No solution found for data sets {} of {}'.format(dname, self.name)) 
                        pars = [np.nan, np.nan, np.nan]
            elif function is 'gauss':
                try:
                    pars, cova = sp.curve_fit(gauss_fn, distance, semivar)
                except:
                    try: 
                        pars, cova = sp.curve_fit(gauss_fn, distance, semivar, ftol=1e-5)
                    except:
                        print('No solution found for data sets {} of {}'.format(dname, self.name)) 
                        pars = [np.nan, np.nan, np.nan]
            else:
                print('Unknown function type..., must be exp or gauss')
                sys.exit()

            # Save parameters
            sill = pars[0]
            sigm = pars[1]
            lamb = pars[2]
            data['Sill'] = sill
            data['Sigma'] = sigm
            data['Lambda'] = lamb

            # Print
            if self.verbose:
                print('Dataset {}:'.format(dname))
                print('     Sill   :  {}'.format(sill))
                print('     Sigma  :  {}'.format(sigm))
                print('     Lambda :  {}'.format(lamb))

            # Compute the covariance function
            data['Covariance'] = sill - semivar

        # All done
        return

    def buildCovarianceMatrix(self, image, dname, write2file=None):
        '''
        Uses the fitted covariance parameters to build a covariance matrix for the dataset
        image of type insarrates or cosicorrrates.
        Args:
            * image     : dataset of type cosicorrrates or insarrates.
            * dname     : Name of the dataset used to estimate the parameters.
                          if image is cosicorrates, the datasets used are "dname East" and "dname North".
            * write2file: Write to a binary file (np.float32).
        '''

        # Get the data position
        x = image.x
        y = image.y

        # Case 1: InSAR
        if image.dtype is 'insarrates':

            # Get the Parameters
            assert 'Sigma' in self.datasets[dname].keys(), 'Need to estimate the covariance function first: {}'.format(dname)
            sigma = self.datasets[dname]['Sigma']
            lamb = self.datasets[dname]['Lambda']
            function = self.datasets[dname]['function']

            # Build the covariance
            Cd = self._buildcov(sigma, lamb, function, x, y)

        # Case 2: Cosicorr
        elif image.dtype is 'cosicorrrates':
            
            # Create the two names
            dnameEast = dname+' East'
            dnameNorth = dname+' North'

            # Get the parameters and Build CdEast
            assert 'Sigma' in self.datasets[dnameEast].keys(), 'Need to estimate the covariance function first: {}'.format(dnameEast)
            sigmaEast = self.datasets[dnameEast]['Sigma']
            lambEast = self.datasets[dnameEast]['Lambda']
            funcEast = self.datasets[dnameEast]['function']
            CdEast = self._buildcov(sigmaEast, lambEast, funcEast, x, y)

            # Get the parameters and Build CdNorth
            assert 'Sigma' in self.datasets[dnameNorth].keys(), 'Need to estimate the covariance function first: {}'.format(dnameNorth)
            sigmaNorth = self.datasets[dnameNorth]['Sigma']
            lambNorth = self.datasets[dnameNorth]['Lambda']
            funcNorth = self.datasets[dnameNorth]['function']
            CdNorth = self._buildcov(sigmaNorth, lambNorth, funcNorth, x, y)

            # Cat matrices
            nd = x.shape[0]
            Cd = np.vstack( (np.hstack((CdEast, np.zeros((nd,nd)))), np.hstack((np.zeros((nd,nd)), CdNorth))) )

        # Write 2 a file?
        if write2file is not None:
            Cd.astype(np.float32).tofile(write2file)

        # All done
        return Cd

    def write2file(self):
        '''
        Writes the results to a text file.
        '''

        # Iterates over the datasets
        for dname in self.datasets:

            # Get data 
            data = self.datasets[dname]

            # continue if nothing has been done
            if 'Semivariogram' not in data.keys():
                print('Nothing to be written for data set {}'.format(dname))
                continue

            # filename
            filename = '{}.cov'.format(dname.replace(' ','_'))

            # Open file
            fout = open(filename, 'w')
        
            # Write stuffs
            fout.write('# Covariance estimated for {}\n'.format(dname))

            # Write fit results
            if 'function' in data.keys():
                fout.write('# Best fit function type {}: \n'.format(data['function']))
                fout.write('#       Sill   : {} \n'.format(data['Sill']))
                fout.write('#       Sigma  : {} \n'.format(data['Sigma']))
                fout.write('#       Lambda : {} \n'.format(data['Lambda']))

            # Write header
            header = '# Distance (km) || Semivariogram '
            if 'Covariance' in data.keys():
                header = header + '|| Covariance'
            header = header + '\n'
            fout.write(header)

            # Write what is in there
            distance = data['Distance']
            semivar = data['Semivariogram']
            if 'Covariance' in data.keys():
                covar = data['Covariance']
            for i in range(distance.shape[0]):
                d = distance[i]
                s = semivar[i]
                line = '{}     {} '.format(d, s)
                if 'Covariance' in data.keys():
                    c = covar[i]
                    line = line + '    {}'.format(c)
                line = line + '\n'
                fout.write(line)

            # Close file
            fout.close()

        # All done
        return

    def plot(self, data='covariance', plotData=False, figure=1, savefig=False):
        '''
        Plots the covariance function.
        Args:
            * data    : Can be covariance or semivariogram or all.
            * plotData: Also plots the image
        '''

        # Plot the data?
        if plotData:
            plt.figure(figure+1)
            plt.clf()
            self.image.plot(show=False, figure=figure+1, ref='lonlat')
            if savefig:
                figname = 'Data_{}.png'.format(self.name.replace(' ','_'))
                plt.savefig(figname)

        # Create a figure
        fig = plt.figure(figure)
        plt.clf()

        # How many data sets
        nData = len(self.datasets)

        # Iterate
        ii = 0
        for dname in self.datasets:

            # Create an axes
            ax = fig.add_subplot(nData, 1, ii)

            # Set its name
            ax.set_title(dname)

            # Plot Semivariogram
            if data in ('semivariogram', 'semi', 'all', 'semivar'):
                semi = self.datasets[dname]['Semivariogram']
                dist = self.datasets[dname]['Distance']
                ax.plot(dist, semi, '.b', markersize=10)
                if 'function' in self.datasets[dname].keys():
                    sill = self.datasets[dname]['Sill']
                    sigm = self.datasets[dname]['Sigma']
                    lamb = self.datasets[dname]['Lambda']
                    function = self.datasets[dname]['function']
                    if function is 'exp':
                        fy = exp_fn(dist, sill, sigm, lamb)
                    elif function is 'gauss':
                        fy = gauss_fn(dist, sill, sigm, lamb)
                    else:
                        print('Unknown function type..., must be exp or gauss')
                        sys.exit()
                    ax.plot(dist, fy, '-k')

            # Plot Covariance
            if data in ('covariance', 'all', 'cov'):
                covar = self.datasets[dname]['Covariance']
                dist = self.datasets[dname]['Distance']
                ax.plot(dist, covar, '.k', markersize=10)
                if 'function' in self.datasets[dname].keys():
                    sill = self.datasets[dname]['Sill']
                    sigm = self.datasets[dname]['Sigma']
                    lamb = self.datasets[dname]['Lambda']
                    function = self.datasets[dname]['function']
                    if function is 'exp':
                        fy = sill - exp_fn(dist, sill, sigm, lamb)
                    elif function is 'gauss':
                        fy = sill - gauss_fn(dist, sill, sigm, lamb)
                    else:
                        print('Unknown function type..., must be exp or gauss')
                        sys.exit()
                    ax.plot(dist, fy, '-r')

            # Axes
            ax.axis('auto')

            # Increase 
            ii += 1

        # Save?
        if savefig:
            figname = '{}.png'.format(self.name.replace(' ','_'))
            plt.savefig(figname)

        # Show me
        plt.show()

        # All done
        return

    def _buildcov(self, sigma, lamb, func, x, y):
        '''
        Returns a matrix of the covariance.
        Args:
            * sigma : Arg #1 of function func
            * lamb  : Arg #2 of function func
            * func  : Function of distance ('exp' or 'gauss')
            * x     : position of data along x-axis
            * y     : position of data along y-axis
        '''

        # Make a distance map matrix
        X1, X2 = np.meshgrid(x,x)
        Y1, Y2 = np.meshgrid(y,y)
        XX = X2-X1
        YY = Y2-Y1
        D = np.sqrt( XX**2 + YY**2)

        # Compute covariance
        Cd = covariance(D, sigma, lamb, covfn=func)

        # All done
        return Cd

#EOF
