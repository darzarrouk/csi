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
            print ("Initialize InSAR downsampling tools {}".format(name))

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
            dname = 'InSAR {}'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.vel}
        elif self.datatype is 'cosicorrrates':
            dname = 'CosiCorr {} East'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.east}
            dname = 'CosiCorr {} North'.format(self.name)
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

    def empiricalSemivariograms(self, frac=0.4, every=1., distmax=50.):
        '''
        Computes the empirical Semivariogram as a function of distance.
        Args:
            * frac      : Size of the fraction of the dataset to take.
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

            # Create a vector
            regular = np.vstack((x.squeeze(),y.squeeze(),d.squeeze())).T
            
            # Take a random permutation
            randomized = np.random.permutation(regular)

            # Take the first frac of it
            Nsamp = np.int(np.floor(frac*x.size))
            x = randomized[:Nsamp,0]
            y = randomized[:Nsamp,1]
            d = randomized[:Nsamp,2]

            # Remove a ramp
            G = np.zeros((Nsamp,3))
            G[:,0] = x
            G[:,1] = y
            G[:,2] = 1.
            pars = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),d) 
            a = pars[0]; b = pars[1]; c = pars[2] 
            d = d - (a*x + b*y + c)
            if self.verbose:
                print('Estimated Orbital Plane: {}x + {}y + {}'.format(a,b,c))  
            # Save it
            data['Ramp'] = [a, b, c]       

            # Build all the permutations
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

    def computeCovariance(self, function='exp', frac=0.4, every=1., distmax=50.):
        '''
        Computes the covariance functions.
        Args:
            * function  : Type of function to fit. Can be 'exp'or 'gauss'.
            * frac      : Size of the fraction of the dataset to take.
            * distmax   : Truncate the covariance function.
            * every     : Binning of the covariance function.
        '''

        # Compute the semivariograms
        if self.verbose:
            print('Computing semivariograms')
        self.empiricalSemivariograms(frac=frac, every=every, distmax=distmax)

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
                pars, cova = sp.curve_fit(exp_fn, distance, semivar)
            elif function is 'gauss':
                pars, cova = sp.curve_fit(gauss_fn, distance, semivar)
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
                print('     Sill   :  {}'.format(sill))
                print('     Sigma  :  {}'.format(sigm))
                print('     Lambda :  {}'.format(lamb))

            # Compute the covariance function
            data['Covariance'] = sill - semivar

        # All done
        return

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

    def plot(self, data='covariance'):
        '''
        Plots the covariance function.
        Args:
            * data  : Can be covariance or semivariogram or all.
        '''

        # Create a figure
        fig = plt.figure()

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

        # Show me
        plt.show()

        # All done
        return

#EOF
