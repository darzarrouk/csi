'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import sys
import h5py
import datetime as dt

# Personals
from .SourceInv import SourceInv
from .insarrates import insarrates

class insartimeseries(SourceInv):

    def __init__(self, name, utmzone='10', ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(insartimeseries,self).__init__(name,utmzone,ellps) 

        # Initialize the data set 
        self.dtype = 'insartimeseries'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR Time Series data set {}".format(self.name))

        # Initialize some things
        self.vel = None
        self.synth = None
        self.err = None
        self.lon = None
        self.lat = None
        self.los = None
        self.corner = None
        self.xycorner = None
        self.Cd = None

        # All done
        return

    def readFromGIAnT(self, h5file, zfile=None, lonfile=None, latfile=None, incidence=None, heading=None, field='recons', keepnan=False):
        '''
        Read the output from a tipycal GIAnT h5 output file.
        Args:
            * h5file        : Input h5file
            * zfile         : File with elevation (float32)
            * lonfile       : File with longitudes (float32)
            * latfile       : File with latitudes (float32)
            * incidence     : Incidence angle (degree)
            * heading       : Heading angle (degree)
            * field         : Name of the field in the h5 file.
        '''

        # open the h5file
        h5in = h5py.File(h5file, 'r')
        self.h5in = h5in

        # Get the data
        data = h5in[field]

        # Get some sizes
        nDates = data.shape[0]
        nLines = data.shape[1]
        nCols  = data.shape[2]

        # Read Lon Lat
        if lonfile is not None:
            self.lon = np.fromfile(lonfile, dtype=np.float32)
        if latfile is not None:
            self.lat = np.fromfile(latfile, dtype=np.float32)
        if zfile is not None:
            self.z = np.fromfile(zfile, dtype=np.float32)

        # Compute utm
        self.x, self.y = self.ll2xy(self.lon, self.lat) 

        # Get the time
        dates = h5in['dates']
        self.dates = []
        for i in range(nDates):
            self.dates.append(dt.date.fromordinal(int(dates[i])))

        # Create a list to hold the dates
        self.timeseries = []

        # Iterate over the dates
        for i in range(nDates):
            
            # Get things
            date = self.dates[i]
            dat = data[i,:,:]

            # Create an insar object
            sar = insarrates(date.isoformat(), utmzone=self.utmzone, verbose=False)

            # Get places where we have finite values
            if not keepnan:
                ii = np.flatnonzero(np.isfinite(dat))
            else:
                ii = np.flatnonzero(dat)

            # Put thing in the insarrate object
            sar.vel = dat.flatten()[ii]
            sar.lon = self.lon[ii]
            sar.lat = self.lat[ii]
            sar.x = self.x[ii]
            sar.y = self.y[ii]

            # Things should remain None
            sar.corner = None
            sar.err = None

            # Set factor
            sar.factor = 1.0

            # Take care of the LOS
            sar.inchd2los(incidence, heading)

            # Store the object in the list
            self.timeseries.append(sar)

        # all done
        return

    def getProfiles(self, prefix, loncenter, latcenter, length, azimuth, width, verbose=False):
        '''
        Get a profile for each time step
        for Arguments, check in insarrates getprofile
        '''

        if verbose:
            print('Get Profile for each time step: ')

        # Simply iterate over the steps
        for date, sar in zip(self.dates, self.timeseries):

            # Make a name
            pname = '{} {}'.format(prefix, date.isoformat())

            # Print
            if verbose:
                sys.stdout.write('\r {} '.format(pname))
                sys.stdout.flush()

            # Get the profile
            sar.getprofile(pname, loncenter, latcenter, length, azimuth, width)
        
        # verbose
        if verbose:
            print('')

        # All done
        return

    def runingAverageProfiles(self, prefix, window, verbose=False, method='mean'):
        '''
        Runs the average window mean on profiles.
        '''

        # Verbose
        if verbose:
            print('Runing Average on profiles: ')

        # Simply iterate over the steps
        for date, sar in zip(self.dates, self.timeseries):

            # make a name
            pname = '{} {}'.format(prefix, date.isoformat())

            # Print
            if verbose:
                sys.stdout.write('\r {} '.format(pname))
                sys.stdout.flush()

            # Smooth the profile
            sar.runingAverageProfile(pname, window, method=method)

        # verbose
        if verbose:
            print('')

        # ALl done
        return

    def referenceProfiles(self, prefix, xmin, xmax, verbose=False):
        '''
        Removes the mean value of points between xmin and xmax for all the profiles.
        '''

        # verbose
        if verbose:
            print('Referencing profiles:')

        # Simply iterate over the steps
        for date, sar in zip(self.dates, self.timeseries):

            # make a name
            pname = '{} {}'.format(prefix, date.isoformat())

            # Print
            if verbose:
                sys.stdout.write('\r {} '.format(pname))
                sys.stdout.flush()

            # Reference
            sar.referenceProfile(pname, xmin, xmax)

        # verbose
        if verbose:
            print('')

        # ALl done
        return

    def cleanProfiles(self, prefix, xlim=None, zlim=None, verbose=False):
        '''
        Wrapper around cleanProfile of insarrates.
        '''

        if verbose:
            print('Clean the profiles:')

        # Simply iterate over the steps
        for date, sar in zip(self.dates, self.timeseries):

            # Make a name
            pname = '{} {}'.format(prefix, date.isoformat())

            # Print
            if verbose:
                sys.stdout.write('\r {} '.format(pname))
                sys.stdout.flush()

            # Cleanup the profile
            sar.cleanProfile(pname, xlim=xlim, zlim=zlim)

        # verbose
        if verbose:
            print('')

        # All done 
        return

    def writeProfiles2Files(self, profileprefix, outprefix, fault=None, verbose=False):
        '''
        Write all the profiles to a file.
        '''

        if verbose:
            print('Write Profile to text files:')

        # Simply iterate over the steps
        for date, sar in zip(self.dates, self.timeseries):

            # make a name
            pname = '{} {}'.format(profileprefix, date.isoformat())

            # Print
            if verbose:
                sys.stdout.write('\r {} '.format(pname))
                sys.stdout.flush()

            # make a filename
            fname = '{}{}{}{}.dat'.format(outprefix, date.isoformat()[:4], date.isoformat()[5:7], date.isoformat()[8:])

            # Write to file
            sar.writeProfile2File(pname, fname, fault=fault)

        # verbose
        if verbose:
            print('')

        # All done
        return

    def writeProfiles2OneFile(self, profileprefix, filename, verbose=False):
        '''
        Write the profiles to one file
        '''

        # verbose
        if verbose:
            print('Write Profiles to one text file: {}'.format(filename))

        # Open the file (this one has no header)
        fout = open(filename, 'w')

        # Iterate over the profiles
        for date, sar in zip(self.dates, self.timeseries):

            # make a name
            pname = '{} {}'.format(profileprefix, date.isoformat())  

            # Print
            if verbose:
                sys.stdout.write('\r {} '.format(pname))
                sys.stdout.flush()      

            # Get the values
            distance = sar.profiles[pname]['Distance'].tolist()
            values = sar.profiles[pname]['LOS Velocity'].tolist()
            
            # Write a starter
            fout.write('> \n')

            # Loop and write
            for d, v in zip(distance, values):
                fout.write('{}T00:00:00 {} {} \n'.format(date.isoformat(), d, v))
                        
        # Close the file
        fout.close()

        # verbose
        if verbose:
            print('')
            
        # All done
        return

    def write2GRDs(self, prefix, interp=100, cmd='surface', oversample=1, tension=None, verbose=False):
        '''
        Write all the dates to GRD files.
        For arg description, see insarrates.write2grd
        '''

        # print stuffs
        if verbose:
            print('Writing each time step to a GRD file')

        # Simply iterate over the insarrates
        i = 1
        for sar, date in zip(self.timeseries, self.dates):

            # Make a filename
            d = '{}{}{}.grd'.format(date.isoformat()[:4], date.isoformat()[5:7], date.isoformat()[8:])
            filename = prefix+d

            # Write things
            if verbose:
                sys.stdout.write('\r {:3d} / {:3d}    Writing to file {}'.format(i, len(self.dates), filename))
                sys.stdout.flush()

            # Use the insarrates routine to write the GRD
            sar.write2grd(filename, oversample=1, interp=interp, cmd=cmd)

            # counter
            i += 1

        if verbose:
            print('')

        # All done
        return

    def plotProfiles(self, prefix, figure=124, show=True, norm=None, xlim=None, zlim=None):
        '''
        Plots the profiles in 3D plot.
        '''

        # Create the figure
        fig = plt.figure(figure)
        ax = fig.add_subplot(111, projection='3d')

        # loop over the profiles to plot these
        for date, sar in zip(self.dates, self.timeseries):

            # Profile name
            pname = '{} {}'.format(prefix, date.isoformat())

            # Clean the profile
            if xlim is not None:
                xmin = xlim[0]
                xmax = xmax[1]
                ii = sar._getindexXlimProfile(pname, xmin, xmax)
            if zlim is not None:
                zmin = zlim[0]
                zmax = zlim[1]
                uu = sar._getindexZlimProfile(pname, zmin, zmax)
            if (xlim is not None) and (zlim is not None):
                jj = np.intersect1d(ii,uu).tolist()
            elif (zlim is None) and (xlim is not None):
                jj = ii
            elif (zlim is not None) and (xlim is None):
                jj = uu
            else:
                jj = range(sar.profiles[pname]['Distance'].shape[0])

            # Get distance
            distance = sar.profiles[pname]['Distance'][jj].tolist()

            # Get values
            values = sar.profiles[pname]['LOS Velocity'][jj].tolist()

            # Get date
            nDate = [date.toordinal() for i in range(len(distance))]
                
            # Plot that
            ax.plot3D(nDate, distance, values, marker='.', color='k', linewidth=0.0 )

        # norm
        if norm is not None:
            ax.set_zlim(norm[0], norm[1])

        # If show
        if show:
            plt.show()

        return
            


#EOF
