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
import scipy.interpolate as sciint
import copy

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

    def readFromGIAnT(self, h5file, zfile=None, lonfile=None, latfile=None, incidence=None, heading=None, field='recons', keepnan=False, mask=None, readVel=None):
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
            * mask          : Adds a common mask to the data.
                                mask is an array the same size as the data with nans and 1
                                It can also be a tuple with a key word in 
                                the h5file, a value and 'above' or 'under'
            * readVel       : If not None, reads the values of parameter given.
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

        # Deal with the mask instructions
        if mask is not None:
            if type(mask) is tuple:
                key = mask[0]
                value = mask[1]
                instruction = mask[2]
                mask = np.ones((nLines, nCols))
                if instruction in ('above'):
                    mask[np.where(h5in[key][:]>value)] = np.nan
                elif instruction in ('under'):
                    mask[np.where(h5in[key][:]<value)] = np.nan
                else:
                    print('Unknow instruction type for Masking...')
                    sys.exit(1)

        # Read Lon Lat
        if lonfile is not None:
            self.lon = np.fromfile(lonfile, dtype=np.float32)
        if latfile is not None:
            self.lat = np.fromfile(latfile, dtype=np.float32)

        # Compute utm
        self.x, self.y = self.ll2xy(self.lon, self.lat) 

        # Elevation
        if zfile is not None:
            self.elevation = insarrates('Elevation', utmzone=self.utmzone, verbose=False)
            self.elevation.read_from_binary(zfile, lonfile, latfile, 
                    incidence=None, heading=None, remove_nan=False, remove_zeros=False)
            self.z = self.elevation.vel

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

            # Mask?
            if mask is not None:
                dat *= mask

            # Create an insar object
            sar = insarrates(date.isoformat(), utmzone=self.utmzone, verbose=False)

            # Put thing in the insarrate object
            sar.vel = dat.flatten()
            sar.lon = self.lon
            sar.lat = self.lat
            sar.x = self.x
            sar.y = self.y

            # Things should remain None
            sar.corner = None
            sar.err = None

            # Set factor
            sar.factor = 1.0

            # Take care of the LOS
            sar.inchd2los(incidence, heading)

            # Store the object in the list
            self.timeseries.append(sar)

        # if readVel
        if readVel is not None:
            u = np.flatnonzero(self.h5in['mName'][:]==readVel)
            if len(u)==1:
                self.param = insarrates('Parameter {}'.format(readVel), utmzone=self.utmzone, 
                        verbose=False)
                self.param.vel = h5in['parms'][u[0]]
                self.param.lon = self.lon
                self.param.lat = self.lat
                self.param.x = self.x
                self.param.y = self.y
                self.param.corner = None
                self.param.err = None
                self.param.factor=1.0
                self.param.inchd2los(incidence, heading)
            else:
                print('{}: No such parameter in {}'.format(readVel,h5file))
                sys.exit()
        else:
            self.param = None

        # Make a common mask if asked
        if not keepnan:
            # Create an array
            checkNaNs = np.array(self.lon.shape)
            checkNaNs[:] = False
            # Trash the pixels where there is only NaNs
            for sar in self.timeseries:
                checkNaNs += np.isfinite(sar.vel)
            uu = np.flatnonzero(not checkNaNs)
            # Keep 'em
            for sar in self.timeseries:
                sar.reject_pixels(uu)
            elevation.reject_pixels(uu)
            if self.param is not None:
                self.param.reject_pixels(uu)

        # all done
        return
    
    def removeDate(self, date):
        '''
        Remove one date from the time series.
        Args:
            * date      : tuple of (year, month, day) or (year, month, day ,hour, min,s)
        '''

        # Make date
        if len(date)==3:
            date = dt.date(date[0], date[1], date[2])
        elif len(date)==6:
            date = dt.datetime(date[0], date[1], date[2], date[3], date[4], date[5])
        else:
            print('Unknow date format...')
            sys.exit(1)

        # Find the date
        i = np.flatnonzero(np.array(self.dates)==date)

        # Remove it
        if len(i)>0:
            del self.timeseries[i[0]]
            del self.dates[i[0]]
        else:
            print('No date to remove')

        # All done
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
        
        # Elevation
        if hasattr(self, 'elevation'):
            pname = 'Elevation {}'.format(prefix)
            self.elevation.getprofile(pname, loncenter,latcenter, length, azimuth, width)

        # Parameter
        if hasattr(self, 'param'):
            if self.param is not None:
                pname = '{} {}'.format(self.param.name,prefix)
                self.param.getprofile(pname, loncenter, latcenter, length, azimuth, width)

        # verbose
        if verbose:
            print('')

        # All done
        return

    def smoothProfiles(self, prefix, window, verbose=False, method='mean'):
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
            try:
                sar.smoothProfile(pname, window, method=method)
            except:
                # Copy the old profile and modify it
                newName = 'Smoothed {}'.format(sar.name)
                sar.profiles[newName] = copy.deepcopy(sar.profiles[sar.name])
                sar.profiles[newName]['LOS Velocity'][:] = np.nan
                sar.profiles[newName]['LOS Error'][:] = np.nan
                sar.profiles[newName]['Distance'][:] = np.nan

        # verbose
        if verbose:
            print('')

        # ALl done
        return

    def referenceProfiles2Date(self, prefix, date):
        '''
        Removes the profile at date 'date' to all the profiles in the time series.
        Args:
            * prefix        : Name of the profiles
            * date          : Tuple of 3 or 6 numbers for the date
                 date = (year(int), month(int), day(int))
              or date = (year(int), month(int), day(int), hour(int), min(int), s(float))
        '''

        # Get the date
        if len(date)==3:
            date = dt.date(date[0], date[1], date[2])
        elif len(date)==6:
            date = dt.datetime(date[0], date[1], date[2],
                               date[3], date[4], date[5])
        else:
            print('Date should be tuple of length 3 or 6')
            sys.exit(1)

        # Get the profile
        i = np.flatnonzero(np.array(self.dates)==date)[0]
        pname = '{} {}'.format(prefix, date)
        refProfile = self.timeseries[i].profiles[pname]

        # Create a linear interpolator
        x = refProfile['Distance']
        y = refProfile['LOS Velocity']
        intProf = sciint.interp1d(x, y, kind='linear', bounds_error=False)

        # Iterate on the profiles
        for date, sar in zip(self.dates, self.timeseries):

            # Get profile
            pname = '{} {}'.format(prefix, date)
            profile = sar.profiles[pname]

            # Copy profile
            newProfile = copy.deepcopy(profile)
            newName = 'Referenced {}'.format(pname)
            sar.profiles[newName] = newProfile

            # Get x-position
            x = sar.profiles[newName]['Distance']

            # Interpolate
            y = intProf(x)

            # Difference
            sar.profiles[newName]['LOS Velocity'] -= y

        # All done
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

    def writeProfiles2Files(self, profileprefix, outprefix, fault=None, verbose=False, smoothed=False):
        '''
        Write all the profiles to a file.
        '''

        if verbose:
            print('Write Profile to text files:')

        # Simply iterate over the steps
        for date, sar in zip(self.dates, self.timeseries):

            # make a name
            pname = '{} {}'.format(profileprefix, date.isoformat())

            # If smoothed
            if smoothed:
                pname = 'Smoothed {}'.format(pname)

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

    def writeProfiles2OneFile(self, profileprefix, filename, verbose=False, smoothed=False):
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

            # Smoothed?
            if smoothed:
                pname = 'Smoothed {}'.format(pname)

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

    def plotProfiles(self, prefix, figure=124, show=True, norm=None, xlim=None, zlim=None, marker='.', color='k'):
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
            ax.plot3D(nDate, distance, values, 
                      marker=marker, color=color, linewidth=0.0 )

        # norm
        if norm is not None:
            ax.set_zlim(norm[0], norm[1])

        # If show
        if show:
            plt.show()

        return
            


#EOF
