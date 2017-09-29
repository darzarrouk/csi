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
from .insar import insar
from .gpstimeseries import gpstimeseries

class insartimeseries(insar):

    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, lat0=None):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * utmzone   : UTM zone. (optional, default is 10 (Western US))
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(insartimeseries,self).__init__(name,
                                             utmzone=utmzone,
                                             ellps=ellps,
                                             lon0=lon0, 
                                             lat0=lat0,
                                             verbose=False) 

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

    def initializeTimeSeries(self, lon, lat, incidence=None, heading=None, elevation=None, Dates=None, Start=None, Increment=None, Steps=None, dtype='f'):
        '''
        Initializes the time series object using a series of dates.
        Args:
            * lon           : Can be an array or a string.
            * lat           : Can be an array or a string.
            * incidence     : Can be an array or a string.
            * heading       : Can be an array or a string.
            * elevation     : Can be an array or a string.
        Can give different inputs:
        Mode 1
            * Dates         : List of dates (datetime object)
        Mode 2
            * Start         : Starting date (datetime object)
            * Increment     : Increment in days
            * Steps         : How many steps
        '''

        # Set up a list of Dates
        if Dates is not None:
            self.time = Dates
        else:
            assert Start is not None, 'Need a starting point...'
            assert Increment is not None, 'Need an increment in days...'
            assert Steps is not None, 'Need a number of steps...'
            self.time = [dt.datetime.fromordinal(Start.toordinal()+Increment*i) for i in range(Steps)] 

        # Get some size
        if type(lon) is str:
            lon = np.fromfile(lon, dtype=dtype)
        nSamples = lon.shape[0]

        # Set the main stuff
        self.read_from_binary(np.ones((nSamples,)), lon, lat, incidence=incidence, heading=heading, dtype=dtype, remove_nan=False, remove_zeros=False)
        self.vel = None

        # Set the elevation if provided
        if elevation is not None:
            self.elevation = insar('Elevation', utmzone=self.utmzone, verbose=False, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
            self.elevation.read_from_binary(elevation, lon, lat, incidence=None, heading=None, remove_nan=False, remove_zeros=False)
            self.z = self.elevation.vel

        # Create timeseries
        self.timeseries = []

        # Create an insarrate instance for each step
        for date in self.time:
            sar = insar(date.isoformat(), utmzone=self.utmzone, verbose=False, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
            sar.read_from_binary(np.zeros((nSamples,)), lon, lat, incidence=incidence, heading=heading, dtype=dtype, remove_nan=False, remove_zeros=False)
            self.timeseries.append(sar)

        # All done
        return
 
    def getInsarAtDate(self, date, verbose=True):
        '''
        Given a datetime instance, returns the corresponding insar instance.

        Args:
            * date          : datetime instance.
        '''

        # Find the right index
        udate = np.flatnonzero(np.array(self.time)==date)

        # All done 
        if udate.size==1:
            # Speak to me
            if verbose:
                print('Returning insar image at date {}'.format(self.time[udate]))
            return self.timeseries[udate]
        else:
            if verbose:
                print('No date found')
            return None

    def setTimeSeries(self, timeseries):
        '''
        Sets the values in the time series.
        Args:
            * timeseries    : List of arrays of the right size.
        '''

        # Assert
        assert type(timeseries) is list, 'Input argument must be a list...'

        # Iterate
        for data, ts in zip(timeseries, self.timeseries):
            # Convert
            data = np.array(data)
            # Check
            assert data.shape==self.lon.shape, 'Wrong size for input for {}...'.format(ts.name)
            # Set
            ts.vel = data

        # All done
        return

    def referenceTimeSeries2Date(self, date):
        '''
        References the time series to the date given as argument.
        Args:
            * date              : Can a be tuple/list of 3 integers (year, month, day)
                                  or a tuple/list of 6 integers (year, month, day, hour, min, sec) 
                                  or a datetime object.
        '''

        # Make date
        if type(date) in (tuple, list):
            if len(date)==3:
                date = dt.datetime(date[0], date[1], date[2])
            elif len(date)==6:
                date = dt.datetime(date[0], date[1], date[2], date[3], date[4], date[5])
            elif len(date)==1:
                print('Unknow date format...')
                sys.exit(1)
        assert (type(date) is type(self.time[0])), 'Provided date can be \n \
                tuple of (year, month, day), \n \
                tuple of (year, month, day ,hour, min,s), \n \
                datetime.datetime object'

        # Get the reference
        i = np.flatnonzero(np.array(self.time)==date)[0]
        reference = copy.deepcopy(self.timeseries[i].vel)

        # Reference
        for ts in self.timeseries:
            ts.vel -= reference

        # All done
        return

    def write2h5file(self, h5file, field='recons'):
        '''
        Writes the time series in a h5file

        Args:
            * h5file        : Output h5file

        Kwargs:
            * field         : name of the field in the h5 file
        '''

        # Open the fike
        h5out = h5py.File(h5file, 'w')

        # Create the data field
        time = h5out.create_dataset('dates', shape=(len(self.timeseries),))
        data = h5out.create_dataset(field, shape=(len(self.timeseries), len(self.timeseries[0].lon), 1))

        # Iterate over time
        for itime, date in enumerate(self.time):

            # Get the time series
            data[itime,:,0] = self.timeseries[itime].vel
            time[itime] = date.toordinal()

        # Close the file
        h5out.close()

        # All done
        return

    def readFromGIAnT(self, h5file, setmaster2zero=None,
                            zfile=None, lonfile=None, latfile=None, 
                            incidence=None, heading=None, inctype='onefloat', 
                            field='recons', keepnan=False, mask=None, readModel=False):
        '''
        Read the output from a tipycal GIAnT h5 output file.
        Args:
            * h5file        : Input h5file
            * setmaster2zero: If index is provided, master will be replaced by zeros (no substraction)
            * zfile         : File with elevation (float32)
            * lonfile       : File with longitudes (float32)
            * latfile       : File with latitudes (float32)
            * incidence     : Incidence angle (degree)
            * heading       : Heading angle (degree)
            * inctype       : Type of the incidence and heading values (see insar.py for details)
                              Can be 'onefloat', 'grd', 'binary', 'binaryfloat'
            * field         : Name of the field in the h5 file.
            * mask          : Adds a common mask to the data.
                                mask is an array the same size as the data with nans and 1
                                It can also be a tuple with a key word in 
                                the h5file, a value and 'above' or 'under'
            * readModel     : Reads the model parameters
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
            self.elevation = insar('Elevation', utmzone=self.utmzone, verbose=False, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
            self.elevation.read_from_binary(zfile, lonfile, latfile, 
                    incidence=None, heading=None, remove_nan=False, remove_zeros=False)
            self.z = self.elevation.vel

        # Get the time
        dates = h5in['dates']
        self.time = []
        for i in range(nDates):
            self.time.append(dt.datetime.fromordinal(int(dates[i])))

        # Create a list to hold the dates
        self.timeseries = []

        # Iterate over the dates
        for i in range(nDates):
            
            # Get things
            date = self.time[i]
            dat = data[i,:,:]

            # Mask?
            if mask is not None:
                dat *= mask

            # check master date
            if i is setmaster2zero:
                dat[:,:] = 0.

            # Create an insar object
            sar = insar('{} {}'.format(self.name,date.isoformat()), utmzone=self.utmzone, 
                        verbose=False, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

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
            sar.inchd2los(incidence, heading, origin=inctype)

            # Store the object in the list
            self.timeseries.append(sar)

        # Keep incidence and heading
        self.incidence = incidence
        self.heading = heading
        self.inctype = inctype

        # if readVel
        if readModel:
            self.readModelFromGIAnT()

        # Make a common mask if asked
        if not keepnan:
            # Create an array
            checkNaNs = np.zeros(self.lon.shape)
            checkNaNs[:] = False
            # Trash the pixels where there is only NaNs
            for sar in self.timeseries:
                checkNaNs += np.isfinite(sar.vel)
            uu = np.flatnonzero(checkNaNs==0)
            # Keep 'em
            for sar in self.timeseries:
                sar.reject_pixel(uu)
            if zfile is not None:
                elevation.reject_pixel(uu)
            self.reject_pixel(uu)

        # all done
        return
 
    def readModelFromGIAnT(self):
        '''
        Read the model parameters from GIAnT after one has read the time series
        '''

        # This step needs tsinsar
        import tsinsar
    
        # Get the representation
        self.rep = tsinsar.mName2Rep(self.h5in['mName'].value)

        # Iterate over the model parameters 
        self.models = []
        for u, mName in enumerate(self.h5in['mName']):
            
            # Build the guy
            param = insar('Parameter {}'.format(mName), 
                          utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps,
                          verbose=False)
            param.vel = self.h5in['parms'][:,:,u].flatten()
            param.lon = self.lon
            param.lat = self.lat
            param.x = self.x
            param.y = self.y
            param.corner = None
            param.err = None
            param.factor=1.0
            param.inchd2los(self.incidence, self.heading, origin=self.inctype)

            # Save it 
            self.models.append(param)

        # All done
        return    

    def removeDate(self, date):
        '''
        Remove one date from the time series.
        Args:
            * date      : tuple of (year, month, day) or (year, month, day ,hour, min,s)
        '''

        # Make date
        if type(date) in (tuple, list):
            if len(date)==3:
                date = dt.datetime(date[0], date[1], date[2])
            elif len(date)==6:
                date = dt.datetime(date[0], date[1], date[2], date[3], date[4], date[5])
            elif len(date)==1:
                print('Unknow date format...')
                sys.exit(1)
        assert (type(date) is type(self.time[0])), 'Provided date can be \n \
                tuple of (year, month, day), \n \
                tuple of (year, month, day ,hour, min,s), \n \
                datetime.datetime object'

        # Find the date
        i = np.flatnonzero(np.array(self.time)==date)

        # Remove it
        if len(i)>0:
            del self.timeseries[i[0]]
            del self.time[i[0]]
        else:
            print('No date to remove')

        # All done
        return

    def removeDates(self, dates):
        '''
        Remove a list of dates from the time series.
        Args:
            * dates     : List of dates to be removed.
        '''
        
        # Iterate
        for date in dates:
            self.removeDate(date)

        # All done
        return

    def dates2time(self, start=0):
        '''
        Computes a time vector in years, starting from the date #start.
        Args:
            * start     : Index of the starting date.
        '''

        # Create a list
        Time = []

        # Iterate over the dates
        for date in self.time:
            Time.append(date.toordinal())

        # Convert to years
        Time = np.array(Time)/365.25

        # Reference
        Time -= Time[start]

        # All done
        return Time

    def extractAroundGPS(self, gps, distance, doprojection=True, reference=False, verbose=False):
        '''
        Returns a gps object with values projected along the LOS around the 
        gps stations included in gps. In addition, it projects the gps displacements 
        along the LOS

        Args:
            * gps           : gps object
            * distance      : distance to consider around the stations
            * doprojection  : Projects the gps enu disp into the los as well
            * reference     : if True, removes to the InSAR the average gps displacemnt in 
                              the LOS for the points overlapping in time.
        '''

        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Projecting GPS into InSAR LOS')

        # Create a gps object 
        out = copy.deepcopy(gps)

        # Initialize time series
        out.initializeTimeSeries(time=np.array(self.time), los=True, verbose=False)

        # Line-of-sight
        los = {}

        # Iterate over time
        for idate,insar in enumerate(self.timeseries):
            if verbose:
                sys.stdout.write('\r Date: {}'.format(self.time[idate].isoformat()))
            # Extract the values at this date
            tmp = insar.extractAroundGPS(gps, distance, doprojection=False)
            # Iterate over the station names to store correctly
            for istation, station in enumerate(out.station):
                assert tmp.station[istation]==station, 'Wrong station name'
                vel, err = tmp.vel_los[istation], tmp.err_los[istation]
                los[station] = tmp.los[istation]
                out.timeseries[station].los.value[idate] = vel
                out.timeseries[station].los.error[idate] = err

        # Print
        if verbose:
            print(' All Done')

        # project
        if reference or doprojection:
            for station in gps.station:
                if los[station] is not None:
                    gps.timeseries[station].project2InSAR(los[station])
                else:
                    gps.timeseries[station].los = None

        # Reference
        if reference:
            for station in gps.station:
                # Get the insar projected time series
                insar = out.timeseries[station].los
                gpspr = gps.timeseries[station].los
                if insar is not None and gpspr is not None:
                    # Find the common dates 
                    diff = []
                    for itime, time in enumerate(insar.time):
                        u = np.flatnonzero(gpspr.time==time)
                        if len(u)>0:
                            diff.append(insar.value[itime]-gpspr.value[u[0]])
                    # Average and correct
                    if len(diff)>0:
                        average = np.nanmean(diff)
                        if not np.isnan(average):
                            insar.value -= average

        # Save the los vectors
        out.losvectors = los

        # All done
        return out
        
    def reference2timeseries(self, gpstimeseries, distance=4.0, verbose=True, parameters=1, 
                                   daysaround=2, propagate='mean'):
        '''
        References the InSAR time series to the GPS time series.
        We estimate a linear function of range and azimuth on the difference 
        between InSAR and GPS at each insar epoch.

        We solve for the a, b, c terms in the equation:
            d_sar = d_gps + a + b*range + c*azimuth + d*azimuth*range 

        Args:
            * gpstimeseries     : A gpstimeseries instance.

        kwArgs:
            * daysaround        : How many days around the date do we consider
            * distance          : Diameter of the circle surrounding a gps station
                                  to gather InSAR points
            * verbose           : Talk to me
            * parameters        : 1, 3 or 4
            * propagate         : 'mean' if no gps data available

        Returns:
            * sar@gps           : a gpstimeseries instance with the InSAR values
                                  around the GPS stations
        '''

        # Save the references
        references = []
        GPS = []

        # Iterate over the InSAR time series
        for sar, date in zip(self.timeseries, self.time):

            # Get the gps displacements at the masterdate
            gps = gpstimeseries.getNetworkAtDate(date, verbose=False)
            GPS.append(gps)

            # Days around
            if daysaround>1:
                ndates = np.ones((gps.vel_enu.shape[0], 3))
                for i in range(1, daysaround):
                    new = gpstimeseries.getNetworkAtDate(date+dt.timedelta(i), verbose=False)
                    ndates[np.isfinite(new.vel_enu[:,0]),:] += 1.
                    gps.vel_enu[np.isfinite(new.vel_enu[:,0]),:] += new.vel_enu[np.isfinite(new.vel_enu[:,0]),:]
                    new = gpstimeseries.getNetworkAtDate(date-dt.timedelta(i), verbose=False)
                    ndates[np.isfinite(new.vel_enu[:,0]),:] += 1.
                    gps.vel_enu[np.isfinite(new.vel_enu[:,0]),:] += new.vel_enu[np.isfinite(new.vel_enu[:,0]),:]
                gps.vel_enu /= ndates

            # Extract the sar displacement around the GPS stations
            saratgps = sar.extractAroundGPS(gps, distance, doprojection=True)

            # Get the position of the GPS stations and SAR points
            x = gps.x - np.mean(gps.x)
            y = gps.y - np.mean(gps.y)

            # Get the difference between the gps and sar
            d = saratgps.vel_los - gps.vel_los

            # Kick out the NaNs if there is some
            u = np.flatnonzero(np.isnan(d))
            x = np.delete(x,u)
            y = np.delete(y,u)
            d = np.delete(d,u)

            # Estimate a linear transform to match them
            if len(d)>=parameters:
                G = np.ones((len(d), parameters))
                if parameters>=3:
                    G[:,1] = x
                    G[:,2] = y
                if parameters==4:
                    G[:,3] = x*y
                m, res, rank, s = np.linalg.lstsq(G, d, rcond=1e-8)
            else:
                m = None

            if m is not None:
                plt.plot(d, '.k', np.dot(G,m), '.r')
                plt.show()

            # Save that
            references.append(m)
 
        # Save
        self.references = references

        # Clean up references
        if propagate=='mean':
            
            # Get mean
            mmean = np.array([m for m in references if m is not None]).mean(axis=0)
            references = [mmean if m is None else m for m in references]

        else:
            assert False, 'No other propagation method implemented yet'

        # Iterate over the frames to reference
        for sar, gps, ref in zip(self.timeseries, GPS, references): 

            # Build G
            G = np.ones((len(sar.x), parameters))
            if parameters>=3:
                G[:,1] = sar.x - np.mean(gps.x)
                G[:,2] = sar.y - np.mean(gps.y)
            if parameters==4:
                G[:,3] = G[:,1]*G[:,2]

            # Correct
            sar.vel -= np.dot(G,ref)

        # Re-do the extraction to return it
        saratgps = self.extractAroundGPS(gpstimeseries, distance, 
                                         verbose=False, reference=False, 
                                         doprojection=True)

        # All done
        return saratgps

    def getProfiles(self, prefix, loncenter, latcenter, length, azimuth, width, verbose=False):
        '''
        Get a profile for each time step
        for Arguments, check in insar getprofile
        '''

        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Get Profile for each time step of time series {}: '.format(self.name))

        # Simply iterate over the steps
        for date, sar in zip(self.time, self.timeseries):

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
        for date, sar in zip(self.time, self.timeseries):

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

        # Make date
        if type(date) in (tuple, list):
            if len(date)==3:
                date = dt.datetime(date[0], date[1], date[2])
            elif len(date)==6:
                date = dt.datetime(date[0], date[1], date[2], date[3], date[4], date[5])
            elif len(date)==1:
                print('Unknow date format...')
                sys.exit(1)
        assert (type(date) is type(self.time[0])), 'Provided date can be \n \
                tuple of (year, month, day), \n \
                tuple of (year, month, day ,hour, min,s), \n \
                datetime.datetime object'

        # Get the profile
        i = np.flatnonzero(np.array(self.time)==date)[0]
        pname = '{} {}'.format(prefix, date.isoformat())
        refProfile = self.timeseries[i].profiles[pname]

        # Create a linear interpolator
        x = refProfile['Distance']
        y = refProfile['LOS Velocity']
        intProf = sciint.interp1d(x, y, kind='linear', bounds_error=False)

        # Iterate on the profiles
        for date, sar in zip(self.time, self.timeseries):

            # Get profile
            pname = '{} {}'.format(prefix, date.isoformat())
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
        for date, sar in zip(self.time, self.timeseries):

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
        Wrapper around cleanProfile of insar.
        '''

        if verbose:
            print('Clean the profiles:')

        # Simply iterate over the steps
        for date, sar in zip(self.time, self.timeseries):

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
        for date, sar in zip(self.time, self.timeseries):

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
        for date, sar in zip(self.time, self.timeseries):

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

    def write2GRDs(self, prefix, interp=100, cmd='surface', oversample=1, tension=None, verbose=False, useGMT=False):
        '''
        Write all the dates to GRD files.
        For arg description, see insar.write2grd
        '''

        # print stuffs
        if verbose:
            print('Writing each time step to a GRD file')

        # Simply iterate over the insar
        i = 1
        for sar, date in zip(self.timeseries, self.time):

            # Make a filename
            d = '{}{}{}.grd'.format(date.isoformat()[:4], date.isoformat()[5:7], date.isoformat()[8:])
            filename = prefix+d

            # Write things
            if verbose:
                sys.stdout.write('\r {:3d} / {:3d}    Writing to file {}'.format(i, len(self.time), filename))
                sys.stdout.flush()

            # Use the insar routine to write the GRD
            sar.write2grd(filename, oversample=oversample, interp=interp, cmd=cmd, useGMT=useGMT)

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
        for date, sar in zip(self.time, self.timeseries):

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
