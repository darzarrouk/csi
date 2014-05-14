'''
A class that deals with seismic catalogs.
This class can also deal with moment tensors.

Written by R. Jolivet, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import copy

# Personals
from .SourceInv import SourceInv

class seismiclocations(SourceInv):

    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name          : Name of the Seismic dataset.
            * utmzone       : UTM zone. Default is 10 (Western US).
        '''

        # Base class init
        super(seismiclocations, self).__init__(name, utmzone, ellps)

        # Initialize the data set type
        self.dtype = 'seismiclocations'

        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize Seismicity data set {}".format(self.name))

        # Initialize the location

        # Initialize some things
        self.time = None
        self.lon = None
        self.lat = None
        self.depth = None
        self.mag = None

        # All done
        return

    def read_from_SCSN(self,filename, header=65):
        '''
        Read the Seismic catalog from the NCSN networks (Template from F. Waldhauser).
        Args:
            * filename      : Name of the input file. 
            * header        : Size of the header.
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Read the header to figure out where is the magnitude
        desc = A[header-2].split()
        #imag = np.flatnonzero(np.array(desc)=='MAG').tolist()[0]
        #imag += 4

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            # Split the string line 
            tmp = A[i].split()

            # Get the values
            yr = np.int(tmp[0])
            mo = np.int(tmp[1])
            da = np.int(tmp[2])
            hr = np.int(tmp[3])
            mi = np.int(tmp[4])
            lat = np.float(tmp[7])
            lon = np.float(tmp[8])
            depth = np.float(tmp[9])
            mag = np.float(tmp[10])

            # Create the time object
            if mi>=60:
                mi = 59
            if mi<=-1:
                mi = 0
            d = dt.datetime(yr, mo, da, hr, mi)
            
            # Store things in self 
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.lonlat2xy()

        # All done
        return

    def read_from_NCSN(self,filename, header=65):
        '''
        Read the Seismic catalog from the NCSN networks (Template from F. Waldhauser).
        Args:
            * filename      : Name of the input file. 
            * header        : Size of the header.
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Read the header to figure out where is the magnitude
        desc = A[header-2].split()
        imag = np.flatnonzero(np.array(desc)=='MAG').tolist()[0]
        imag += 4

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            # Split the string line 
            tmp = A[i].split()

            # Get the values
            yr = np.int(tmp[0])
            mo = np.int(tmp[1])
            da = np.int(tmp[2])
            hr = np.int(tmp[3])
            mi = np.int(tmp[4])
            lat = np.float64(tmp[6])
            lon = np.float64(tmp[7])
            depth = np.float64(tmp[8])
            mag = np.float64(tmp[imag])

            # Create the time object
            d = dt.datetime(yr, mo, da, hr, mi)
            
            # Store things in self 
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.lonlat2xy()

        # All done
        return

    def read_ascii(self, infile, header=0):
        '''
        Reads data from an ascii file.
        columns are time (isoformat, lat, lon, depth, mag).
        '''

        # open the file
        fin = open(infile, 'r')

        # Read all
        All = fin.readlines()

        # Initialize things
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Loop 
        for i in range(header, len(All)):

            # Get the splitted string
            tmp = All[i].split()

            # Get values
            time = dt.datetime.strptime(tmp[0], "%Y-%m-%dT%H:%M:%S.%fZ")
            if len(tmp)>=5:
                lon = np.float(tmp[2])
                lat = np.float(tmp[1])
                depth = np.float(tmp[3])
                mag = np.float(tmp[4])
                self.time.append(time)
                self.lon.append(lon)
                self.lat.append(lat)
                self.depth.append(depth)
                self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm
        self.lonlat2xy()

        # All done
        return

    def read_CMTSolutions(self, infile):
        '''
        Reads data and moment tensors from an ascii file listing CMT solutions format
        Args:
            infile: Input file.
        '''

        # open the file
        fin = open(infile, 'r')

        # Read all
        All = fin.readlines()

        # Initialize things
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []
        self.CMTinfo = []

        # Initialize counter
        i = 0

        # Loop over the lines
        while i<len(All):
            
            # split the line
            line = All[i].split()

            # Check if line is empty
            if len(line)>0:
            
                # Check the first character
                if line[0][0] in ('P'):
                
                    # Time
                    yr = np.int(line[0][4:])
                    mo = np.int(line[1])
                    da = np.int(line[2])
                    hr = np.int(line[3])
                    mn = np.int(line[4])
                    sd = np.int(np.float(line[5]))
                    time = dt.datetime(yr, mo, da, hr, mn, sd)

                    # cmt informations
                    info = {}
                    i += 1
                    for j in range(12):
                        line = All[i].split(':')
                        name = line[0]
                        value = line[1].split()[0]
                        if name not in ('event name'):
                            value = np.float(value)
                        info[name] = value
                        i += 1

                    # Get values
                    lat = info['latitude']
                    lon = info['longitude']
                    depth = info['depth']

                    # set in self
                    self.time.append(time)
                    self.lon.append(lon)
                    self.lat.append(lat)
                    self.depth.append(depth)
                    self.CMTinfo.append(info)

                # Else
                else:
                    i += 1

            # Else
            else:
                i += 1
    
        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.depth = np.array(self.depth)

        # Compute the magnitudes
        self.Cmt2Dislocation(size=1e-1, mu=44e9, choseplane='nochoice', moment_from_tensor=True)
        self.Mo2mag()

        # Create the utm
        self.lonlat2xy()

        # All done
        return

    def selectbox(self, minlon, maxlon, minlat, maxlat, depth=100000.):
        ''' 
        Select the earthquakes in a box defined by min and max, lat and lon.
        
        Args:
            * minlon        : Minimum longitude.
            * maxlon        : Maximum longitude.
            * minlat        : Minimum latitude.
            * maxlat        : Maximum latitude.
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        print( "Selecting the earthquakes in the box Lon: {} to {} and Lat: {} to {}".format(minlon, maxlon, minlat, maxlat))
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon) & (self.depth < depth))

        # make the selection
        self._select(u)

        # All done
        return


    def selecttime(self, start=[2001, 1, 1], end=[2101, 1, 1]):
        '''
        Selects the earthquake in between two dates. Dates can be datetime.datetime or lists.
        Args:
            * start     : Beginning of the period.
            * end       : End of the period.
        '''

        # check start and end
        if (start.__class__ is float) or (start.__class__ is int) :
            st = dt.datetime(start, 1, 1)
        if (start.__class__ is list):
            if len(start) == 1:
                st = dt.datetime(start[0], 1, 1)
            elif len(start) == 2:
                st = dt.datetime(start[0], start[1], 1)
            elif len(start) == 3:
                st = dt.datetime(start[0], start[1], start[2])
            elif len(start) == 4:
                st = dt.datetime(start[0], start[1], start[2], start[3])
            elif len(start) == 5:
                st = dt.datetime(start[0], start[1], start[2], start[3], start[4])
            elif len(start) == 6:
                st = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5])
        if start.__class__ is dt.datetime:
            st = start

        if (end.__class__ is float) or (end.__class__ is int) :
            ed = dt.datetime(np.int(end), 1, 1)
        if (end.__class__ is list):
            if len(end) == 1:
                ed = dt.datetime(end[0], 1, 1)
            elif len(end) == 2:
                ed = dt.datetime(end[0], end[1], 1)
            elif len(end) == 3:
                ed = dt.datetime(end[0], end[1], end[2])
            elif len(end) == 4:
                ed = dt.datetime(end[0], end[1], end[2], end[3])
            elif len(end) == 5:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4])
            elif len(end) == 6:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5])
        if end.__class__ is dt.datetime:
            ed = end

        # Get values
        print ("Selecting earthquake between {} and {}".format(st.isoformat(),ed.isoformat()))
        u = np.flatnonzero((self.time > st) & (self.time < ed))

        # Selection 
        self._select(u)

        # All done
        return  

    def selectmagnitude(self, minimum, maximum=10):
        '''
        Selects the earthquakes between two magnitudes.
        Args:
            * minimum   : Minimum earthquake magnitude wanted.
            * maximum   : Maximum earthquake magnitude wanted.
        '''
        
        # Get the magnitude
        mag = self.mag

        # get indexes
        print ("Selecting earthquake between magnitudes {} and {}".format(minimum, maximum))
        u = np.flatnonzero((self.mag > minimum) & (self.mag < maximum))

        # Selection 
        self._select(u)

        # All done
        return

    def computeGR(self, plot=False, bins=20):
        ''' 
        Plots the Gutemberg-Richter distribution.
        Args:
            * ion       : Turns on the plt.ion().
        '''

        # Get the magnitude
        mag = self.mag

        # Get the histogram
        h, x = np.histogram(self.mag, bins=bins)
        x = (x[1:] + x[:-1])/2.

        # Store that somewhere
        self.Histogram = [x, h]

        # plot the values
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.semilogy(x, h, '.r', markersize=10, linewidth=1)
            plt.show()

        # All done
        return
    
    def fitBvalue(self, b=None):
        '''
        Fits a B-value to a Gutemberg-Righter distribution.
        option: if b is provided, then the fit is forced to have a slope b.
        '''

        # All done
        return

    def distance2fault(self, faults, distance=5.):
        '''
        Selects the earthquakes that are located less than distance away from the fault plane.
        Args:
            * faults    : List of faults
            * distance  : Threshold
        '''

        # Create the list
        u = []

        # Loop over the faults
        for fault in faults:
            dis = np.array(self._getDistance2FaultPlane(fault))
            ut = np.flatnonzero( dis < distance ) 
            for i in ut:
                u.append(i)

        # make u an array
        u = np.array(u)
        u = np.unique(u)

        # Selection
        self._select(u)

        # All done
        return

    def distance2trace(self, faults, distance=5.):
        '''
        Selects the earthquakes that are located less than 'distance' km away from a given surface fault trace.
        Args:
            * faults    : list of structures created from verticalfault.
            * distance  : threshold distance.
        '''

        # Shapely (Change for matplotlib path later)
        import shapely.geometry as sg

        # Create a list with the earthquakes locations
        LL = np.vstack((self.x, self.y)).T.tolist()

        # Create a MultiPoint object 
        PP = sg.MultiPoint(LL)

        # Loop over faults
        u = []
        for fault in faults:
            dis = []
            # Build a line object
            FF = np.vstack((fault.xf, fault.yf)).T.tolist()
            trace = sg.LineString(FF)
            # Get the distance between each point and this line
            for uu in PP.geoms:
                dis.append(trace.distance(uu))
            dis = np.array(dis)
            # Get the indexes of the ones that are close to the fault
            ut = np.flatnonzero( dis < distance )
            # Fill in u
            for i in ut:
                u.append(i)

        # make u an array
        u = np.array(u)
        u = np.unique(u)

        # selection
        self._select(u)

        # All done
        return  

    def delete2Close2Trace(self, faults, distance=1.):
        '''
        Deletes the earthquakes that are too close from the fault trace.
        '''

        # Shapely (Change for matplotlib path later)
        import shapely.geometry as sg

        # Create a list with the earthquakes locations
        LL = np.vstack((self.x, self.y)).T.tolist()

        # Create a MultiPoint object 
        PP = sg.MultiPoint(LL)

        # Loop over faults
        u = []
        for fault in faults:
            dis = []
            # Build a line object
            FF = np.vstack((fault.xf, fault.yf)).T.tolist()
            trace = sg.LineString(FF)
            # Get the distance between each point and this line
            for uu in PP.geoms:
                dis.append(trace.distance(uu))
            dis = np.array(dis)
            # Get the indexes of the ones that are away from the fault
            ut = np.flatnonzero( dis < distance )
            # Fill in u
            for i in ut:
                u.append(i)

        # Selection
        self._select(u)

        # All done
        return  

    def ProjectOnFaultTrace(self, fault, discretized=True, filename=None):
        '''
        Projects the location of the earthquake along the fault trace. 
        This routine is not a 3D one, it just deals with the surface trace of the fault.
        Args:
            fault:       Fault object that has a surface trace
            discretized: If True, then it uses the discretized fault, not the trace. Never tested with False.
        '''

        # Import needed stuff
        import scipy.spatial.distance as scidis

        # Check that we are in the same utmzone
        assert (fault.utmzone==self.utmzone), 'Fault {} utmzone is not seismiclocation {} utmzone: {} <==> {}'.format(
                    fault.name, self.name, fault.utmzone, self.utmzone)

        # discretized?
        if discretized:
            # Check
            assert (fault.xi is not None), 'Fault {} needs a discretized surface trace, Abort....'.format(fault.name)
            # Get x, y
            x = fault.xi
            y = fault.yi
            nf = fault.xi.shape[0]
        else:
            # Check
            assert (fault.xf is not None), 'Fault {} needs a surface trace, Abort....'.format(fault.name)
            # Get x, y
            x = fault.xf
            y = fault.yf
            nf = fault.xf.shape[0]

        # Compute the cumulative distance along the fault trace
        dis = fault.cumdistance(discretized=discretized)

        # If you want to store that in a file
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | time | depth | mag | AlongStrikeDistance | DistanceToFault \n')

        # Create the structure that'll hold everything
        if not hasattr(self, 'Projected'):
            self.Projected = {}

        # Create the structure that holds this particular one
        self.Projected['{}'.format(fault.name)] = {}
        proj = self.Projected['{}'.format(fault.name)]
        proj['x'] = []
        proj['y'] = []
        proj['lon'] = []
        proj['lat'] = []
        proj['time'] = []
        proj['depth'] = []
        proj['mag'] = []
        proj['AlongStrikeDistance'] = []
        proj['DistanceToFault'] = []

        # Iterate on the earthquakes
        for i in range(self.time.shape[0]):
            # Get earthquake
            qx = self.x[i]
            qy = self.y[i]
            qlon = self.lon[i]
            qlat = self.lat[i]
            qtime = self.time[i]
            qz = self.depth[i]
            qmag = self.mag[i]
            # Get distances
            d = scidis.cdist([[qx, qy]], [ [x[j], y[j]] for j in range(nf)])[0]
            # Get the smallest
            imin1 = d.argmin()
            dmin1 = d.min()
            d[imin1] = 9999999999.
            imin2 = d.argmin()
            dmin2 = d.min()
            dtot= dmin1 + dmin2
            # Do the spatial position
            qx = (x[imin1]*dmin1 + x[imin2]*dmin2)/dtot
            qy = (y[imin1]*dmin1 + y[imin2]*dmin2)/dtot
            # Put into lon lat
            qlon, qlat = self.xy2ll(qx, qy)
            # Compute the AlongStrike Distance
            if dmin1<dmin2:
                jm = imin1
            else:
                jm = imin2
            qdis = dis[jm] + np.sqrt( (qx-x[jm])**2 + (qy-y[jm])**2 )
            # Compute the distance to the fault
            dl = np.sqrt( (x[imin1]-x[imin2])**2 + (y[imin1]-y[imin2])**2 ) # 3 side of the triangle 
            semiperi = (dmin1 + dmin2 + dl)/2.                              # Semi-perimeter of the triangle
            A = semiperi*(semiperi-dmin1)*(semiperi-dmin2)*(semiperi-dl)    # Area of the triangle (Heron's formula)
            qh = 2*A/dl                                                     # Height of the triangle
            # Store all that in a structure
            proj['x'].append(qx)
            proj['y'].append(qy)
            proj['lon'].append(qlon)
            proj['lat'].append(qlat)
            proj['time'].append(qtime)
            proj['depth'].append(qz)
            proj['mag'].append(qmag)
            proj['AlongStrikeDistance'].append(qdis)
            proj['DistanceToFault'].append(qh)
            # Write to file?
            if filename is not None:
                fout.write('{} {} {} {} {} {} {} \n'.format(qlon, qlat, qtime, qz, qmag, qdis, qh))

        # Close the file
        if filename is not None:
            fout.close()

        # All done
        return

    def _getDistance2Fault(self, fault):
        '''
        Computes the distance between the fault trace and all the earthquakes.
        '''
        
        # Import
        import shapely.geometry as sg

        # Create a list with earthquakes
        LL = np.vstack((self.x, self.y)).T.tolist()
        PP = sg.MultiPoint(LL)

        # Build a line object
        FF = np.vstack((fault.xf, fault.yf)).T.tolist()
        trace = sg.LineString(FF)

        # Distance
        dis = []
        for p in PP.geoms:
            dis.append(trace.distance(p))

        # All done
        return dis

    def _getDistance2FaultPlane(self, fault):
        '''
        Computes the distance between the fault plane and all the earthquakes.
        '''

        # import scipy
        import scipy.spatial.distance as scidis
        
        # Create a list
        dis = []

        # Create the list of vertices
        if fault.patchType in ('triangle'):
            vertices = fault.gocad_vertices
        elif fault.patchType in ('rectangle'):
            vertices = []
            for p in fault.patch:
                for i in range(4):
                    vertices.append(p[i])

        # Loop on the earthquakes
        for i in range(self.mag.shape[0]):
            
            # Get position
            x = self.x[i]
            y = self.y[i]
            z = self.depth[i]

            # Get the min distance
            d = scidis.cdist([[x, y, z]], vertices).min()

            # Append
            dis.append(d)

        # All done
        return dis

    def MapHistogram(self, binwidth=1.0, plot=False, normed=True):
        '''
        Builds a 2D histogram of the earthquakes locations.
        binwidth: width of the bins used for histogram.
        '''

        # Build x- and y-bins
        xbins = np.arange(self.x.min(), self.x.max(), binwidth)
        ybins = np.arange(self.y.min(), self.y.max(), binwidth)

        # Build the histogram
        hist, xedges, yedges = np.histogram2d(self.x, self.y, bins=[xbins, ybins], normed=normed)

        # Store the histogram
        self.histogram = {}
        self.histogram['xedges'] = xedges
        self.histogram['yedges'] = yedges
        self.histogram['values'] = hist
        
        # Store (x,y) locations
        x = (xedges[1:] - xedges[:-1])/2. + xedges[:-1]
        y = (yedges[1:] - yedges[:-1])/2. + yedges[:-1]
        x, y = np.meshgrid(x,y)
        x = x.flatten()
        y = y.flatten()
        self.histogram['x'] = x
        self.histogram['y'] = y

        # Pass it in lon lat
        lon, lat = self.xy2ll(x,y)
        self.histogram['lon'] = lon
        self.histogram['lat'] = lat

        # Plot
        if plot:
            plt.figure()
            plt.imshow(hist, interpolation='nearest', extent=[lon.min(), lon.max(), lat.min(), lat.max()])
            plt.colorbar(orientation='horizontal', shrink=0.6)
            plt.show()

        # All done
        return

    def BuildHistogramsAlongFaultTrace(self, fault, filename, normed=True, width=10.0, bins=50, plot=False, planemode='verticalfault', Range=(-5.0, 5.0), reference=None):
        '''
        Builds a histogram of the earthquake distribution along the fault trace.
        width: Width of the averaging cell (distance along strike)
        '''

        # Import needed stuffs
        import scipy.stats as stats

        # Need a projected earthquake set
        assert hasattr(self, 'Projected'), 'No dictionary of Projected earthquakes is available'
        assert ('{}'.format(fault.name)), 'No projection of earthquakes associated with fault {} is available'.format(fault.name)

        # Need a discretized fault trace
        assert hasattr(fault, 'xi'), 'No discretized fault trace is available'

        # Open the file
        frough = open('Rough_'+filename, 'w')
        frough.write('# Along Strike Distance (km) | Distance to the fault plane (km) | Counts \n')
        fsmooth = open(filename, 'w')
        fsmooth.write('# Along Strike Distance (km) | Distance to the fault plane (km) | Counts \n')

        # Get the projected earthquakes
        x = self.x
        y = self.y
        lon = self.lon
        lat = self.lat
        z = self.depth
        Das = self.Projected['{}'.format(fault.name)]['AlongStrikeDistance']
        Dff = self.Projected['{}'.format(fault.name)]['DistanceToFault']

        # Get the fault trace
        xf = fault.xi
        yf = fault.yi

        # And the corresponding distance along the fault
        df = fault.cumdistance(discretized=True)

        # Reference
        if reference is not None:
            xr, yr = self.ll2xy(reference[0], reference[1])
            RefD = np.sqrt( (xr-xf)**2 + (yr-yf)**2 )
            dmin = np.argmin(RefD)
            Ref = df[dmin]
        else:
            Ref = 0.0

        # On every point of the fault
        for i in range(len(xf)):
        
            # 1. Get the earthquakes in between x-width/2. and x+width/2.
            U = np.flatnonzero( (Das<=df[i]+width/2.) & (Das>=df[i]-width/2.) ).tolist()

            # 2. Get the corresponding earthquakes
            xe = np.array([x[u] for u in U])
            ye = np.array([y[u] for u in U])
            ze = np.array([z[u] for u in U])
            de = np.array([Dff[u] for u in U])
            
            if planemode in ('bestfit'):

                if xe.shape[0]>3: # Only do the thing if we have more than 3 earthquakes
            
                    # 3.1 Get those earthquakes in between -2.0 and 2.0 km from the fault trace
                    #     A reasonable thing to do would be to code a L1-regression, but for some places it is a nigthmare
                    M = np.flatnonzero( (de<=2.) & (de>=-2.) ).tolist()
                    xee = np.array([xe[m] for m in M])    
                    yee = np.array([ye[m] for m in M])    
                    zee = np.array([ze[m] for m in M])    

                    # 3.2 Find the center of the best fitting plane 
                    c = [xee.sum()/len(xee), yee.sum()/len(yee), zee.sum()/len(zee)]

                    # 3.3 Find the normal of the best fitting plane
                    sumxx = ((xee-c[0])*(xee-c[0])).sum() 
                    sumyy = ((yee-c[1])*(yee-c[1])).sum() 
                    sumzz = ((zee-c[2])*(zee-c[2])).sum() 
                    sumxy = ((xee-c[0])*(yee-c[1])).sum() 
                    sumxz = ((xee-c[0])*(zee-c[2])).sum() 
                    sumyz = ((yee-c[1])*(zee-c[2])).sum() 
                    M = np.array([ [sumxx, sumxy, sumxz],
                                   [sumxy, sumyy, sumyz],
                                   [sumxz, sumyz, sumzz] ])
                    w, v = np.linalg.eig(M)
                    u = w.argmin()
                    N = v[:,u]
                    
                    # 3.4 If the normal points toward the west, rotate it 180 deg
                    if N[0]<0.0:
                        N[0] *= -1.0
                        N[1] *= -1.0
                        N[2] *= -1.0

                    # 3.5 Project earthquakes on that plane and get the distance between the EQ and the plane
                    vecs = np.array([[xe[u]-c[0],ye[u]-c[1], ze[u]-c[2]] for u in range(len(xe))]).T
                    distance = np.dot(N, vecs)
                    xn = xe - distance*N[0]
                    yn = ye - distance*N[1]
                    zn = ze - distance*N[2]

                else:
                    distance = []
                    xn = []
                    yn = []
                    zn = []

            elif planemode in ('verticalfault'):

                # 3.1 Get the fault points 
                F = np.flatnonzero( (df<=df[i]+10.) & (df>=df[i]-10.) ).tolist()
                xif = np.array([xf[u] for u in F])
                yif = np.array([yf[u] for u in F])

                # 3.2 Get the center of the fault points
                c = [xif.sum()/len(xif), yif.sum()/len(yif), ze.mean()]
                 
                # 3.3 Get the normal to the fault
                sumxx = ((xif-c[0])*(xif-c[0])).sum() 
                sumyy = ((yif-c[1])*(yif-c[1])).sum()
                sumxy = ((xif-c[0])*(yif-c[1])).sum()
                M = np.array([ [sumxx, sumxy],
                               [sumxy, sumyy] ])
                w, v = np.linalg.eig(M)
                u = w.argmin()
                N = np.zeros(3,)
                N[:2] = v[:,u]

                # 3.4 Get the projected earthquakes
                xn = [self.Projected['{}'.format(fault.name)]['x'][u] for u in U]
                yn = [self.Projected['{}'.format(fault.name)]['y'][u] for u in U]
                zn = [self.Projected['{}'.format(fault.name)]['depth'][u] for u in U]

                # 3.5 If the normal points toward the west, rotate it 180 deg
                if N[0]<0.0:
                    N[0] *= -1.0
                    N[1] *= -1.0
                    N[2] *= -1.0

                # 3.6 Get the distance to the fault
                vecs = np.array([[xe[u]-c[0],ye[u]-c[1], ze[u]-c[2]] for u in range(len(xe))]).T
                if vecs.shape[0]>0:
                    distance = np.dot(N, vecs)
                else:
                    distance = []

            # 4. Compute the histogram of the distance
            if len(distance)>10:
                hist, edges = np.histogram(distance, bins=bins, normed=normed, range=Range)
            else:
                hist, edges = np.histogram(distance, bins=bins, normed=False, range=Range)

            # 5. Store the rough histograms in the xy file
            xi = df[i] - Ref
            for k in range(len(hist)):
                yi = (edges[k+1] - edges[k])/2. + edges[k]
                zi = hist[k] 
                frough.write('{} {} {} \n'.format(xi, yi, zi))

            # 5. Fit the histogram with a Gaussian and center it
            if len(distance)>10:
                Mu, Sigma = stats.norm.fit(distance)
                Sdis = distance - Mu
                hist, edges = np.histogram(Sdis, bins=bins, normed=normed, range=Range)
                yis = (edges[1:] - edges[:-1])/2. + edges[:-1]
                wis = stats.norm.pdf(yis, Mu, Sigma)
                # Write the smoothed and entered histograms in the file
                for k in range(len(hist)):
                    yi = yis[k]
                    zi = hist[k]
                    wi = wis[k]
                    fsmooth.write('{} {} {} {} \n'.format(xi, yi, zi, wi))
            else:
                hist, edges = np.histogram(distance, bins=bins, normed=False, range=Range)

            # check 
            if plot and df[i]>107. and df[i]<108.:
                import matplotlib.pyplot as plt
                fig = plt.figure(23)
                ax3 = fig.add_subplot(211, projection='3d')
                ax3.scatter3D(xe, ye, ze, s=5.0, color='k')
                ymin, ymax = ax3.get_ylim(); xmin, xmax = ax3.get_xlim()
                ax3.plot(xf, yf, '-r')
                ax3.scatter3D(xn, yn, zn, s=2.0, color='r')
                ax3.plot([c[0], c[0]+N[0]], [c[1], c[1]+N[1]], [c[2], c[2]+N[2]], '-r')
                ax3.set_xlim([xmin, xmax]); ax3.set_ylim([ymin, ymax])
                axh = fig.add_subplot(212)
                T = axh.hist(distance, bins=bins, normed=True, range=Range)
                edge = T[1]
                hist = T[0]
                cent = (edge[1:] - edge[:-1])/2. + edge[:-1]
                aa = np.arange(cent[0], cent[-1], 0.1)
                g = np.zeros((hist.size,1))
                g[:,0] = stats.norm.pdf(cent, Mu, Sigma)
                A, res, rank, s = np.linalg.lstsq(g, hist)
                axh.plot(aa, A*stats.norm.pdf(aa, Mu, Sigma), '-r')
                plt.show()

        # Close the file
        frough.close()
        fsmooth.close()

        # All done
        return
        
    def getClosestFaultPatch(self, fault):
        '''
        Returns a list of index for all the earthquakes containing the index of the closest fault patch.
        '''

        # Create a list
        ipatch = []

        # scipy.distance
        import scipy.spatial.distance as distance

        # Create a list of patch centers
        Centers = [fault.getpatchgeometry(i, center=True)[:3] for i in range(len(fault.patch))]

        # Create a list of points
        Earthquakes = [[self.x[i], self.y[i], self.depth[i]] for i in range(self.x.shape[0])]

        # Iterate on the earthquakes
        for eq in Earthquakes:

            # Create a list of distances
            dis = distance.cdist([eq], Centers)

            # Get the index of the smallest distance
            ipatch.append((np.array(dis)).argmin())

        # All done
        return ipatch

    def mag2Mo(self):
        '''
        Compute the moment from the magnitude.
        Result in N.m
        '''

        # Compute
        self.Mo = 10.**(1.5*self.mag + 9.1)     

        # All done
        return

    def Mo2mag(self):
        '''
        Compute the magnitude from the moment.
        '''

        self.mag = 2./3. * (np.log10(self.Mo) - 9.1)

        # All done
        return

    def momentEvolution(self, plot=False, outfile=None):
        '''
        Computes the evolution of the moment with time.
        '''

        # Make sure these are sorted
        self.sortInTime()

        # Lets consider self.mag is the moment magnitude :-)
        self.mag2Mo()

        # Compute the cumulative moment
        self.cumMo = np.cumsum(self.Mo)

        # Compute the cumulative number of earthquakes
        self.cumEQ = np.cumsum(np.ones(len(self.mag)))

        # Output?
        if outfile is not None:
            fout = open(outfile, 'w')
            fout.write('# Time | Cum. Moment (N.m) | Cum. Num. of Eq. \n')
            # First point 
            t = self.time[0].isoformat()
            Mo = 0.1
            Ec = 0
            fout.write('{} {} {} \n'.format(t, Mo, Ec))
            # All the rest
            for i in range(1,len(self.time)):
                t = self.time[i].isoformat()
                Mo = self.cumMo[i-1]
                Ec = self.cumEQ[i-1]
                fout.write('{} {} {} \n'.format(t, Mo, Ec))
                t = self.time[i].isoformat()
                Mo = self.cumMo[i]
                Ec = self.cumEQ[i]
                fout.write('{} {} {} \n'.format(t, Mo, Ec))
            fout.close()

        # Plot?
        if plot:
            fig = plt.figure(1)
            axmo = fig.add_subplot(111)
            axec = axmo.twinx()
            axmo.plot(self.time, self.cumMo, '-', color='black', label='Cum. Moment (N.m)')
            axec.plot(self.time, self.cumEQ, '-', color='gray', label='Cum. # of Eq')
            #axmo.legend()
            #axec.legend()
            plt.show()

        # All done
        return

    def sortInTime(self):
        '''
        Sorts the earthquakes in Time
        '''

        # Get the ordering
        i = np.argsort(self.time)

        # selection
        self._select(i)

        # All done
        return

    def write2file(self, filename, add_column=None):
        '''
        Write the earthquakes to a file.
        Args:
            * filename      : Name of the output file.
        '''

        # open the file
        fout = open(filename, 'w')

        # Write a header
        fout.write('# Lon | Lat | Depth (km) | Mw | time \n')

        # Loop over the earthquakes
        for u in range(len(self.lon)):
            if add_column is not None:
                last = '{} {}'.format(self.time[u].isoformat(), add_column[u])
            else:
                last = '{}'.format(self.time[u].isoformat())
            fout.write('{} {} {} {} {} \n'.format(self.lon[u], self.lat[u], self.depth[u], self.mag[u], last))
        
        # Close the file
        fout.close()

        # all done
        return

    def writeSelectedMagRange(self, filename, minMag=5.0, maxMag=10.):
        '''
        Write to a file the earthquakes with a magnitude larger than minMag and 
        smaller than maxMag.
        Args:
            * filename  : Name of the output file
            * minMag    : minimum Magnitude.
            * maxMag    : maximum Magnitude.
        '''

        # Create a new object
        eq = copy.deepcopy(self)

        # Remove the small earthquakes
        eq.selectmagnitude(minMag, maximum=maxMag)

        # Write to a file
        eq.write2file(filename)

        # Delete the object
        del eq

        # All done
        return

    def Cmt2Dislocation(self, size=1, mu=30e9, choseplane='nochoice', moment_from_tensor=False, verbose=True):
        '''
        Returns a single square patch fault from the cmt solutions.
        If no condition is given, it returns the first value.

        Args:
            * size          : Size of one side of the fault patch (km).
            * mu            : Shear modulus (Pa).
            * choseplane    : Choice of the focal plane to use (can be 'smallestdip', 'highestdip', 'nochoice')
            * moment_from_tensor: Computes the scalar moment from the cmt.
        '''

        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Convert CMTs to dislocation')

        # Import what is needed
        from .planarfault import planarfault

        # Create a list of faults
        self.faults = []

        # Check something
        if not hasattr(self, 'Mo'):
            self.Mo = np.zeros(self.lon.shape)

        # Loop on the earthquakes
        for i in range(len(self.CMTinfo)):

            # Get the event
            eq = self.CMTinfo[i]

            # Get the event name
            event = eq['event name']

            # Get the moment tensor
            cmt = [ [eq['Mrr'], eq['Mrt'], eq['Mrp']], 
                    [eq['Mrt'], eq['Mtt'], eq['Mtp']],
                    [eq['Mrp'], eq['Mtp'], eq['Mpp']] ]

            # Get strike dip rake
            sdr1, sdr2, Mo = self._cmt2strikediprake(cmt, returnMo=True)
            self.CMTinfo[i]['cmt'] = cmt
            
            # Moment
            if moment_from_tensor:
                self.Mo[i] = Mo

            # Condition to chose strike dip rake
            if choseplane in ('smallestdip'):
                if sdr1[1]<sdr2[1]:
                    sdr = sdr1
                else:
                    sdr = sdr2
            elif choseplane in ('highestdip'):
                if sdr1[1]>sdr2[1]:
                    sdr = sdr1
                else:
                    sdr = sdr2
            elif choseplane in ('nochoice'):
                sdr = sdr1
            strike, dip, rake = sdr
            self.CMTinfo[i]['sdr1'] = sdr1
            self.CMTinfo[i]['sdr2'] = sdr2

            # Get the depth
            depth = eq['depth']

            # Shear Modulus (I should code PREM here)
            if (mu.__class__ is float):
                Mu = mu

            # Build a planar fault
            fault = planarfault(event, utmzone=self.utmzone, verbose=False)
            fault.buildPatches(self.lon[i], self.lat[i], depth, strike*180./np.pi, dip*180./np.pi, size, size, 1, 1, verbose=False)

            # Components of slip
            ss = np.cos(rake) * self.Mo[i] / (Mu * size * size * 1000. * 1000.)
            ds = np.sin(rake) * self.Mo[i] / (Mu * size * size * 1000. * 1000.)

            # Set slip
            fault.slip[0,0] = ss
            fault.slip[0,1] = ds
            fault.slip[0,2] = 0.0

            # Put the fault in the list
            self.faults.append(fault)

            # Save the strike, dip rake infos
            eq['strike'] = strike
            eq['dip'] = dip
            eq['rake'] = rake

        # all done
        return

    def mergeCatalog(self, catalog):
        '''
        Merges another catalog into this one.
        Args:
            catalog:    Seismic location object.
        '''

        # Merge variables
        self.mag = np.hstack((self.mag, catalog.mag))
        self.lat = np.hstack((self.lat, catalog.lat))
        self.lon = np.hstack((self.lon, catalog.lon))
        self.depth = np.hstack((self.depth, catalog.depth))
        self.time = np.hstack((self.time, catalog.time))

        # Compute the xy
        self.lonlat2xy()

        # all done 
        return

    def lonlat2xy(self):
        '''
        Pass the position into the utm coordinate system.
        '''

        x, y = self.putm(self.lon, self.lat)
        self.x = x/1000.
        self.y = y/1000.

        # All done
        return

    def xy2lonlat(self):
        '''
        Pass the position from utm to lonlat.
        '''

        lon, lat = self.putm(x*1000., y*1000.)
        self.lon = lon
        self.lat = lat

        # all done
        return

    def _select(self, u):
        '''
        Makes a selection.
        '''

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.time = self.time[u]
        self.depth = self.depth[u]
        self.mag = self.mag[u]

        # Conditional
        if hasattr(self, 'CMTinfo'):
            self.CMTinfo = np.array(self.CMTinfo)
            self.CMTinfo = self.CMTinfo[u]
            self.CMTinfo = self.CMTinfo.tolist()

        # Conditional
        if hasattr(self, 'Mo'):
            self.Mo = self.Mo[u]

        # All done
        return

    def _cmt2strikediprake(self, cmt, returnMo=False):
        '''
        From a moment tensor in Harvard convention, returns 2 tuples of (strike, dip, rake)
        Args:
            * cmt   : Array (3,3) with the CMT.
        '''

        # 1. Compute the eigenvalues and eigenvectors
        EigValues, EigVectors = np.linalg.eig(cmt)

        # 2. Sort them => T = max(Eig)
        #                 N = Neutral
        #                 P = min(Eig)
        #    Then, n = (T+P)/sqrt(2)    # Normal
        #          s = (P-T)/sqrt(2)    # Slip
        T = EigVectors[:,np.argmax(EigValues)]
        P = EigVectors[:,np.argmin(EigValues)]
        n = (T+P)/np.sqrt(2.)
        s = (T-P)/np.sqrt(2.)

        # 3. Compute the moment
        Mo = (np.abs(np.min(EigValues)) + np.abs(np.max(EigValues)))
        Mo /= 2e7

        # 4. Get strike, dip and rake from vectors
        sdr1 = self._ns2sdr(n,s)
        sdr2 = self._ns2sdr(s,n)

        # All done
        if returnMo:
            return sdr1, sdr2, Mo
        else:
            return sdr1, sdr2

    def _ns2sdr(self, n, s, epsilon=0.0001):
        '''
        From the normal and the slip vector, returns the strike, dip and rake.
        Args:
            * n     : Normal vector.
            * s     : Slip vector.
        '''

        # Case: If normal downwards, flip them
        if n[0]<0.:
            n = -1.0*n
            s = -1.0*s

        # Case: if normal is vertical (i.e. if the plane is horizontal)
        if n[0]>(1-epsilon):
            strike = 0.0
            dip = 0.0
            rake = np.arctan2(-s[2], -s[1])

        # Case: if normal is horizontal (i.e. plane if vertical)
        elif n[0]<epsilon:
            strike = np.arctan2(n[1], n[2])
            dip = np.pi/2.
            rake = np.arctan2(s[0], -s[1]*n[2] + s[2]*n[1])

        # General Case:
        else:
            strike = np.arctan2(n[1], n[2])
            dip = np.arccos(n[0])
            rake = np.arctan2(-s[1]*n[1]-s[2]*n[2], (-s[1]*n[2]+s[2]*n[1])*n[0])
        
        # Strike
        if strike < 0.:
            strike += 2*np.pi

        # All done
        return strike, dip, rake

# EOF
