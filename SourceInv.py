''' 
A base class for faults and datasets

Written by Z. Duputel, November 2013.
'''

import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt

class SourceInv(object):
    
    def __init__(self,name,utmzone=None,ellps='WGS84',lon0=None, lat0=None):
        '''
        Args:
            * name      : Instance Name 
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Initialization
        self.name = name
        
        # Set the utm zone
        self.utmzone = utmzone
        self.ellps   = ellps
        self.lon0 = lon0
        self.lat0 = lat0
        self.set_utmzone(utmzone=utmzone, 
                         ellps = ellps, 
                         lon0 = lon0,
                         lat0 = lat0)

        # All done
        return

    def ll2xy(self, lon, lat):
        '''
        Do the lat/lon 2 utm transform
        '''

        # Transpose 
        x, y = self.putm(lon, lat)

        # Put it in Km
        x = x/1000.
        y = y/1000.

        # All done
        return x, y

    def xy2ll(self, x, y):
        '''
        Do the utm to lat/lon transform
        '''

        # Transpose and return
        return self.putm(x*1000., y*1000., inverse=True)

    def set_utmzone(self, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        '''
        Set the utm zone of the fault.

        Args:
            * utm           : UTM zone of the fault.
        '''

        # Cases
        if utmzone is not None:
            self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps=ellps)
        else:
            assert lon0 is not None, 'Please specify a 0 longitude'
            assert lat0 is not None, 'Please specify a 0 latitude'
            string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(lat0, lon0, ellps)
            self.putm = pp.Proj(string)

        # Set utmzone
        self.utmzone = utmzone
        self.lon0 = lon0
        self.lat0 = lat0
        
        # Set Geod
        self.geod = pp.Geod(ellps=ellps)

        # All done
        return        

    def _checkLongitude(self):
        '''
        Iterates over the longitude array and checks if longitude is between 
        0 and 360
        '''

        # Check 
        if len(self.lon[self.lon<0.])>0:
            self.lon[self.lon<0.] += 360.

        # All done
        return

#EOF
