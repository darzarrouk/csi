''' 
A base class for faults and datasets

Written by Z. Duputel, November 2013.
'''

import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt

class SourceInv(object):
    
    def __init__(self,name,utmzone=None,ellps='WGS84'):
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
        if utmzone is not None:
            self.set_utmzone(utmzone, ellps)

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

    def set_utmzone(self, utmzone, ellps='WGS84'):
        '''
        Set the utm zone of the fault.

        Args:
            * utm           : UTM zone of the fault.
        '''

        # Set utmzone
        self.utmzone = utmzone
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps=ellps)

        # All done
        return        
