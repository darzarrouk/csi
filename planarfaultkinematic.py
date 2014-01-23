'''
A class that deals planar kinematic faults

Written by Z. Duputel, January 2014
'''

## Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys

## Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

# Rectangular patches Fault class
from .planarfault import planarfault



class planarfaultkinematic(planarfault):

    def __init__(self, name, utmzone=None, ellps='WGS84'):
        '''
        Args:
            * name      : Name of the fault.
            * f_strike: strike angle in degrees (from North)
            * f_dip:    dip angle in degrees (from horizontal)
            * f_length: length of the fault (i.e., along strike)
            * f_width: width of the fault (i.e., along dip)        
            * utmzone   : UTM zone.
        '''
        
        super(self.__class__,self).__init__(name,utmzone,ellps)

        # Hypocenter coordinates
        self.hypo_x   = None
        self.hypo_y   = None
        self.hypo_lon = None
        self.hypo_lat = None
                
        # Fault size
        self.f_length = None
        self.f_width  = None
        self.f_strike = None
        self.f_dip    = None
        
        # Patch objects
        self.patch = None
        self.grid  = None
        
        # All done
        return

    def setHypoXY(self,x,y, UTM=True):
        '''
        Set hypocenter attributes from x,y
        Outputs: East/West UTM/Lon coordinates, depth attributes
        Args:
            * x:   east  coordinates 
            * y:   north coordinates
            * UTM: default=True, x and y is in UTM coordinates (in km)
                   if    ==False x=lon and y=lat (in deg)
        '''
        
        # Check strike/dip assigmement
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip    must be assigned'
        f_strike_rad = self.f_strike * np.pi/180.
        f_dip_rad    = self.f_dip    * np.pi/180.

        # If UTM==False, convert x=lon/y=lat to UTM
        if not UTM:
            self.hypo_x,self.hypo_y = self.ll2xy(x,y)

        # Get distance from the fault trace axis (in km)
        dist_from_trace = self.hypo_x * np.cos(f_strike_rad) - self.hypo_y * np.sin(f_strike_rad)

        # Get depth on the fault
        self.hypo_z = dist_from_trace * np.tan(f_dip_rad)
        
        # UTM to lat/lon conversion        
        self.hypo_lon,self.hypo.lat = self.xy2ll(self.hypo.x,self.hypo.y)

        # All done
        return
        
    def buildFault(self, lat, lon, dep, strike, dip, f_length, f_width, grid_size, p_nstrike, p_ndip):
        '''
        Build fault patches/grid
        Args:
            * lat,lon,dep:  coordinates at the center of the top edge of the fault (in degrees)
            * strike:       strike angle in degrees (from North)
            * dip:          dip angle in degrees (from horizontal)
            * f_length: Fault length, km
            * f_width:  Fault width, km
            * grid_size:    Spacing between point sources within each patch
            * p_nstrike:   Number of subgrid points per patch along strike (multiple pts src per patch)
            * p_ndip:      Number of subgrid points per patch along dip    (multiple pts src per patch)
        '''
        
        # Orientation
        self.f_strike = f_strike
        self.f_dip    = f_dip

        # Patch size = nb of pts along dip/strike * spacing
        patch_length  = grid_size * p_nstrike
        patch_width   = grid_size * p_ndip

        # Number of patches along strike / along dip
        f_nstrike = int(np.round(f_length/patch_length))
        f_ndip    = int(np.round(f_width/patch_width))

        # Correct the fault size to match n_strike and n_dip
        self.f_length = p_nstrike * patch_length
        self.f_width  = p_ndip    * patch_width
        if self.f_length != f_length or self.f_dip != f_dip:
            sys.stderr.write('!!! Fault size changed to %.2f x %.2f km'%(self.f_length,self.f_width))
                    
        # build patches
        self.buildPatches(lon, lat, dep, self.f_strike, self.f_dip, self.f_length, self.f_width, f_nstrike, f_ndip)
        
        # build subgrid
        self.buildSubGrid(grid_size,p_nstrike,p_ndip)

        # All done
        return

        
    
    def buildSubGrid(self,grid_size,nbp_strike,nbp_dip):
        '''
        Define a subgrid of point sources on the fault (multiple point src per patches)
        Args: 
            * grid_size:    Spacing between point sources within each patch
            * p_nstrike:   Number of subgrid points per patch along strike 
            * p_ndip:      Number of subgrid points per patch along dip            
        '''
        
        # Check prescribed assigments
        assert self.f_strike != None, 'Fault length must be assigned'
        assert self.f_dip    != None, 'Fault length must be assigned'
        assert self.patch    != None, 'Patch objects must be assigned'
        
        # Loop over patches
        self.grid = []
        deg2rad   = np.pi/180.
        for p in range(len(self.patch)):
            # Get patch location/size
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p)

            # Set grid points coordinates on fault
            grid_strike = np.arange(0.5*grid_size,p_length,grid_size) - p_length/2.
            grid_dip    = np.arange(0.5*grid_size,p_length,grid_size) - p_width/2.

            # Check that everything is correct
            assert p_strike==self.f_strike and p_dip==self.f_dip, 'Fault must be planar'     
            assert nbp_strike == len(grid_strike), 'Incorrect length for patch %d'%(p)
            assert nbp_dip    == len(grid_dip),    'Incorrect width for patch  %d'%(p)

            # Initial deg2rad conversion
            if not p:
                p_dipdir_rad = ((p_strike + 90)%360) * deg2rad
                p_strike_rad = p_strike * deg2rad
                p_dip_rad    = p_dip    * deg2rad

            # Get grid points coordinates in UTM  
            xt = grid_strike * np.sin(p_strike_rad)
            yt = grid_strike * np.cos(p_strike_rad)
            zt = np.ones(xt.shape) * p_z 
            x = np.array([],dtype='float64')
            y = np.array([],dtype='float64')
            z = np.array([],dtype='float64')
            for i in range(ndip):
                x = np.append(x, xt + grid_dip[i] * np.cos(dip_rad) * np.sin(dipdir_rad))
                y = np.append(y, yt + grid_dip[i] * np.cos(dip_rad) * np.cos(dipdir_rad))
                z = np.append(z, zt + grid_dip[i] * np.sin(dip_rad))
            self.grid.append([x,y,z])
                
        # All done
        return

