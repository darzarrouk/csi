'''
A parent class that deals with rectangular patches fault

Written by R. Jolivet, Z. Duputel and Bryan Riel November 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys
import os

# Personals
from .RectangularPatches import RectangularPatches
from .stressfield import stressfield
from . import okadafull


class RectangularPatchesKin(RectangularPatches):
    
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
        
        # Parent class init
        super(RectangularPatchesKin,self).__init__(name,utmzone,ellps)

        # Hypocenter coordinates
        self.hypo_x   = None
        self.hypo_y   = None
        self.hypo_z   = None
        self.hypo_lon = None
        self.hypo_lat = None
        self.hypo_patch_index = None
                
        # Patch objects
        self.patch = None
        self.grid  = None
        self.vr    = None
        self.tr    = None
        
        # Patch index mapping along strike and along dip
        self.fault_map = None

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

        # If UTM==False, convert x=lon/y=lat to UTM
        if not UTM:
            self.hypo_x,self.hypo_y = self.ll2xy(x,y)
        else:
            self.hypo_x = x
            self.hypo_y = y

        # Check if within a patch
        hypo_point = np.array([self.hypo_x,self.hypo_y])
        for p in self.patch:
            Reg = []
            for v in p:
                Reg.append([v[0],v[1]])
            Reg = np.array(Reg)
            region = path.Path(Reg,closed=False)
            if region.contains_point(hypo_point):
                x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(p, center=True)
                self.hypo_z = x3
                self.hypo_patch_index = self.getindex(p)

        # UTM to lat/lon conversion        
        self.hypo_lon,self.hypo_lat = self.xy2ll(self.hypo_x,self.hypo_y)
        
        # All done
        return


    def getHypoToCenter(self, p, sd_dist=False):
        ''' 
        Get patch center coordinates from hypocenter
        Args:
            * p      : Patch number.
            * sd_dist: If true, will return along dip and along strike distances
        '''

        # Check strike/dip/hypo assigmement
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip    must be assigned'
        assert self.hypo_x   != None, 'Hypocenter   must be assigned'
        assert self.hypo_y   != None, 'Hypocenter   must be assigned'
        assert self.hypo_z   != None, 'Hypocenter   must be assigned'

        # Get center
        p_x, p_y, p_z = self.getcenter(self.patch[p])

        # Along dip and along strike distance to hypocenter
        if sd_dist:
            assert self.hypo_patch_index != None, 'Must provide a hypocenter patch index'
            assert self.fault_map        != None, 'Must provide a fault map'
            
            cc_strike = self.fault_map[self.hypo_patch_index]
                                       
            dip_d = z / np.sin(self.f_dip)
            strike_d = x * np.sin(self.f_strike) + y * np.cos(self.f_strike)
            return dip_d, strike_d
        else:
            x = p_x - self.hypo_x
            y = p_y - self.hypo_y
            z = p_z - self.hypo_z
            return x,y,z

    def setFaultMap(self,Nstrike,Ndip,leading='strike',check_depth=True):
        '''
        Set along dip and along strike indexing for patches
        Args:
            * Nstrike: number of patches along strike
            * Ndip   : number of patches along dip
            * leading: leadinf index of self.patch (can be 'strike' or 'dip'
        '''

        # Check input parameters
        if leading=='strike':
            Nx=Nstrike
            Ny=Ndip
        else:
            Nx=Ndip
            Ny=Nstrike
        assert Nx*Ny==len(self.patch), 'Incorrect Nstrike and Ndip'
        
        # Loop over patches
        self.fault_map = []
        for ny in range(Ny):
            for nx in range(Nx):
                p = ny * Nx + nx
                if leading=='strike':
                    self.fault_map.append([nx,ny])
                elif leading=='dip':
                    self.fault_map.append([ny,nx])
        self.fault_map = np.array(self.fault_map)
        
        for n in range(Ndip):
            i = np.where(self.fault_map[:,1]==n)[0]
            assert len(i)==Nstrike, 'Mapping error'

        for n in range(Nstrike):
            i = np.where(self.fault_map[:,0]==n)[0]
            assert len(i)==Ndip, 'Mapping error'

        if check_depth:
            for n in range(Ndip):
                indexes = np.where(self.fault_map[:,1]==n)[0]
                flag = True
                for i in indexes:
                    x,y,z = self.getcenter(self.patch[i])
                    if flag:
                        depth = np.round(z,1)
                        flag  = False
                    assert depth==np.round(z,1), 'Mapping error: inconsistent depth'


    def initializekinmodel(self, n=None):
        '''
        Re-initializes the fault slip array to zero values.
        Args:
            * n     : Number of slip values. If None, it'll take the number of patches.
        '''
        self.initializeslip(n=n)
        self.tr = np.zeros((self.N_slip,))
        self.vr = np.zeros((self.N_slip,))
        
        # All done
        return


    def buildSubGrid(self,nbp_strike,nbp_dip):
        '''
        Define a subgrid of point sources on the fault (multiple point src per patches)
        All patches must have the same size
        Args: 
            * p_nstrike:   Number of subgrid points per patch along strike 
            * p_ndip:      Number of subgrid points per patch along dip            
        '''
        
        # Init Grid size
        grid_size_strike = None
        grid_size_dip    = None
        
        # Loop over patches        
        self.grid = []
        for p in range(len(self.patch)):
            # Get patch location/size
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            
            # Dip direction
            dipdir = (p_strike+np.pi/2.)%(2.*np.pi)

            # grid-size
            if grid_size_strike==None:
                grid_size_strike = p_length/nbp_strike
            else:
                dum = p_length/nbp_strike
                errmsg = 'Heteogeneous grid size not implemented (%f,%f)'%(grid_size_strike,dum)
                assert np.round(grid_size_strike,2) == np.round(dum,2), errmsg

            if grid_size_dip==None:
                grid_size_dip = p_length/nbp_dip
            else:
                errmsg = 'Heteogeneous grid size not implemented (dip)'
                assert np.round(grid_size_dip,2) ==np.round(p_length/nbp_dip,2), errmsg

            # Set grid points coordinates on fault
            grid_strike = np.arange(0.5*grid_size_strike,p_length,grid_size_strike) - p_length/2.
            grid_dip    = np.arange(0.5*grid_size_dip   ,p_width ,grid_size_dip   ) - p_width/2.

            # Check that everything is correct
            assert nbp_strike == len(grid_strike), 'Incorrect length for patch %d'%(p)
            assert nbp_dip    == len(grid_dip),    'Incorrect width for patch  %d'%(p)

            # Get grid points coordinates in UTM  
            xt = p_x + grid_strike * np.sin(p_strike)
            yt = p_y + grid_strike * np.cos(p_strike)
            zt = p_z * np.ones(xt.shape)
            g  = []
            for i in range(nbp_dip):
                x = xt + grid_dip[i] * np.cos(p_dip) * np.sin(dipdir)
                y = yt + grid_dip[i] * np.cos(p_dip) * np.cos(dipdir)
                z = zt + grid_dip[i] * np.sin(p_dip)
                for j in range(x.size):                    
                    g.append([x[j],y[j],z[j]])
            self.grid.append(g)
                
        # All done
        return    


