'''
A parent class that deals with rectangular patches fault

Written by R. Jolivet, Z. Duputel and Bryan Riel November 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy.signal as signal
from   glob import glob
import shutil as sh
import copy
import sys
import os

# Personals
from .RectangularPatches import RectangularPatches
from .stressfield import stressfield
from . import okadafull


class RectangularPatchesKin(RectangularPatches):
    
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
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
        super(RectangularPatchesKin,self).__init__(name,
                                                   utmzone=utmzone,
                                                   ellps=ellps,
                                                   lon0=lon0,
                                                   lat0=lat0)

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
        self.mu    = None

        # bigG and bigD
        self.bigG = None
        self.bigD = None
        
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
        assert self.hypo_x   != None, 'Hypocenter   must be assigned'
        assert self.hypo_y   != None, 'Hypocenter   must be assigned'
        assert self.hypo_z   != None, 'Hypocenter   must be assigned'

        # Get center
        p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)

        # Along dip and along strike distance to hypocenter
        if sd_dist:
            assert self.hypo_patch_index != None, 'Must provide a hypocenter patch index'
            assert self.fault_map        != None, 'Must provide a fault map'

            hp_x, hp_y, hp_z, hp_W, hp_L, hp_S, hp_D = self.getpatchgeometry(self.hypo_patch_index,center=True)

            assert np.round(p_width,2)  == np.round(hp_W,2), 'Patch width  must be homogeneous over the fault'
            assert np.round(p_length,2) == np.round(hp_L,2), 'Patch length must be homogeneous over the fault'            

            hp_strike,hp_dip = self.fault_map[self.hypo_patch_index]
            p_strike ,p_dip = self.fault_map[p]
            
            strike_d = (p_strike - hp_strike) * p_length
            dip_d    = (p_dip    - hp_dip   ) * p_width
            
            dip_d    += (self.hypo_z-hp_z) / np.sin(hp_D)            
            strike_d += (self.hypo_x-hp_x) * np.sin(hp_S) + (self.hypo_y-hp_y) * np.cos(hp_S)

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
        self.fault_inv_map = np.zeros((Nstrike,Ndip),dtype='int')
        for ny in range(Ny):
            for nx in range(Nx):
                p = ny * Nx + nx
                if leading=='strike':
                    self.fault_map.append([nx,ny])
                    self.fault_inv_map[nx,ny] = p
                elif leading=='dip':
                    self.fault_map.append([ny,nx])
                    self.fault_inv_map[ny,nx] = p
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

    def setMu(self,model_file):
        '''
        Set shear modulus values for seismic moment calculation
        from model_file:
        Thickness  Vp  Vs  Rho (...)
        '''
        
        # Read model file
        mu = []
        depth  = 0.
        depths = []
        with open(model_file) as f:
            for l in f:
                if l.strip()[0]=='#':
                    continue
                items = l.strip().split()
                H   = float(items[0])
                VS  = float(items[2])
                RHO = float(items[3])
                mu.append(VS*VS*RHO*1.0e9)
                if H==0.:
                    H = np.inf
                depths.append([depth,depth+H])
                depth += H
        Nd = len(depths)
        Np = len(self.patch)
        
        # Set Mu for each patch
        self.mu = np.zeros((Np,))
        for p in range(Np):
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)
            for d in range(Nd):
                if p_z>=depths[d][0] and p_z<depths[d][1]:
                    self.mu[p] = mu[d]

        # All done
        return

    def buildKinGFsFromDB(self, data, wave_engine, slip, rake, Mu = None, filter_coef=None):
        '''
        Build Kinematic Green's functions based on the discretized fault and a pre-calculated GF database. 
        Green's functions will be calculated for a given shear modulus and a given slip (cf., slip) 
        along a given rake angle (cf., rake)
        Args:
            * data:        Seismic data object
            * wave_engine: waveform generator
            * slip:        slip amplitude (in m)
            * rake:        rake angle (in deg)
            * Mu:          Shear modulus (optional)
            * filter_coef: Filter coefficient [a, b] (optional)
        '''        
        
        

        print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))
        print ("Using GF_path: {}".format(wave_engine.GF_path))        
        
        # Check Mu
        Np = len(self.patch)
        if Mu!=None:
            self.mu = np.ones((Np,)) * Mu
        else:
            assert self.mu != None

        # Check the patch attribute
        assert self.patch != None, 'Patch object should be assigned'

        # Init Green's functions
        if not self.G.has_key(data.name):
            self.G[data.name] = {}
        self.G[data.name][rake] = []
        
        # Init station lat/lon
        assert len(data.lat)>0, 'Station lat must be assigned'
        assert len(data.lon)>0, 'Station lon must be assigned'
        assert len(data.lon)==len(data.lat), 'Inconsistent station lat/lon'
        assert len(data.sta_name)==len(data.lat), 'Inconsistent station name/lat/lon'
        Ns = len(data.sta_name)
        s_name = data.sta_name
        s_lat  = data.lat
        s_lon  = data.lon

        # Get delta
        delta = data.d[data.sta_name[0]].delta
        
        # Loop over each patch
        G = self.G[data.name][rake]
        for p in range(Np):
            
            # Get point source location and patch geometry
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)  
            p_lon,p_lat = self.xy2ll(p_x,p_y)
            strike = strike_rad*180./np.pi
            dip    = dip_rad*180./np.pi

            # Seismic moment
            M0 = self.mu[p] * slip * width * length * 1.0e13 # M0  
            
            # Compute GFs for each station
            synth = {}
            for s in range(Ns):
                # Get station name and component
                dkey = data.sta_name[s]
                ori  = data.d[dkey].kcmpnm[2]         
                # Station Azimuth and distance                
                [az,baz,dist] = self.geod.inv(p_lon,p_lat,s_lon[s],s_lat[s])
                dist /= 1000. # km -> m
                # Compute synthetics
                o_sac,L_sac,T_sac = wave_engine.synthSDR(p_z,az,dist,M0,strike,dip,rake)
                if ( ori == 'N' or ori == 'E' or ori == '1' or ori == '2' ):                    
                    o_sac = wave_engine.rotTraces(L_sac,T_sac,baz,data.d[dkey].cmpaz)
                # Check delta
                assert data.d[dkey].delta == delta, 'Sampling frequency must be identical for each station'
                assert o_sac.delta == delta,        'Sampling frequency must be identical for each GFs'
                # GFs filtering
                if filter_coef != None:
                    assert len(filter_coef)==2, 'Incorrect filter_coef, must include [a,b]'
                    a,b = filter_coef
                    o_sac.depvar = signal.lfilter(b,a,o_sac.depvar)
                # GFs time-windowing
                b = data.d[dkey].b - data.d[dkey].o
                npts = data.d[dkey].npts
                t = np.arange(o_sac.npts)*o_sac.delta+o_sac.b-o_sac.o
                dtb = np.absolute(t-b)
                ib  = np.where(dtb==dtb.min())[0][0]           
                o_sac.depvar = o_sac.depvar[ib:ib+npts]
                # Sac headers
                o_sac.kstnm  = data.d[dkey].kstnm
                o_sac.kcmpnm = data.d[dkey].kcmpnm
                o_sac.knetwk = data.d[dkey].knetwk
                o_sac.khole  = data.d[dkey].khole
                o_sac.stlo   = data.d[dkey].stlo
                o_sac.stla   = data.d[dkey].stla
                o_sac.npts   = npts
                o_sac.b      = t[ib]+o_sac.o
                # Assemble GFs
                synth[s_name[s]] = o_sac.copy()
            G.append(copy.deepcopy(synth))

        # All done
        return        
        
    def buildBigGD(self,eik_solver,data,rakes,vmax,Nt,Dt):
        '''
        Build BigG and bigD matrices from Green's functions and data dictionaries
        Args:
            eik_solver: Eikonal solver (e.g., FastSweep)
            data:       Seismic data object
            rakes:      List of rake angles
            vmax:       Maximum rupture velocity
            Nt:         Number of rupture time-steps
            Dt:         Rupture time-steps
        '''

        # Check vmax
        assert vmax > 0., 'vmax must be positive'
        
        # Set eikonal solver grid for vmax
         
        if vmax != np.inf:
            vr = copy.deepcopy(self.vr)
            self.vr[:] = vmax 
            eik_solver.setGridFromFault(self,1.0)
            eik_solver.fastSweep()
            self.vr[:] = copy.deepcopy(vr)
        
            # Get tmin for each patch
            tmin = []
            for p in range(len(self.patch)):
                # Location at the patch center
                dip_c, strike_c = self.getHypoToCenter(p,True)
                tmin.append(eik_solver.getT0([dip_c],[strike_c])[0])
        
        # Build up bigD
        self.bigD = []
        for dkey in data.sta_name:
            self.bigD.extend(data.d[dkey].depvar)
        self.bigD = np.array(self.bigD)
        self.bigD = self.bigD.reshape(len(self.bigD),1)
        
        # Get tmin for each patch
        Np = len(self.patch)
        self.bigG = np.zeros((len(self.bigD),Nt*Np*len(rakes)))
        j  = 0
        for nt in range(Nt):
            for r in rakes:
                for p in range(Np):
                    tshift = tmin[p] + nt * Dt
                    di = 0
                    for dkey in data.sta_name:
                        b    = self.G[data.name][r][p][dkey].b
                        o    = self.G[data.name][r][p][dkey].o
                        npts = self.G[data.name][r][p][dkey].npts
                        depvar = self.G[data.name][r][p][dkey].depvar
                        t = np.arange(npts) + b - o                         
                        i = np.where(t>=tshift)[0] + di
                        self.bigG[i,j] = depvar[:len(i)]
                        di += npts
                    j += 1
                
        # All done
        return tmin
            
    def saveBigGD(self, bigDfile = 'data.kin', bigGfile='gf.kin', dtype='float64'):
        '''
        Save bigG and bigD to binary file
        Args:
            * bigDfile: bigD filename (optional)
            * bigGfile: bigG gilename (optional)
        '''
        
        # Check bigG and bigD
        assert self.bigD != None
        assert self.bigG != None

        # Write files
        self.bigD.tofile(bigDfile, dtype=dtype)
        self.bigG.T.tofile(bigGfile, dtype=dtype)
        
        # All done
        return


    def loadBigGD(self, bigDfile = 'data.kin', bigGfile='gf.kin', dtype='float64'):
        '''
        Load bigG and bigD to binary file
        Args:
            * bigDfile: bigD filename (optional)
            * bigGfile: bigG gilename (optional)
        '''
        
        # Write files
        self.bigD = np.fromfile(bigDfile, dtype=dtype)
        self.bigG = np.fromfile(bigGfile, dtype=dtype)
        Nd = self.bigD.size
        assert self.bigG.size%Nd == 0
        Nm = self.bigG.size/Nd

        # Reshape matrices
        self.bigG = self.bigG.reshape(Nm,Nd).T
        
        # All done
        return

    def saveKinGF(self, data, outputDir = 'GFs'):
        '''
        Save kinematic Green's functions in outputDir
        Args:
            data:      seismic data object
            outputDir: output directory where GFs will be stored
        '''
           
        # Print stuff
        print('Writing Kinematic GFs to directory {} for fault {}'.format(outputDir,self.name))

        # Check i_path
        assert os.path.exists(outputDir), '%s: No such directory'%(i_path)

        # Main loop
        G = self.G[data.name]
        Np = len(self.patch)
        for r in G.keys():
            o_dir = os.path.join(outputDir,'rake_%.1f'%(r))
            if os.path.exists(o_dir):
                sh.rmtree(o_dir)
            os.mkdir(o_dir)
            for p in range(Np):
                for dkey in data.sta_name:                    
                    o_file = os.path.join(o_dir,'gf_p%d_%s.kin'%(p,dkey))
                    self.G[data.name][r][p][dkey].wsac(o_file)
    
        # All done
        return

    def loadKinGF(self, data, rakes, inputDir = 'GFs'):
        '''
        Load kinematic Green's functions in i_path
        Args:
            data:     seismic data object
            rakes:    list of rake angles
            inputDir: input directory where GFs are stored
        '''

        # Print stuff
        print('Loading Kinematic GFs from directory {} for fault {}'.format(inputDir,self.name))        

        # Import sacpy 
        import sacpy
        i_sac = sacpy.sac()
        
        # Check the patch attribute
        assert self.patch != None, 'Patch object should be assigned'
        
        # Check inputDir
        assert os.path.exists(inputDir), '%s: No such directory'%(inputDir)
        
        # Init Green's functions
        if not self.G.has_key(data.name):
            self.G[data.name] = {}

        # Main loop
        Np = len(self.patch)
        for r in rakes:

            # Check subdirectories
            i_dir = os.path.join(inputDir,'rake_%.1f'%(r))
            assert os.path.exists(i_dir), '%s: no such directory'%(i_dir)

            self.G[data.name][r] = []
            for p in range(Np):
                # Read GFs for each station
                synth = {}
                for dkey in data.sta_name:
                    # Read sac
                    i_file = os.path.join(i_dir,'gf_p%d_%s.kin'%(p,dkey))
                    i_sac.rsac(i_file)
                    synth[dkey] = i_sac.copy()
                self.G[data.name][r].append(copy.deepcopy(synth))
        
        # All done
        return
        
#EOF
