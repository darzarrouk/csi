''' 
A class that deals with seismic or high-rate GPS data (not finished)

Written by Z. Duputel, April 2013.
'''

# Externals
import os
import sys
import copy
import shutil
import numpy  as np
import pyproj as pp
import matplotlib.pyplot as plt


# Personals
#xfrom WaveMod    import sac
from .SourceInv import SourceInv

class seismic(SourceInv):
    
    def __init__(self,name,dtype='seismic',utmzone=None,ellps='WGS84', lon0=None, lat0=None):
        '''
        Args:
            * name      : Name of the dataset.
            * dtype     : data type (optional, default='seismic')
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''
        
        super(seismic,self).__init__(name,
                                     utmzone=utmzone,
                                     ellps=ellps,
                                     lon0=lon0,
                                     lat0=lat0) 

        # Initialize the data set 
        self.dtype = dtype
        
        # Initialize Waveform Engine
        self.waveform_engine = None

        # Initialize some things
        self.sta_name = []
        self.lat  = np.array([],dtype='float64')
        self.lon  = np.array([],dtype='float64')
        self.x    = np.array([],dtype='float64')
        self.y    = np.array([],dtype='float64')
    
        # Data
        self.d = {}

        # Covariance matrix
        self.Cd = None

        # All done
        return

    def setStat(self,sta_name,x,y,loc_format='LL'):
        '''
        Set station names and locations attributes
        Args:
            * sta_name: station names
            * x: x coordinate (longitude or UTM) 
            * y: y coordinate (latitude or UTM)
            * loc_format: location format ('LL' for lon/lat or 'XY' for UTM)
        '''

        # Check input parameters
        assert len(sta_name)==len(x)==len(y), 'sta_name, x and y must have the same length'
        assert loc_format=='LL' or loc_format=='XY', 'loc_format can be LL or XY'        
        if type(x)==list:
            x = np.array(x)
        if type(y)==list:
            y = np.array(y)

        # Assign input parameters to station attributes
        self.sta_name = copy.deepcopy(sta_name)
        if loc_format=='LL':            
            self.lon = np.append(self.lon,x)
            self.lat = np.append(self.lat,y)
            self.x, self.y = self.ll2xy(self.lon,self.lat)
        else:
            self.x = np.append(self.x,x)
            self.y = np.append(self.y,y)
            self.lon, self.lat = self.ll2xy(self.x,self.y)            

        # All done
        return

    def buildDiagCd(self,std):
        '''
        Build a diagonal Cd from standard deviations
        Args:
            std: array of standard deviations
        '''

        assert len(std) == len(self.sta_name)

        # Set variance vector
        var_vec = np.array([])
        for i in range(len(self.sta_name)):
            stanm = self.sta_name[i]
            var_vec_sta = np.ones((self.d[stanm].npts,))*std[i]*std[i]
            var_vec = np.append(var_vec,var_vec_sta)
        
        # Build Cd from variance vector
        self.Cd = np.diag(var_vec)
        
        # All done
        return

    def buildCdFromRes(self,fault,model_file,n_ramp_param,eik_solver,npt=4,relative_error=0.2,
                       add_to_previous_Cd=False):
        '''
        Build Cd from residuals
        Args:
            * model_file: model file name
            * n_ramp_param: number of model parameters
            * eik_solver: eikonal solver
            * npt**2: numper of point sources per patch 
            * relative_error: standard deviation = relative_error * max(data)
        '''
        
        print('Computing Cd from residuals')

        Dtriangles = 1. # HACK: We assume Dtriangles=1 !!!
        
        Np = len(fault.patch)        
        # Read model file
        post     = np.loadtxt(model_file)
        assert len(post)==4*Np + n_ramp_param + 2
        
        # Assign fault parameters
        fault.slip[:,0] = post[:Np]
        fault.slip[:,1] = post[Np:2*Np]
        fault.tr = post[2*Np+n_ramp_param:3*Np+n_ramp_param]
        fault.vr = post[3*Np+n_ramp_param:4*Np+n_ramp_param]
        h_strike = post[4*Np+n_ramp_param]
        h_dip    = post[4*Np+n_ramp_param+1]
        fault.setHypoOnFault(h_strike,h_dip)
        
        # Eikonal resolution
        eik_solver.setGridFromFault(fault,1.0)
        eik_solver.fastSweep()
        
        # BigG x BigM (on the fly time-domain convolution)
        Ntriangles = fault.bigG.shape[1]/(2*Np)
        G = fault.bigG
        D = fault.bigD
        m = np.zeros((G.shape[1],))
        for p in range(len(fault.patch)):
            # Location at the patch center
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = fault.getpatchgeometry(p,center=True)
            dip_c, strike_c = fault.getHypoToCenter(p,True)
            # Grid location
            grid_size_dip = p_length/npt
            grid_size_strike = p_length/npt
            grid_strike = strike_c+np.arange(0.5*grid_size_strike,p_length,grid_size_strike) - p_length/2.
            grid_dip    = dip_c+np.arange(0.5*grid_size_dip   ,p_width ,grid_size_dip   ) - p_width/2.
            time = np.arange(Ntriangles)*Dtriangles+Dtriangles
            T  = np.zeros(time.shape)
            Tr2 = fault.tr[p]/2.
            for i in range(npt):
                for j in range(npt):
                    t = eik_solver.getT0([grid_dip[i]],[grid_strike[j]])[0]
                    tc = t+Tr2
                    ti = np.where(np.abs(time-tc)<Tr2)[0]            
                    T[ti] += (1/Tr2 - np.abs(time[ti]-tc)/(Tr2*Tr2))*Dtriangles
            for nt in range(Ntriangles):
                m[2*nt*Np+p]     = T[nt] * fault.slip[p,0]/float(npt*npt)
                m[(2*nt+1)*Np+p] = T[nt] * fault.slip[p,1]/float(npt*npt)
        P = np.dot(G,m)

        # Compute residual autocorrelation for each station
        n = 0
        R = P - D # Residual vector
        print(R.shape)
        Cd = np.zeros((len(D),len(D)))
        for dkey in self.sta_name:
            print(('Cd for %s'%dkey))
            npts = self.d[dkey].npts
            print(npts)
            res  = R[n:n+npts]
            obs  = self.d[dkey].depvar
            cor = signal.correlate(res,res)
            cor /= cor.max()
            std = obs.max()*relative_error
            C = np.zeros((npts,npts))
            for k1 in range(npts):
                for k2 in range(npts):
                    dk = k1-k2
                    C[k1,k2] = cor[npts+dk-1]*std*std
            Cd[n:n+npts,n:n+npts] = C.copy()
            print(len(obs),len(P[n:n+npts]))
            #plt.figure()
            #plt.plot(obs)
            #plt.plot(P[n:n+npts])
            #plt.plot(res)
            #plt.show()
            n += npts
        
        # Assign Cd attribute
        if add_to_previous_Cd:
            self.Cd += Cd
        else:
            self.Cd = Cd.copy()

        # All done return
        return

    def writeCd2BinaryFile(self,outfile='kinematicG.Cd',dtype='float64'):
        '''
        Write Kinematic Cd to an output file
        Args:
            * outfile: Name of the output file
            * dtype:   Type of data to write. 
        '''
        
        # Check if Cd exists
        assert self.Cd != None, 'Cd must be assigned'
        
        # Convert Cd
        Cd = self.Cd.astype(dtype)
        
        # Write t file
        Cd.tofile(outfile)
        
        # All done
        return    

    def readStat(self,station_file,loc_format='LL'):
        '''
        Read station file and populate the Xr attribute (station coordinates)
        Args:
            * station_file: station filename including station coordinates
            * loc_format:  station file format (default= 'LL')
        file format:
        STNAME  X_COORD Y_COORD (if loc_format='XY')
        STNAME  LON LAT (if loc_format='LL')
        '''
        
        # Assert if station file exists
        assert os.path.exists(station_file), 'Cannot read %s (no such file)'%(station_file)

        # Assert file format
        assert loc_format=='LL' or loc_format=='XY', 'loc_format can be either LL or XY'
        
        # Read the file 
        X = []
        Y = []
        sta_name = []
        for l in open(station_file):
            if (l.strip()[0]=='#'):
                continue
            items = l.strip().split()
            sta_name.append(items[0].strip())
            X.append(float(items[1]))
            Y.append(float(items[2]))

        # Set station attributes
        self.setStat(sta_name,X,Y,loc_format)

        # All done
        return    

    def readSac(self,sacfiles):
        '''
        Read sac data files
        '''
        # Import personnal sac module
        import sacpy

        # Read sac files
        self.lon  = []
        self.lat  = []
        self.d = {}
        for sacfile in sacfiles:
            sac = sacpy.sac()
            sac.rsac(sacfile)
            self.lon.append(sac.stlo)
            self.lat.append(sac.stla)
            stanm = sac.kstnm+'_'+sac.kcmpnm
            self.sta_name.append(stanm)
            #if not self.d.has_key(sac.kstnm):
            #    self.d = {}
            #if not self.d[sac.kstnm].has_key(sac.kcmpnm):
            #    self.d[sac.kstnm][sac.kcmpnm] = {}
            #self.d[sac.kstnm][sac.kcmpnm[-1]] = sac.copy()
            assert stanm not in self.d, 'Multiple data for {}'.format(stanm)
            self.d[stanm] = sac.copy()

        # All done
        return


    def initWave(self,waveform_engine):
        '''
        Initialize Green's function database engine
        '''
        
        # Assign reference to waveform_engine
        self.waveform_engine = copy.deepcopy(waveform_engine)

        # All done
        return


    def initWaveInt(self,waveform_engine):
        '''
        Initialize Bob Hermann's wavenumber integration engine
        '''
        
        # Assign reference to waveform_engine
        self.initWave(waveform_engine)

        # Assign receiver location
        self.waveform_engine.setXr(self.sta_name,self.x,self.y)

        # All done
        return


    def calcSynthetics(self,dir_name,strike,dip,rake,M0,rise_time,stf_type='triangle',rfile_name=None,
                       out_type='D',src_loc=None,cleanup=True,ofd=sys.stdout,efd=sys.stderr):
        '''
        Build Green's functions for a particular source location
        Args:
            * dir_name:  Name of the directory where synthetics will be created
            * strike:    Fault strike (in deg)
            * dip:       Fault dip (in deg)
            * rake:      Fault rake (in deg)
            * M0:        Seismic moment
            * rise_time: Rise time (in sec)
            * stf_type: 
            * src_loc:  Point source coordinates (ndarray)
            * rfile_name: pulse file name if stf_type='rfile'
            * ofd:       stream for standard output (optional, default=sys.stdout)
            * efd:       stream for standard error  (optional, default=sys.stdout)        
        '''
        
        # Check Waveform Engine
        assert self.waveform_engine != None, 'waveform_engine must be assigned'
        if src_loc == None:
            assert self.waveform_engine.Xs != None, 'Source location must be assigned'
        else:
            self.waveform_engine.Xs = copy.deepcopy(src_loc)

        # Assign receiver locations
        assert self.waveform_engine.Xr != None, 'Recever locations must be assigned'

        # Go in dir_name
        cwd = os.getcwd()
        if cleanup and os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        os.chdir(dir_name)

        # Waveform simulation        
        self.waveform_engine.synthSDR(out_type,strike,dip,rake,M0,stf_type,rise_time,rfile_name,True,ofd,efd)
        
        # Go back
        os.chdir(cwd)
        
        if cleanup:
            shutil.rmtree(dir_name)

        # All done
        return
        
    def plot(self,synth_vector=None,nc=3,nl=5, title = 'Seismic data', sta_lst=None, basename=None,
             figsize=[11.69,8.270],xlims=None,bottom=0.06,top=0.87,left=0.06,right=0.95,wspace=0.25,
             hspace=0.35,grid=True,axis_visible=True,inc=False,Y_max=False,Y_units='mm'):
        '''
        Plot seismic traces
        Args:
           synth_vector: concatenated synthetic waveforms
           nc: number of collumns per page
           nl: number of rows per page
           title: figure title
           sta_lst: station list
        '''
        # Station list
        if sta_lst==None:
            sta_name=self.sta_name
        else:
            sta_name=sta_lst
        # Base name
        if basename==None:
            basename=self.name
        # Set station index limits in synth_vector:
        if synth_vector!=None:
            i = 0
            sta_lims  = {}
            for dkey in self.sta_name:
                sta_lims[dkey] = i
                i += self.d[dkey].npts
        # Plots per page
        perpage = nl*nc
        # Figure object
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(bottom=bottom,top=top,left=left,right=right,wspace=wspace,hspace=hspace)
        # Number of pages
        count = 1; pages = 1; nchan = 1
        ntot   = len(sta_name)
        npages = np.ceil(float(ntot)/float(perpage))
        # Main loop
        sa = 0.; sb = 0.
        for dkey in sta_name:
            # Data vector
            data  = self.d[dkey].depvar
            nsamp = self.d[dkey].npts            
            # Page change
            if count > perpage:
                if title!=None:
                    plt.suptitle(title+ ',   p %d/%d'%(pages,npages), fontsize=16, y=0.95)
                fig.set_rasterized(True)
                plt.savefig('%s_page_%d.pdf'%(basename,pages),orientation='landscape')
                pages += 1
                count = 1
                fig = plt.figure(figsize=[11.69,8.270])
                fig.subplots_adjust(bottom=0.06,top=0.87,left=0.06,right=0.95,wspace=0.25,hspace=0.35)
            t1 = np.arange(nsamp,dtype='double')*self.d[dkey].delta + self.d[dkey].b - self.d[dkey].o
            ax = plt.subplot(nl,nc,count)
            ax.plot(t1,data*1000.,'k')
            if synth_vector!=None:                
                i = sta_lims[dkey]
                synth = synth_vector[i:i+nsamp]   
                ax.plot(t1,synth*1000.,'r')  
                sa = synth.min()*1000.
                sb = synth.max()*1000.            
            a = data.min()*1000.
            b = data.max()*1000.
            if sa<a:
                ymin = 1.1*sa
            else:
                ymin = 1.1*a
            if sb>b:
                ymax = 1.1*sb
            else:
                ymax = 1.1*b                
            if ymin>-20.:
                ymin = -20.
            if ymax<20.:
                ymax=20.
            ax.set_ylim([ymin,ymax])
            if Y_max:                
                label = r'%s %s %s %s $(\phi,\Delta, A) = %6.1f^{\circ}, %6.1f^{\circ}, %.0f%s$'%(
                    self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], self.d[dkey].khole,
                    self.d[dkey].az, self.d[dkey].gcarc,b,Y_units)                
            elif self.d[dkey].kcmpnm[2] == 'Z' or inc==False:
                label = r'%s %s %s %s $(\phi,\Delta) = %6.1f^{\circ}, %6.1f^{\circ}$'%(
                    self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], self.d[dkey].khole,
                    self.d[dkey].az, self.d[dkey].gcarc)
            else:
                label  = r'%s %s %s %s $(\phi,\Delta,\alpha) = %6.1f^{\circ},'
                label += '%6.1f^{\circ}, %6.1f^{\circ}$'
                label  = label%(self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], self.d[dkey].khole,
                                self.d[dkey].az, self.d[dkey].gcarc, self.d[dkey].cmpaz)	
            plt.title(label,fontsize=9.0,va='center',ha='center')                        
            if not (count-1)%nc:
                plt.ylabel(Y_units,fontsize=10)
            if (count-1)/nc == nl-1 or nchan+nc > ntot:
                plt.xlabel('time, sec',fontsize=10) 
            elif not axis_visible:
                ax.xaxis.set_visible(False)
            if not axis_visible:
                ax.yaxis.set_visible(False)
            if xlims!=None:
                plt.xlim(xlims)
            if grid:
                plt.grid()                
            count += 1
            nchan += 1
        fig.set_rasterized(True)
        if title!=None:
            plt.suptitle(title + ',    p %d/%d'%(pages,npages), fontsize=16, y=0.95)
        plt.savefig('%s_page_%d.pdf'%(basename,pages),orientation='landscape')
        plt.close()
    
#EOF
