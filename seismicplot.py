'''
Class that plots Kinematic faults.

Written by Z. Duputel, January 2014.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as colls
import numpy as np

# Base class
from .geodeticplot import geodeticplot

class seismicplot(geodeticplot):

    def __init__(self, figure=130, ref='utm', pbaspect=None):
        '''
        Args:
            * figure        : Number of the figure.
            * ref           : 'utm' or 'lonlat'.
        '''

        # Base class init
        super(seismicplot,self).__init__(figure,ref,pbaspect)        
        
    def faultPatchesGrid(self, fault, slip=None, Norm=None, colorbar=True, 
                         plot_on_2d=False, revmap=False, data=None):
        '''
        Args:
            * fault         : Fault class from verticalfault.
            * slip          : Can be 'strikeslip', 'dipslip' or 'opening'
            * Norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
        '''

        # Get slip
        if slip!=None:
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)

        # norm
        if Norm is None and slip != None:
            vmin=0
            vmax=slip.max()
        else:
            vmin=Norm[0]
            vmax=Norm[1]

        # set z axis
        self.setzaxis(fault.depth+5., zticklabels=fault.z_patches)

        # set color business
        if revmap:
            cmap = plt.get_cmap('jet_r')
        else:
            cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        Xs = np.array([])
        Ys = np.array([])
        for p in range(len(fault.patch)):
            ncorners = len(fault.patch[0])
            x = []
            y = []
            z = []
            for i in range(ncorners):
                if self.ref is 'utm':
                    x.append(fault.patch[p][i][0])
                    y.append(fault.patch[p][i][1])
                    z.append(-1.0*fault.patch[p][i][2])
                else:
                    x.append(fault.patchll[p][i][0])
                    y.append(fault.patchll[p][i][1])
                    z.append(-1.0*fault.patchll[p][i][2])
            x.append(x[0])
            y.append(y[0])
            z.append(z[0])
            verts = [zip(x, y, z)]            
            if slip!=None:
                rect = art3d.Poly3DCollection(verts)
                rect.set_color(scalarMap.to_rgba(slip[p]))                
            else:
                grid = np.array(fault.grid[p])      
                self.faille.scatter3D(grid[:,0],grid[:,1],-grid[:,2],color='b',s=20,zorder=1000)
                rect = art3d.Line3DCollection(verts)
            rect.set_edgecolors('k')
            self.faille.scatter3D(fault.hypo_x,fault.hypo_y,-fault.hypo_z,color='k',marker=(5,1,0),s=100,zorder=1000)
            self.faille.add_collection3d(rect)            
            Xs = np.append(Xs,x)
            Ys = np.append(Ys,y)

        if data!=None and slip==None:
            for x,y in zip(data.x,data.y):
                self.faille.scatter3D(x,y,0.,color='b',marker='v',s=20,zorder=1000)
                Xs = np.append(Xs,x)
                Ys = np.append(Ys,y)

        self.faille.set_xlim3d(Xs.min(),Xs.max())
        self.faille.set_ylim3d(Ys.min(),Ys.max())

        # If 2d.
        if plot_on_2d:
            for patch in fault.patch:
                x1 = patch[0][0]
                y1 = patch[0][1]
                x2 = patch[1][0]
                y2 = patch[1][1]
                x3 = patch[2][0]
                y3 = patch[2][1]
                x4 = patch[3][0]
                y4 = patch[3][1]
                self.carte.plot([x1, x2, x3, x4,x1], [y1, y2, y3, y4,y1], '-k', linewidth=1)
                self.carte.plot([(x1+x3)/2.], [(y1+y3)/2.], '.r', markersize=5)

        # put up a colorbar        
        if colorbar:
            scalarMap.set_array(slip)
            self.fig1.colorbar(scalarMap, shrink=0.6, orientation='horizontal')

        # All done
        return Xs,Ys


    def faulttrace(self, fault, color='r', add=False, data=None):
        '''
        Args:
            * fault         : Fault class from verticalfault.
            * color         : Color of the fault.
            * add           : plot the faults in fault.addfaults    
        '''

        # Plot the added faults before
        if add and (self.ref is 'utm'):
            for f in fault.addfaultsxy:
                self.faille.plot(f[0], f[1], '-k')
                self.carte.plot(f[0], f[1], '-k')
        elif add and (self.ref is not 'utm'):
            for f in fault.addfaults:
                self.faille.plot(f[0], f[1], '-k')
                self.carte.plot(f[0], f[1], '-k')

        # Plot the surface trace
        print fault.top
        if self.ref is 'utm':
            if fault.xf is None:
                fault.trace2xy()
            self.faille.plot3D(fault.xf, fault.yf,-fault.top, '-{}'.format(color), linewidth=2)
            self.carte.plot(fault.xf, fault.yf, '-{}'.format(color), linewidth=2)
        else:
            self.faille.plot3D(fault.lon, fault.lat,-fault.top, '-{}'.format(color), linewidth=2)
            self.carte.plot(fault.lon, fault.lat, '-{}'.format(color), linewidth=2)
            
            
        if data!=None:
            self.carte.plot(data.x,data.y,'bv',zorder=1000)

        # All done
        return





