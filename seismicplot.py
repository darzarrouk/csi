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
        
    def faultPatchesGrid(self, fault, slip='strikeslip', Norm=None, colorbar=True, 
                     plot_on_2d=False, revmap=False):
        '''
        Args:
            * fault         : Fault class from verticalfault.
            * slip          : Can be 'strikeslip', 'dipslip' or 'opening'
            * Norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
        '''

        # Get slip
        if slip in ('strikeslip'):
            slip = fault.slip[:,0]
        elif slip in ('dipslip'):
            slip = fault.slip[:,1]
        elif slip in ('opening'):
            slip = fault.slip[:,2]
        elif slip in ('total'):
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        else:
            print ("Unknown slip direction")
            return

        # norm
        if Norm is None:
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
            verts = [zip(x, y, z)]
            rect = art3d.Poly3DCollection(verts)
            rect.set_color(scalarMap.to_rgba(slip[p]))
            rect.set_edgecolors('k')
            self.faille.add_collection3d(rect)
            Xs = np.append(Xs,x)
            Ys = np.append(Ys,y)
            grid = np.array(fault.grid[p])
            self.faille.scatter3D(grid[:,0],grid[:,1],1.-grid[:,2],'ko',s=20,zorder=1000)
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


