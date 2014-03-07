'''
Class that plots the class verticalfault, gpsrates and insarrates in 3D.

Written by R. Jolivet and Z. Duputel, April 2013.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as colls
import numpy as np

class geodeticplot(object):

    def __init__(self, figure=130, ref='utm', pbaspect=None):
        '''
        Args:
            * figure        : Number of the figure.
            * ref           : 'utm' or 'lonlat'.
        '''

        # Open a figure
        fig1 = plt.figure(figure)
        faille = fig1.add_subplot(111, projection='3d')
        fig2  = plt.figure(figure+1)
        carte = fig2.add_subplot(111)

        # Set the axes
        if ref is 'utm':
            faille.set_xlabel('Easting (km)')
            faille.set_ylabel('Northing (km)')
            carte.set_xlabel('Easting (km)')
            carte.set_ylabel('Northing (km)')
        else:
            faille.set_xlabel('Longitude')
            faille.set_ylabel('Latitude')
            carte.set_xlabel('Longitude')
            carte.set_ylabel('Latitude')

        faille.set_zlabel('Depth (km)')

        # store plots
        self.faille = faille
        self.carte  = carte
        self.fig1 = fig1
        self.fig2 = fig2
        self.ref  = ref

    def show(self, mapaxis='equal', triDaxis='auto'):
        ''' 
        Show to screen 
        '''

        # Change axis of the map
        self.carte.axis(mapaxis)

        # Change the axis of the 3d projection
        self.faille.axis(triDaxis)

        # Show
        plt.show()

        # All done
        return

    def savefig(self, prefix, mapaxis='equal', dpi=None, bbox_inches=None):
        '''
        Save to file.
        '''

        # Change axis of the map
        self.carte.axis(mapaxis)

        # Save
        if dpi is None:
            self.fig1.savefig('%s_fig1.pdf' % (prefix))
            self.fig2.savefig('%s_fig2.pdf' % (prefix))
        else:
            self.fig1.savefig('%s_fig1.png' % (prefix), dpi=dpi, bbox_inches=bbox_inches)
            self.fig2.savefig('%s_fig2.png' % (prefix), dpi=dpi, bbox_inches=bbox_inches)

        # All done
        return

    def clf(self):
        '''
        Clears the figures
        '''
        self.fig1.clf()
        self.fig2.clf()
        return

    def titlemap(self, titre):
        '''
        Sets the title of the map.
        '''

        self.carte.set_title(titre)

        # All done
        return

    def titlefault(self, titre):
        '''
        Sets the title of the fault model.
        '''

        self.faille.set_title(titre)

        # All done
        return

    def set_view(self, elevation, azimuth):
        '''
        Sets azimuth and elevation angle for the 3D plot.
        '''
        # Set angles
        self.faille.view_init(elevation,azimuth)
        #all done
        return

    def equalize3dAspect(self):
        """
        Make the 3D axes have equal aspect. Not working yet (maybe never).
        """
        #self.faille.set_aspect('equal')
        xlim = self.faille.get_xlim3d()
        ylim = self.faille.get_ylim3d()
        zlim = self.faille.get_zlim3d()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]

        x0 = 0.5 * (xlim[1] + xlim[0])
        y0 = 0.5 * (ylim[1] + ylim[0])
        z0 = 0.5 * (zlim[1] + zlim[0])

        max_range = 0.5 * np.array([x_range, y_range, z_range]).max()

        self.faille.set_xlim3d([x0-max_range, x0+max_range])
        self.faille.set_ylim3d([y0-max_range, y0+max_range])
        #max_range *= 0.35
        #self.faille.set_zlim3d([z0-max_range, z0+max_range])
        self.faille.set_zlim3d(zlim)

        self.fig1.set_size_inches((14,6))


        #for direction in (-1, 1):
        #    for point in np.diag(direction * max_range * np.array([1,1,1])):
        #        self.faille.plot([point[0]+x0], [point[1]+y0], [point[2]+z0], 'w')
        
        return 

    def set_xymap(self, xlim, ylim):
        '''
        Sets the xlim and ylim on the map.
        '''

        self.carte.xlim(xlim)
        self.carte.ylim(ylim)

        # All done
        return

    def faulttrace(self, fault, color='r', add=False):
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
        if self.ref is 'utm':
            if fault.xf is None:
                fault.trace2xy()
            self.faille.plot(fault.xf, fault.yf, '-{}'.format(color), linewidth=2)
            self.carte.plot(fault.xf, fault.yf, '-{}'.format(color), linewidth=2)
        else:
            self.faille.plot(fault.lon, fault.lat, '-{}'.format(color), linewidth=2)
            self.carte.plot(fault.lon, fault.lat, '-{}'.format(color), linewidth=2)

        # All done
        return

    def faultdiscretized(self, fault, color='r'):
        '''
        Args:
            * fault         : Fault class from verticalfault.
        '''

        # Plot the surface trace
        if self.ref is 'utm':
            self.faille.plot(fault.xi, fault.yi, '.{}'.format(color), linewidth=2)
            self.carte.plot(fault.xi, fault.yi, '.{}'.format(color), linewidth=2)
        else:
            self.faille.plot(fault.loni, fault.lati, '.{}'.format(color), linewidth=2)
            self.carte.plot(fault.loni, fault.lati, '.{}'.format(color), linewidth=2)

        # All done
        return

    def faultpatches(self, fault, slip='strikeslip', Norm=None, colorbar=True, 
                     plot_on_2d=False, revmap=False, linewidth=1.0):
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
            rect.set_linewidth(linewidth)
            self.faille.add_collection3d(rect)
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

    def setzaxis(self, depth, zticklabels=None):
        '''
        Set the z-axis.
        Args:
            * depth     : Maximum depth.
            * zticks    : ticks along z.
        '''

        self.faille.set_zlim3d([-1.0*(depth+5), 0])
        if zticklabels is None:
            zticks = []
            zticklabels = []
            for z in np.linspace(0,depth,5):
                zticks.append(-1.0*z)
                zticklabels.append(z)
        else:
            zticks = []
            for z in zticklabels:
                zticks.append(-1.0*z)
        self.faille.set_zticks(zticks)
        self.faille.set_zticklabels(zticklabels)
        
        # All done
        return

    def surfacestress(self, stress, component='normal', linewidth=0.0, Norm=None, colorbar=True):
        '''
        Plots the stress on the map.
        Args:
            * stress        : Stressfield object.
            * component     : If string, can be normal, shearstrike, sheardip
                              If tuple or list, can be anything specifying the indexes of the Stress tensor.
            * linewidth     : option of scatter.
            * Norm          : Scales the color bar.
            * colorbar      : if true, plots a colorbar
        '''
        
        # Get values
        if component.__class__ is str:
            if component in ('normal'):
                val = stress.Sigma
            elif component in ('shearstrike'):
                val = stress.TauStrike
            elif component in ('sheardip'):
                val = stress.TauDip
            else:
                print ('Unknown component of stress to plot')
                return
        else:
            val = stress.Stress[component[0], component[1]]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # Norm
        if Norm is not None:
            vmin = Norm[0]
            vmax = Norm[1]
        else:
            vmin = val.min()
            vmax = val.max()

        # Plot
        if self.ref is 'utm':
            sc = self.carte.scatter(stress.x, stress.y, s=20, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth)
        else:
            sc = self.carte.scatter(stress.lon, stress.y, s=20, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth)

        # colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')   

        # All don
        return

    def gpsvelocities(self, gps, color='k', colorsynth='b', scale=None, legendscale=10., linewidths=.1, name=False, data='both'):
        '''
        Args:
            * gps           : gps object from gpsrates.
            * color         : Color of the gps velocity arrows.
            * colorsynth    : Color of the synthetics.
            * scale         : Scales the arrows
            * legendscale   : Length of the scale.
            * linewidths    : Width of the arrows.
            * name          : Plot the name of the stations (True/False).
            * data          : If both, plots data and synthetics, if 'res', plots the residuals.
        '''

        if data is 'both':
            # Plot the GPS velocities
            if self.ref is 'utm':
                p = self.carte.quiver(gps.x, gps.y, gps.vel_enu[:,0], gps.vel_enu[:,1], width=0.0025, color=color, scale=scale, linewidths=linewidths)
                self.psave = p
                q = self.carte.quiverkey(p, 0.1, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color=color)
            else:
                p = self.carte.quiver(gps.lon, gps.lat, gps.vel_enu[:,0], gps.vel_enu[:,1], width=0.0025, color=color, scale=scale, linewidths=linewidths)
                q = self.carte.quiverkey(p, 0.1, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color=color)

            # If there is some synthetics
            if gps.synth is not None:                                                        
                if self.ref is 'utm':                                                              
                    p = self.carte.quiver(gps.x, gps.y, gps.synth[:,0], gps.synth[:,1], color=colorsynth, scale=scale, width=0.0025, linewidths=linewidths)   
                    q = self.carte.quiverkey(p, 0.1, 0.8, legendscale, "{}".format(legendscale), coordinates='axes', color=colorsynth)
                else:                                                                         
                    p = self.carte.quiver(gps.lon, gps.lat, gps.synth[:,0], gps.synth[:,1], color=colorsynth, scale=scale, width=0.0025, linewidths=linewidths)  
                    q = self.carte.quiverkey(p, 0.1, 0.8, legendscale, "{}".format(legendscale), coordinates='axes', color=colorsynth)
        elif data is 'res':
            if self.ref is 'utm':
                p = self.carte.quiver(gps.x, gps.y, gps.vel_enu[:,0]-gps.synth[:,0], gps.vel_enu[:,1]-gps.synth[:,1], width=0.0025, color=color, scale=scale, linewidths=linewidths)
                self.psave = p
                q = self.carte.quiverkey(p, 0.1, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color=color)
            else:
                p = self.carte.quiver(gps.lon, gps.lat, gps.vel_enu[:,0]-gps.synth[:,0], gps.vel_enu[:,1]-gps.synth[:,1], width=0.0025, color=color, scale=scale, linewidths=linewidths)
                q = self.carte.quiverkey(p, 0.1, 0.9, legendscale, "{}".format(legendscale), coordinates='axes', color=color)

        # Plot the name of the stations if asked
        if name:
            if self.ref is 'utm':
                for i in range(len(gps.x)):
                    self.carte.text(gps.x[i], gps.y[i], gps.station[i], fontsize=12)
            else:
                for i in range(len(gps.lat)):
                    self.carte.text(gps.lon[i], gps.lat[i], gps.station[i], fontsize=12)

        # All done
        return

    def gpsverticals(self, gps, norm=None, colorbar=True, markersize=20.):
        '''
        Scatter plot of the vertical displacement of the GPS.
        '''

        # Get xy
        if self.ref is 'utm':
            x = gps.x
            y = gps.y
        else:
            x = gps.lon
            y = gps.lat

        # Get the verticals
        V = gps.vel_enu[:,2]

        # Get a colormap
        cmap = plt.get_cmap('jet')

        # Norm
        if norm is not None:
            vmin = norm[0]
            vmax = norm[1]
        else:
            vmin = V.min()
            vmax = V.max()

        # Plot that on the map
        sc = self.carte.scatter(x, y, s=markersize, c=V, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.01)

        # Colorbar
        if colorbar:
            self.fig2.colorbar(sc, orientation='horizontal', shrink=0.6)

        return

    def gps_projected(self, gps, norm=None, colorbar=True):
        '''
        Args:
            * gps       : gpsrate object
            * norm      : List of lower and upper bound of the colorbar.
            * colorbar  : activates the plotting of the colorbar.
        '''

        # Get the data
        d = gps.vel_los

        # Prepare the color limits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # Plot
        if self.ref is 'utm':
            sc = self.carte.scatter(gps.x, gps.y, s=100, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5)
        else:
            sc = self.carte.scatter(gps.lon, gps.lat, s=100, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # All done
        return

    def insar_scatter(self, insar, norm=None, colorbar=True, data='data'):
        '''
        Args:
            * insar     : insar object from insarrates.
            * norm      : List of lower and upper bound of the colorbar.
            * colorbar  : activates the plotting of the colorbar.
            * data      : plots either the 'data' or the 'synth'.
        '''

        if data is 'data':
            d = insar.vel
        elif data is 'synth':
            d = insar.synth
        elif data is 'res':
            d = insar.vel-insar.synth

        # Prepare the color limits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # Plot the insar
        if self.ref is 'utm':
            sc = self.carte.scatter(insar.x, insar.y, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0)
        else:
            sc = self.carte.scatter(insar.lon, insar.lat, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # All done
        return

    def earthquakes(self, earthquakes, plot='2d3d', color='k', markersize=5, norm=None, colorbar=False):
        ''' 
        Args:
            * earthquakes   : object from seismic locations.
            * plot          : any combination of 2d and 3d.
            * color         : color of the earthquakes. Can be an array.
            * markersize    : size of each dots. Can be an array.
            * norm          : upper and lower bound for the color code.
            * colorbar      : Draw a colorbar.
        '''

        # set vmin and vmax
        vmin = None
        vmax = None
        if (color.__class__ is np.ndarray):
            if norm is not None:
                vmin = norm[0]
                vmax = norm[1]
            else:
                vmin = color.min()
                vmax = color.max()
            import matplotlib.cm as cmx
            import matplotlib.colors as colors
            cmap = plt.get_cmap('jet')
            cNorm = colors.Normalize(vmin=color.min(), vmax=color.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
            scalarMap.set_array(color)
        else:
            cmap = None

        # plot the earthquakes on the map if ask
        if '2d' in plot:
            if self.ref is 'utm':
                sc = self.carte.scatter(earthquakes.x, earthquakes.y, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.0)
            else:
                sc = self.carte.scatter(earthquakes.lon, earthquakes.lat, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.0)

            if colorbar:
                self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # plot the earthquakes in the volume if ask
        if '3d' in plot:
            if self.ref is 'utm':
                sc = self.faille.scatter3D(earthquakes.x, earthquakes.y, -1.*earthquakes.depth, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.0)
            else:
                sc = self.faille.scatter3D(earthquakes.lon, earthquakes.lat, -1.*earthquakes.depth, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.0)

            if colorbar:
                self.fig1.colorbar(sc, shrink=0.6, orientation='horizontal')
         
        # All done
        return

    def faultsimulation(self, fault, norm=None, colorbar=True, direction='north'):
        '''
        Args:
            * fault     : fault object from verticalfault.
            * norm      : List of lower and upper bound of the colorbar.
            * colorbar  : activates the plotting of the colorbar.
            * direction : which direction do we plot ('east', 'north', 'up', 'total' or a given azimuth in degrees, or an array to project in the LOS).
        '''

        if direction is 'east':
            d = fault.sim.vel_enu[:,0]
        elif direction is 'north':
            d = fault.sim.vel_enu[:,1]
        elif direction is 'up':
            d = fault.sim.vel_enu[:,2]
        elif direction is 'total':
            d = np.sqrt(fault.sim.vel_enu[:,0]**2 + fault.sim.vel_enu[:,1]**2 + fault.sim.vel_enu[:,2]**2)
        elif direction.__class__ is float:
            d = fault.sim.vel_enu[:,0]/np.sin(direction*np.pi/180.) + fault.sim.vel_enu[:,1]/np.cos(direction*np.pi/180.)
        elif direction.shape[0]==3:
            d = np.dot(fault.sim.vel_enu,direction)
        else:
            print ('Unknown direction')
            return

        # Prepare the color limits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # Plot the insar
        if self.ref is 'utm':
            sc = self.carte.scatter(fault.sim.x, fault.sim.y, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.1)
        else:
            sc = self.carte.scatter(fault.sim.lon, fault.sim.lat, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.1)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # All done
        return

    def insar_decimate(self, insar, norm=None, colorbar=True, data='data',
                       plotType='rect', gmtCmap=None):
        ''' 
        Args:
            * insar     : insar object from insarrates.
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * data      : plot either 'data' or 'synth' or 'res'.
        '''

        # Choose data type
        if data == 'data':
            d = insar.vel
        elif data == 'synth':
            d = insar.synth
        elif data == 'res':
            d = insar.vel - insar.synth
        else:
            print('Unknown data type')
            return

        # Prepare the colorlimits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        if gmtCmap is not None:
            try:
                import basemap_utils as bu
                cmap = bu.gmtColormap(gmtCmap)
            except ImportError:
                cmap = plt.get_cmap('jet') 
        else:
            cmap = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        if plotType is 'rect':
            for i in range(insar.xycorner.shape[0]):
                x = []
                y = []
                # upper left
                x.append(insar.xycorner[i,0])
                y.append(insar.xycorner[i,1])
                # upper right
                x.append(insar.xycorner[i,2])
                y.append(insar.xycorner[i,1])
                # down right
                x.append(insar.xycorner[i,2])
                y.append(insar.xycorner[i,3])
                # down left
                x.append(insar.xycorner[i,0])
                y.append(insar.xycorner[i,3])
                verts = [zip(x, y)]
                rect = colls.PolyCollection(verts)
                rect.set_color(scalarMap.to_rgba(d[i]))
                rect.set_edgecolors('k')
                self.carte.add_collection(rect)
            
        elif plotType is 'scatter':
            for i in range(insar.x.size):
                color = scalarMap.to_rgba(d[i])
                self.carte.plot(insar.x[i], insar.y[i], 'o', color=color)

        else:
            assert False, 'unsupported plot type. Must be rect or scatter'

        # plot colorbar
        if colorbar:
            scalarMap.set_array(d)
            plt.colorbar(scalarMap)

        # All done
        return

    def cosicorr_decimate(self, corr, norm=None, colorbar=True, data='dataEast',
                          plotType='rect', gmtCmap=None, decimate=None):
        ''' 
        Args:
            * insar     : insar object from insarrates.
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * data      : plot either 'dataEast', 'dataNorth', 'synthNorth', 'synthEast',
                          'resEast', or 'resNorth'
            * plotType  : plot either rectangular patches (rect) or scatter (scatter)
        '''

        # Choose the data
        if data == 'dataEast':
            d = corr.east
        elif data == 'dataNorth':
            d = corr.north
        elif data == 'synthEast':
            d = corr.east_synth
        elif data == 'synthNorth':
            d = corr.north_synth
        elif data == 'resEast':
            d = corr.east - corr.east_synth
        elif data == 'resNorth':
            d = corr.north - corr.north_synth
        else:
            print('unknown data type')
            return

        # Prepare the colorlimits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        if gmtCmap is not None:
            try:
                import basemap_utils as bu
                cmap = bu.gmtColormap(gmtCmap)
            except ImportError:
                cmap = plt.get_cmap('jet') 
        else:
            cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        if plotType is 'rect':
            for i in range(corr.xycorner.shape[0]):
                x = []
                y = []
                # upper left
                x.append(corr.xycorner[i,0])
                y.append(corr.xycorner[i,1])
                # upper right
                x.append(corr.xycorner[i,2])
                y.append(corr.xycorner[i,1])
                # down right
                x.append(corr.xycorner[i,2])
                y.append(corr.xycorner[i,3])
                # down left
                x.append(corr.xycorner[i,0])
                y.append(corr.xycorner[i,3])
                verts = [zip(x, y)]
                rect = colls.PolyCollection(verts)
                rect.set_color(scalarMap.to_rgba(d[i]))
                rect.set_edgecolors('k')
                self.carte.add_collection(rect)
            
        elif plotType is 'scatter':
            self.carte.scatter(corr.x[::decimate], corr.y[::decimate], s=10., c=d[::decimate], cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0)

        else:
            assert False, 'unsupported plot type. Must be rect or scatter'

        # plot colorbar
        if colorbar:
            scalarMap.set_array(d)
            plt.colorbar(scalarMap)

        # All done
        return

    def slipdirection(self, fault, linewidth=1., color='k', scale=1.):
        '''
        Plots the segment in slip direction of the fault.
        '''

        # Check if it exists
        if not hasattr(fault,'slipdirection'):
            fault.computeSlipDirection(scale=scale)

        # Loop on the vectors
        for v in fault.slipdirection:
            # Z increase downward
            v[0][2] *= -1.0
            v[1][2] *= -1.0
            # Make lists
            x, y, z = zip(v[0],v[1])
            # Plot
            self.faille.plot3D(x, y, z, color=color, linewidth=linewidth)

        # All done
        return
            

