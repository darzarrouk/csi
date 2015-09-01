'''
Class that plots the class verticalfault, gps and insar in 3D.

Written by R. Jolivet and Z. Duputel, April 2013.
'''

# Numerics
import numpy as np
import scipy.interpolate as sciint

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as colls

# mpl_toolkits
import mpl_toolkits.basemap as basemap
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

# CSI
from .SourceInv import SourceInv

class geodeticplot(object):

    def __init__(self, figure=None, pbaspect=None, 
                 projection='cyl',
                 lonmin=None, latmin=None, lonmax=None, latmax=None,
                 resolution='i',
                 figsize=[None,None]):
        '''
        Args:
            * figure        : Number of the figure.
            * ref           : 'utm' or 'lonlat'.
        '''

        # Check longitude
        if lonmin is not None:
           if lonmin < 0.:
               lonmin += 360.
               lonmax += 360.

        # Save 
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax

        # Open a figure
        fig1 = plt.figure(figure, figsize=figsize[0])
        faille = fig1.add_subplot(111, projection='3d')
        if figure is None:
            nextFig = np.max(plt.get_fignums())+1
        else:
            nextFig=figure+1
        fig2  = plt.figure(nextFig, figsize=figsize[1])
        ax = fig2.add_subplot(111)
        carte = basemap.Basemap(projection=projection,
                                llcrnrlon=lonmin, 
                                llcrnrlat=latmin, 
                                urcrnrlon=lonmax, 
                                urcrnrlat=latmax, 
                                resolution=resolution, ax=ax)

        # Set the axes
        faille.set_xlabel('Longitude')
        faille.set_ylabel('Latitude')
        faille.set_zlabel('Depth (km)')
        carte.ax.set_xlabel('Longitude')
        carte.ax.set_ylabel('Latitude')

        # store plots
        self.faille = faille
        self.fig1 = fig1
        self.carte  = carte
        self.fig2 = fig2
        
        # All done
        return

    def close(self, fig2close=['map', 'fault']):
        '''
        Closes all the figures
        '''

        # Check
        if type(fig2close) is not list:
            fig2close = [fig2close]

        # Figure 1
        if 'fault' in fig2close:
            plt.close(self.fig1)
        if 'map' in fig2close:
            plt.close(self.fig2)

        # All done
        return

    def show(self, mapaxis=None, triDaxis=None, showFig=['fault', 'map'], fitOnBox=True):
        '''
        Show to screen
        Args:
            * mapaxis   : Specify the axis type for the map (see matplotlib)
            * triDaxis  : Specify the axis type for the 3D projection (see mpl_toolkits)
            * showFig   : List of plots to show on screen ('fault' and/or 'map')
            * fitOnBox  : If True, fits the horizontal axis to the one asked at initialization 
                          even if data fall outside the box.
        '''

        # Change axis of the map
        if mapaxis is not None:
            self.carte.ax.axis(mapaxis)

        # Change the axis of the 3d projection
        if triDaxis is not None:
            self.faille.axis(triDaxis)

        # Fits the horizontal axis to the asked values
        if fitOnBox:
            self.carte.ax.set_xlim([self.lonmin, self.lonmax])
            self.carte.ax.set_ylim([self.latmin, self.latmax])
            self.faille.set_xlim(self.carte.ax.get_xlim())
            self.faille.set_ylim(self.carte.ax.get_ylim())

        # Delete figures
        if 'map' not in showFig:
            plt.close(self.fig2)
        if 'fault' not in showFig:
            plt.close(self.fig1)

        # Show
        plt.show()

        # All done
        return

    def savefig(self, prefix, mapaxis='equal', ftype='pdf', dpi=None, bbox_inches=None, triDaxis='auto', saveFig=['fault', 'map']):
        '''
        Save to file.
        ftype can be: 'eps', 'pdf', 'png'
        '''

        # Change axis of the map
        if mapaxis is not None:
            self.carte.ax.axis(mapaxis)

        # Change the axis of the 3d proj
        if triDaxis is not None:
            self.faille.axis(triDaxis)

        # Save
        if (ftype is 'png') and (dpi is not None) and (bbox_inches is not None):
            if 'fault' in saveFig:
                self.fig1.savefig('%s_fault.png' % (prefix), 
                        dpi=dpi, bbox_inches=bbox_inches)
            if 'map' in saveFig:
                self.fig2.savefig('%s_map.png' % (prefix), 
                        dpi=dpi, bbox_inches=bbox_inches)
        else:
            if 'fault' in saveFig:
                self.fig1.savefig('{}_fault.{}'.format(prefix, ftype))
            if 'map' in saveFig:
                self.fig2.savefig('{}_map.{}'.format(prefix, ftype))

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

        self.carte.ax.set_title(titre, y=1.08)

        # All done
        return

    def titlefault(self, titre):
        '''
        Sets the title of the fault model.
        '''

        self.faille.set_title(titre, title=1.08)

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
        self.faille.set_zlim3d(zlim)

        self.fig1.set_size_inches((14,6))

        return

    def set_xymap(self, xlim, ylim):
        '''
        Sets the xlim and ylim on the map.
        '''

        self.carte.ax.xlim(xlim)
        self.carte.ax.ylim(ylim)

        # All done
        return

    def drawCoastlines(self, color='k', linewidth=1.0, linestyle='solid', 
            resolution='i', drawLand=True, drawMapScale=None, 
            parallels=4, meridians=4, drawOnFault=True, drawCountries=True,
            zorder=1):
        '''
        Draws the coast lines in the desired area.
        Args:
            * color         : Color of lines
            * linewidth     : Width of lines
            * linestyle     : Style of lines
            * resolution    : Resolution of the coastline. 
                              Can be c (crude), l (low), i (intermediate), h (high), f (full)
            * drawLand      : Fill the continents (True/False)
            * drawMapScale  : Draw a map scale (None or length in km)
            * drawCountries : Draw County boundaries?
            * parallels     : If int -> Number of parallels 
                              If float -> spacing in degrees between paralles
                              If np.array -> array of parallels
            * meridians     : Number of meridians to draw or array of meridians
            * drawOnFault   : Draw on 3D fault as well
        '''

        # Draw landmask
        if drawLand:
            continents = self.carte.fillcontinents(color='0.9', alpha=0.7, zorder=zorder)

        # MapScale
        if drawMapScale is not None:
            lon = self.lonmin + (self.lonmax-self.lonmin)/6.
            lat = self.latmin + (self.latmax-self.latmin)/6.
            try:
                self.carte.drawmapscale(lon, lat, 
                                        self.lonmin + (self.lonmax-self.lonmin)/2., 
                                        self.latmin + (self.latmax-self.latmin)/2., 
                                        drawMapScale, units='km', barstyle='simple', zorder=zorder)
            except:
                print('Map Scale cannot be plotted with this projection')

        # Draw and get the line object
        coasts = self.carte.drawcoastlines(color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
        if drawOnFault:
            segments = []
            for path in coasts.get_paths():
                segments.append(np.hstack((path.vertices,np.zeros((path.vertices.shape[0],1)))))
            if len(segments)>0:
                cote = art3d.Line3DCollection(segments)
                cote.set_edgecolor(coasts.get_color())
                cote.set_linestyle(coasts.get_linestyle())
                cote.set_linewidth(coasts.get_linewidth())
                self.faille.add_collection3d(cote)

        # Draw countries
        if drawCountries:
            countries = self.carte.drawcountries(linewidth=linewidth/2., color='darkgray', zorder=zorder)
            if drawOnFault:
                segments = []
                for path in coasts.get_paths():
                    segments.append(np.hstack((path.vertices,np.zeros((path.vertices.shape[0],1)))))
                if len(segments)>0:
                    border = art3d.Line3DCollection(segments)
                    border.set_edgecolor(countries.get_color())
                    border.set_linestyle(countries.get_linestyle())
                    border.set_linewidth(countries.get_linewidth())
                    self.faille.add_collection3d(border)

        # Draw parallels 
        lmin = self.carte.latmin
        lmax = self.carte.latmax 
        if type(parallels) is int:
            parallels = np.linspace(lmin, lmax, parallels+1)
        elif type(parallels) is float:
            parallels = np.arange(lmin, lmax+parallels, parallels)
        parallels = np.round(parallels, decimals=2)
        parDir = self.carte.drawparallels(parallels, labels=[0,1,0,0], linewidth=0.4, color='gray', zorder=zorder)
        if drawOnFault and parDir!={}:
            segments = []
            colors = []
            linestyles = []
            linewidths = []
            for p in parDir:
                par = parDir[p][0][0]
                segments.append(np.hstack((par.get_path().vertices,np.zeros((par.get_path().vertices.shape[0],1)))))
                colors.append(par.get_color())
                linestyles.append(par.get_linestyle())
                linewidths.append(par.get_linewidth())
            parallel = art3d.Line3DCollection(segments, colors=colors, linestyles=linestyles, linewidths=linewidths)
            self.faille.add_collection3d(parallel)

        # Draw meridians
        lmin = self.carte.lonmin
        lmax = self.carte.lonmax
        if type(meridians) is int:
            meridians = np.linspace(lmin, lmax, meridians+1)
        elif type(meridians) is float:
            meridians = np.arange(lmin, lmax+meridians, meridians)
        meridians = np.round(meridians, decimals=2)
        merDir = self.carte.drawmeridians(meridians, labels=[0,0,1,0], linewidth=0.4, color='gray', zorder=zorder)
        if drawOnFault and merDir!={}:
            segments = []
            colors = []
            linestyles = []
            linewidths = []
            for m in merDir:
                mer = merDir[m][0][0]
                segments.append(np.hstack((mer.get_path().vertices,np.zeros((mer.get_path().vertices.shape[0],1)))))
                colors.append(mer.get_color())
                linestyles.append(mer.get_linestyle())
                linewidths.append(mer.get_linewidth())
            meridian = art3d.Line3DCollection(segments, colors=colors, linestyles=linestyles, linewidths=linewidths)
            self.faille.add_collection3d(meridian)

        # Restore axis
        self.faille.set_xlim(self.carte.ax.get_xlim())
        self.faille.set_ylim(self.carte.ax.get_ylim())

        # All done
        return
    
    def faulttrace(self, fault, color='r', add=False, discretized=False, zorder=4):
        '''
        Args:
            * fault         : Fault class from verticalfault.
            * color         : Color of the fault.
            * add           : plot the faults in fault.addfaults
            * discretized   : Plot the discretized fault
        '''

        # discretized?
        if discretized:
            lon = fault.loni
            lat = fault.lati
        else:
            lon = fault.lon
            lat = fault.lat

        # Plot the added faults before
        if add:
            for f in fault.addfaults:
                if f[0]<0.:
                    f += 360.
                self.carte.plot(f[0], f[1], '-k', zorder=zorder)
            for f in fault.addfaults:
                if self.faille_flag:
                    self.faille.plot(f[0], f[1], '-k')

        # Plot the surface trace
        lon[lon<0.] += 360.
        self.faille.plot(lon, lat, '-{}'.format(color), linewidth=2)
        self.carte.plot(lon, lat, '-{}'.format(color), linewidth=2, zorder=2)

        # All done
        return

    def faultpatches(self, fault, slip='strikeslip', Norm=None, colorbar=True,
            plot_on_2d=False, revmap=False, linewidth=1.0, transparency=0.0, factor=1.0, zorder=0):
        '''
        Args:
            * fault         : Fault class from verticalfault.
            * slip          : Can be 'strikeslip', 'dipslip', 'tensile', 'total' or 'coupling'
            * Norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
            * factor        : scale factor for fault slip values
        '''

        # Get slip
        if slip in ('strikeslip'):
            slip = fault.slip[:,0].copy()
        elif slip in ('dipslip'):
            slip = fault.slip[:,1].copy()
        elif slip in ('tensile'):
            slip = fault.slip[:,2].copy()
        elif slip in ('total'):
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        elif slip in ('coupling'):
            slip = fault.coupling.copy()
        else:
            print ("Unknown slip direction")
            return
        slip *= factor

        # norm
        if Norm is None:
            vmin=slip.min()
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
            ncorners = len(fault.patchll[0])
            x = []
            y = []
            z = []
            for i in range(ncorners):
                x.append(fault.patchll[p][i][0])
                y.append(fault.patchll[p][i][1])
                z.append(-1.0*fault.patchll[p][i][2])
            verts = []
            for xi,yi,zi in zip(x,y,z):
                if xi<0.:
                    xi += 360.
                verts.append((xi,yi,zi))
            rect = art3d.Poly3DCollection([verts])
            rect.set_facecolor(scalarMap.to_rgba(slip[p]))
            rect.set_edgecolors('gray')
            alpha = 1.0 - transparency
            if alpha<1.0:
                rect.set_alpha(alpha)
            rect.set_linewidth(linewidth)
            self.faille.add_collection3d(rect)

        # Reset x- and y-lims 
        self.faille.set_xlim(self.carte.ax.get_xlim())
        self.faille.set_ylim(self.carte.ax.get_ylim())

        # If 2d.
        if plot_on_2d:
            for p, patch in zip(range(len(fault.patchll)), fault.patchll):
                x = []
                y = []
                for i in range(ncorners):
                    x.append(patch[i][0])
                    y.append(patch[i][1])
                verts = []
                for xi,yi in zip(x,y):
                    if xi<0.:
                        xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection([verts])
                rect.set_facecolor(scalarMap.to_rgba(slip[p]))
                rect.set_edgecolors('gray')
                rect.set_linewidth(linewidth)
                rect.set_zorder(zorder)
                self.carte.ax.add_collection(rect)
                
        # put up a colorbar
        if colorbar:
            scalarMap.set_array(slip)
            self.fphbar = self.fig1.colorbar(scalarMap, shrink=0.6, orientation='horizontal')

        # All done
        return 

    def faultTents(self, fault, slip='strikeslip', Norm=None, colorbar=True, method='surface',
            plot_on_2d=False, revmap=False, factor=1.0, npoints=10, xystrides=[100, 100], zorder=0):
        '''
        Args:
            * fault         : Fault class from verticalfault.
            * slip          : Can be 'strikeslip', 'dipslip', 'tensile', 'total' or 'coupling'
            * Norm          : Limits for the colorbar.
            * method        : Can be 'scatter' --> Plots all the sub points as a colored dot.
                                     'surface' --> Interpolates a 3D surface (can be ugly)
            * colorbar      : if True, plots a colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
            * revmap        : Reverse the default colormap
            * factor        : Scale factor for fault slip values
            * npoints       : Number of subpoints per patch. This number is only indicative of the 
                              actual number of points that is picked out by the dropSourcesInPatch
                              function of EDKS.py. It only matters to make the interpolation finer.
                              Default value is generally alright.
            * xystrides     : If method is 'surface', then xystrides is going to be the number of 
                              points along x and along y used to interpolate the surface in 3D and 
                              its color.
        '''

        # Get slip
        if slip in ('strikeslip'):
            slip = fault.slip[:,0].copy()
        elif slip in ('dipslip'):
            slip = fault.slip[:,1].copy()
        elif slip in ('tensile'):
            slip = fault.slip[:,2].copy()
        elif slip in ('total'):
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        elif slip in ('coupling'):
            slip = fault.coupling.copy()
        else:
            print ("Unknown slip direction")
            return
        slip *= factor

        # norm
        if Norm is None:
            vmin=slip.min()
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

        # Get the variables we need
        vertices = fault.Vertices.tolist()
        vertices_ll = fault.Vertices_ll.tolist()
        patches = fault.patch
        faces = fault.Faces

        # Plot the triangles
        for face in faces:
            verts = [vertices_ll[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [-1.0*v[2] for v in verts]
            x.append(x[0]); y.append(y[0]); z.append(z[0])
            x = np.array(x); x[x<0.] += 360.
            self.faille.plot3D(x, y, z, '-', color='gray', linewidth=1)
            if plot_on_2d:
                self.carte.plot(x, y, '-', color='gray', linewidth=1, zorder=zorder)

        # Plot the color for slip
        # 1. Get the subpoints for each triangle
        from .EDKS import dropSourcesInPatches as Patches2Sources
        if hasattr(fault, 'plotSources'):
            if fault.sourceNumber==npoints:
                print('Using precomputed sources for plotting')
        else:
            fault.sourceNumber = npoints
            Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(fault, verbose=False)
            fault.plotSources = [Ids, xs, ys, zs, strike, dip, Areas]

        # Get them
        Ids = fault.plotSources[0]
        X = fault.plotSources[1]
        Y = fault.plotSources[2]
        Z = fault.plotSources[3]

        # 2. Interpolate the slip on each subsource
        Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, slip)
        
        # Check Method:
        if method is 'surface':

            # Do some interpolation
            intpZ = sciint.LinearNDInterpolator(np.vstack((X, Y)).T, Z, fill_value=np.nan) 
            intpC = sciint.LinearNDInterpolator(np.vstack((X, Y)).T, Slip, fill_value=np.nan)
            x = np.linspace(np.nanmin(X), np.nanmax(X), xystrides[0])
            y = np.linspace(np.nanmin(Y), np.nanmax(Y), xystrides[1])
            x,y = np.meshgrid(x,y)
            z = intpZ(np.vstack((x.flatten(), y.flatten())).T).reshape(x.shape)
            slip = intpC(np.vstack((x.flatten(), y.flatten())).T).reshape(x.shape)

            # Do the surface plot
            cols = np.empty(x.shape, dtype=tuple)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    cols[i,j] = scalarMap.to_rgba(slip[i,j])

            lon, lat = fault.xy2ll(x, y)
            lon[lon<0.] += 360.
            self.faille.plot_surface(lon, lat, -1.0*z, facecolors=cols, rstride=1, cstride=1, antialiased=True, linewidth=0)

            # On 2D?
            if plot_on_2d:
                lon, lat = fault.xy2ll(X, Y)
                lon[lon<0.] += 360.
                self.carte.scatter(lon, lat, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax, zorder=zorder) 

            # Color Bar
            if colorbar:
                scalarMap.set_array(slip)
                self.fphbar = self.fig1.colorbar(scalarMap, shrink=0.6, orientation='horizontal')

        elif method is 'scatter':
            # Do the scatter ploto
            lon, lat = fault.xy2ll(X, Y)
            lon[lon<0.] += 360.
            cb = self.faille.scatter3D(lon, lat, zs=-1.0*Z, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)

            # On 2D?
            if plot_on_2d:
                self.carte.scatter(lon, lat, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax, zorder=zorder) 

            # put up a colorbar
            if colorbar:
                self.fphbar = self.fig1.colorbar(cb, shrink=0.6, orientation='horizontal')

        # All done
        return lon, lat, Z, Slip

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

    def setview(self, elev=30.0, azim=180.0):
        '''
        Set the 3D view direction
        Args:
            * elevation : in degree, angle from horizontal plane, positive looking downwards
            * azimuth   : in degree, 0 for North-to-South direction
        '''
        self.faille.view_init(elev=elev, azim=azim)
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
        sc = self.carte.scatter(stress.lon, stress.lat, s=20, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth)

        # colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # All don
        return

    def gps(self, gps, data=['data'], color=['k'], scale=None, legendscale=10., linewidths=.1, name=False, zorder=5):
        '''
        Args:
            * gps           : gps object from gps.
            * data          : List of things to plot:
                              Can be any list of 'data', 'synth', 'res', 'strain', 'transformation'
            * color         : List of the colors of the gps velocity arrows.
                              Must be the same size as data
            * scale         : Scales the arrows
            * legendscale   : Length of the scale.
            * linewidths    : Width of the arrows.
            * name          : Plot the name of the stations (True/False).
        '''

        # Assert
        if (type(data) is not list) and (type(data) is str):
            data = [data]
        if (type(color) is not list):
            color = [color]
        if len(color)==1 and len(data)>1:
            color = [color[0] for d in data]

        # Get lon lat
        lon = gps.lon
        lat = gps.lat
        lon[lon<0.] += 360.

        # Make the dictionary of the things to plot
        Data = {}
        for dtype,col in zip(data, color):
            if dtype is 'data':
                dName = '{} Data'.format(gps.name)
                Values = gps.vel_enu
            elif dtype is 'synth':
                dName = '{} Synth.'.format(gps.name)
                Values = gps.synth
            elif dtype is 'res':
                dName = '{} Res.'.format(gps.name)
                Values = gps.vel_enu - gps.synth
            elif dtype is 'strain':
                dName = '{} Strain'.format(gps.name)
                Values = gps.Strain
            elif dtype is 'transformation':
                dName = '{} Trans.'.format(gps.name)
                Values = gps.transformation
            else:
                assert False, 'Data name not recognized'
            Data[dName] = {}
            Data[dName]['Values'] = Values
            Data[dName]['Color'] = col

        # Plot these
        for dName in Data:
            values = Data[dName]['Values']
            c = Data[dName]['Color']
            p = self.carte.quiver(lon, lat, values[:,0], values[:,1], width=0.005, color=c, scale=scale, linewidths=linewidths, zorder=zorder)
#            if np.isfinite(self.err_enu[:,0]).all() and np.isfinite(self.err_enu[:,1]).all():
                # Extract the location of the arrow head

                # Create an ellipse of the good size at that location

                # Add it to collection, under the arrow

        # Plot Legend
        q = plt.quiverkey(p, 0.1, 0.1, legendscale, '{}'.format(legendscale), coordinates='axes', color='k', zorder=10)

        # Plot the name of the stations if asked
        if name:
            font = {'family' : 'serif',
                    'color'  : 'k',
                    'weight' : 'normal',
                    'size'   : 15}
            for lo, la, sta in zip(lon.tolist(), lat.tolist(), gps.station):
                self.carte.ax.text(lo, la, sta, fontdict=font)

        # All done
        return

    def gpsverticals(self, gps, norm=None, colorbar=True, data=['data'], markersize=[10], linewidth=0.1, zorder=4, cmap='jet'):
        '''
        Scatter plot of the vertical displacement of the GPS.
        '''

        # Assert
        if (type(data) is not list) and (type(data) is str):
            data = [data]
        if (type(markersize) is not list):
            markersize = [markersize]
        if len(markersize)==1 and len(data)>1:
            markersize = [markersize[0] for d in data]

        # Get lon lat
        lon = gps.lon
        lat = gps.lat
        lon[lon<0.] += 360.

        # Initiate
        vmin = 999999999.
        vmax = -999999999.

        # Make the dictionary of the things to plot
        from collections import OrderedDict
        Data = OrderedDict()
        for dtype,mark in zip(data, markersize):
            if dtype is 'data':
                dName = '{} Data'.format(gps.name)
                Values = gps.vel_enu[:,2]
            elif dtype is 'synth':
                dName = '{} Synth.'.format(gps.name)
                Values = gps.synth[:,2]
            elif dtype is 'res':
                dName = '{} Res.'.format(gps.name)
                Values = gps.vel_enu[:,2] - gps.synth[:,2] 
            elif dtype is 'strain':
                dName = '{} Strain'.format(gps.name)
                Values = gps.Strain[:,2]
            elif dtype is 'transformation':
                dName = '{} Trans.'.format(gps.name)
                Values = gps.transformation[:,2]
            Data[dName] = {}
            Data[dName]['Values'] = Values
            Data[dName]['Markersize'] = mark
            vmin = np.min([vmin, np.min(Values)])
            vmax = np.max([vmax, np.max(Values)])

        # Get a colormap
        cmap = plt.get_cmap(cmap)

        # Norm
        if norm is not None:
            vmin = norm[0]
            vmax = norm[1]

        # Plot that on the map
        for dName in Data:
            mark = Data[dName]['Markersize']
            V = Data[dName]['Values']
            sc = self.carte.scatter(lon, lat, s=mark, c=V, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth, zorder=zorder)

        # Colorbar
        if colorbar:
            self.fig2.colorbar(sc, orientation='horizontal', shrink=0.6)

        return

    def gpsprojected(self, gps, norm=None, colorbar=True, zorder=4):
        '''
        Args:
            * gps       : gpsrate object
            * norm      : List of lower and upper bound of the colorbar.
            * colorbar  : activates the plotting of the colorbar.
        '''

        # Get the data
        d = gps.vel_los
        lon = gps.lon; lon[lon<0.] += 360.
        lat = gps.lat

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
        sc = self.carte.scatter(lon, lat, s=100, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, zorder=zorder)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # All done
        return

    def earthquakes(self, earthquakes, plot='2d3d', color='k', markersize=5, norm=None, colorbar=False, zorder=2):
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

        # Get lon lat
        lon = earthquakes.lon; lon[lon<0.] += 360.
        lat = earthquakes.lat

        # plot the earthquakes on the map if ask
        if '2d' in plot:
            sc = self.carte.scatter(lon, lat, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.1, zorder=zorder)
            if colorbar:
                self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # plot the earthquakes in the volume if ask
        if '3d' in plot:
            sc = self.faille.scatter3D(lon, lat, -1.*earthquakes.depth, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.1)
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

        # Get lon lat
        lon = fault.sim.lon; lon[lon<0.] += 360.
        lat = fault.sim.lat

        # Plot the insar
        sc = self.carte.scatter(lon, lat, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.1)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.6, orientation='horizontal')

        # All done
        return

    def insar(self, insar, norm=None, colorbar=True, data='data',
                       plotType='decimate', gmtCmap=None, 
                       decim=1, zorder=3):
        '''
        Args:
            * insar     : insar object from insar.
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * data      : plot either 'data' or 'synth' or 'res'.
            * plotType  : Can be 'decimate' or 'scatter'
            * decim     : In case plotType='scatter', decimates the data by a factor decim.
        '''

        # Assert
        assert data in ('data', 'synth', 'res', 'poly'), 'Data type to plot unknown'
        
        # Choose data type
        if data == 'data':
            d = insar.vel
        elif data == 'synth':
            d = insar.synth
        elif data == 'res':
            d = insar.vel - insar.synth
        elif data == 'poly':
            d = insar.orb
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

        if plotType is 'decimate':
            for corner, disp in zip(insar.corner, d):
                x = []
                y = []
                # upper left
                x.append(corner[0])
                y.append(corner[1])
                # upper right
                x.append(corner[2])
                y.append(corner[1])
                # down right
                x.append(corner[2])
                y.append(corner[3])
                # down left
                x.append(corner[0])
                y.append(corner[3])
                verts = []
                for xi,yi in zip(x,y):
                    if xi<0.:
                        xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection([verts])
                rect.set_color(scalarMap.to_rgba(disp))
                rect.set_edgecolors('k')
                rect.set_zorder(zorder)
                self.carte.ax.add_collection(rect)

        elif plotType is 'scatter':
            lon = insar.lon; lon[lon<0.] += 360.
            lat = insar.lat
            sc = self.carte.scatter(lon[::decim], lat[::decim], s=30, c=d[::decim], cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0, zorder=zorder)

        else:
            print('Unknown plot type: {}'.format(plotType))
            return

        # plot colorbar
        if colorbar:
            scalarMap.set_array(d)
            plt.colorbar(scalarMap,shrink=0.6, orientation='horizontal')

        # All done
        return

    def cosicorr(self, corr, norm=None, colorbar=True, data='dataEast',
                plotType='decimate', gmtCmap=None, decim=1, zorder=3):
        '''
        Args:
            * corr      : instance of the class cosicorr
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * data      : plot either 'dataEast', 'dataNorth', 'synthNorth', 'synthEast',
                          'resEast', 'resNorth', 'data', 'synth' or 'res'
            * plotType  : plot either rectangular patches (decimate) or scatter (scatter)
            * decim     : decimation factor if plotType='scatter'
        '''

        # Assert
        assert data in ('dataEast', 'dataNorth', 'synthEast', 'synthNorth', 'resEast', 'resNorth', 'data', 'synth', 'res'), 'Data type to plot unknown'

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
        elif data == 'data':
            d = np.sqrt(corr.east**2+corr.north**2)
        elif data == 'synth':
            d = np.sqrt(corr.east_synth**2 + corr.north_synth**2)
        elif data == 'res':
            d = np.sqrt( (corr.east - corr.east_synth)**2 + \
                    (corr.north - corr.north_synth)**2 )

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

        if plotType is 'decimate':
            for corner, disp in zip(corr.corner, d):
                x = []
                y = []
                # upper left
                x.append(corner[0])
                y.append(corner[1])
                # upper right
                x.append(corner[2])
                y.append(corner[1])
                # down right
                x.append(corner[2])
                y.append(corner[3])
                # down left
                x.append(corner[0])
                y.append(corner[3])
                verts = []
                for xi,yi in zip(x,y):
                    if xi<0.:
                        xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection(verts)
                rect.set_color(scalarMap.to_rgba(disp))
                rect.set_edgecolors('k')
                rect.set_zorder(zorder)
                self.carte.add_collection(rect)

        elif plotType is 'scatter':
            lon = corr.lon; lon[lon<0.] += 360.
            lat = corr.lat
            self.carte.scatter(lon[::decim], lat[::decim], s=10., c=d[::decim], cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0, zorder=zorder)

        else:
            assert False, 'unsupported plot type. Must be rect or scatter'

        # plot colorbar
        if colorbar:
            scalarMap.set_array(d)
            plt.colorbar(scalarMap, shrink=0.6, orientation='horizontal')

        # All done
        return

    def slipdirection(self, fault, linewidth=1., color='k', scale=1.):
        '''
        Plots the segment in slip direction of the fault.
        '''

        # Check utmzone
        assert self.utmzone==fault.utmzone, 'Fault object {} not in the same utmzone...'.format(fault.name)
        
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

#EOF
