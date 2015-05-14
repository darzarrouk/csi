import numpy as np
import scipy.interpolate as sciint
try:
    from netCDF4 import Dataset as netcdf
except:
    from scipy.io.netcdf import netcdf_file as netcdf

#----------------------------------------------------------------
#----------------------------------------------------------------
# A routine to write netcdf files

def write2netCDF(filename, lon, lat, z, increments=None, nSamples=None, title='CSI product', name='z', scale=1.0, offset=0.0, xyunits=['Lon', 'Lat'], units='None', interpolation=True):
    '''
    Creates a netCDF file  with the arrays in Z. 
    Z can be list of array or an array, the size of lon.
                
    .. Args:
        
        * filename -> Output file name
        * lon      -> 1D Array of lon values
        * lat      -> 1D Array of lat values
        * z        -> 2D slice to be saved
   
    .. Kwargs:
               
        * title    -> Title for the grd file
        * name     -> Name of the field in the grd file
        * scale    -> Scale value in the grd file
        * offset   -> Offset value in the grd file
                
    .. Returns:
          
        * None'''

    if interpolation:

        # Check
        if nSamples is not None:
            if type(nSamples) is int:
                nSamples = [nSamples, nSamples]
            dlon = (lon.max()-lon.min())/nSamples[0]
            dlat = (lat.max()-lat.min())/nSamples[1]
        if increments is not None:
            dlon, dlat = increments

        # Resample on a regular grid
        olon, olat = np.meshgrid(np.arange(lon.min(), lon.max(), dlon),
                                 np.arange(lat.min(), lat.max(), dlat))
    else:
        # Get lon lat
        olon = lon
        olat = lat
        dlon = olon[0,1]-olon[0,0]
        dlat = olat[1,0]-olat[0,0]

    # Create a file
    fid = netcdf(filename,'w')

    # Create a dimension variable
    fid.createDimension('side',2)
    fid.createDimension('xysize', np.prod(olon.shape))

    # Range variables
    fid.createVariable('x_range','d',('side',))
    fid.variables['x_range'].units = xyunits[0]

    fid.createVariable('y_range','d',('side',))
    fid.variables['y_range'].units = xyunits[1]
    
    # Spacing
    fid.createVariable('spacing','d',('side',))
    fid.createVariable('dimension','i4',('side',))

    # Informations
    if title is not None:
        fid.title = title
    fid.source = 'CSI.utils.write2netCDF'

    # Filing rnage and spacing
    fid.variables['x_range'][0] = olon[0,0]
    fid.variables['x_range'][1] = olon[0,-1]
    fid.variables['spacing'][0] = dlon

    fid.variables['y_range'][0] = olat[0,0]
    fid.variables['y_range'][1] = olat[-1,0]
    fid.variables['spacing'][1] = dlat
    
    if interpolation:
        # Interpolate
        interpZ = sciint.LinearNDInterpolator(np.vstack((lon, lat)).T, z, fill_value=np.nan)
        oZ = interpZ(olon, olat)
    else:
        # Get values
        oZ = z

    # Range
    zmin = np.nanmin(oZ)
    zmax = np.nanmax(oZ)
    fid.createVariable('{}_range'.format(name),'d',('side',))
    fid.variables['{}_range'.format(name)].units = units
    fid.variables['{}_range'.format(name)][0] = zmin
    fid.variables['{}_range'.format(name)][1] = zmax

    # Create Variable
    fid.createVariable(name,'d',('xysize',))
    fid.variables[name].long_name = name
    fid.variables[name].scale_factor = scale
    fid.variables[name].add_offset = offset
    fid.variables[name].node_offset=0

    # Fill it
    fid.variables[name][:] = np.flipud(oZ).flatten()

    # Set dimension
    fid.variables['dimension'][:] = oZ.shape[::-1]

    # Synchronize and close
    fid.sync()
    fid.close()

    # All done
    return
