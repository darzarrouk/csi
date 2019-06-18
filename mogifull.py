'''
A group of routines that runs mogi.

Written by T. Shreve, June 2019.

References:
Mogi, K., 1958, Relations between the eruptions of various volcanoes and the deformations of the ground surfaces around them
Beauducel, Francois, 2011, Mogi: point source in elastic half-space, https://fr.mathworks.com/matlabcentral/fileexchange/25943-mogi-point-source-in-elastic-half-space

'''

# External
import numpy as np
import sys

# Personals
#import okada4py as ok92


#--------------------------------------------------
# Check inputs
def ArraySizes(*args):
    '''
    Only requirement is that each arguments has the same size and can be converted to a numpy array
    Returns : Numpy arrays
    '''

    # Create a list of sizes
    Sizes = []
    Arrays = []

    # Check class
    for arg in args:
        if arg.__class__ in (list, tuple):
            arg = np.array(arg)
        elif arg.__class__ in (float, np.float64, int):
            arg = np.array([arg])
        Arrays.append(arg)
        Sizes.append(arg.shape)

    # Assert sizes
    assert (len(np.unique(Sizes))==1), 'The {} provided arrays are not the same size'.format(len(args))

    # All done
    return Arrays

#--------------------------------------------------
# Displacements only
def displacement(xs, ys, zs, xc, yc, zc, a, DP, nu=0.25):
    '''
    Returns the displacements at the stations located on (xs, ys, zs) for spheroid pressure source
        with center on (xc, yc, zc). All arguments can be float, list or array.

    Note :
        This equations are only correct if the radius of curvature of the upper surface is less than or equal to its depth.


    Args:
            * (xs, ys, zs)      : data point locations
            * (xc, yc, zc)      : center of pressure source
            * a                 : semi-major axis
            * DP                : dimensionless pressure
            * nu                : poisson's ratio

    Returns:
            * u       : Displacement array
    '''

    mu = 30e9

    # Nu does matter here, and it is by default 0.25

    #Define parameters correctly
    lambd = 2.*mu*nu/(1.-2.*nu)        #first Lame's elastic modulus
    P = DP*mu                       #Excess pressure

    # Run mogi
    Ux, Uy, Uz = runMogi_disp(xs, ys, zs, xc, yc, zc, a, P, mu, nu, lambd)

    # Reshape the displacement
    ##u = u.reshape((len(xs), 3))

    # All Done
    return Ux, Uy, Uz

#--------------------------------------------------
# Displacements only
def runMogi_disp(xs, ys, zs, xc, yc, zc, a, P, mu, nu, lambd):
    '''
    Mogi formulation for 3D displacements at the surface (yangdisp.m).

    Args:
            * (xs, ys, zs)      : data point locations
            * (xc, yc, zc)      : center of pressure source
            * a                 : semi-major axis
            * P                 : excess pressure
            * mu                : shear modulus
            * nu                : poisson's ratio
            * lambd             : lame's constant

    Returns:
            * Ux, Uy, Uz        : horizontal and vertical displacements

    '''

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    xxn = xs - xc; yyn = ys - yc
    [rho, phi] = cart2pol(xxn,yyn)
    #rrn = np.sqrt((xxn)**2+(yyn)**2)
    R = np.sqrt((zc)**2+(rho)**2)

    Ur = a**3*P*(1.0-nu)*rho/(mu*R**3)
    Uz = a**3*P*(1.0-nu)*zc/(mu*R**3)

    [Ux, Uy] = pol2cart(Ur,phi)

    #xc + Ux ? #yc + Uy ?
    return Ux, Uy, Uz

#--------------------------------------------------
# Strain only
def runMogi_strain(xs, ys, zs, xc, yc, zc, A, dip, strike, DP, mu, nu):
    '''
    Yang formulation adapted from dMODELS.
    Maurizio Battaglia, et al, dMODELS: A MATLAB software package for modeling crustal deformation near active faults and volcanic centers, JVGR, Volume 254, 2013.
    '''

    return u, d, s
#
# #--------------------------------------------------
# # Stress only
# def stress(xs, ys, zs, xc, yc, zc, width, length, strike, dip, ss, ds, ts, mu=30e9, nu=0.25, full=False):
#     '''
#     Returns the stress at the stations located on (xs, ys, zs) for patches
#         with centers on (xc, yc, zc). All arguments can be float, list or array.
#     if Full is True, returns the full strain tensor,
#             is False, returns and array [nstations, 6] = [Sxx, Sxy, Sxz, Syy, Syz, Szz]
#     '''
#
#     # Mu and Nu do matter here, there is default values, but feel free to change...
#
#     # Check
#     xs, ys, zs = ArraySizes(xs, ys, zs)
#     xc, yc, zc, width, length, strike, dip, ss, ds, ts = ArraySizes(xc, yc, zc, width, length, strike, dip, ss, ds, ts)
#
#     # Normally, StaticInv does angles in Radians
#     dip = dip*180./np.pi
#     strike = strike*180./np.pi
#
#     # Run okada
#     u, d, s, flag, flag2 = ok92.okada92(xs, ys, zs, xc, yc, zc, length, width, dip, strike, ss, ds, ts, mu, nu)
#
#     # Check if things went well
#     if not (flag==0.).all():
#         if not np.where(flag!=0)==[]:
#             print('Something went wrong in okada4py... You should check...Problem with stress')
#             print(' Error: {}'.format(tuple(np.where(flag!=0.))))
#
#     # Reshape the displacement
#     s = s.reshape((len(xs), 6))
#
#     if not full:
#         return s, flag, flag2
#     else:
#         Stress = np.zeros((3, 3, len(xs)))
#         Stress[0,0,:] = s[:,0]  # Sxx
#         Stress[1,1,:] = s[:,3]  # Syy
#         Stress[2,2,:] = s[:,5]  # Szz
#         Stress[0,1,:] = s[:,1]  # Sxy
#         Stress[1,0,:] = s[:,1]  # Sxy
#         Stress[0,2,:] = s[:,2]  # Sxz
#         Stress[2,0,:] = s[:,2]  # Sxz
#         Stress[1,2,:] = s[:,4]  # Syz
#         Stress[2,1,:] = s[:,4]  # Syz
#         return Stress, flag, flag2
#
# #EOF
