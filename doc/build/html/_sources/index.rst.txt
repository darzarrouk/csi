.. csi documentation master file, created by
   sphinx-quickstart on Thu Apr 19 18:31:04 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CSI user manual
===============================

.. toctree::
   :maxdepth: 1

   installation
   examples
   basictutorial
   CSIpublications
   classes

CSI stands for Classic Slip Inversion. In brief, it allows to quickly build scripts that will allow to obtain the slip distribution along a fault for an earthquake given some geodetic and seismological data. Here is an example of what kind of model you can get.

Put a nice figure here

Overview 
===============================

This library has been designed to handle what is simple in fault slip inversions and especiall what should not be re-written everytime a new students comes to work with us. No one should have to decide how to implement a fault or insar data in python before trying to run an inverse problem because it is not interesting (the inverse problem is interesting and should be delt with great care). Therefore, we have put together this set of python classes that can interact together and allow anyone to set up his own problem.

Because we do not aim at solving everyone's problems, this code is open and can be augmented if you feel like it. Do not hesitate to contact us and share potentially new capabilities.

This manual is minimal... We are aware of it... But until someone manages to lock the authors for enough time in a room with a computer and enough food, there will not be a better one. Luckily, we answer emails and if we are interested in the project, we can help.

Here is a non-exhaustive list of basic capabilities:

  - Simplifies construction of multi-patch faults with rectangular, triangular and node based parameterization
  - For each of the fault parameterizations, allows construction of GFs (homogeneous or stratified elastic structure)
  - Handles various geodetic data types

    - SAR (offsets or interferometry)
    - GPS (hence tide gauges which is a vertical GPS, as everyone knows)
    - Optical cross-correlation measurements 
    - Manipulate many of these

        - Read, plot, make profiles of InSAR time series (no time-dependent modeling yet).
        - Extracts empirical covariance functions from satellite imagery data
        - Downsample satellite imagery data: gradient based (Jonson et al 2002), curvature based (Simons et al 2002) or model based (Lohman et al 2005)

  - Makes profiles across displacements fields (any kind), plot faults in 2-3D, plot displacement maps and vectors, coastlines, political boundaries, etc
  - Tools for fault slip inversions
    - Solves for slip distribution using constrained LSQ or classic MCMC (less than ~40 parameters). 
    - Can provide the input files for AlTar (Bayesian solver of M. Simons' group @CalTech)
    - Some tools to handle kinematic slip inversions.

To cite CSI, please use the doi of the code provided by Github.

Contacts
===============================

To contact us, please visit our websites:

- Romain Jolivet: http://www.geologie.ens.fr/~jolivet
- Zacharie Duputel: http://wphase.unistra.fr/zacharie
- Mark Simons: http://www.gps.caltech.edu/~simons


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
