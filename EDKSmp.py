'''
A bunch of routines to handle EDKS

Written by F. Ortega in 2010.
Modified by R. Jolivet in 2014.
'''

# Externals
import os
import struct 
import sys
import numpy as np
import multiprocessing as mp

# Initialize a class to allow multiprocessing
class pointdropper(mp.Process):

    def __init__(self, fault, queue, charArea, istart, iend):
        '''
        Initialize the multiprocessing class to run the point dropper

        Args:
            * fault     : Instance of Fault.py
            * queue     : INstance of mp.Queue
            * charArea  : Characteristic area of the subfaults
            * istart    : Index of the first patch to deal with
            * iend      : Index of the last pacth to deal with
        '''

        # Save the fault
        self.fault = fault
        self.charArea = charArea
        self.istart = istart
        self.iend = iend

        # Save the queue
        self.queue = queue

        # Initialize the Process
        super(pointdropper, self).__init__()

        # All done
        return

    def run(self):
        '''
        Run the subpatch construction
        '''

        # Create lists
        Ids, Xs, Ys, Zs, Strike, Dip, Area = [], [], [], [], [], [], []
        allSplitted = []

        # Iterate overthe patches
        for i in range(self.istart, self.iend):

            # Get patch
            patch = self.fault.patch[i]

            # Check if the Area is bigger than the target
            area = self.fault.patchArea(patch)
            if area>self.charArea[i]:
                keepGoing = True
                tobeSplitted = [patch]
                splittedPatches = []
            else: 
                keepGoing = False
                print('Be carefull, patch {} has not been refined into point sources'.format(self.fault.getindex(patch)))
                print('Possible causes: Area = {}, Nodes = {}'.format(area, patch))
                tobeSplitted = []
                splittedPatches = [patch]

            # Iterate
            while keepGoing:
                
                # Take a patch
                p = tobeSplitted.pop()

                # Split into 4 patches
                Splitted = self.fault.splitPatch(p)

                # Check the area
                for splitted in Splitted:
                    # get area
                    area = self.fault.patchArea(splitted)
                    # check 
                    if area<self.charArea[i]:
                        splittedPatches.append(splitted)
                    else:
                        tobeSplitted.append(splitted)

                # Do we continue?
                if len(tobeSplitted)==0:
                    keepGoing = False

                # Do we have a limit
                if hasattr(self.fault, 'maximumSources'):
                    if len(splittedPatches)>=self.fault.maximumSources:
                        keepGoing = False

            # When all done get their centers
            geometry = [self.fault.getpatchgeometry(p, center=True)[:3] for p in splittedPatches]
            x, y, z = zip(*geometry)
            strike, dip = self.fault.getpatchgeometry(patch)[5:7] 
            strike = np.ones((len(x),))*strike
            strike = strike.tolist()
            dip = np.ones((len(x),))*dip
            dip = dip.tolist()
            areas = [self.fault.patchArea(p) for p in splittedPatches]
            ids = np.ones((len(x),))*(i)
            ids = ids.astype(np.int).tolist()

            # Save
            Ids += ids
            Xs += x
            Ys += y
            Zs += z
            Strike += strike
            Dip += dip
            Area += areas
            allSplitted += splittedPatches

        # Put in the Queue
        self.queue.put([Ids, Xs, Ys, Zs, Strike, Dip, Area, splittedPatches])

        # all done
        return



def dropSourcesInPatches(fault, verbose=False, returnSplittedPatches=False):
    '''
    From a fault object, returns sources to be given to sum_layered_sub.
    The number of sources is determined by the spacing provided in fault.
    Args:
        fault                   : instance of Fault (Rectangular or Triangular).
        verbose                 : Talk to me
        returnSplittedPactches  : Returns a triangularPatches object with the splitted 
                                  patches.
    '''

    # Create lists
    Ids, Xs, Ys, Zs, Strike, Dip, Area = [], [], [], [], [], [], []
    allSplitted = []

    # Check
    if (not hasattr(fault, 'sourceSpacing')) and (not hasattr(fault, 'sourceNumber')) and (not hasattr(fault, 'sourceArea')):
        print('EDKS: Need to provide area, spacing or number of sources...')
        sys.exit(1)
    if hasattr(fault, 'sourceSpacing') and hasattr(fault, 'sourceNumber') and hasattr(fault, 'sourceArea'):
        print('EDKS: Please delete sourceSpacing, sourceNumber or sourceArea...')
        print('EDKS: I do not judge... You decide...')
        sys.exit(1)

    # show me
    if verbose:
        print('Dropping point sources') 

    # Spacing
    if hasattr(fault, 'sourceArea'):
        area = fault.sourceArea
        charArea = np.ones((len(fault.patch),))*area
    if hasattr(fault, 'sourceSpacing'):
        spacing = fault.sourceSpacing
        if fault.patchType == 'rectangle':
            charArea = np.ones((len(fault.patch),))*spacing**2
        elif fault.patchType in ('triangle', 'triangletent'):
            charArea = np.ones((len(fault.patch),))*spacing**2/2.
    if hasattr(fault, 'sourceNumber'):
        number = fault.sourceNumber
        fault.computeArea()
        charArea = np.array(fault.area)/np.float(number)

    # Create a queue
    output = mp.Queue()

    # how many workers
    try:
        nworkers = int(os.environ['OMP_NUM_THREADS'])
    except:
        nworkers = mp.cpu_count()

    # how many patches
    npatches = len(fault.patch)

    # Create them
    workers = [pointdropper(fault, output, charArea, 
                            i*npatches/nworkers, 
                            (i+1)*npatches/nworkers) for i in range(nworkers)]
    workers[-1].iend = npatches

    # Start them
    for worker in workers: worker.start()
    for worker in workers: worker.join()

    # Get things from the queue
    for i in range(nworkers):
        ids, xs, ys, zs, strike, dip, area, splitted = output.get()
        Ids += ids
        Xs += xs 
        Ys += ys
        Zs += zs 
        Strike += strike
        Dip += dip
        Area += area
        allSplitted += splitted

    # Make arrays
    Ids = np.array(Ids)
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Zs = np.array(Zs)
    Strike = np.array(Strike)
    Dip = np.array(Dip)
    Area = np.array(Area)

    # All done
    if returnSplittedPatches:
        from .TriangularPatches import TriangularPatches as trianglePatches
        splitFault = trianglePatches('Splitted {}'.format(fault.name), 
                                     utmzone=fault.utmzone, 
                                     lon0=fault.lon0,
                                     lat0=fault.lat0,
                                     ellps=fault.ellps,
                                     verbose=verbose)
        # set up patches
        splitFault.patch = [np.array(p) for p in allSplitted]
        splitFault.patch2ll()
        # Patches 2 vertices
        splitFault.setVerticesFromPatches()
        # Depth
        splitFault.setdepth()
        return Ids, Xs, Ys, Zs, Strike, Dip, Area, splitFault
    else:
        return Ids, Xs, Ys, Zs, Strike, Dip, Area

def sum_layered(xs, ys, zs, strike, dip, rake, slip, width, length,\
                npw, npy,\
                xr, yr, edks,\
                prefix, \
                BIN_EDKS = 'EDKS_BIN',
                cleanUp=True, verbose=True):
    '''
    --- INPUT ---
    --- SOURCE INFO
    --- 1D NUMPY arrays, length = number of fault patches
    xs       m, east coord to center of fault patch
    ys       m, north coord to center of fault patch
    zs       m,depth coord to center of fault patch (+ down) 
    strike   deg, clockwise from north 
    dip      deg, 90 is vertical 
    rake     deg, 0 left lateral strike slip, 90 up-dip slip 
    slip     m, slip in the rake direction
    width    m, width of the patch
    length   m, length of the patch
    npw      integers, number of sources along strike
    npy      integers, number of sources along dip 
    --- RECEIVER INFO
    1D arrays, length = number of receivers
    xr       m, east coordinate of receivers 
    yr       m, north coordinate of receivers 
    --- ELASTIC STRUCTURE INFO
    edks     string, full name of edks file, e.g., halfspace.edks
    --- FILE NAMING 
    prefix   string, prefix for the IO files generated by sum_layered
    BIN_EDKS Name of the environement variable where EDKS executables are.
    --- OUTPUT ---
    --- 2D arrays, (receivers, fault patches)
    ux     m, east displacement
    uy     m, west displacement
    uz     m, up displacement (+ up)
    '''

    # Get executables
    BIN_EDKS = os.environ[BIN_EDKS]

    # Some initializations
    Np = len(xs)            # number of sources
    nrec = len(xr)          # number of receivers
    A = length*width        # Area of the patches

    # Some formats
    BIN_FILE_FMT = 'f' # python float = C/C++ float = Fortran 'real*4' 
    NBYTES_FILE_FMT = 4  # a Fortran (real*4) uses 4 bytes.

    # convert sources from center to top edge of fault patch ("sum_layered" needs that)
    sind = np.sin( dip * np.pi / 180.0 )
    cosd = np.cos( dip * np.pi / 180.0 )
    sins = np.sin( strike * np.pi / 180.0 )
    coss = np.cos( strike * np.pi / 180.0 )

    # displacement in local coordinates (phi, delta)
    dZ = (width/2.0) * sind
    dD = (width/2.0) * cosd

    # rotation to global coordinates 
    xs = xs - dD * coss
    ys = ys + dD * sins
    zs = zs - dZ

    # Define filenames:
    file_rec = prefix + '.rec'
    file_pat = prefix + '.pat'
    file_dux = prefix + '_ux.dis'
    file_duy = prefix + '_uy.dis'
    file_duz = prefix + '_uz.dis'

    # Clean the file if they exist
    cmd = 'rm -f {} {} {} {} {}'.format(file_rec, file_pat, file_dux, file_duy, file_duz)
    os.system(cmd) 
    
    # write receiver location file (observation points)
    temp = [xr, yr]
    file = open(file_rec, 'wb') 
     
    for k in range(0, nrec):
       for i in range(0, len(temp)):
          file.write( struct.pack( BIN_FILE_FMT, temp[i][k] ) )       
    file.close() 
  
    # write point sources information
    temp = [xs, ys, zs, strike, dip, rake, width, length, slip]
    file = open(file_pat, 'wb');
    for k in range(0, Np):
       for i in range(0, len(temp)):
          file.write( struct.pack( BIN_FILE_FMT, temp[i][k] ) )
    file.close()
  
    # call sum_layered
    cmd = '{}/sum_layered {} {} {} {} {} {}'.format(BIN_EDKS, edks, prefix, nrec, Np, npw, npy)
    if verbose:
        print(cmd)
    os.system(cmd)
     
    # read sum_layered output Greens function
    # ux
    ux = np.fromfile(file_dux, 'f').reshape((nrec, Np), order='FORTRAN')

    #file = open(file_dux, 'rb')
    #ux = np.zeros((nrec, Np))
    #for j in range(0, Np):
    #   for i in range(0, nrec):
    #      byteVal = file.read(NBYTES_FILE_FMT)
    #      if byteVal != '':
    #         ux[i][j] = struct.unpack('f', byteVal)[0]
    #      else:
    #         raise ValueError(' Premature EOF in %s, something nasty happened'%(file_dux))
    #file.close()
 
    # uy
    uy = np.fromfile(file_duy, 'f').reshape((nrec, Np), order='FORTRAN')

    #file = open(file_duy, 'rb')
    #uy = np.zeros((nrec, Np))
    #for j in range(0, Np):
    #   for i in range(0, nrec):
    #      byteVal = file.read(NBYTES_FILE_FMT)
    #      if byteVal != '':
    #         uy[i][j] = struct.unpack('f', byteVal)[0]
    #      else:
    #         raise ValueError('Premature EOF in %s, something nasty happened'%(file_duy))
    #file.close()
 
    # uz
    uz = np.fromfile(file_duz, 'f').reshape((nrec, Np), order='FORTRAN')

    #file = open(file_duz, 'rb')
    #uz = np.zeros((nrec, Np))
    #for j in range(0, Np):
    #   for i in range(0, nrec):
    #      byteVal = file.read(NBYTES_FILE_FMT)
    #      if byteVal != '':
    #         uz[i][j] = struct.unpack('f', byteVal)[0]
    #      else:
    #         raise ValueError('Premature EOF in %s, something nasty happened'%(file_duz))
    #file.close()
 
    # remove IO files.
    if cleanUp:
        cmd = 'rm -f {} {} {} {} {}'.format(file_rec, file_pat, file_dux, file_duy, file_duz)
        os.system(cmd)  
 
    # return the GF matrices
    return [ux, uy, uz]

def sum_layered_sub(IDs, xs, ys, zs, strike, dip, rake, slip, A, \
                       xr, yr, edks,\
                       prefix, \
                       BIN_EDKS = 'EDKS_BIN', tensile=False,
                       cleanUp=True):
    '''
    --- INPUT ---
    --- SOURCE INFO
    --- 1D NUMPY arrays, length = number of fault patches
    IDs      list of strings,IDs of  point sources (see below for a detailed explanation)
    xs       m, east coord to center of fault patch
    ys       m, north coord to center of fault patch
    zs       m,depth coord to center of fault patch (+ down) 
    strike   deg, clockwise from north 
    dip      deg, 90 is vertical 
    rake     deg, 0 left lateral strike slip, 90 up-dip slip 
    slip     m, slip in the rake direction
    A        m2, surface area of fault patch 
    --- RECEIVER INFO
    1D arrays, length = number of receivers
    xr       m, east coordinate of receivers 
    yr       m, north coordinate of receivers 
    --- ELASTIC STRUCTURE INFO
    edks     string, full name of edks file, e.g., halfspace.edks
    --- FILE NAMING 
    prefix   string, prefix for the IO files generated by sum_layered
    BIN_EDKS Name of environement variable pointing to EDKS binaries
    --- OPTIONS
    tensile  boolean specifying whether to compute tensile displacements
    --- OUTPUT ---
    --- 2D arrays, (receivers, fault patches)
    ux     m, east displacement
    uy     m, west displacement
    uz     m, up displacement (+ up)

    Explanation of IDs:
      the source ID (IDs) is used to be able to represent a finite source by a set
      of "well defined" point sources (ex: point sources modeling a triangular or 
      rectangular finite source). 
      - If you want to use this code only to calculate independent point sources, just
      give a different ID to all the sources.
      - If you want to use this code to approximate several finite dislocations, you need
      to define and assign a different ID to each finite source. The sources with 
      equal IDs will be added to compute the surface displacements of the finite 
      dislocation. Then the code will return only the displacements corresponding to 
      the one of the finite dislocation, in the same order as the specified IDs.
      IMPORTANT: The equal IDs must be contiguous in order to ensure that the order
      in which the output is computed is the same. 
      Ex: -  a good list of source IDs is 
             [id1, id1,..., id1, id2, id2,..., id2, idj,..., idj, ..., idN,..,idN] 
          - a BAD list of source (you should not do this) would be:
             [id1,id2, id3, ... , idN, id1, id2, .. idN, id1, id3, id8] 

    NOTE ON THE NUMBER OF CORES USED BY sum_layered_sub:
       in order to set the number of cores used (by default openMP uses all the 
       available cores) you must set the environment variable OMP_NUM_THREADS
       with the number of cores (threads) that you want. 

    '''

    # Get where the binaries are
    BIN_EDKS = os.environ[BIN_EDKS]

    # Define a few things 
    nrec = len(xr)          # number of receivers
    Np = len(set(IDs))      # total number of finite faults 
    ntsp = len(xs)          # total number of point sources (sub sources)

    # compute mapping between the string IDs and non decreasing positive integer number
    setOfAlreadyStoredIDs = set() # this is just for testing the right order of the IDs.
    sortedListOfFiniteFaultIDs = []
    ident = []
    i = 1
    IDprev = IDs[0]
    ident.append(i)
    sortedListOfFiniteFaultIDs.append(IDprev)
    setOfAlreadyStoredIDs.add(IDprev)
    NumSubSources = {}
    NumSubSources[IDprev] = 1
    for k in range(1, ntsp):
        if IDs[k] == IDprev:
            ident.append(i)
            NumSubSources[IDprev] += 1
        else: # the current ID is a new one.
            if IDs[k] in setOfAlreadyStoredIDs: # this is an error
                raise ValueError('Source IDs are not in the right order...')
            else:
                IDprev = IDs[k]
                i += 1
                ident.append(i)
                sortedListOfFiniteFaultIDs.append(IDs[k])
                setOfAlreadyStoredIDs.add(IDs[k])
                NumSubSources[IDs[k]] = 1
    nspp = np.max(list(NumSubSources.values())) # maximum number of subsources

    # Some format
    BIN_FILE_FMT_real4 = 'f' # python float = C/C++ float = Fortran 'real*4' 
    BIN_FILE_FMT_int4 = 'i' # python int, fortran 'integer*4''
    NBYTES_FILE_FMT = 4  # a Fortran (real*4) uses 4 bytes.

    # Define filenames:
    file_rec = prefix + '.rec'
    file_pat = prefix + '.pat'
    file_dux = prefix + '_ux.dis'
    file_duy = prefix + '_uy.dis'
    file_duz = prefix + '_uz.dis'

    # Clean files if they exist
    cmd = 'rm -f {} {} {} {} {}'.format(file_rec, file_pat, file_dux, file_duy, file_duz)
    os.system(cmd)
   
    # write receiver location file (observation points)
    temp = [xr, yr]
    file = open(file_rec, 'wb') 
    for k in range(0, nrec):
        for i in range(0, len(temp)):
            file.write( struct.pack( BIN_FILE_FMT_real4, temp[i][k] ) )       
    file.close() 
 
    # write point sources information
    temp = [xs, ys, zs, strike, dip, rake, A, slip, ident]
    file = open(file_pat, 'wb');
    for k in range(0, ntsp):
        for i in range(0, len(temp)-1):
            file.write( struct.pack( BIN_FILE_FMT_real4, temp[i][k] ) )
        file.write( struct.pack( BIN_FILE_FMT_int4, temp[-1][k] ) )
    file.close()
 
    # call sum_layered
    if tensile:
        cmd = '{}/sum_layered_tensile {} {} {} {} {} {} '.format(BIN_EDKS, 
                                    edks, prefix, nrec, Np, ntsp, nspp)
    else:
        cmd = '{}/sum_layered_sub {} {} {} {} {} {} '.format(BIN_EDKS,
                                    edks, prefix, nrec, Np, ntsp, nspp)
    print(cmd)
    os.system(cmd)
    
    # read sum_layered output Greens function
    
    # ux
    file = open(file_dux, 'rb')
    ux = np.zeros((nrec, Np))
    for j in range(0, Np):
        for i in range(0, nrec):
            byteVal = file.read(NBYTES_FILE_FMT)
            if byteVal != '':
                ux[i][j] = struct.unpack('f', byteVal)[0]
            else:
                raise ValueError(' Premature EOF in %s, something nasty happened'%(file_dux))
    file.close()

    # uy
    file = open(file_duy, 'rb')
    uy = np.zeros((nrec, Np))
    for j in range(0, Np):
        for i in range(0, nrec):
            byteVal = file.read(NBYTES_FILE_FMT)
            if byteVal != '':
                uy[i][j] = struct.unpack('f', byteVal)[0]
            else:
                raise ValueError('Premature EOF in %s, something nasty happened'%(file_duy))
    file.close()

    # uz
    file = open(file_duz, 'rb')
    uz = np.zeros((nrec, Np))
    for j in range(0, Np):
        for i in range(0, nrec):
            byteVal = file.read(NBYTES_FILE_FMT)
            if byteVal != '':
                uz[i][j] = struct.unpack('f', byteVal)[0]
            else:
                raise ValueError('Premature EOF in %s, something nasty happened'%(file_duz))
    file.close()

    # remove IO files.
    if cleanUp:
        cmd = 'rm -f {} {} {} {} {}'.format(file_rec, file_pat, file_dux, file_duy, file_duz)
        os.system(cmd)  

    # return the computed displacements for each sources
    return [ux, uy, uz]

