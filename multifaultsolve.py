'''
A Class to assemble several faults into a single inverse problem. All the faults must have been intialized and constructed using the same data set.
This class allows then to:
    1. Spit the G, m, Cm, and Cd elements for a third party solver (such as Altar, for instance)
    2. Proposes a simple solution based on a least-square optimization.

Written by R. Jolivet, April 2013.

'''

import copy
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt

class multifaultsolve(object):

    def __init__(self, name, faults):
        '''
        Class initialization routine.

        Args:
            * name          : Name of the project.
            * faults        : List of faults from verticalfault.
        '''

        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initializing solver object")

        # Ready to compute?
        self.ready = False

        # Store things into self
        self.name = name
        self.faults = faults

        # check the utm zone
        self.utmzone = faults[0].utmzone
        for fault in faults:
            if fault.utmzone is not self.utmzone:
                print("UTM zones are not equivalent, this is a problem")
                self.ready = False
                return
        self.ptum = faults[0].putm

        # check that G and d have been assembled prior to initialization
        for fault in faults:
            if fault.Gassembled is None:
                self.ready = False
                print("G has not been assembled in fault structure {}".format(fault.name))
                return
            if fault.dassembled is None:
                self.ready = False
                print("d has not been assembled in fault structure {}".format(fault.name))

        # Check that the sizes of the data vectors are consistent
        self.d = faults[0].dassembled
        for fault in faults:
            if (fault.dassembled != self.d).all():
                print("Data vectors are not consistent, please re-consider your data in fault structure {}".format(fault.name))
                return

        # Check that the data covariance matrix is the same
        self.Cd = faults[0].Cd
        for fault in faults:
            if (fault.Cd != self.Cd).all():
                print("Data Covariance Matrix are not consistent, please re-consider your data in fault structure {}".format(fault.name))
                return

        # Initialize things
        self.fault_indexes = None

        # Store an array of the patch areas
        patchAreas = []
        for fault in faults:
            if fault.patchType == 'triangletent':
                fault.computeTentArea()
                for tentIndex in range(len(fault.tent)):
                    patchAreas.append(fault.area_tent[tentIndex])
            else:
                fault.computeArea()
                for patchIndex in range(len(fault.patch)):
                    patchAreas.append(fault.area[patchIndex])
        self.patchAreas = np.array(patchAreas)

        # All done
        return

    def assembleGFs(self):
        '''
        Assembles the Green's functions matrix G for the concerned faults.
        '''

        # Get the faults
        faults = self.faults

        # Get the size of the total G matrix
        Nd = self.d.size
        Np = 0
        st = []
        se = []
        if self.fault_indexes is None:
            self.fault_indexes = {}
        for fault in faults:
            st.append(Np)
            Np += fault.Gassembled.shape[1]
            se.append(Np)
            self.fault_indexes[fault.name] = [st[-1], se[-1]]

        # Allocate the big G matrix
        self.G = np.zeros((Nd, Np))

        # Store the guys
        for fault in faults:
            # get the good indexes
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            # Store the G matrix
            self.G[:,st:se] = fault.Gassembled
            # Keep track of indexing
            self.affectIndexParameters(fault)

        # self ready
        self.ready = True

        # Set the number of parameters
        self.Nd = Nd
        self.Np = Np

        # Describe which parameters are what
        self.describeParams(faults)

        # All done
        return


    def sensitivity(self):
        '''
        Calculates sqrt(diag(G^t Cd^-1 G))
        '''
        # Import things
        import scipy.linalg as scilin
        # Invert Cd
        iCd = scilin.inv(self.Cd)
        s = np.diag(np.dot(self.G.T,np.dot(iCd,self.G)))

        # All done
        return s


    def describeParams(self, faults):
        '''
        Prints to screen  which parameters are what...
        '''

        # Prepare the table
        print('          Fault Name          ||   Strike Slip   ||   Dip Slip   ||   Tensile   ||   Orbits   ')

        # initialize the counters
        ns = 0
        ne = 0

        # Loop over the faults
        for fault in faults:

            # Where does this fault starts
            nfs = copy.deepcopy(ns)

            # Initialize the values
            ss = 'None'
            ds = 'None'
            ts = 'None'

            # Conditions on slip
            if 's' in fault.slipdir:
                ne += len(fault.patch)
                ss = '{} - {}'.format(ns,ne)
                ns += len(fault.patch)
            if 'd' in fault.slipdir:
                ne += len(fault.patch)
                ds = '{} - {}'.format(ns, ne)
                ns += len(fault.patch)
            if 't' in fault.slipdir:
                ne += len(fault.patch)
                ts = '{} - {}'.format(ns, ne)
                ns += len(fault.patch)

            # conditions on orbits (the rest is orbits)
            np = ne - nfs
            no = fault.Gassembled.shape[1] - np
            if no>0:
                ne += no
                op = '{} - {}'.format(ns, ne)
                ns += no
            else:
                op = 'None'

            # print things
            print('{:30s}||{:17s}||{:14s}||{:13s}||{:12s}'.format(fault.name, ss, ds, ts, op))

        # all done
        return

    def assembleCm(self):
        '''
        Assembles the Model Covariance Matrix for the concerned faults.
        '''

        # Get the faults
        faults = self.faults

        # Get the size of Cm
        Np = 0
        st = []
        se = []
        if self.fault_indexes is None:
            self.fault_indexes = {}
        for fault in faults:
            st.append(Np)
            Np += fault.Gassembled.shape[1]
            se.append(Np)
            self.fault_indexes[fault.name] = [st[-1], se[-1]]

        # Allocate Cm
        self.Cm = np.zeros((Np, Np))

        # Store the guys
        for fault in faults:
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            self.Cm[st:se, st:se] = fault.Cm

        # Store the number of parameters
        self.Np = Np

        # All done
        return

    def affectIndexParameters(self, fault):
        '''
        Build the index parameter for a fault.
        '''

        # Get indexes
        st = self.fault_indexes[fault.name][0]
        se = self.fault_indexes[fault.name][1]

        # Save the fault indexes
        fault.index_parameter = np.zeros((fault.slip.shape))
        fault.index_parameter[:,:] = 9999999
        if 's' in fault.slipdir:
            fault.index_parameter[:,0] = range(st, st+fault.slip.shape[0])
            st += fault.slip.shape[0]
        if 'd' in fault.slipdir:
            fault.index_parameter[:,1] = range(st, st+fault.slip.shape[0])
            st += fault.slip.shape[0]
        if 'u' in fault.slipdir:
            fault.index_parameter[:,2] = range(st, st+fault.slip.shape[0])

        # All done
        return

    def distributem(self, verbose=False):
        '''
        After computing the m_post model, this routine distributes the m parameters to the faults.
        '''

        # Get the faults
        faults = self.faults

        # Loop over the faults
        for fault in faults:

            if verbose:
                print("Distribute the slip values to fault {}".format(fault.name))

            # Store the mpost
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            fault.mpost = self.mpost[st:se]

            # Affect the indexes
            self.affectIndexParameters(fault)

            # put the slip values in slip
            st = 0
            if 's' in fault.slipdir:
                se = st + fault.slip.shape[0]
                fault.slip[:,0] = fault.mpost[st:se]
                st += fault.slip.shape[0]
            if 'd' in fault.slipdir:
                se = st + fault.slip.shape[0]
                fault.slip[:,1] = fault.mpost[st:se]
                st += fault.slip.shape[0]
            if 'u' in fault.slipdir:
                se = st + fault.slip.shape[0]
                fault.slip[:,2] = fault.mpost[st:se]
                st += fault.slip.shape[0]

            # Get the polynomial/orbital/helmert values if they exist
            fault.polysol = {}
            fault.polysolindex = {}
            for dset in fault.datanames:
                if dset in fault.poly.keys():
                    if (fault.poly[dset].__class__ is not str):
                        if (fault.poly[dset] > 0):
                            se = st + fault.poly[dset]
                            fault.polysol[dset] = fault.mpost[st:se]
                            fault.polysolindex[dset] = range(st,se)
                            st += fault.poly[dset]
                    elif (fault.poly[dset].__class__ is str):
                        if fault.poly[dset] is 'full':
                            nh = fault.helmert[dset]
                            se = st + nh
                            fault.polysol[dset] = fault.mpost[st:se]
                            fault.polysolindex[dset] = range(st,se)
                            st += nh
                        if fault.poly[dset] in ('strain', 'strainnorotation'):
                            nh = fault.strain[dset]
                            se = st + nh
                            fault.polysol[dset] = fault.mpost[st:se]
                            fault.polysolindex[dset] = range(st,se)
                            st += nh
                    else:
                        fault.polysol[dset] = None

        # All done
        return

    def SetSolutionFromExternal(self, soln, dtype=np.float32):
        '''
        Takes a vector where the solution of the problem is and affects it to mpost.
        By default, the input has to be an array.
        It can be a string with a file name.
        '''

        # If it is a file, read it (binary)
        if type(soln) is str:
            soln = np.fromfile(soln, dtype=dtype)
        
        # Put it where it needs to be
        self.mpost = soln

        # All done
        return

    def NonNegativeBruteSoln(self):
        '''
        Solves the least square problem argmin_x || Ax - b ||_2 for x>=0.
        No Covariance can be used here, maybe in the future.
        '''

        # Import what is needed
        import scipy.optimize as sciopt

        # Get things
        d = self.d
        G = self.G

        # Solution
        mpost, rnorm = sciopt.nnls(G, -1*d)

        # Store results
        self.mpost = mpost
        self.rnorm = rnorm

        # All done
        return

    def SimpleLeastSquareSoln(self):
        '''
        Solves the simple least square problem:

            m_post = (G.T * G)-1 * G.T * d

        '''

        # Import things
        import scipy.linalg as scilin

        # Print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Computing the Simple Least Squares")

        # Get the matrixes and vectors
        G = self.G
        d = self.d

        # Copmute
        mpost = np.dot( np.dot( scilin.inv(np.dot( G.T, G )), G.T ), d)

        # Store mpost
        self.mpost = mpost

        # All done
        return

    def UnregularizedLeastSquareSoln(self, mprior=None):
        '''
        Solves the unregularized generalized least-square problem using the following formula (Tarantolla, 2005, "Inverse Problem Theory", SIAM):

            m_post = m_prior + (G.T * Cd-1 * G)-1 * G.T * Cd-1 * (d - G*m_prior)

        Args:

            * mprior        : A Priori model. If None, then mprior = np.zeros((Nm,)).

        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # Import things
        import scipy.linalg as scilin

        # Print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Computing the Unregularized Least Square Solution")

        # Get the matrixes and vectors
        G = self.G
        d = self.d
        Cd = self.Cd

        # Get the number of model parameters
        Nm = G.shape[1]

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        print ("Computing the inverse of the data covariance")
        iCd = scilin.inv(Cd)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Compute mpost
        print ("Computing m_post")
        One = scilin.inv(np.dot(  np.dot(G.T, iCd), G ) )
        Res = d - np.dot( G, mprior )
        Two = np.dot( np.dot( G.T, iCd ), Res )
        mpost = mprior + np.dot( One, Two )

        # Store m_post
        self.mpost = mpost

        # All done
        return

    def GeneralizedLeastSquareSoln(self, mprior=None, rcond=None):
        '''
        Solves the generalized least-square problem using the following formula (Tarantolla, 2005, "Inverse Problem Theory", SIAM):

            m_post = m_prior + (G.T * Cd-1 * G + Cm-1)-1 * G.T * Cd-1 * (d - G*m_prior)

        Args:
            * mprior        : A Priori model. If None, then mprior = np.zeros((Nm,)).
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # Import things
        import scipy.linalg as scilin

        # Print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Computing the Generalized Inverse")

        # Get the matrixes and vectors
        G = self.G
        d = self.d
        Cd = self.Cd
        Cm = self.Cm

        # Get the number of model parameters
        Nm = Cm.shape[0]

        # Check If Cm is symmetric and positive definite
        if (Cm.transpose() != Cm).all():
            print("Cm is not symmetric, Return...")
            return

        # Get the inverse of Cm
        print ("Computing the inverse of the model covariance")
        iCm = scilin.inv(Cm)

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        print ("Computing the inverse of the data covariance")
        if rcond is None:
            iCd = scilin.inv(Cd)
        else:
            iCd = np.linalg.pinv(Cd, rcond=rcond)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Compute mpost
        print ("Computing m_post")
        One = scilin.inv(np.dot(  np.dot(G.T, iCd), G ) + iCm )
        Res = d - np.dot( G, mprior )
        Two = np.dot( np.dot( G.T, iCd ), Res )
        mpost = mprior + np.dot( One, Two )

        # Store m_post
        self.mpost = mpost

        # All done
        return

    def ConstrainedLeastSquareSoln(self, mprior=None, Mw_thresh=10., bounds=None,
                                   method='SLSQP', rcond=None):
        """
        Solves the least squares problem:

            min (d - G*m).T * Cd-1 * (d - G*m) + m.T * Cm-1 * m
            subject to:
                Mw <= Mw_bound

        Args:
            mprior          : a priori model; if None, mprior = np.zeros((Nm,))
            Mw_thresh       : upper bound on moment magnitude
            bounds          : list of tuple bounds for every parameter
            method          : solver for constrained minimization: SLSQP or COBYLA
                              SLSQP is recommended
        """
        assert self.ready, 'You need to assemble the GFs'

        # Import things
        import scipy.linalg as scilin
        from scipy.optimize import minimize

        # Check the provided method is valid
        assert method == 'SLSQP' or method == 'COBYLA', 'unsupported minimizing method'

        # Print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Computing the Constrained least squares solution")

        # Get the matrixes and vectors
        G = self.G
        d = self.d
        Cd = self.Cd
        Cm = self.Cm

        # Get the number of model parameters
        Nm = Cm.shape[0]
        Npatch = len(self.patchAreas)

        # Check If Cm is symmetric and positive definite
        if (Cm.transpose() != Cm).all():
            print("Cm is not symmetric, Return...")
            return

        # Get the inverse of Cm
        print ("Computing the inverse of the model covariance")
        iCm = scilin.inv(Cm)

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        print ("Computing the inverse of the data covariance")
        if rcond is None:
            iCd = scilin.inv(Cd)
        else:
            iCd = np.linalg.pinv(Cd, rcond=rcond)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Define the cost function
        def costFunction(m, G, d, iCd, iCm, mprior):
            """
            Compute data + prior misfits.
            """
            dataMisfit = d - np.dot(G,m)
            dataLikely = np.dot(dataMisfit, np.dot(iCd, dataMisfit))
            priorMisfit = m - mprior
            priorLikely = np.dot(priorMisfit, np.dot(iCm, priorMisfit))
            return 0.5 * dataLikely + 0.5 * priorLikely

        # Define the moment magnitude inequality constraint function
        def computeMwDiff(m, Mw_thresh, patchAreas, Npatch):
            """
            Ahhhhh hard coded shear modulus.
            """
            shearModulus = 22.5e9
            moment = shearModulus * np.abs(np.dot(patchAreas, m[:Npatch]))
            if moment>0.:
                Mw = 2.0 / 3.0 * np.log10(moment) - 6.0
            else:
                Mw = -6.0
            return np.array([Mw_thresh - Mw])

        # Define the constraints dictionary
        if Mw_thresh is not None:
            constraints = {'type': 'ineq',
                           'fun': computeMwDiff,
                           'args': (Mw_thresh, self.patchAreas, Npatch)}
        else:
            constraints = None

        # Call solver
        print("Performing constrained minimzation")
        res = minimize(costFunction, mprior, args=(G,d,iCd,iCm,mprior),
                       constraints=constraints, method=method, bounds=bounds,
                       options={'disp': True})

        # Store result
        self.mpost = res.x

        # All done
        return

    def writeMpost2File(self, outfile):
        '''
        Writes the solution to a file.
        '''

        # Check
        assert (hasattr(self, 'mpost')), 'Compute mpost first, you idiot...'

        # Open file
        fout = open(outfile, 'w')

        # Write header
        fout.write('# Param Number | Mean (mm/yr) | Std (mm/yr) \n')

        # Loop over mpost
        for i in range(self.mpost.shape[0]):
            fout.write('{:3d} {} 0.0000 \n'.format(i, self.mpost[i]))

        # Close file
        fout.close()

        # All done
        return

    def writeMpost2BinaryFile(self, outfile):
        '''
        Writes the solution to a binary file.
        '''

        self.mpost.tofile(outfile)

        # all done
        return

    def writeGFs2BinaryFile(self, outfile='GF.dat', dtype='f'):
        '''
        Writes the assembled GFs to the file outfile.
        Args:
            * outfile       : Name of the output file.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = np.float64

        # Convert the data
        G = self.G.astype(dtype)

        # Write to file
        G.tofile(outfile)

        # Keep track of the file
        self.Gfile = outfile

        # Print stuff
        print("Writing Green's functions to file {}".format(outfile))
        print("Green's functions matrix size: {} ; {}".format(G.shape[0], G.shape[1]))

        # All done
        return

    def writeData2BinaryFile(self, outfile='d.dat', dtype='f'):
        '''
        Writes the assembled data vector to an output file.
        Args:
            * outfile       : Name of the output file.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = np.float64

        # Convert the data
        d = self.d.astype(dtype)

        # Write to file
        d.tofile(outfile)

        # Keep track of the file
        self.dfile = outfile

        # Print stuff
        print("Data vector size: {}".format(d.shape[0]))

        # All done
        return

    def writeCd2BinaryFile(self, outfile='Cd.dat', dtype='f', scale=1.):
        '''
        Writes the assembled Data Covariance matrix to a binary file.
        Args:
            * outfile       : Name of the output file.
            * scale         : Multiply the data covariance.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = np.float64

        # Convert the data
        Cd = self.Cd.astype(dtype) * scale

        # Write to file
        Cd.tofile(outfile)

        # keep track of the file
        self.Cdfile = outfile

        # print stuff
        print("Data Covariance Size: {} ; {}".format(Cd.shape[0], Cd.shape[1]))

        # All done
        return

    def RunAltar(self, tasks=2, chains=1024, steps=100, support=(-10, 10)):
        '''
        Runs Altar on the d = Gm problem with a Cd covariance matrix.
        Args:
            * tasks         : Number of mpi tasks.
            * chains        : Number of chains.
            * steps         : Number of metropolis steps.
            * support       : Upper and Lower bounds of the parameter exploration.
        '''

        # Create the cfg and py file
        self.writeAltarCfgFile(prefix=self.name, tasks=tasks, chains=chains, steps=steps, support=support)

        # Create the line
        import subprocess as subp
        com = ['python3.3', self.name+'.py']
        subp.call(com)

        # return
        return

    def writeAltarCfgFile(self, prefix='linearfullcov', tasks=2, chains=1024, steps=100, support=(-10, 10), minimumratio=0.000001):
        '''
        Writes a cfg and a py file to be used by altar.
        Args:
            * outfile       : Prefix of problem
            * tasks         : Number of mpi tasks.
            * chains        : Number of chains.
            * steps         : Number of metropolis steps.
            * support       : Upper and Lower bounds of the parameter exploration.
            * minimumratio  : Minimum Eignevalue to cut in the metropolis covariance matrix.
        '''

        # Open the file and print the opening credits
        fout = open(prefix+'.cfg', 'w')
        fout.write('; \n')
        fout.write('; R Jolivet \n')
        fout.write('; california institute of technology \n')
        fout.write('; (c) 2010-2013 all rights reserved \n')
        fout.write('; \n')
        fout.write(' \n')

        fout.write('; exercising the sequential controller \n')
        fout.write('[ {} ] \n'.format(prefix))
        fout.write('shell = mpi \n')
        fout.write('model = altar.models.lineargm.linearfullcov \n')
        fout.write('controller = catmip.annealer \n')
        fout.write('rng.algorithm = mt19937 \n')
        fout.write(' \n')

        fout.write('; model configuration \n')
        fout.write('[ altar.models.lineargm.linearfullcov #{}.model ] \n'.format(prefix))
        fout.write('dof = {} \n'.format(self.G.shape[1]))
        fout.write('nd = {} \n'.format(self.G.shape[0]))
        fout.write('support = ({}, {}) \n'.format(support[0], support[1]))
        fout.write('Gfile={} \n'.format(self.Gfile))
        fout.write('dfile={} \n'.format(self.dfile))
        fout.write('covfile={} \n'.format(self.Cdfile))
        fout.write(' \n')

        fout.write('; mpi application shell \n')
        fout.write('[ mpi.shells.mpirun #{}.shell ] \n'.format(prefix))
        fout.write('tasks = {} \n'.format(tasks))
        fout.write(' \n')

        fout.write('; annealing schedule\n')
        fout.write('[ catmip.controllers.annealer #{}.controller ]\n'.format(prefix))
        fout.write('chains = {} \n'.format(chains))
        fout.write('tolerance = .005 \n')
        fout.write('scheduler = cov \n')
        fout.write('sampler = metropolis \n')
        fout.write(' \n')

        fout.write('; metropolis sampler\n')
        fout.write('[ catmip.samplers.metropolis #{}.controller.sampler ]\n'.format(prefix))
        fout.write('steps = {} \n'.format(steps))
        fout.write('scaling = .1 \n')
        fout.write(' \n')

        fout.write('; COV schedule\n')
        fout.write('[ catmip.schedulers.cov #{}.controller.scheduler ] \n'.format(prefix))
        fout.write('tolerance = .01 \n')
        fout.write('MinimumRatio = {} \n'.format(minimumratio))
        fout.write(' \n')

        fout.write('; end of file')

        # Close file
        fout.close()

        # Open the py file
        fout = open(prefix+'.py', 'w')

        # Write things
        fout.write('# -*- coding: utf-8 -*- \n')
        fout.write('# \n')
        fout.write('# R Jolivet \n')
        fout.write('# california institute of technology \n')
        fout.write('# (c) 2010-2013 all rights reserved\n')
        fout.write('# \n')

        fout.write('""" \n')
        fout.write("Exercises the linear Gm model with the full data covariance matrix \n")
        fout.write('""" \n')

        fout.write('def test(): \n')
        fout.write('    # externals \n')
        fout.write('    import catmip \n')
        fout.write('    print(catmip.__file__) \n')
        fout.write('    # instantiate the default application\n')
        fout.write("    app = catmip.application(name='{}')\n".format(prefix))
        fout.write('    # run it\n')
        fout.write('    app.run()\n')
        fout.write('    # and return the app object \n')
        fout.write('    return app\n')

        fout.write('# main \n')
        fout.write('if __name__ == "__main__":\n')
        fout.write('    import journal\n')

        fout.write('    # altar\n')
        fout.write("    journal.debug('altar.beta').active = True\n")
        fout.write("    journal.debug('altar.controller').active = False\n")
        fout.write("    journal.debug('altar.initialization').active = False\n")

        fout.write('    # catmip\n')
        fout.write("    journal.debug('catmip.annealing').active = True\n")

        fout.write('    # do...\n')
        fout.write('    test()\n')

        fout.write('# end of file ')

        # Close file
        fout.close()

        # All done
        return


    def writePatchAreasFile(self, outfile='PatchAreas.dat', dtype='d',
                            npadStart=None, npadEnd=None):
        """
        Write a binary file for the patch areas to be read into altar.

        Args:
            * outfile               : output file name
            * dtype                 : output data type
            * npadStart             : number of starting zeros to pad output
            * npadEnd               : number of ending zeros to pad output
        """
        # Construct output vector of patch areas
        vout = self.patchAreas.astype(dtype)
        if npadStart is not None:
            vpad = np.zeros((npadStart,), dtype=dtype)
            vout = np.hstack((vpad, vout))
        if npadEnd is not None:
            vpad = np.zeros((npadEnd,), dtype=dtype)
            vout = np.hstack((vout, vpad))

        # Write to file and return
        vout.tofile(outfile)
        return

