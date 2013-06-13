'''
A Class to assemble several faults into a single inverse problem. All the faults must have been intialized and constructed using the same data set.
This class allows then to:
    1. Spit the G, m, Cm, and Cd elements for a third party solver (such as Altar, for instance)
    2. Proposes a simple solution based on a least-square optimization.

Written by R. Jolivet, April 2013.

'''

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
                print("G has not been assembled in fault structure %s"%fault.name)
                return
            if fault.dassembled is None:
                self.ready = False
                print("d has not been assembled in fault structure %s"%fault.name)

        # Check that the sizes of the data vectors are consistent
        self.d = faults[0].dassembled
        for fault in faults:
            if (fault.dassembled != self.d).all():
                print("Data vectors are not consistent, please re-consider your data in fault structure %s"%fault.name)
                return

        # Check that the data covariance matrix is the same 
        self.Cd = faults[0].Cd
        for fault in faults:
            if (fault.Cd != self.Cd).all():
                print("Data Covariance Matrix are not consistent, please re-consider your data in fault structure %s"%fault.name)
                return

        # Initialize things
        self.fault_indexes = None

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
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            self.G[:,st:se] = fault.Gassembled

        # self ready
        self.ready = True

        # All done
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
            print fault.name
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            self.Cm[st:se, st:se] = fault.Cm

        # All done
        return

    def distributem(self):
        '''
        After computing the m_post model, this routine distributes the m parameters to the faults.
        '''

        # Get the faults
        faults = self.faults

        # Loop over the faults
        for fault in faults:
            
            print ("Distribute the slip values to fault %s"%fault.name)

            # Affect the model parameters
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            fault.mpost = self.mpost[st:se]
            
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

            # Get the polynomial values if they exist
            fault.polysol = {}
            for dset in fault.poly.keys():
                if fault.poly[dset] > 0:
                    se = st + fault.poly[dset]
                    fault.polysol[dset] = fault.mpost[st:se]
                    st += fault.poly[dset]
                else:
                    fault.polysol[dset] = None

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

    def GeneralizedLeastSquareSoln(self, mprior=None):
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
        iCd = scilin.inv(Cd)

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
        print("Green's functions matrix size: %i ; %i"%(G.shape))

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
        print("Data vector size: %i"%(d.shape[0]))

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
        print("Data Covariance Size: %i ; %i"%(Cd.shape))

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
        fout.write('[ %s ] \n'%(prefix))
        fout.write('shell = mpi \n')
        fout.write('model = altar.models.lineargm.linearfullcov \n')
        fout.write('controller = catmip.annealer \n')
        fout.write('rng.algorithm = mt19937 \n')
        fout.write(' \n')

        fout.write('; model configuration \n')
        fout.write('[ altar.models.lineargm.linearfullcov #%s.model ] \n'%(prefix))
        fout.write('dof = %i \n'%(self.G.shape[1]))
        fout.write('nd = %i \n'%(self.G.shape[0]))
        fout.write('support = (%f, %f) \n'%(support[0], support[1]))
        fout.write('Gfile=%s \n'%self.Gfile)
        fout.write('dfile=%s \n'%(self.dfile))
        fout.write('covfile=%s \n'%(self.Cdfile))
        fout.write(' \n')

        fout.write('; mpi application shell \n')
        fout.write('[ mpi.shells.mpirun #%s.shell ] \n'%(prefix))
        fout.write('tasks = %i \n'%(tasks))
        fout.write(' \n')

        fout.write('; annealing schedule\n')
        fout.write('[ catmip.controllers.annealer #%s.controller ]\n'%(prefix))
        fout.write('chains = %i \n'%chains)
        fout.write('tolerance = .005 \n')
        fout.write('scheduler = cov \n')
        fout.write('sampler = metropolis \n')
        fout.write(' \n')

        fout.write('; metropolis sampler\n')
        fout.write('[ catmip.samplers.metropolis #%s.sampler ]\n'%(prefix))
        fout.write('steps = %i \n'%steps)
        fout.write('scaling = .1 \n')
        fout.write('MinimumRatio = %e \n'%minimumratio)
        fout.write(' \n')

        fout.write('; COV schedule\n')
        fout.write('[ catmip.schedulers.cov #%s.controller.scheduler ] \n'%(prefix))
        fout.write('tolerance = .01 \n')
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
        fout.write("    app = catmip.application(name='%s')\n"%prefix)
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
    

