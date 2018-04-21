Installation
===============================

Repositories

Installation requires you to get access to the bzr repositories.
There is 2 repositories for CSI and associated libraries: one at Caltech, one at ENS Paris.
To get access to the repository at CalTech, contact Mark and Scott Dungan.\\
To get access to the repository at ENS, contact Romain.\\
These repos should be identical.\\

__**Current Install instructions**__:

1. Create a directory where you will drop all the codes and go in this directory. In the following, I will call this directory "modeling". WARNING: Don't call this directory csi!!!!!\\
Prepend the directory "modeling" to your PYTHONPATH environment variable. If you are operating bash shell, you would add the following line to your .bashrc file:
   >> export PYTHONPATH=/absolute/path/to/modeling:$PYTHONPATH

2. To install CSI, in the "modeling" directory, do:
  >> bzr init csi
  >> cd csi
- If you want to get codes from the ENS repository:
  >> bzr pull --remember bzr+ssh://csi@bzr.geologie.ens.fr/~
- If you want to get codes from the Caltech repository:
  >> bzr pull --remember bzr+ssh://csi@bzr.gps.caltech.edu

For those who know bzr, the ~ is important for the ENS repository (I don't know why, I probably have badly configured the repository).

3. To install EDKS, go in the modeling directory, then do:
  >> bzr init edks
  >> cd edks
- If you want to get codes from the ENS repository:
  >> bzr pull --remember bzr+ssh://edks@bzr.geologie.ens.fr/~
- If you want to get codes from the Caltech repository:
  >> bzr pull --remember bzr+ssh://edks@bzr.gps.caltech.edu

Then add the following lines to your .bashrc file (if you operate in bash):
  export EDKS_HOME='/path/to/modeling/directory/edks'
  export PYTHONPATH=${EDKS_HOME}/MPI_EDKS:$PYTHONPATH
  export PATH=${EDKS_HOME}/MPI_EDKS:$PATH
  export EDKS_BIN=${EDKS_HOME}/bin 
  export OMP_NUM_THREADS=4

4. To install okada4py, go in the modeling directory and do:
  >> bzr init okada
  >> cd okada
- If you want to get codes from the ENS repository:
  >> bzr pull --remember bzr+ssh://okada4py@bzr.geologie.ens.fr/~ 
- If you want to get codes from the Caltech repository:
  >> bzr pull --remember bzr+ssh://okada4py@bzr.gps.caltech.edu
Then, do:
  >> cd okada
  >> sudo python setup.py install
  
This should do it!

