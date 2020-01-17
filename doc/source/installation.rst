Installation
===============================

Dependencies:
-------------

CSI relies on a lot of amazing librairies written by smart people. Please install:

- python3
- gcc
- numpy
- spicy
- pyproj
- matplotlib
- cartopy
- multiprocessing
- h5py
- okada4py

Repositories:
-------------

To clone csi:

>> git clone git+ssh://csi@git.geologie.ens.fr/ csi

Okada4py is also one thing you will need:

>> git clone git+ssh://okada4py@git.geologie.ens.fr/~ okada4py

Install:
--------

There is nothing to compile for CSI. It is pure python and we haven't written a proper install script.
Therefore, the easiest way to go is to add the directory where you have cloned CSI to your PYTHONPATH environment variable:

For instance, in Bash, add to your .bashrc or .bash_profile:

>> export PYTHONPATH=/where/I/did/drop/the/code:$PYTHONPATH

This should do it!

