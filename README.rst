=========
obliquity
=========

Infer the stellar obliquity distribution of transiting planet systems, following Morton & Winn (2014).
Makes use of the `simpledist <https://github.com/timothydmorton/simpledist>`_ package, which will be installed 
as a dependency with installation of this package.

There are two main tasks this package does:

1. Calculate posteriors of cos(I) given measurements of Rstar, Prot, Vsin(I).

2. Infer the Fisher distribution parameter $\kappa$ given a sample of cos(I) posteriors.

Installation
------------

::

   $ pip install [--user] obliquity
   
Or clone the repository and install:

::

    $ git clone https://github.com/timothydmorton/obliquity.git
    $ cd obliquity
    $ python setup.py install [--user]

Basic usage
-----------

.. code-block:: python

    from obliquity.distributions import Cosi_Distribution
    cosi_dist = Cosi_Distribution((1.3,0.1),(15,0.3),(3.5,0.5)) #Radius, Prot, VsinI
    cosi_dist.summary_plot()

Command-line scripts
--------------------

In addition to the ``obliquity`` module, this package also installs a few command-line scripts.  

``write_cosi_dist``: This calculates a ``Cosi_Distribution`` given input parameters, and writes the distribution to 
file (`.h5` format that can be easily re-loaded back). e.g.,

::

    $ write_cosi_dist test.h5 -R 1.3 0.1 -P 14 0.3 -V 4 0.5

After having done this, you could launch up python and read in the distribution as follows:

.. code-block:: python

    >>> from obliquity import Cosi_Distribution_FromH5
    >>> cosi_dist = Cosi_Distribution_FromH5('test.h5')
    >>> cosi_dist.summary_plot()

This is particularly useful for running batch jobs and doing more analysis later.


