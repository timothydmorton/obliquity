=========
obliquity
=========

Infer the stellar obliquity distribution of transiting planet systems, following Morton & Winn (2014).

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

**Command-line scripts**

In addition to the `obliquity` module, this package also installs a few command-line scripts.  

`write_cosi_dist`: This calculates a `Cosi_Distribution` given input parameters, and writes the distribution to 
file (`.h5` format that can be easily re-loaded back).


