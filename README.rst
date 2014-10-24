obliquity
=========

Infer the stellar obliquity distribution of transiting planet systems, following Morton & Winn (2014).

**Installation**

::

   $ pip install [--user] obliquity
   
Or clone the repository and install:

::

    $ git clone https://github.com/timothydmorton/obliquity.git
    $ cd obliquity
    $ python setup.py install [--user]

**Usage from python shell**

.. code-block::python

    from obliquity.distributions import Cosi_Distribution
    cosi_dist = Cosi_Distribution((1.3,0.1),(15,0.3),(3.5,0.5)) #Radius, Prot, VsinI
    cosi_dist.summary_plot()


