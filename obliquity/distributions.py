from __future__ import print_function,division
import logging
import numpy as np

import simpledist.distributions as dists
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline as interpolate

from .kappa_inference import cosi_posterior

from astropy import constants as const
RSUN = const.R_sun.cgs.value
MSUN = const.M_sun.cgs.value
DAY = 86400

class Cosi_Distribution(dists.Distribution):
    """A Distribution for cos(I*) given Rstar, Prot, Vsini measurements

    Measurements may be passed as either (val,err) tuples/lists or 
    as `distribution` (or `stats.continuous`) objects
    """
    def __init__(self,R_dist,Prot_dist,vsini_dist,
                 vgrid=None,npts=100,vgrid_pts=1000,
                 N_veq_samples=1e4,alpha=0.23,l0=20,sigl=20,
                 veq_bandwidth=0.03):

        if type(R_dist) in [type([]),type((1,))]:
            R_dist = stats.norm(*R_dist)
        if type(Prot_dist) in [type([]),type((1,))]:
            Prot_dist = stats.norm(*Prot_dist)
        if type(vsini_dist) in [type([]),type((1,))]:
            vsini_dist = stats.norm(*vsini_dist)

        veq_dist = Veq_Distribution(R_dist,Prot_dist,N=N_veq_samples,
                                    alpha=alpha,l0=l0,sigl=sigl,
                                    bandwidth=veq_bandwidth)
        self.R_dist = R_dist
        self.Prot_dist = Prot_dist
        self.vsini_dist = vsini_dist
        self.veq_dist = veq_dist


        cs,Ls = cosi_posterior(vsini_dist, veq_dist, vgrid=vgrid,
                             npts=npts, vgrid_pts=vgrid_pts)
        pdf = interpolate(cs,Ls,s=0)
        
        dists.Distribution.__init__(self,pdf,name='cos(I)',
                                    minval=0,maxval=1)

def diff_Prot_factor(l,alpha=0.23):
    return (1 - alpha*(np.sin(np.deg2rad(l)))**2)

def veq_samples(R_dist,Prot_dist,N=1e4,alpha=0.23,l0=20,sigl=20):
    """Source for diff rot
    """
    ls = stats.norm(l0,sigl).rvs(N)
    Prots = Prot_dist.rvs(N)
    Prots *= diff_Prot_factor(ls,alpha)
    return R_dist.rvs(N)*2*np.pi*RSUN/(Prots*DAY)/1e5 
      
        
class Veq_Distribution(dists.KDE_Distribution):
    def __init__(self,R_dist,Prot_dist,N=1e4,alpha=0.23,l0=20,sigl=20,
                 adaptive=False,bandwidth=0.03,**kwargs):
        self.R_dist = R_dist
        self.Prot_dist = Prot_dist
        self.alpha = alpha
        self.l0 = l0
        self.sigl = sigl
        veqs = veq_samples(R_dist,Prot_dist,N=N,alpha=alpha,l0=l0,sigl=sigl)
        self.samples = veqs
        dists.KDE_Distribution.__init__(self,veqs,adaptive=adaptive,
                                        bandwidth=bandwidth,
                                        **kwargs)
    
