from __future__ import print_function,division
import logging
import numpy as np
import pandas as pd

import simpledist.distributions as dists
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline as interpolate

from .kappa_inference import cosi_posterior,sample_posterior

from astropy import constants as const
RSUN = const.R_sun.cgs.value
MSUN = const.M_sun.cgs.value
DAY = 86400

class Cosi_Distribution(dists.Distribution):
    """A Distribution for cos(I*) given Rstar, Prot, Vsini measurements

    Measurements may be passed as either (val,err) tuples/lists or 
    as `distribution` (or `stats.continuous`) objects
    """
    def __init__(self,R_dist,Prot_dist,vsini_dist,nsamples=1e3,
                 vgrid=None,npts=100,vgrid_pts=1000,
                 N_veq_samples=1e4,alpha=0.23,l0=20,sigl=20,
                 veq_bandwidth=0.03):

        if type(R_dist) == type(''):
            R_dist = dists.Distribution_FromH5(R_dist)
        elif type(R_dist) in [type([]),type((1,))]:
            if len(R_dist)==2:
                R_dist = dists.Gaussian_Distribution(float(R_dist[0]),float(R_dist[1]),name='radius')
            elif len(R_dist)==3:
                R_dist = dists.fit_doublegauss(float(R_dist[0]),float(R_dist[1]),float(R_dist[2]),name='radius')
        if type(Prot_dist) in [type([]),type((1,))]:
            Prot_dist = dists.Gaussian_Distribution(float(Prot_dist[0]),
                                                    float(Prot_dist[1]),
                                                    name='Prot')
        if type(vsini_dist) in [type([]),type((1,))]:
            vsini_dist = dists.Gaussian_Distribution(float(vsini_dist[0]),
                                                     float(vsini_dist[1]),
                                                     name='vsini')

        veq_dist = Veq_Distribution(R_dist,Prot_dist,N=N_veq_samples,
                                    alpha=alpha,l0=l0,sigl=sigl,
                                    bandwidth=veq_bandwidth)
        self.vsini_dist = vsini_dist
        self.veq_dist = veq_dist


        cs,Ls = cosi_posterior(vsini_dist, veq_dist, vgrid=vgrid,
                             npts=npts, vgrid_pts=vgrid_pts)
        self.samples = sample_posterior(cs,Ls,nsamples=nsamples)
        pdf = interpolate(cs,Ls,s=0)
        
        dists.Distribution.__init__(self,pdf,name='cos(I)',
                                    minval=0,maxval=1)

    def save_hdf(self,filename,path='',**kwargs):
        dists.Distribution.save_hdf(self,filename,path,**kwargs)
        self.vsini_dist.save_hdf(filename,'{}/vsini'.format(path))
        self.veq_dist.save_hdf(filename,'{}/veq'.format(path))

class Cosi_Distribution_FromH5(Cosi_Distribution,dists.Distribution_FromH5):
    def __init__(self,filename,path='',**kwargs):
        dists.Distribution_FromH5.__init__(self,filename,path,
                                     **kwargs)
        self.vsini_dist = dists.Distribution_FromH5(filename,'vsini')
        self.veq_dist = Veq_Distribution_FromH5(filename,'veq')

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
                                        bandwidth=bandwidth,name='v_eq',
                                        **kwargs)

    def save_hdf(self,filename,path='',**kwargs):
        dists.KDE_Distribution.save_hdf(self,filename,path,**kwargs)
        self.R_dist.save_hdf(filename,'{}/radius'.format(path))
        self.Prot_dist.save_hdf(filename,'{}/Prot'.format(path))

class Veq_Distribution_FromH5(Veq_Distribution,dists.Distribution_FromH5):
    def __init__(self,filename,path='',**kwargs):
        dists.Distribution_FromH5.__init__(self,filename,path,
                                     **kwargs)
        self.R_dist = dists.Distribution_FromH5(filename,'{}/radius'.format(path))
        self.Prot_dist = dists.Distribution_FromH5(filename,'{}/Prot'.format(path))


