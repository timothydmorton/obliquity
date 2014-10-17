from __future__ import print_function,division
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simpledist.distributions as dists
from plotutils import setfig
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline as interpolate
from scipy.special import erf
from numpy import pi

from .kappa_inference import cosi_posterior,sample_posterior

from astropy import constants as const
RSUN = const.R_sun.cgs.value
MSUN = const.M_sun.cgs.value
DAY = 86400

class Cosi_Distribution(dists.Distribution):
    """A Distribution for cos(I*) given Rstar, Prot, Vsini measurements

    Measurements may be passed as either (val,err) tuples/lists or 
    as `distribution` (or `stats.continuous`) objects

    if vsini_dist is passed as a tuple/list, and vsini_corrected is False,
    then by default the value of vsini will be corrected by dividing by 
    1 - (alpha/2), to correct for differential rotation, as in 
    Hirano et al. (2013/14)
    """
    def __init__(self,R_dist,Prot_dist,vsini_dist,nsamples=1e3,
                 vgrid=None,npts=100,vgrid_pts=1000,vsini_corrected=False,
                 N_veq_samples=1e4,alpha=0.23,l0=20,sigl=20,
                 veq_bandwidth=0.03):

        if type(R_dist) == type(''):
            R_dist = dists.Distribution_FromH5(R_dist,'radius')
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
            if not vsini_corrected:
                vsini_dist[0] = float(vsini_dist[0])/(1 - (alpha/2))
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

    def summary_plot(self,fig=None,title=''):
        setfig(fig)
        
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)

        plt.sca(ax1)
        self.plot(fig=0)
        plt.sca(ax2)
        self.veq_dist.R_dist.plot(fig=0)
        plt.sca(ax3)
        self.veq_dist.plot(fig=0,label='V_eq')
        self.vsini_dist.plot(fig=0,label='VsinI')
        #plt.legend()
        plt.xlabel('Rotational Velocity [km/s]')

        plt.annotate('$P = {:.2f}\pm{:.2f}$d'.format(self.veq_dist.Prot_dist.mu,
                                                     self.veq_dist.Prot_dist.sig),
                     xy=(0.6,0.3),xycoords='figure fraction',fontsize=24) 


        plt.suptitle(title)

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


def fveq(z,R,dR,P,dP):
    """z=veq in km/s.  Breaks on errors < 6% for some numerical overflow reason
    """

    R *= 2*pi*RSUN
    dR *= 2*pi*RSUN
    P *= DAY
    dP *= DAY

    exp1 = -P**2/(2*dP**2) - R**2/(2*dR**2)
    exp2 = (dR**2*P + dP**2*R*(z*1e5))**2/(2*dP**2*dR**2*(dR**2 + dP**2*(z*1e5)**2))

    nonexp_term = 2*dP*dR*np.sqrt(dR**2 + dP**2*(z*1e5)**2)

    return 1e5/(4*pi*(dR**2 + dP**2*(z*1e5)**2)**(3/2))*np.exp(exp1 + exp2) *\
        (dR**2 * P*np.sqrt(2*pi) + 
         dP**2 * np.sqrt(2*pi)*R*(z*1e5) + 
         nonexp_term * np.exp(-exp2) +
         np.sqrt(2*pi)*(dR**2*P + dP**2*R*(z*1e5))*erf((dR**2*P + dP**2*R*(z*1e5)) *
                                                       (np.sqrt(2)*dP*dR*
                                                        np.sqrt(dR**2 + dP**2*(z*1e5)**2))))



class Veq_Distribution_Simple(dists.Distribution):
    def __init__(self,R,dR,P,dP):
        self.R = R
        self.dR = dR
        #if dR < 0.05*self.R:
        #    self.dR = 0.05*self.R
        #else:
        #    self.dR = dR
        self.P = P
        self.dP = dP
        #if dP < 0.05*self.P:
        #    self.dP = 0.05*self.P
        #else:
        #    self.dP = dP
        self.veq_nominal = (RSUN*2*np.pi*self.R)/(self.P*DAY)/1e5
        self.e_veq_nominal = self.veq_nominal*(np.sqrt((self.dR/self.R)**2 +
                                                       (self.dP/self.P)**2))
        minveq = self.veq_nominal - 5*self.e_veq_nominal
        maxveq = self.veq_nominal + 5*self.e_veq_nominal

        def pdf(x):
            return fveq(x,self.R,self.dR,self.P,self.dP)

        dists.Distribution.__init__(self,pdf,name='Veq',
                                    minval=minveq,maxval=maxveq)

