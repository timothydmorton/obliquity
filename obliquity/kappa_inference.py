from __future__ import print_function,division
import logging
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy.random as rand
from scipy.misc import logsumexp
from numpy import pi
from scipy.special import erf
from scipy.interpolate import LinearNDInterpolator as interpnd

import pandas as pd

from pkg_resources import resource_filename

from sini_inference_fns import cosi_integrand
#from simpledist import distributions as dists

def fisher(x,k):
    """Fisher distribution 
    """
    return k/(2*np.sinh(k)) * np.exp(k*np.cos(x))*np.sin(x)

def rayleigh(x,sigma):
    """Rayleigh distribution
    """
    return x/sigma**2 * np.exp(-x**2/(2*sigma**2))

def rayleigh_k(x,k):
    """Rayleigh distribution parameterized by Fisher k parameter.
    """
    return k*x * np.exp(-k*x**2/(2))

def sin_fisher(y,k):
    """pdf for y=sin(x) if x is fisher-distributed with parameter k.  Support is [0,1).
    """
    return 1/np.sqrt(1-y**2) * (k/(np.sinh(k)) * y * (np.cosh(k*np.sqrt(1-y**2))))

def cos_uniform(x):
    """ probability distribution of the cosine of a uniform angle, on [0,1]
    """
    return 2./(np.pi*np.sqrt(1-x**2))

    
def cosi_pdf(z,k=1):
    """Equation (11) of Morton & Winn (2014)
    """
    return 2*k/(np.pi*np.sinh(k)) * quad(cosi_integrand,z,1,args=(k,z))[0] #calling cython function

def draw_fisher(n,k=0.1,res=1e4,comp=False):
    psi = np.linspace(0,np.pi,res)
    pdf = fisher(psi,k)
    cdf = pdf.cumsum()
    cdf /= cdf.max()
    r = rand.random(n)
    inds = np.digitize(r,cdf)
    psis = psi[inds]
    if comp:
        return np.pi/2 - psis
    return psi[inds]

RSUN = 6.96e10
DAY = 86400

def generate_veq(R=1.3, dR=0.1, Prot=6, dProt=0.1,nsamples=1e4,plot=False,
                 R_samples=None,Prot_samples=None):
    """ Returns the mean and std equatorial velocity given R,dR,Prot,dProt

    Assumes all distributions are normal.  This will be used mainly for
    testing purposes; I can use MC-generated v_eq distributions when we go for real.
    """
    if R_samples is None:
        R_samples = R*(1 + rand.normal(size=nsamples)*dR)
    else:
        inds = rand.randint(len(R_samples),size=nsamples)
        R_samples = R_samples[inds]

    if Prot_samples is None:
        Prot_samples = Prot*(1 + rand.normal(size=nsamples)*dProt)
    else:
        inds = rand.randint(len(Prot_samples),size=nsamples)
        Prot_samples = Prot_samples[inds]

    veq_samples = 2*np.pi*R_samples*RSUN/(Prot_samples*DAY)/1e5
    
    if plot:
        plt.hist(veq_samples,histtype='step',color='k',bins=50,normed=True)
        d = stats.norm(scale=veq_samples.std(),loc=veq_samples.mean())
        vs = np.linspace(veq_samples.min(),veq_samples.max(),1e4)
        plt.plot(vs,d.pdf(vs),'r')
    
    return veq_samples.mean(),veq_samples.std()

def like_cosi(cosi,vsini_dist,veq_dist,vgrid=None):
    """likelihood of Data (vsini_dist, veq_dist) given cosi
    """
    sini = np.sqrt(1-cosi**2)
    def integrand(v):
        return vsini_dist(v)*veq_dist(v/sini)
    if vgrid is None:
        return quad(integrand,0,np.inf)[0]
    else:
        return np.trapz(integrand(vgrid),vgrid)
    
def cosi_posterior(vsini_dist,veq_dist,vgrid=None,npts=100,vgrid_pts=1000):
    """returns posterior of cosI given dists for vsini and veq (incorporates unc. in vsini)
    """
    if vgrid is None:
        vgrid = np.linspace(vsini_dist.ppf(0.005),vsini_dist.ppf(0.995),vgrid_pts)
    cs = np.linspace(0,1,npts)
    Ls = cs*0
    for i,c in enumerate(cs):
        Ls[i] = like_cosi(c,vsini_dist.pdf,veq_dist.pdf,vgrid=vgrid)
    if np.isnan(Ls[-1]): #hack to prevent nan when cos=1
        Ls[-1] = Ls[-2]
    Ls /= np.trapz(Ls,cs)
    return cs,Ls


def sample_posterior(x,post,nsamples=1):
    """ Returns nsamples from a tabulated posterior (not necessarily normalized)
    """
    cdf = post.cumsum()
    cdf /= cdf.max()
    u = rand.random(size=nsamples)
    inds = np.digitize(u,cdf)
    return x[inds]

def kappa_prior(k):
    return (1+k**2)**(-3/4)

def uniform_samples_from_kappa_prior(kmin,kmax,n,res=1e4):
    ks = np.linspace(kmin,kmax,res)
    pdf = kappa_prior(ks)
    cdf = pdf.cumsum()/pdf.cumsum().max()
    u = np.linspace(0,1,n)
    inds = np.digitize(u,cdf)
    inds = inds.clip(0,len(ks)-1)
    return ks[inds]


def cosi_samples_from_posteriors(vsini_posts,veq_posts,nsamples=100,vsini_upperlim=2):
    samples = np.zeros((len(vsini_posts),nsamples))
    N = len(vsini_posts)
    for i,vsini_post,veq_post in zip(np.arange(len(vsini_posts)),
                                     vsini_posts,veq_posts):
        samples[i,:] = cosi_samples(vsini_dist=vsini_post,
                                    veq_dist=veq_post,nsamples=nsamples)
    return samples


def v_posts_from_dataframe(df,N=1e4,alpha=0.23,l0=20,sigl=20):
    """ names: Prot, e_Prot, R, e_R, vsini, e_vsini
    """
    vsini_posts = []
    veq_posts = []
    if 'ep_R' in df:
        for R,dR_p,dR_m,P,dP,v,dv in zip(df['R'],df['ep_R'],df['em_R'],
                                  df['Prot'],df['e_Prot'],
                                  df['vsini'],df['e_vsini']):
            vsini_posts.append(stats.norm(v,dv))
            if dR_p==dR_m:
                veq_posts.append(Veq_Posterior(R,dR_p,P,dP))
            else:
                R_dist = dists.fit_doublegauss(R,dR_m,dR_p)
                Prot_dist = stats.norm(P,dP)
                veq_posts.append(Veq_Posterior_General(R_dist,Prot_dist,N=N,
                                                       l0=l0,sigl=sigl))
    else:
        for R,dR,P,dP,v,dv in zip(df['R'],df['e_R'],
                                  df['Prot'],df['e_Prot'],
                                  df['vsini'],df['e_vsini']):
            vsini_posts.append(stats.norm(v,dv))
            veq_posts.append(Veq_Posterior(R,dR,P,dP))
        
    return vsini_posts,veq_posts

def all_cosi_samples(vsinis,veqs,dveqs,dvsinis=None,nsamples=100,vsini_upperlim=4):
    if dvsinis is None:
        dvsinis = [None]*len(vsinis)
    if np.size(dvsinis)==1:
        dvsinis = [dvsinis]*len(vsinis)
    samples = np.zeros((len(vsinis),nsamples))
    for i,vsini,dvsini,veq,dveq in zip(range(len(vsinis)),vsinis,dvsinis,veqs,dveqs):
        samples[i,:] = cosi_samples(vsini,veq,dveq,dvsini,nsamples=nsamples,
                                    vsini_upperlim=vsini_upperlim)
    return samples

def build_interpfn(ks=None,nz=100,return_vals=False,filename='cosi_pdf_grid.h5',
                   kmax=300, save=False):
    """
    """
    if ks is None:
        ks = np.concatenate((np.logspace(-2,0,100),np.linspace(1.1,kmax+1,300)))

    nk = len(ks)
    zs = np.linspace(0,0.9999,nz)
    vals = np.zeros(nk*nz)
    allks = vals*0
    allzs = vals*0
    i=0
    for k in ks:
        for z in zs:
            vals[i] = cosi_pdf(z,k)
            allks[i] = k
            allzs[i] = z
            i += 1
    pts = np.array([allzs,allks]).T
    if save:
        df = pd.DataFrame({'z':allzs, 'k':allks, 'val': vals})
        df.to_hdf(filename,'grid')
    if return_vals:
        return pts,vals
    else:
        return interpnd(pts,vals)

#def cosi_pdf_fn(filename=resource_filename('data','cosi_pdf_grid.h5'),
#                recalc=False,**kwargs):
#    if recalc:
#        return build_interpfn(**kwargs)
#    else:
#        df = pd.read_hdf(filename,'grid')
#        pts = np.array([df['z'],df['k']]).T
#        vals = np.array(df['val'])
#        return interpnd(pts,vals)


COSI_PDF_FN = build_interpfn()

def lnlike_kappa(k,samples,prior=None):
    if prior is None:
        prior = kappa_prior
    like = COSI_PDF_FN(samples.ravel(),k).reshape(samples.shape)
    return np.log(np.prod(np.sum(like,axis=1)/samples.shape[1])*prior(k))
                  
def kappa_posterior(samples,kmin=0.01,kmax=100,npts=200):
    #ks = uniform_samples_from_kappa_prior(kmin,kmax,npts)
    ks = np.linspace(kmin,kmax,npts)
    lnlikes = ks*0
    for i,k in enumerate(ks):
        lnlikes[i] = lnlike_kappa(k,samples)
    post = np.exp(lnlikes)
    return ks,post/np.trapz(post,ks)


def lnlike_twofisher(f,k1,k2,samples,prior=None):
    if prior is None:
        prior = kappa_prior
    if f < 0 or f > 1:
        return -np.inf
    #if k1 > k2:
    #    return -np.inf
    else:
        term1 = f*COSI_PDF_FN(samples.ravel(),k1)
        term2 = (1-f)*COSI_PDF_FN(samples.ravel(),k2)
        print(term1.shape,term2.shape)
        like = (f*COSI_PDF_FN(samples.ravel(),k1) + 
                (1-f)*COSI_PDF_FN(samples.ravel(),k2)).reshape(samples.shape)
        return np.log(np.prod(np.sum(like,axis=1)/samples.shape[1])*prior(k1)*prior(k2))



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


class Veq_Posterior(object):
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

    def __call__(self,x):
        return fveq(x,self.R,self.dR,self.P,self.dP)

    pdf = __call__



