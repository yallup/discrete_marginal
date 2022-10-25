#!/usr/bin/env python3
import argparse
import os
import sys

import anesthetic
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fgivenx.mass import PMF
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from mpi4py import MPI
from pypolychord import settings
from pypolychord.priors import SortedUniformPrior, UniformPrior,GaussianPrior
import scipy
from scipy.stats import poisson

import pypolychord as pc
import fgivenx
from scipy.stats import cauchy,norm


def plot_signal_corner(signal_root,signal_fb_root,plotdir):
    ns = anesthetic.read_chains(root=signal_root)
    ns_fb = anesthetic.read_chains(root=signal_fb_root)
    ns.set_label(0,r"$\mu$")
    ns.set_label(1,r"$\sigma$")
    ns.set_label(2,r"$A$")
    f,a=ns.plot_2d([0,1,2],kinds=dict(lower='kde_2d'),label="Discrete",lower_kwargs=dict(legend=None,alpha=0.7,levels=[.68],zorder=2),diagonal_kwargs=dict(legend=None))
    ns_fb.plot_2d(a,kinds=dict(lower='kde_2d'),label="Fixed",lower_kwargs=dict(legend=None,alpha=0.7,levels=[.68],zorder=2),diagonal_kwargs=dict(legend=None))
    # a.axlines([0,1,2],[true_loc,true_scale,true_xs],alpha=1,color="black",linestyle=(1,(1,1)),linewidth=.5)
    f.savefig(os.path.join(plotdir,"psi_corner.pdf"))
    f.savefig(os.path.join(plotdir,"psi_corner.png"))

def plot_signal_analysis(signal_root,bg_root,plotdir,model):
    gs = {"hspace": 0.05, "wspace": 0.05, "height_ratios":(.5,.5,.5,.3)}
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex="col",
                            figsize=[2.5, 5], gridspec_kw=gs)
    ns = anesthetic.read_chains(root=signal_root)
    ns_bg = anesthetic.read_chains(root=bg_root)
    samp=ns.sample(1000)
    samp_bg=ns_bg.sample(1000)
    w_s=samp.get_weights()
    w_s_bg =samp_bg.get_weights()
    sig_mod_params=samp.to_numpy()[...,:3]
    back_mod_params=samp.to_numpy()[...,3:-3]
    bg_back_mod_params=samp_bg.to_numpy()[...,:-3]

    hist_signal_only=np.zeros_like(model.x)
    for i,s in enumerate(sig_mod_params):
        y_sig=s[2] * norm(loc=s[0],scale=s[1]).pdf(model.x)
        hist_signal_only+= y_sig* w_s[i]
        axs[2].plot(model.x,y_sig,alpha=0.5,color="C0",zorder=-1,label="Candidate signals")

    # hist_signal_only=/1000
    hbins=np.linspace(100,180,81)
    axs[1].hist(model.x, hbins, weights=hist_signal_only / hist_signal_only.sum(), histtype='step')
    axs[1].set_ylim([0,0.3])
    y_bg_samps=model.posterior_predict(back_mod_params, model.x)
    y_bg_null_samps=model.posterior_predict(bg_back_mod_params, model.x)
        

    axs[0].scatter(model.x,model.y,color="black",marker=".",s=1,label="Data")
    axs[0].plot(model.x,y_bg_samps.mean(axis=0),color="C1",zorder=-1,label="Background model",linewidth=.8)
    axs[0].plot(model.x,y_bg_null_samps.mean(axis=0),color="C5",zorder=-1,label="Background Null model",linewidth=.8)
    axs[0].set_ylim([0,model.y.max()])
    axs[0].legend()
    axs[1].set_yticks(axs[1].get_yticks()[:-1])
    axs[2].set_yticks(axs[2].get_yticks()[:-1])
    axs[3].set_xlim([100,180])
    axs[3].set_yticks(axs[3].get_yticks()[1:-1])
    axs[3].set_ylabel("Residuals")
    axs[3].hlines(0,xmin=100,xmax=180,color="C1")
    evidence_ratio=ns.logZ() - ns_bg.logZ()
    axs[1].text(150, 0.21, r"$\ln\frac{\mathcal{Z}_\psi}{\mathcal{Z}_0}=%s$" % float(np.round(evidence_ratio,decimals=3)),fontsize=5)


    axs[3].hist(model.x,hbins,weights=(model.y-y_bg_samps.mean(axis=0))/np.sqrt(model.y),color="black",histtype='step',linewidth=0.6)
    axs[3].hist(model.x,hbins,weights=(y_bg_null_samps.mean(axis=0)-y_bg_samps.mean(axis=0))/np.sqrt(model.y),color="C5",histtype='step',linewidth=0.6)
    axs[3].set_ylim([-3,3])
    axs[3].set_yticks([-1,0,1])
    axs[3].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir,"psi_predict.pdf"))
    fig.savefig(os.path.join(plotdir,"psi_predict.png"),dpi=200)

    print

    return 0

def powerlaw(basis, x):
    return (np.atleast_2d(basis[..., 0]).T * (x + 1) ** -(1.5 * np.atleast_2d(basis[..., 1])).T).sum(axis=0)


def exponential(basis, x):
    return (np.atleast_2d(basis[..., 0]).T * np.exp(-(x) * ( np.atleast_2d(basis[..., 1])).T)).sum(axis=0)
    

def polylog(basis, x):
    return ((np.atleast_2d(basis[..., 0]).T) * np.log((x + np.e)) ** (-(4 *  np.atleast_2d(basis[..., 1])).T)).sum(
        axis=0)


class marginal_model(object):
    allowed_funcs = [exponential, powerlaw, polylog]

    def __init__(self, x, y, signal=True,hier=True,family=None, n_basis=4, n_hier=2, logzero=-1e25):
        self._init_data(x, y)
        self.signal=signal
        self.x, self.y = x,y
        self.x_whiten,self.y_whiten=self.whiten(x,y)
        self.n_basis = n_basis
        self.n_family = len(self.allowed_funcs)
        self.params_per_basis = 2
        self.hier=hier
        self.n_hier = n_hier
        self.family=family
        self.dict_shape = (self.n_basis, self.params_per_basis)
        self.shape = self.dict_shape
        self.logzero = logzero

    def _init_data(self, x, y):
        self.xmin = x.min()
        self.xrange = x.max() - x.min()
        self.ystd = y.std()
        # self.ystd=1.0

    def whiten(self, x, y):
        x = (x - self.xmin) / self.xrange
        y = y / self.ystd
        return x, y

    def unwhiten(self, x, y):
        y = y * self.ystd
        x = (x * self.xrange) + self.xmin
        return x, y

    def dims(self):
        if self.hier:
            base=int(np.prod(self.dict_shape) + self.n_hier)
        else:
            base=int(np.prod(self.dict_shape))
        if self.signal:
            return base + 3
        else:
            return base

    def fold_flat_prior(self, theta):
        n_active_bases = theta[1].astype(int)
        family_choice = theta[0].astype(int)
        basis_params = theta[2:].reshape(self.shape)
        return n_active_bases, family_choice, basis_params

    def fold_flat_prior_vec(self, theta):
        n_active_bases = theta[..., 1].astype(int)
        family_choice = theta[..., 0].astype(int)
        basis_params = theta[..., 2:].reshape(theta.shape[0], *self.shape)
        return n_active_bases, family_choice, basis_params

    def predict_background(self, basis_params, n_active, family_choice, x):
        chosen_family = self.allowed_funcs[family_choice]
        active_basis_params = basis_params[:n_active, ...]
        return chosen_family(active_basis_params, x)
    

    def predict_signal(self,sig_theta,x):
        return sig_theta[2] * norm(loc=sig_theta[0],scale=sig_theta[1]).pdf(x)

    def posterior_predict(self, theta, x):
        x, _ = self.whiten(x, 0)
        n_active_bases, family_choice, basis_param = self.fold_flat_prior_vec(
            theta)
        y = [self.predict_background(basis_params=basis_param[i], n_active=n_active_bases[i],
                          family_choice=family_choice[i], x=x) for i, v in enumerate(n_active_bases)]
        _, y = self.unwhiten(0, np.asarray(y))
        return np.asarray(y)

    def fgivenx(self, x, theta):
        x, _ = self.whiten(x, 0)
        theta = np.atleast_2d(theta)
        n_active_bases = theta[..., 1].astype(int)
        family_choice = theta[..., 0].astype(int)
        basis_params = theta[..., 2:].reshape(theta.shape[0], *self.shape)
        y = [self.predict_background(basis_params=basis_params[i], n_active=n_active_bases[i],
                          family_choice=family_choice[i], x=x) for i, v in enumerate(n_active_bases)]
        _, y = self.unwhiten(0, np.asarray(y))
        return y.squeeze()

    def likelihood(self, theta, **kwargs):
        if self.signal:
            sig_theta,theta=np.split(theta,[3])

        if self.hier:
            family_choice = theta[0].astype(int)
            n_active_bases = theta[1].astype(int)
            basis_params=theta[2:].reshape(self.shape)
        else:
            n_active_bases=self.n_basis
            family_choice=self.family
            basis_params=theta.reshape(self.shape)


        # n_active_bases, family_choice, basis_params = self.fold_flat_prior(
        #     theta)
        y = self.predict_background(basis_params, n_active_bases, family_choice, self.x_whiten)
        
        #unwhiten data so the poisson errors make sense
        _, y = self.unwhiten(0, y)
        #Poisson likelihood

        if self.signal:
            y+=self.predict_signal(sig_theta,self.x)

        logl = poisson.logpmf(np.round(self.y),y).sum()

        #Gaussian likelihood
        # logl=sum(norm.logpdf(y,loc=self.y,scale=np.sqrt(self.y)))
        if np.isnan(logl):
            logl = self.logzero
        if np.isinf(logl):
            logl = self.logzero
        return float(logl), []

class prior(object):
    def __init__(self, shape, n_families,signal=True,hier=True):
        self.amplitude_prior = SortedUniformPrior(-10,10)
        self.exponent_prior = UniformPrior(0, 5)

        self.family_prior = UniformPrior(0, n_families)
        self.n_prior = UniformPrior(1, shape[0] + 1)
        #optional exponential prior on N to promote sparsity
        # self.n_prior=ExponentialPrior(.5)
        self.shape = shape
        self.signal=signal
        self.hier=hier
        #Priors on signal parameters
        #Amplitude of signal
        self.signal_xs_prior=UniformPrior(0,500)
        #Location of peak
        self.signal_mean_prior=UniformPrior(100,180)
        #variance of peak
        self.signal_scale_prior=UniformPrior(0.5,3.0)
        # self.signal_scale_prior=UniformPrior(0.9,1.1)


    def __call__(self, theta):
        if self.signal:
            sig_theta,theta=np.split(theta,[3])
            s=np.zeros_like(sig_theta)
            s[0]=self.signal_mean_prior(sig_theta[0])
            s[1] = self.signal_scale_prior(sig_theta[1])
            s[2] = self.signal_xs_prior(sig_theta[2])
        if self.hier:
            hier_param,theta = np.split(theta, [2])
            x = np.zeros_like(hier_param)
            x[0] = self.family_prior(hier_param[0])
            # x[1]=self.n_prior(hier_param[1])+1
            x[1] = self.n_prior(hier_param[1])


        t = theta.reshape(self.shape)
        y = np.zeros_like(t)
        
        y[..., 0] = self.amplitude_prior(t[..., 0])[::-1]
        y[..., 1] = self.exponent_prior(t[..., 1])

        y=y.flatten()

        if self.hier:
            y=np.concatenate([x,y])
        if self.signal:
            y=np.concatenate([s,y])
        return y


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #nlive is the main control param here, nlive=1k is hi res, nlive=100 typical exploratory value
    nlive = 100
    repeat_frac = 5  # 10
    prior_boost = 10


    #load event generator curve from file
    events = pd.read_csv("./events/events_rebin.csv",
                         delim_whitespace=True, header=None)

    #scale the dataset size
    lumi = 10000


    x = ((events[0] + events[1]) / 2).to_numpy()
    
    true_loc=125
    true_scale=1.0
    true_xs=150

    #work out the signal shape
    sig=norm(loc=true_loc,scale=true_scale).pdf(x)*true_xs


    #poisson noise
    y = poisson.rvs(events[2] * lumi + sig , random_state=3)
    # y = poisson.rvs(events[2] * lumi , random_state=3)


    ######################################################################################################################
    #fixed bg model
    run_name = "signal_fb"
    file_name = os.path.join(base_dir, run_name)
    sigfbdir=file_name
    
    model_sig_fb = marginal_model(x, y, n_basis=3,family=0,hier=False)
    p = prior(model_sig_fb.shape, model_sig_fb.n_family,model_sig_fb.signal,hier=model_sig_fb.hier)

    pc_settings = settings.PolyChordSettings(nDims=model_sig_fb.dims(), nDerived=0,
                                             nlive=nlive, num_repeats=repeat_frac * model_sig_fb.dims(),
                                             base_dir=base_dir, file_root=run_name)
    pc_settings.read_resume = False
    pc_settings.write_resume = False
    pc_settings.nprior = pc_settings.nlive * prior_boost
    # pc_settings.max_ndead=1
    pc_settings.cluster_posteriors = False
    pc_settings.boost_posterior = pc_settings.num_repeats
    # pc_settings.precision_criterion=0.00001

    output_signal_fb=pc.run_polychord(model_sig_fb.likelihood, prior=p, nDims=model_sig_fb.dims(), nDerived=0,
                     settings=pc_settings)

    ######################################################################################################################
    #first the hypothesis with signal model

    run_name = "signal"
    file_name = os.path.join(base_dir, run_name)
    sigdir=file_name

    model_sig = marginal_model(x, y, n_basis=3)
    p = prior(model_sig.shape, model_sig.n_family,model_sig.signal)

    pc_settings = settings.PolyChordSettings(nDims=model_sig.dims(), nDerived=0,
                                             nlive=nlive, num_repeats=repeat_frac * model_sig.dims(),
                                             base_dir=base_dir, file_root=run_name)
    pc_settings.read_resume = False
    pc_settings.write_resume = False
    pc_settings.nprior = pc_settings.nlive * prior_boost
    # pc_settings.max_ndead=1
    pc_settings.cluster_posteriors = False
    pc_settings.boost_posterior = pc_settings.num_repeats
    # pc_settings.precision_criterion=0.00001

    output_signal=pc.run_polychord(model_sig.likelihood, prior=p, nDims=model_sig.dims(), nDerived=0,
                     settings=pc_settings)

    ######################################################################################################################
    #repeat with background model
    run_name = "background"
    file_name = os.path.join(base_dir, run_name)
    bgdir=file_name

    model = marginal_model(x, y, n_basis=3,signal=False)
    p = prior(model.shape, model.n_family,model.signal)

    pc_settings = settings.PolyChordSettings(nDims=model.dims(), nDerived=0,
                                             nlive=nlive, num_repeats=repeat_frac * model.dims(),
                                             base_dir=base_dir, file_root=run_name)
    pc_settings.read_resume = False
    pc_settings.write_resume = False
    pc_settings.nprior = pc_settings.nlive * prior_boost
    # pc_settings.max_ndead=1
    pc_settings.cluster_posteriors = False
    pc_settings.boost_posterior = pc_settings.num_repeats
    # pc_settings.precision_criterion=0.00001

    output_background=pc.run_polychord(model.likelihood, prior=p, nDims=model.dims(), nDerived=0,
                     settings=pc_settings)


    if rank==0:
        
        os.makedirs(plotdir,exist_ok=True)
        plot_signal_corner(sigdir,sigfbdir,plotdir)
        plot_signal_analysis(sigdir,bgdir,plotdir,model_sig)


if __name__ == '__main__':
    sys.exit(main())
