#!/usr/bin/env python
# coding: utf-8


import phoebe

import emcee
import matplotlib.pyplot as plt


from scipy.stats import norm

from matplotlib.pyplot import cm 
import schwimmbad
import corner
from timeit import default_timer as timer
import pickle
import sys
#Turn on multiprocessing
import os
#Some builds of NumPy will automatically parallelize some operations using something like the MKL linear algebra.
#This interferes with multithreading. so we turn it off with this variable, os.environ["OMP_NUM_THREADS"] = "1".
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

#Turn off multithreading
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')



print(list(reversed(range(1,11))))



#intialize binary
b = phoebe.default_binary()

#set known parameters
b['period@orbit'] = 0.5
b['sma@orbit'] = 3.5
b['incl@orbit'] = 83.5
b['requiv@primary'] = 1.2
b['requiv@secondary'] = 0.8
b['teff@primary'] = 6500.
b['teff@secondary'] = 5500.

#what we want to predict.
true_values=(83.5,1.2,0.8,5500)

b.add_dataset('lc', times=np.linspace(0, 1, 51))

b.run_compute()

b.plot(show=True)


np.savetxt('data.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model']+np.random.normal(0, 0.001, 51))).T)




#loading 'data.lc' file into an LC dataset
(times, fluxes) = np.loadtxt('data.lc', unpack=True)
sigma_arr = np.zeros(len(times))+0.001

plt.plot(times,fluxes)
plt.errorbar(times,fluxes,yerr=sigma_arr)


#number of walkers and iterations
nwalkers = 20
niter = 100
#Order(incl,r1,r2,teff2)
init_dist = [(83,84),(1.15,1.25),(0.75,0.85),(5450,5550)]

#uniform priors
priors = [(83,84),(1.15,1.25),(0.75,0.85),(5450,5550)]

#initiating new bundle
c = phoebe.default_binary()
c.add_dataset(phoebe.dataset.lc, times=times, fluxes=fluxes, sigmas=sigma_arr, compute_phases=np.linspace(0,1.,51), passband='Kepler:mean', dataset='lc02', overwrite=True)


c.get_parameter(context='compute', qualifier='irrad_method').set_value("none")
c.get_parameter(context='compute', component='primary', qualifier='ntriangles').set_value(100)
c.get_parameter(context='compute', component='secondary', qualifier='ntriangles').set_value(100)


c.set_value_all('ld_mode', 'manual')
c.set_value_all('ld_func', 'linear')
c.set_value_all('ld_coeffs', [0.])
c.set_value_all('ld_mode_bol', 'manual')
c.set_value_all('ld_func_bol', 'linear')
c.set_value_all('ld_coeffs_bol', [0.])

#autoscale data 
c.set_value('pblum_mode',value='dataset-scaled') 





#setting values of the period, sma and primary teff
c['period@orbit'] = 0.5
c['sma@orbit'] = 3.5
c['teff@primary'] = 6500.



def rpars(init_dist):
    '''
    Pick a starting point from a uniform distributions
    '''
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist]


def lnprior(priors, values):
    '''
    Using a uniform prior, so values outside the prior range are set to -infinity
    '''  
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp

def lnprob(model_params,sigma_arr):
    '''
    Calculate the probability for each dataset. 
    '''
    #Model_params order(incl,r1,r2,teff2)

    c['incl@binary@orbit@component'] = model_params[0]
    c['requiv@primary@star@component'] = model_params[1]
    c['requiv@secondary@star@component'] = model_params[2]
    c['teff@secondary@star@component'] = model_params[3]
    
    #Get the log prior probabilities
    lnp = lnprior(priors,model_params)
    if not np.isfinite(lnp):
            return -np.inf
    
    try: 
        c.run_compute()

        # use chi^2 to compare the model to the data:
        chi2 = 0.
        #compute_residuals only works on LC and RV dataset types
        #This will make the final chi2 a sum across all of these.
        for dataset in c.get_model().datasets:
            chi2+=np.sum(c.calculate_residuals(dataset=dataset, as_quantity=False)**2/sigma_arr**2)
        # calculate lnprob
        lnprob = -0.5*chi2 + lnp
        return lnprob
    except:
        return -np.inf

#replace xrang with range, 
#matplotlib.mlab.normpdf is deprecated 
#used scipy.stats.norm.pdf instead, 
#print needed ()

def run(init_dist, nwalkers, niter, sigma_arr, true_values):
    # Specify the number of dimensions for mcmc
    ndim = len(init_dist)

    # Generate initial guesses for all parameters for all chains
    p0 = np.array([rpars(init_dist) for i in range(nwalkers)])

    # Generate the emcee sampler. Here the inputs provided include the lnprob function. With this setup, the value z in the lnprob function, is the output from the sampler.
    with schwimmbad.MultiPool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob, args=(sigma_arr,),pool=pool)
    
    #pos is the position of the walkers in parameter space
    #prob is the probability of the given "pos" positions
    #state is the state of the random number generator
        start = timer()
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        print("Total time: {:.2f}".format((timer() - start)/60.0 ))


    return pos, sampler


def hist_fig(sampler,ndim,true_val):
    labels = ["$incl$", "$r_1$", "$r_2$","$teff_2$"]
    labels2 = ["incl", "r_1", "r_2","teff_2"]
    for i in range(ndim):
        fig = plt.figure()
        y = sampler.flatchain[:,i]
        n, bins, patches = plt.hist(y, 200, density=True, color="b", alpha=0.45)#, histtype="step"
        plt.title("Dimension {}".format(i))
        
        mu = np.average(y)
        sigma = np.std(y)
        
        print ("mu = ", mu)
        print ("sigma = ",sigma)

        bf = norm.pdf(bins, mu, sigma)
        l = plt.plot(bins, bf, 'k--', linewidth=2.0)
        plt.gca().axvline(true_val[i])
        plt.savefig("{}.pdf".format(labels2[i]))
        
    plt.show()
    return fig 


def walker_fig(sampler,ndim,true_val):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.chain
    print(samples.shape)
    labels = ["$incl$", "$r_1$", "$r_2$","$teff_2$"]
    for i in range(ndim):
        ax = axes[i]
        for j in range(len(samples)):
            if(i == 2):
                ax.plot(samples[j, :, i], "k", alpha=0.1)
            else:
                ax.plot(samples[j, :, i], "k", alpha=0.1)
        ax.set_xlim(0, len(samples[1]))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.axhline(true_val[i])

    axes[-1].set_xlabel("step number")
    plt.show()
    
    return fig 


def fit_plot(mod,sampler,nwalkers,times,true_value,ndim):
    
    #Plot actual data
    times = c.get_value(dataset='lc02',qualifier='times',context='dataset')
    fluxes = c.get_value(dataset='lc02',qualifier='fluxes',context='dataset')
    fig= plt.plot(times,fluxes,'.k')
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    
    #Limit number of samples to 100
    size=len(samples) #For small numbers of samples    
    if size > 20:
        size = 20
    for model_params in samples[np.random.randint(len(samples), size=size)]:
        #Model_params order(incl,r1,r2,teff2)
        c.set_value(qualifier='incl',context='component', component='binary',value=model_params[0])
        c.set_value(qualifier='requiv',context='component', component='primary',value=model_params[1])
        c.set_value(qualifier='requiv',context='component', component='secondary',value=model_params[2])
        c.set_value(qualifier='teff',component='secondary',value=model_params[3])
        
        c.run_compute(overwrite=True)
        
        model = c.get_parameter(context='model',qualifier='fluxes').interp_value(times=times)
        plt.savefig("fit_plot.pdf")

        plt.plot(times,model,c='k',alpha=0.01)
        
  
    plt.xlabel("Times")
    plt.ylabel("Flux")
    plt.show()
    return fig
    


pos,sampler = run(init_dist, nwalkers, niter, sigma_arr,true_values)



#pickle option

#pickle.dump(sampler,open('emcee_newfit_20w10i100t.p','wb'))



ndim = len(init_dist)
#ndim = 100
fig = hist_fig(sampler,ndim,true_values)

ndim = len(init_dist)
samples = sampler.chain[:, :, :].reshape((-1, ndim))
corner.corner(samples, labels=["$incl$", "$r_1$", "$r_2$","$teff_2$"],truths=true_values)
plt.savefig("cornerplot.pdf")
fig = plt.show()

fig = fit_plot(c,sampler,nwalkers,times,true_values,ndim)


fig= walker_fig(sampler,ndim,true_values)
fig= fig.savefig("walkerfig.pdf")

num = 100
incl_arr=np.linspace(83,84,num)
r1_arr=np.zeros(num) + true_values[1]
r2_arr=np.zeros(num) + true_values[2]
teff2_arr=np.zeros(num) + true_values[3]
print(incl_arr)

prob_list = list()
for i in np.arange(len(incl_arr)):
    prob_list.append(lnprob([incl_arr[i],r1_arr[i],r2_arr[i],teff2_arr[i]],sigma_arr))

prob_arr = np.array(prob_list)
plt.plot(incl_arr,prob_arr)
plt.gca().axvline(true_values[0])

num = 100
r1_arr=np.linspace(1.15,1.25,num)
incl_arr=np.zeros(num) + true_values[0]
r2_arr=np.zeros(num) + true_values[2]
teff2_arr=np.zeros(num) + true_values[3]
print(r1_arr)

prob_list = list()
for i in np.arange(len(incl_arr)):
    prob_list.append(lnprob([incl_arr[i],r1_arr[i],r2_arr[i],teff2_arr[i]],sigma_arr))

prob_arr = np.array(prob_list)
plt.plot(r1_arr,prob_arr)
plt.gca().axvline(true_values[1])


num = 100
r2_arr=np.linspace(.75,.85,num)
incl_arr=np.zeros(num) + true_values[0]
r1_arr=np.zeros(num) + true_values[1]
teff2_arr=np.zeros(num) + true_values[3]
print(r2_arr)

prob_list = list()
for i in np.arange(len(r2_arr)):
    prob_list.append(lnprob([incl_arr[i],r1_arr[i],r2_arr[i],teff2_arr[i]],sigma_arr))

prob_arr = np.array(prob_list)
plt.plot(r2_arr,prob_arr)
plt.gca().axvline(true_values[2])



num = 100
teff2_arr=np.linspace(5450,5550,num)
incl_arr=np.zeros(num) + true_values[0]
r1_arr=np.zeros(num) + true_values[1]
r2_arr=np.zeros(num) + true_values[2]
print(teff2_arr)

prob_list = list()
for i in np.arange(len(incl_arr)):
    prob_list.append(lnprob([incl_arr[i],r1_arr[i],r2_arr[i],teff2_arr[i]],sigma_arr))

prob_arr = np.array(prob_list)
plt.plot(teff2_arr,prob_arr)
plt.gca().axvline(true_values[3])





