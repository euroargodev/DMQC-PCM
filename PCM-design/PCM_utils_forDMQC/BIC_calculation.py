# BIC calculation functions file
import xarray as xr
import numpy as np

import pyxpcm
from pyxpcm.models import pcm

import matplotlib.pyplot as plt

from .data_processing import get_regulargrid_dataset


def BIC_calculation(ds, corr_dist, pcm_features, features_in_ds, z_dim, Nrun=10, NK=20):
    '''Calculation of BIC (Bayesian Information Criteria) for a training dataset.
        The calculation is parallelised using ThreadPoolExecutor.

           Parameters
           ----------
               ds: dataset
               corr_dist: correlation distance
               pcm_features: dictionary with pcm features {'temperature': z vector}
               features_in_ds: dictionary with the name of feaures in the model and in the dataste
                    {temperature: thetao} 
               z_dim: name of the z variable
               Nrun: number of runs
               NK: max number of classes

           Returns
           ------
               BIC: matrix with BIC value for each run and number of classes
               BIC_min: minimun BIC value calculated from the mean of each number of classes

               '''
    BIC = np.zeros((NK-1, Nrun))

    for run in range(Nrun):
        print("run %i/%i" % ((run+1), Nrun))
    
        # get sub-sampling dataset
        #ds_run = pcm_utils.get_regulargrid_dataset(ds, corr_dist)
        ds_run = get_regulargrid_dataset(ds, corr_dist)
    
        # bic calculation
        BICi=[]
        for k in range(1,NK):
            m = pcm(K=k, features=pcm_features)
            m.fit(ds_run, features=features_in_ds, dim=z_dim)
            BICi.append(m.bic(ds_run))
        
        BIC[:, run] = np.array([i for i in BICi])

    BIC_min = np.argmin(np.mean(BIC, axis=1)) + 1

    return BIC, BIC_min


def plot_BIC(BIC, NK):
    '''Plot of mean BIC (Bayesian Information Criteria) and standard deviation.

           Parameters
           ----------
               BIC: BIC values obtained using BIC_calculation function
               NK: maximum number of classes

           Returns
           ------
                Figure showing mean 

               '''
    BICmean = np.mean(BIC, axis=1)
    BICstd = np.std(BIC, axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(
                6, 6), dpi=120, facecolor='w', edgecolor='k')
    ax.plot(np.arange(NK-1) + 1, BICmean, label='BIC mean')
    ax.plot(np.arange(NK-1) + 1, BICmean + BICstd,
                 color=[0.7] * 3, linewidth=0.5, label='BIC std')
    ax.plot(np.arange(NK-1) + 1, BICmean - BICstd, color=[0.7] * 3, linewidth=0.5)
    plt.ylabel('BIC')
    plt.xlabel('Number of classes')
    plt.xticks(np.arange(NK) + 1)
    plt.title('Bayesian information criteria (BIC)')
