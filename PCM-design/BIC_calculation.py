# BIC calculation functions file
import xarray as xr
import numpy as np

import pyxpcm
from pyxpcm.models import pcm

import matplotlib.pyplot as plt

import concurrent.futures
from tqdm import tqdm

from datetime import datetime

import warnings


def mapping_corr_dist(corr_dist, start_point, grid_extent):
    '''Remapping longitude/latitude grid using a start point. It creates a new grid from the start point 
       where each point is separated the given correlation distance.

           Parameters
           ----------
               corr_dist: correlation distance
               start_point: latitude and longitude of the start point 
               grid_extent: max and min latitude and longitude of the grid to be remapped 
                    [min lon, max lon, min let, max lat]

           Returns
           ------
               new_lats: new latitude vector with points separeted the correlation distance 
               new_lons: new longitude vector with points separeted the correlation distance

               '''

    # angular distance d/earth's radius (km)
    delta = corr_dist / 6371

    # all in radians (conversion at the end)
    grid_extent = grid_extent * np.pi / 180
    start_point = start_point * np.pi / 180

    ### while loop for lat nord ###
    max_lat = grid_extent[3]
    lat2 = -np.pi / 2
    lat1 = start_point[1]
    # bearing = 0 donc cos(0)=1 and sin(0)=0
    new_lats = [lat1]
    while lat2 < max_lat:
        lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                         np.cos(lat1) * np.sin(delta))
        new_lats.append(lat2)
        lat1 = lat2

    ### while loop for lat sud ###
    min_lat = grid_extent[2]
    lat2 = np.pi / 2
    lat1 = start_point[1]
    # bearing = pi donc cos(pi)=-1 and sin(pi)=0
    while lat2 > min_lat:
        lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) -
                         np.cos(lat1) * np.sin(delta))
        new_lats.append(lat2)
        lat1 = lat2

    new_lats = np.sort(new_lats) * 180 / np.pi

    ### while loop for lon east ###
    max_lon = grid_extent[1]
    lon2 = -np.pi
    lon1 = start_point[0]
    lat1 = start_point[1]
    # bearing = pi/2 donc cos(pi/2)=0 and sin(pi/2)=1
    new_lons = [lon1]
    dlon = np.arctan2(np.sin(delta) * np.cos(lat1),
                      np.cos(delta) - np.sin(lat1) * np.sin(lat1))
    while lon2 < max_lon:
        lon2 = lon1 + dlon
        new_lons.append(lon2)
        lon1 = lon2

    ### while loop for lon west ###
    min_lon = grid_extent[0]
    lon2 = np.pi
    lon1 = start_point[0]
    lat1 = start_point[1]
    # bearing = -pi/2 donc cos(-pi/2)=0 and sin(-pi/2)=-1
    dlon = np.arctan2(-np.sin(delta) * np.cos(lat1),
                      np.cos(delta) - np.sin(lat1) * np.sin(lat1))
    while lon2 > min_lon:
        lon2 = lon1 + dlon
        new_lons.append(lon2)
        lon1 = lon2

    new_lons = np.sort(new_lons) * 180 / np.pi

    return new_lats, new_lons


def mapping_corr_time(corr_time, start_point, time_extent):
    '''Remapping time vector using a start point. It creates a new vector from the start point 
       where elements are separated the time correlation.

           Parameters
           ----------
               corr_time: correlation time
               start_point: date of the start point 
               time_extent: max and min time of the vector to be remapped 
                    [min time, max time]

           Returns
           ------
               new_time: new time vector with points separeted the time correlation

               '''

    # we are supossing that 1 month is 30 days for using np.timedelta64

    ### while loop for bigger dates ###
    max_time = time_extent[1]
    time2 = time_extent[0]
    time1 = start_point[0]
    # bearing = 0 donc cos(0)=1 and sin(0)=0
    new_time = [time1]
    while time2 < max_time:
        time2 = time1 + np.timedelta64(corr_time * 30, 'D')
        new_time.append(time2)
        time1 = time2

    ### while loop for smaller dates ###
    min_time = time_extent[0]
    time2 = time_extent[1]
    time1 = start_point[0]
    # bearing = pi donc cos(pi)=-1 and sin(pi)=0
    while time2 > min_time:
        time2 = time1 - np.timedelta64(corr_time * 30, 'D')
        new_time.append(time2)
        time1 = time2

    new_time = np.sort(new_time)

    return new_time


def BIC_cal(X, k, pcm_features):
    ''' Function that calculates BIC for a number of classes k

            Parameters
            ----------
                X: dataset after preprocessing
                k: number of classes

            Returns
            ------
                BIC: BIC value
                k: number of classes

           '''

    # create model
    m = pcm(K=k + 1, features=pcm_features)
    # fit model
    m._classifier.fit(X)

    # calculate LOG LIKEHOOD
    llh = m._classifier.score(X)

    # calculate Nb of independant parameters to estimate
    # we suppose m._classifier.covariance_type == 'full'
    _, n_features = m._classifier.means_.shape
    cov_params = m._classifier.n_components * \
                 n_features * (n_features + 1) / 2.
    mean_params = n_features * m._classifier.n_components
    Nf = int(cov_params + mean_params + m._classifier.n_components - 1)

    # calculate bic
    N_samples = X.shape[0]
    BIC = (-2 * llh * N_samples + Nf * np.log(N_samples))
    # BIC = m._classifier.bic(X)

    return BIC, k


def BIC_calculation(ds, corr_dist, coords_dict, time_steps, pcm_features, features_in_ds, z_dim, Nrun=10, NK=20):
    '''Calculation of BIC (Bayesian Information Criteria) for a training dataset.
        The calculation is parallelised using ThreadPoolExecutor.

           Parameters
           ----------
               ds: dataset
               corr_dist: correlation distance
               coords_dict: dictionary with coordinates names
                    {'depth': 'depth', 'latitude': 'latitude', 'time': 'time', 'longitude': 'longitude'}
               time_steps: time steps to be taken in to acount (format %Y-%m)
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

    # TODO: latitude and longitude values
    # TODO: automatic detection of variables names
    # TODO: If only one time step?

    # grid extent
    grid_extent = np.array([ds[coords_dict.get('longitude')].values.min(), ds[coords_dict.get('longitude')].values.max(
    ), ds[coords_dict.get('latitude')].values.min(), ds[coords_dict.get('latitude')].values.max()])

    # check time steps
    if len(time_steps) > 1:
        d1 = datetime.strptime(time_steps[0], "%Y-%m")
        d2 = datetime.strptime(time_steps[1], "%Y-%m")
        num_months = abs(d1.year - d2.year) * 12 + abs(d1.month - d2.month)
        # print(num_months)
        if num_months < 6:
            warnings.warn(
                "Chosen time steps are too near, you may not obtain a minimum in the BIC function. If this is the case, please try with more distant time steps")

    # this is the list of arguments to iterate over, for instance nb of classes for a PCM
    class_list = np.arange(0, NK)

    ## Here was the previously nested BIC_cal function

    BIC = np.zeros((NK, Nrun))
    for run in range(Nrun):

        for itime in range(len(time_steps)):  # time loop
            # random fist point
            latp = np.random.choice(ds[coords_dict.get('latitude')].values, 1, replace=False)
            lonp = np.random.choice(ds[coords_dict.get('longitude')].values, 1, replace=False)
            # remapping
            new_lats, new_lons = mapping_corr_dist(
                corr_dist=corr_dist, start_point=np.concatenate((lonp, latp)), grid_extent=grid_extent)

            ds.sel({'latitude': 30, 'longitude': 30}, method='nearest')
            ds_run_i = ds.sel({'latitude': list(new_lats), 'longitude': list(new_lons)}, method='nearest')
            ds_run_i = ds_run_i.sel({'time': time_steps[itime]})

            # change lat and lot dimensions by index to be able to merge the datasets (it is not necessary to have lat lon information)
            n_lati = ds_run_i[coords_dict.get('latitude')].size
            n_loni = ds_run_i[coords_dict.get('longitude')].size
            # concat results
            if itime == 0:
                # change lat and lon to index
                ds_run_i[coords_dict.get('latitude')] = np.arange(0, n_lati)
                ds_run_i[coords_dict.get('longitude')] = np.arange(0, n_loni)
                n_lat = n_lati
                n_lon = n_loni
                ds_run = ds_run_i
            else:
                # change lat and lon to index
                ds_run_i[coords_dict.get('latitude')] = np.arange(n_lat, n_lat + n_lati)
                ds_run_i[coords_dict.get('longitude')] = np.arange(n_lon, n_lon + n_loni)
                n_lat = n_lat + n_lati
                n_lon = n_lon + n_loni
                # concat time steps
                ds_run = xr.concat([ds_run, ds_run_i], dim=coords_dict.get('time'))

        # pre-processing
        # K=4 it is not important, it is only used for preprocess data
        m = pcm(K=4, features=pcm_features)
        X, sampling_dims = m.preprocessing(
            ds_run, features=features_in_ds, dim=z_dim, action='fit')

        # serial computation of BIC
        results = []
        for k in class_list:
            results.append(BIC_cal(X, k, pcm_features))

        results = [r for r in results if r is not None]
        results.sort(key=lambda x: x[1])
        BIC[:, run] = np.array([i[0] for i in results])

    BIC_min = np.argmin(np.mean(BIC, axis=1)) + 1

    return BIC, BIC_min


def plot_BIC(BIC, NK, bic_min=-1):
    '''Plot of mean BIC (Bayesian Information Criteria) and standard deviation.

           Parameters
           ----------
               BIC: BIC values obtained using BIC_calculation function
               NK: maximum number of classes

           Returns
           ------
                Figure showing mean 

               '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), dpi=90)
    BICmean = np.mean(BIC, axis=1)
    BICstd = np.std(BIC, axis=1)
    normBICmean = (BICmean - np.mean(BICmean)) / np.std(BICmean)
    # normBICstd = np.std(normBICmean)
    # plt.plot(np.arange(kmax)+1,(BIC-np.mean(BIC))/np.std(BIC),label='Raw BIC')
    plt.plot(np.arange(NK) + 1, BICmean, label='BIC mean')
    plt.plot(np.arange(NK) + 1, BICmean + BICstd,
             color=[0.7] * 3, linewidth=0.5, label='BIC std')
    plt.plot(np.arange(NK) + 1, BICmean - BICstd, color=[0.7] * 3, linewidth=0.5)
    plt.ylabel('BIC')
    plt.xlabel('Number of classes')
    plt.xticks(np.arange(NK) + 1)
    plt.legend()
    plt.title('Bayesian information criteria (BIC)')
    if bic_min > -1:
        ax.annotate('min = ' + str(bic_min), xy=(bic_min, BICmean[bic_min - 1]), xycoords='data',
                    xytext=(0.8, 0.8), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=3),
                    horizontalalignment='right', verticalalignment='top',
                    )
