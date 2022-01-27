# function for data processing
import xarray as xr
import numpy as np

import scipy as sp
from scipy import interpolate


def interpolate_standard_levels(ds, std_lev):
    """ Interpolate data to given presure standard levels (from interp_std_levels in argopy)

        Parameters
        ----------
        ds: dataset to be interpolated
        std_lev: array of pres standard values

        Returns
        -------
        :class:`xarray.DataArray`
            Dataset array with interpolated values
    """

    # Selecting profiles that have a max(pressure) > max(std_lev) to avoid extrapolation in that direction
    # For levels < min(pressure), first level values of the profile are extended to surface.
    i1 = (ds['pres'].max('n_pres') >= std_lev[-1])
    ds = ds.where(i1, drop=True)

    # check if any profile is left, ie if any profile match the requested depth
    if (len(ds['n_profiles']) == 0):
        raise Warning(
                'None of the profiles can be interpolated (not reaching the requested depth range).')
        return None

    # add new vertical dimensions, this has to be in the datasets to apply ufunc later
    ds['z_levels'] = xr.DataArray(std_lev, dims={'z_levels': std_lev})

    # init
    ds_out = xr.Dataset()

    # vars to interpolate
    datavars = [dv for dv in list(ds.variables) if set(['n_pres', 'n_profiles']) == set(
            ds[dv].dims) and 'ptmp' not in dv]
    # coords
    coords = [dv for dv in list(ds.coords)]
    # vars depending on N_PROF only
    solovars = [dv for dv in list(
            ds.variables) if dv not in datavars and dv not in coords and 'QC' not in dv and 'ERROR' not in dv]
    for dv in datavars:
        ds_out[dv] = linear_interpolation_remap(
                ds.pres, ds[dv], ds['z_levels'], z_dim='n_pres', z_regridded_dim='z_levels')
    ds_out = ds_out.rename({'remapped': 'PRES_INTERPOLATED'})
    for sv in solovars:
        ds_out[sv] = ds[sv]

    for co in coords:
        ds_out.coords[co] = ds[co]

    # ds_out = ds_out.drop_vars(['n_pres', 'z_levels'])
    ds_out = ds_out[np.sort(ds_out.data_vars)]
    ds_out.attrs = ds.attrs  # Preserve original attributes
    # ds_out.argo._add_history('Interpolated on standard levels')
    
    # some format
    #pres should be negative for the PCM
    ds_out['PRES_INTERPOLATED'] = -np.abs(ds_out['PRES_INTERPOLATED'].values)
    #axis attributtes for plotter class
    ds_out.PRES_INTERPOLATED.attrs['axis'] = 'Z'
    ds_out.lat.attrs['axis'] = 'Y'
    ds_out.long.attrs['axis'] = 'X'
    ds_out.dates.attrs['axis'] = 'T'

    return ds_out


def linear_interpolation_remap(
    z, data, z_regridded, z_dim=None, z_regridded_dim="regridded", output_dim="remapped"
):

    # interpolation called in xarray ufunc
    def _regular_interp(x, y, target_values):
        # remove all nans from input x and y
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~idx]
        y = y[~idx]

        # Need at least 5 points in the profile to interpolate, otherwise, return NaNs
        if len(y) < 5:
            interpolated = np.empty(len(target_values))
            interpolated[:] = np.nan
        else:
            # replace nans in target_values with out of bound Values (just in case)
            target_values = np.where(
                ~np.isnan(target_values), target_values, np.nanmax(x) + 1
            )
            # Interpolate with fill value parameter to extend min pressure toward 0
            interpolated = interpolate.interp1d(
                x, y, bounds_error=False, fill_value=(y[0], y[-1])
            )(target_values)
        return interpolated

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError(
                "if z_dim is not specified,x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a dataset input just apply the function for each appropriate array

    kwargs = dict(
        input_core_dims=[[dim], [dim], [z_regridded_dim]],
        output_core_dims=[[output_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
        output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
    )
    remapped = xr.apply_ufunc(_regular_interp, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped

def order_class_names(ds_out, K):
    """ Rename class from south to nord to have always the same class name for the same dataset

        Parameters
        ----------
        ds_out: dataset including PCM_LABELS variable
        K: number of classes

        Returns
        -------
        :class:`xarray.DataArray`
            Dataset ordered class names in PCM_LABELS variable
    """

    def assign_cluster_number(x):
        if x == x:
            return float(order[int(x)])
        else:
            return x

    max_lat_class = []
    for ik in range(K):
        max_lat_class.append(
            np.nanmax(ds_out['lat'].where(ds_out['PCM_LABELS'] == ik)))
    order = np.argsort(max_lat_class)

    vfunc = np.vectorize(assign_cluster_number)
    ds_out['PCM_LABELS'].values = vfunc(ds_out['PCM_LABELS'].values)

    return ds_out

def cal_dist_matrix(lats, lons):
    '''Calculate distance matrix

           Parameters
           ----------
               lats: latitude vector
               lons: longitude vector

           Returns
           ------
               Distance maytrix in int16

               '''    
    from sklearn.metrics.pairwise import haversine_distances
    from math import radians
    
    lats_in_radians = np.array([radians(_) for _ in lats])
    lons_in_radians = np.array([radians(_) for _ in lons])
    coords_in_radians = np.column_stack((lats_in_radians, lons_in_radians))
    dist_matrix = haversine_distances(coords_in_radians).astype(np.float32)
    dist_matrix = dist_matrix * 6371  # multiply by Earth radius to get kilometers
    dist_matrix = dist_matrix.astype(np.int16)
    
    return dist_matrix

def get_regulargrid_dataset(ds, corr_dist, season='all'):
    '''Re-sampling od the dataset selecting profiles separated the correlation distance

           Parameters
           ----------
               ds: reference profiles dataset
               corr_dist: correlation distance
               season: choose season: 'DJF', 'MAM', 'JJA','SON' (default: 'all')

           Returns
           -------
           :class:`xarray.DataArray`
               Re-sampled dataset

               '''
    
    ds['n_profiles'] = np.arange(len(ds['n_profiles']))
    # create mask
    mask_s = np.empty((1,len(ds['n_profiles'].values)))
    mask_s[:] = np.NaN
    ds["mask_s"]=(['n_profiles'],  np.squeeze(mask_s))
    
    plus_degrees = corr_dist/111 +1 # from km to degrees
    
    #loop
    n_iterations = range(len(ds['n_profiles'].values))

    for i in n_iterations:
        
        # choose random profile
        random_p = np.random.choice(ds['n_profiles'].where(np.isnan(ds['mask_s']), drop=True).values, 1, replace=False)
        random_p = int(random_p[0])
        lat_p = ds['lat'].sel(n_profiles = random_p).values
        long_p = ds['long'].sel(n_profiles = random_p).values
        
        # dataset arround random point
        ds_slice = ds[['lat', 'long', 'mask_s']]
        ds_slice = ds_slice.where(ds['lat'] > (lat_p - plus_degrees), drop=True)
        ds_slice = ds_slice.where(ds_slice['lat'] < (lat_p + plus_degrees), drop=True)
        ds_slice = ds_slice.where(ds_slice['long'] > (long_p - plus_degrees), drop=True)
        ds_slice = ds_slice.where(ds_slice['long'] < (long_p + plus_degrees), drop=True)
        random_p_i = np.argwhere(ds_slice['n_profiles'].values == random_p)
        
        # calculate distance matrix
        dist_matrix = cal_dist_matrix(ds_slice['lat'].values, ds_slice['long'].values)
        
        # points near than corr_dist = 1
        mask_dist = np.isnan(ds_slice['mask_s'].values)*1
        dist_vector = np.array(np.squeeze(dist_matrix[:,random_p_i])).astype('float')*np.array(mask_dist)
        dist_vector[dist_vector == 0] = np.NaN
        bool_near_points = (dist_vector < corr_dist)
        n_profiles_near_points = ds_slice['n_profiles'].values[bool_near_points]
        
        # change mask
        ds['mask_s'][random_p] = 1
        ds['mask_s'][n_profiles_near_points] = 0
        
        # stop condition
        #print(sum(np.isnan(ds['mask_s'].values)))
        if np.any(np.isnan(ds['mask_s'])) == False:
            #print('no more points to delate')
            #print(i)
            break
        
                            
    # choose season
    if 'all' not in season:
        season_idxs = ds.groupby('dates.season').groups

        season_select = []
        for key in season:
            season_select = np.concatenate(
                (season_select, np.squeeze(season_idxs.get(key))))

        if len(season) == 1:
            season_select = np.array(season_select)

        season_select = np.sort(season_select.astype(int))
        ds = ds.isel(n_profiles=season_select)
    
    ds_t = ds.where(ds['mask_s']== 1, drop=True)
    
    del dist_matrix

    return ds_t