#function for classification
import xarray as xr
import numpy as np
import pandas as pd

import scipy as sp
from scipy.io import loadmat
from scipy import interpolate

def interpolate_standard_levels(ds, std_lev):
    """ Interpolate data to given presure standard levels (from interp_std_levels in argopy)
    
        Parameters
        ----------
        ds: dataset to be interpolated
        std_lev: array of pres standard values
        
        Returns
        -------
        Datset array with interpolated values
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

    #ds_out = ds_out.drop_vars(['n_pres', 'z_levels'])
    ds_out = ds_out[np.sort(ds_out.data_vars)]
    ds_out.attrs = ds.attrs  # Preserve original attributes
    #ds_out.argo._add_history('Interpolated on standard levels')

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
            raise RuntimeError("if z_dim is not specified,x must be a 1D array.")
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

def get_refdata(geo_extent, WMOboxes_latlon, wmo_boxes, ref_path, season='all'):
    """ Get data from argo reference database
    
        Parameters
        ----------
        geo_extent: array with geographical extent [min lon, max lon, min lat, max lat]
        WMOboxes_latlon: WMOboxes_latlon file name
        wmo_boxes: wmo_boxes file name
        ref_path: path to argo reference database
        
        Returns
        -------
        Dataset with data
    """
    
    # Read wmo boxes latlon: load txt file
    WMOboxes_latlon = np.loadtxt(WMOboxes_latlon, skiprows=1)
    
    # select boxes
    boo_array = (WMOboxes_latlon[:, 1] >= geo_extent[0]-9) & (
    WMOboxes_latlon[:, 2] <= geo_extent[1]+9) & (
    WMOboxes_latlon[:, 3] >= geo_extent[2]-9) & (
    WMOboxes_latlon[:, 4] <= geo_extent[3]+9)
    boxes_list = WMOboxes_latlon[boo_array, 0]
    
    # Read wmo_boxes.mat
    wmo_boxes = sp.io.loadmat(wmo_boxes)
    wmo_boxes = wmo_boxes.get('la_wmo_boxes')
    
    # check if CTD, argo or both
    argo_data = np.any(wmo_boxes[:,3]==1)
    ctd_data = np.any(wmo_boxes[:,1]==1)
    
    # look if boxes has data
    wmo_boxes_selec = wmo_boxes[np.isin(wmo_boxes[:,0], boxes_list),:]
    boxes_list = wmo_boxes_selec[np.logical_or(wmo_boxes_selec[:,1],wmo_boxes_selec[:,3]==1), 0]
    last_file = boxes_list[-1]
    
    if argo_data & ctd_data:
        # duplicate boxes list
        boxes_list = np.concatenate((boxes_list, boxes_list))
        # start with argo data
        file_str = 'argo_'
        folder = '/argo_profiles/'
    elif argo_data:
        file_str = 'argo_'
        folder = '/argo_profiles/'
    elif ctd_data:
        file_str = 'ctd_'
        folder = '/historical_ctd/'

    #load from .mat files
    #files loop
    cnt = 0
    iprofiles = 0
    for ifile in boxes_list:

        print(ref_path + folder + file_str + str(int(ifile)) + '.mat')
        
        try:
            mat_dict_load = sp.io.loadmat(ref_path + folder + file_str + str(int(ifile)) + '.mat')
        except FileNotFoundError:
            print('file not found')
            continue
        
        # source should be a str list
        new_source = []
        for isource in mat_dict_load['source'][0]:
            new_source.append(isource[0])
        mat_dict_load['source'] = new_source
        
        #concat
        if cnt == 0:
            mat_dict = mat_dict_load
        else:
            #check pres_levels
            pres_levels_load = np.shape(mat_dict_load['pres'])[0]
            pres_levels = np.shape(mat_dict['pres'])[0]
        
            if pres_levels_load > pres_levels:
                # add NaNs in mat_dict
                #create nan matrix
                n_prof = np.shape(mat_dict['pres'])[1]
                nan_matrix = np.empty((pres_levels_load - pres_levels, n_prof))
                nan_matrix.fill(np.nan)
            
                #concat to mat_load
                mat_dict['pres'] = np.concatenate((mat_dict['pres'], nan_matrix))
                mat_dict['temp'] = np.concatenate((mat_dict['temp'], nan_matrix))
                mat_dict['ptmp'] = np.concatenate((mat_dict['ptmp'], nan_matrix))
                mat_dict['sal'] = np.concatenate((mat_dict['sal'], nan_matrix))
            
            elif pres_levels_load < pres_levels:
                # add NaNs in mat_dict_load
                #create nan matrix
                n_prof = np.shape(mat_dict_load['pres'])[1]
                nan_matrix = np.empty((pres_levels - pres_levels_load, n_prof))
                nan_matrix.fill(np.nan)
            
                #concat to mat_load
                mat_dict_load['pres'] = np.concatenate((mat_dict_load['pres'], nan_matrix))
                mat_dict_load['temp'] = np.concatenate((mat_dict_load['temp'], nan_matrix))
                mat_dict_load['ptmp'] = np.concatenate((mat_dict_load['ptmp'], nan_matrix))
                mat_dict_load['sal'] = np.concatenate((mat_dict_load['sal'], nan_matrix))
            
            #concatenate
            mat_dict['pres'] = np.concatenate((mat_dict['pres'], mat_dict_load['pres']), axis=1)
            mat_dict['temp'] = np.concatenate((mat_dict['temp'], mat_dict_load['temp']), axis=1)
            mat_dict['ptmp'] = np.concatenate((mat_dict['ptmp'], mat_dict_load['ptmp']), axis=1)
            mat_dict['sal'] = np.concatenate((mat_dict['sal'], mat_dict_load['sal']), axis=1)
            #mat_dict['source'] = np.concatenate((mat_dict['source'], mat_dict_load['source']), axis=1)
            mat_dict['source'][len(mat_dict['source']):] = mat_dict_load['source']
            mat_dict['long'] = np.concatenate((mat_dict['long'], mat_dict_load['long']), axis=1)
            mat_dict['lat'] = np.concatenate((mat_dict['lat'], mat_dict_load['lat']), axis=1)
            mat_dict['dates'] = np.concatenate((mat_dict['dates'], mat_dict_load['dates']), axis=1)
        
        cnt = cnt+1
        
        # if ctd + argo
        if ifile == last_file:
            file_str = 'ctd_'
            folder = '/historical_ctd/'
    
    #convert from dict to xarray
    ds = xr.Dataset(
         data_vars=dict(
             pres=(["n_pres", "n_profiles"], mat_dict['pres']),
             temp=(["n_pres", "n_profiles"], mat_dict['temp']),
             ptmp=(["n_pres", "n_profiles"], mat_dict['ptmp']),
             sal=(["n_pres", "n_profiles"], mat_dict['sal']),
             source=(["n_profiles"], mat_dict['source']),
         ),
         coords=dict(
             long=(["n_profiles"], np.squeeze(mat_dict['long'])),
             lat=(["n_profiles"], np.squeeze(mat_dict['lat'])),
             dates=(["n_profiles"], pd.to_datetime(list(map(str, map(int, np.squeeze(mat_dict['dates'])))))),
         ),
         attrs=dict(
             __header__=mat_dict['__header__'],
             __version__=mat_dict['__version__'],
             __globals__=mat_dict['__version__'],
         )
     )
    
    
    # Change lat values from [0-360] to [-180,180]
    ds.long.values = np.mod((ds.long.values+180),360)-180
    
    # chose profiles in geo_extent
    ds = ds.where(ds.long >= geo_extent[0], drop = True)
    ds = ds.where(ds.long <= geo_extent[1], drop = True)
    ds = ds.where(ds.lat >= geo_extent[2], drop = True)
    ds = ds.where(ds.lat <= geo_extent[3], drop = True)
    
    # drop ptmp variable
    ds=ds.drop('ptmp')
    
    # convert dimension to coordinates
    ds['n_profiles'] = ds.n_profiles.values
    ds['n_pres'] = ds.n_pres.values
    
    #choose season
    if 'all' not in season:
        season_idxs = ds.groupby('dates.season').groups
        
        season_select = []
        for key in season:
            season_select = np.concatenate((season_select, np.squeeze(season_idxs.get(key))))
            
        if len(season) == 1:
            season_select = np.array(season_select)
            
        season_select = np.sort(season_select.astype(int))
        ds = ds.isel(n_profiles = season_select)
    
    return ds

def add_floatdata(float_WMO, ds):
    """ Add selected float profiles to reference daataset
    
        Parameters
        ----------
        float_WMO: float number
        ds: dataset with reference data from get_refdata function
        
        Returns
        -------
        Dataset including float profiles
    """
    
    # load float profiles using argopy with option localftp
    import argopy
    argopy.set_options(src='localftp', local_ftp='/home/coriolis_exp/spool/co05/co0508/')
    from argopy import DataFetcher as ArgoDataFetcher
    argo_loader = ArgoDataFetcher()
    
    ds_f = argo_loader.float([float_WMO]).to_xarray()
    ds_f = ds_f.argo.point2profile()
    #print(ds_f)
    
    #delete float profiles in reference dataset
    cnt = 0
    drop_index = []
    for isource in ds['source'].values:
        if str(float_WMO) in isource[0]:
            #print(isource)
            drop_index.append(cnt)
        cnt = cnt +1
    ds = ds.drop_sel(n_profiles = drop_index)
    ds['n_profiles'] = np.arange(len(ds['n_profiles'].values))
    
    # create a dataset similar to ds for concatenation
    nan_matrix = np.empty((len(ds_f['N_PROF'].values), len(ds['n_pres'].values) - len(ds_f['N_LEVELS'].values)))
    nan_matrix.fill(np.nan)
    source_matrix = ['selected_float'] * len(ds_f['N_PROF'].values)
    ds_fc = xr.Dataset(
             data_vars=dict(
                 pres=(["n_profiles", "n_pres"], np.concatenate((ds_f['PRES'].values, nan_matrix),axis=1)),
                 temp=(["n_profiles", "n_pres"], np.concatenate((ds_f['TEMP'].values, nan_matrix),axis=1)),
                 sal=(["n_profiles", "n_pres"], np.concatenate((ds_f['PSAL'].values, nan_matrix),axis=1)),
                 source=(["n_profiles"], source_matrix),
             ),
             coords=dict(
                 long=(["n_profiles"], ds_f['LONGITUDE'].values),
                 lat=(["n_profiles"], ds_f['LATITUDE'].values),
                 dates=(["n_profiles"], ds_f['TIME'].values),
             )
         )
    ds_fc['n_profiles'] = ds_fc.n_profiles.values + len(ds.n_profiles.values)
    ds_fc['n_pres'] = ds_fc.n_pres.values
    #print(ds_fc)
    
    # combine datasets
    ds_out = xr.combine_by_coords([ds, ds_fc])
    
    return ds_out

def order_class_names(ds_out, K):
    """ Rename class from south to nord to have always the same class name for the same dataset
    
        Parameters
        ----------
        ds_out: dataset including PCM_LABELS variable
        K: number of classes
        
        Returns
        -------
        Dataset ordered class names in PCM_LABELS variable
    """

    def assign_cluster_number(x):
        if x==x:
            return float(order[int(x)])
        else:
            return x

    max_lat_class = []
    for ik in range(K):
        max_lat_class.append(np.nanmax(ds_out['lat'].where(ds_out['PCM_LABELS'] == ik)))
    order = np.argsort(max_lat_class)
    
    vfunc = np.vectorize(assign_cluster_number)
    ds_out['PCM_LABELS'].values = vfunc(ds_out['PCM_LABELS'].values)
    
    return ds_out