#function for classification
import xarray as xr
import numpy as np

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

def get_refdata(geo_extent, WMOboxes_latlon, wmo_boxes, ref_path):
    #TODO: explain function
    
    # Read wmo boxes latlon: load txt file
    WMOboxes_latlon = np.loadtxt(WMOboxes_latlon, skiprows=1)
    
    # select boxes
    boo_array = (WMOboxes_latlon[:, 1] >= geo_extent[0]-9) & (
    WMOboxes_latlon[:, 2] <= geo_extent[1]+9) & (
    WMOboxes_latlon[:, 3] >= geo_extent[2]-9) & (
    WMOboxes_latlon[:, 4] <= geo_extent[3]+9)
    boxes_list = WMOboxes_latlon[boo_array, 0]
    
    # Read wmo_boxes.mat
    wmo_boxes = sp.io.loadmat('wmo_boxes.mat')
    wmo_boxes = wmo_boxes.get('la_wmo_boxes')
    
    # look if boxes has data
    wmo_boxes_selec = wmo_boxes[np.isin(wmo_boxes[:,0], boxes_list),:]
    # TODO: only argo data (think on it)
    boxes_list = wmo_boxes_selec[wmo_boxes_selec[:,1]==1, 0]

    #load from .mat files
    #files loop
    cnt = 0
    iprofiles = 0
    for ifile in boxes_list:
        
        mat_dict_load = sp.io.loadmat(ref_path + 'argo_' + str(int(ifile)) + '.mat')
    
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
            mat_dict['source'] = np.concatenate((mat_dict['source'], mat_dict_load['source']), axis=1)
            mat_dict['long'] = np.concatenate((mat_dict['long'], mat_dict_load['long']), axis=1)
            mat_dict['lat'] = np.concatenate((mat_dict['lat'], mat_dict_load['lat']), axis=1)
            mat_dict['dates'] = np.concatenate((mat_dict['dates'], mat_dict_load['dates']), axis=1)
        
        cnt = cnt+1
    
    #convert from dict to xarray
    ds = xr.Dataset(
         data_vars=dict(
             pres=(["n_pres", "n_profiles"], mat_dict['pres']),
             temp=(["n_pres", "n_profiles"], mat_dict['temp']),
             ptmp=(["n_pres", "n_profiles"], mat_dict['ptmp']),
             sal=(["n_pres", "n_profiles"], mat_dict['sal']),
             source=(["n_profiles"], np.squeeze(mat_dict['source'])),
         ),
         coords=dict(
             long=(["n_profiles"], np.squeeze(mat_dict['long'])),
             lat=(["n_profiles"], np.squeeze(mat_dict['lat'])),
             dates=(["n_profiles"], np.squeeze(mat_dict['dates'])),
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
    
    return ds