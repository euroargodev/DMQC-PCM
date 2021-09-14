# function for classification
import xarray as xr
import numpy as np
import pandas as pd

import scipy as sp
from scipy.io import loadmat
from scipy import interpolate

import seawater as sw

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


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

    # ds_out = ds_out.drop_vars(['n_pres', 'z_levels'])
    ds_out = ds_out[np.sort(ds_out.data_vars)]
    ds_out.attrs = ds.attrs  # Preserve original attributes
    # ds_out.argo._add_history('Interpolated on standard levels')

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
    argo_data = np.any(wmo_boxes[:, 3] == 1)
    ctd_data = np.any(wmo_boxes[:, 1] == 1)

    # look if boxes has data
    wmo_boxes_selec = wmo_boxes[np.isin(wmo_boxes[:, 0], boxes_list), :]
    boxes_list = wmo_boxes_selec[np.logical_or(
        wmo_boxes_selec[:, 1], wmo_boxes_selec[:, 3] == 1), 0]
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

    # load from .mat files
    # files loop
    cnt = 0
    iprofiles = 0
    for ifile in boxes_list:

        print(ref_path + folder + file_str + str(int(ifile)) + '.mat')

        try:
            mat_dict_load = sp.io.loadmat(
                ref_path + folder + file_str + str(int(ifile)) + '.mat')
        except FileNotFoundError:
            print('file not found')
            continue

        # source should be a str list
        new_source = []
        for isource in mat_dict_load['source'][0]:
            new_source.append(isource[0])
        mat_dict_load['source'] = new_source

        # concat
        if cnt == 0:
            mat_dict = mat_dict_load
        else:
            # check pres_levels
            pres_levels_load = np.shape(mat_dict_load['pres'])[0]
            pres_levels = np.shape(mat_dict['pres'])[0]

            if pres_levels_load > pres_levels:
                # add NaNs in mat_dict
                # create nan matrix
                n_prof = np.shape(mat_dict['pres'])[1]
                nan_matrix = np.empty((pres_levels_load - pres_levels, n_prof))
                nan_matrix.fill(np.nan)

                # concat to mat_load
                mat_dict['pres'] = np.concatenate(
                    (mat_dict['pres'], nan_matrix))
                mat_dict['temp'] = np.concatenate(
                    (mat_dict['temp'], nan_matrix))
                mat_dict['ptmp'] = np.concatenate(
                    (mat_dict['ptmp'], nan_matrix))
                mat_dict['sal'] = np.concatenate((mat_dict['sal'], nan_matrix))

            elif pres_levels_load < pres_levels:
                # add NaNs in mat_dict_load
                # create nan matrix
                n_prof = np.shape(mat_dict_load['pres'])[1]
                nan_matrix = np.empty((pres_levels - pres_levels_load, n_prof))
                nan_matrix.fill(np.nan)

                # concat to mat_load
                mat_dict_load['pres'] = np.concatenate(
                    (mat_dict_load['pres'], nan_matrix))
                mat_dict_load['temp'] = np.concatenate(
                    (mat_dict_load['temp'], nan_matrix))
                mat_dict_load['ptmp'] = np.concatenate(
                    (mat_dict_load['ptmp'], nan_matrix))
                mat_dict_load['sal'] = np.concatenate(
                    (mat_dict_load['sal'], nan_matrix))

            # concatenate
            mat_dict['pres'] = np.concatenate(
                (mat_dict['pres'], mat_dict_load['pres']), axis=1)
            mat_dict['temp'] = np.concatenate(
                (mat_dict['temp'], mat_dict_load['temp']), axis=1)
            mat_dict['ptmp'] = np.concatenate(
                (mat_dict['ptmp'], mat_dict_load['ptmp']), axis=1)
            mat_dict['sal'] = np.concatenate(
                (mat_dict['sal'], mat_dict_load['sal']), axis=1)
            mat_dict['source'][len(mat_dict['source'])                                   :] = mat_dict_load['source']
            mat_dict['long'] = np.concatenate(
                (mat_dict['long'], mat_dict_load['long']), axis=1)
            mat_dict['lat'] = np.concatenate(
                (mat_dict['lat'], mat_dict_load['lat']), axis=1)
            mat_dict['dates'] = np.concatenate(
                (mat_dict['dates'], mat_dict_load['dates']), axis=1)

        cnt = cnt+1

        # if ctd + argo
        if ifile == last_file:
            file_str = 'ctd_'
            folder = '/historical_ctd/'

    # convert from dict to xarray
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
             dates=(["n_profiles"], pd.to_datetime(
                 list(map(str, map(int, np.squeeze(mat_dict['dates'])))))),
         ),
         attrs=dict(
             __header__=mat_dict['__header__'],
             __version__=mat_dict['__version__'],
             __globals__=mat_dict['__version__'],
         )
     )

    # chose profiles in geo_extent
    ds = ds.where(np.mod((ds.long+180), 360)-180 >= geo_extent[0], drop=True)
    ds = ds.where(np.mod((ds.long+180), 360)-180 <= geo_extent[1], drop=True)
    ds = ds.where(ds.lat >= geo_extent[2], drop=True)
    ds = ds.where(ds.lat <= geo_extent[3], drop=True)

    # drop ptmp variable
    ds = ds.drop('ptmp')

    # convert dimension to coordinates
    ds['n_profiles'] = ds.n_profiles.values
    ds['n_pres'] = ds.n_pres.values

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

    return ds


def add_floatdata(float_WMO, float_mat_path, ds):
    """ Add selected float profiles to reference daataset

        Parameters
        ----------
        float_WMO: float reference number
        float_mat_path: path to float mat file
        ds: dataset with reference data from get_refdata function

        Returns
        -------
        Dataset including float profiles
    """

    # load float profiles from .mat file
    mat_dict_float = sp.io.loadmat(float_mat_path)

    # delete float profiles in reference dataset
    cnt = 0
    drop_index = []
    for isource in ds['source'].values:
        if str(float_WMO) in isource[0]:
            # print(isource)
            drop_index.append(cnt)
        cnt = cnt + 1
    ds = ds.drop_sel(n_profiles=drop_index)
    ds['n_profiles'] = np.arange(len(ds['n_profiles'].values))

    # create a dataset similar to ds for concatenation

    nan_matrix = np.empty(
        (len(ds['n_pres'].values) - len(mat_dict_float['PRES'][:, 1]), len(mat_dict_float['PRES'][1, :])))
    nan_matrix.fill(np.nan)
    source_matrix = ['selected_float'] * len(mat_dict_float['PRES'][1, :])
    ds_fc = xr.Dataset(
             data_vars=dict(
                 pres=(["n_pres", "n_profiles"], np.concatenate(
                     (mat_dict_float['PRES'], nan_matrix), axis=0)),
                 temp=(["n_pres", "n_profiles"], np.concatenate(
                     (mat_dict_float['TEMP'], nan_matrix), axis=0)),
                 sal=(["n_pres", "n_profiles"], np.concatenate(
                     (mat_dict_float['SAL'], nan_matrix), axis=0)),
                 source=(["n_profiles"], source_matrix),
             ),
             coords=dict(
                 long=(["n_profiles"], np.squeeze(mat_dict_float['LONG'])),
                 lat=(["n_profiles"], np.squeeze(mat_dict_float['LAT'])),
                 dates=(["n_profiles"], pd.to_datetime(
                     list(map(str, map(int, np.squeeze(mat_dict_float['DATES'])))))),
             )
         )

    ds_fc['n_profiles'] = ds_fc.n_profiles.values + len(ds.n_profiles.values)
    ds_fc['n_pres'] = ds_fc.n_pres.values

    # Change lat values from [0-360] to [-180,180]
    # ds_fc.long.values = np.mod((ds_fc.long.values+180),360)-180
    # print(ds_fc)

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


def get_regulargrid_dataset(ds, corr_dist, grid_extent, gridplot=True):

    # random fist point
    latp = np.random.choice(ds['lat'].values, 1, replace=False)
    lonp = np.random.choice(ds['long'].values, 1, replace=False)

    print(grid_extent)
    grid_extent = np.array(grid_extent)

    # remapping
    grid_lats, grid_lons = mapping_corr_dist(
                corr_dist=corr_dist, start_point=np.concatenate((lonp, latp)), grid_extent=grid_extent)

    # dataset with profiles coordinates
    ds_coords = ds[['long', 'lat']]
    ds_coords['long'].values = np.mod((ds_coords['long'].values+180), 360)-180
    print([len(grid_lats), len(grid_lons)])

    # for each point in new grid calculate distance in km and minimun
    new_profiles = np.empty((len(grid_lons)-1, len(grid_lats)-1))
    new_profiles[:] = np.NaN

    for ilat in range(len(grid_lats)-1):
        #print(ilat)
        for ilon in range(len(grid_lons)-1):
            # print(ilon)
            # get profiles only near the point
            ds_coordsi = ds_coords

            # select data in grid
            ds_coordsi = ds_coordsi.where((ds_coordsi.long > grid_lons[ilon]) & (
                ds_coordsi.long < grid_lons[ilon + 1]), drop=True)
            ds_coordsi = ds_coordsi.where((ds_coordsi.lat > grid_lats[ilat]) & (
                ds_coordsi.lat < grid_lats[ilat + 1]), drop=True)
            if ds_coordsi['n_profiles'].values.size == 0:
                #print('no data in grid')
                continue

            # select randon ref profile
            random_prof = np.random.choice(
                ds_coordsi['n_profiles'].values, 1, replace=False)

            # calculate distance to nearest points
            for i in range(20):

                # choose nearest points
                if ilon == 0 & ilat == 0:
                    # first point
                    break
                elif ilon == 0:
                    nearest_profs=[new_profiles[ilon+1, ilat-1],
                        new_profiles[ilon, ilat-1]]
                elif ilat == 0:
                    nearest_profs=[new_profiles[ilon-1, ilat]]
                elif ilon == len(grid_lons)-2 or ilat == len(grid_lats)-2:
                    nearest_profs=[new_profiles[ilon-1, ilat],
                        new_profiles[ilon, ilat-1], new_profiles[ilon-1, ilat-1]]
                else:
                    nearest_profs=[new_profiles[ilon-1, ilat], new_profiles[ilon+1,
                        ilat-1], new_profiles[ilon, ilat-1], new_profiles[ilon-1, ilat-1]]

                # check if nearest profile is nan
                nearest_profs=np.array(nearest_profs)
                if np.all(np.isnan(nearest_profs)):
                    #print('all nearest profiles are Nan')
                    break
                elif np.any(np.isnan(nearest_profs)):
                    nearest_profs=nearest_profs[np.logical_not(
                        np.isnan(nearest_profs))]

                nearest_profs=np.array(nearest_profs).astype(int)
                # print(nearest_profs)

                distances=np.array([])
                for iprof in nearest_profs:
                    idistance=sw.dist([ds_coords['lat'].sel(n_profiles=random_prof).values[0], ds_coords['lat'].sel(n_profiles=iprof).values],
                                        [ds_coords['long'].sel(n_profiles=random_prof).values[0], ds_coords['long'].sel(n_profiles=iprof).values])
                    distances=np.append(distances, idistance[0])

                # check if point is farther than correlation
                if np.any(distances < 0.8*corr_dist):
                    # choose another profile and check again
                    random_prof=np.random.choice(
                        ds_coordsi['n_profiles'].values, 1, replace=False)
                else:
                    #print('ok')
                    #print(i)
                    break
                if i == 19:
                    #print('end of loop, all profiles are very near')
                    random_prof=np.NaN

            # get profile number
            new_profiles[ilon, ilat]=random_prof
            # print(new_profiles)

    # convert to 1D vector
    new_profiles=new_profiles.flatten()
    new_profiles=new_profiles[np.logical_not(np.isnan(new_profiles))]
    new_profiles=np.unique(new_profiles)
    new_profiles=new_profiles.astype(int)

    # select profiles in dataset
    ds_rg=ds.sel(n_profiles=new_profiles)

    if gridplot:
        proj=ccrs.PlateCarree()
        subplot_kw={'projection': proj}
        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(
            12, 12), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)

        Mlons, Mlats=np.meshgrid(grid_lons, grid_lats)
        p1=ax.scatter(Mlons, Mlats, s=3, transform=proj, label='grid')
        p1=ax.scatter(ds_rg['long'].values, ds_rg['lat'].values,
                      s=3, transform=proj, label='selected ref data')
        # p2 = ax.plot(ds_p['long'].isel(n_profiles = selected_float_index), ds_p['lat'].isel(n_profiles = selected_float_index),
        #         'ro-', transform=proj, markersize = 3, label = str(float_WMO) + ' float trajectory')

        land_feature=cfeature.NaturalEarthFeature(
            category='physical', name='land', scale='50m', facecolor=[0.9375, 0.9375, 0.859375])
        ax.add_feature(land_feature, edgecolor='black')

        defaults={'linewidth': .5, 'color': 'gray',
            'alpha': 0.5, 'linestyle': '--'}
        gl=ax.gridlines(crs=ax.projection, draw_labels=True, **defaults)
        gl.xlocator=mticker.FixedLocator(np.arange(-180, 180+1, 4))
        gl.ylocator=mticker.FixedLocator(np.arange(-90, 90+1, 4))
        gl.xformatter=LONGITUDE_FORMATTER
        gl.yformatter=LATITUDE_FORMATTER
        gl.xlabel_style={'fontsize': 5}
        gl.ylabel_style={'fontsize': 5}
        gl.xlabels_top=False
        gl.ylabels_right=False

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    return ds_rg
