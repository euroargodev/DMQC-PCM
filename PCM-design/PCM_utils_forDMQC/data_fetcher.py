# function for data fetching
import xarray as xr
import numpy as np
import pandas as pd

import scipy as sp
from scipy.io import loadmat
from scipy import interpolate

import copy
import struct

def get_refdata(float_mat_path, wmo_boxes, ref_path, config, map_pv_use):
    """ Get data from argo reference database

        Parameters
        ----------
        geo_extent: array with geographical extent [min lon, max lon, min lat, max lat]
        wmo_boxes: wmo_boxes file name
        ref_path: path to argo reference database
        config: dict with config parameter in ow_config.txt file
        map_pv_use: use potential vorticity to calculate ellipses or not

        Returns
        -------
        Dataset with data
    """
    
    # get float trajectory
    mat_dict_float = sp.io.loadmat(float_mat_path)
    # calculate geographical extent
    plus_box = 10 #degrees
    longitude_large = float(config['mapscale_longitude_large'])
    latitude_large = float(config['mapscale_latitude_large'])
    lon_float_180 = np.mod((mat_dict_float['LONG']+180),360)-180
    geo_extent = [lon_float_180.min() - longitude_large - plus_box, 
                  lon_float_180.max() + longitude_large + plus_box, 
                  mat_dict_float['LAT'].min() - latitude_large - plus_box, 
                  mat_dict_float['LAT'].max() + latitude_large + plus_box]
    
    # Read wmo boxes latlon: load txt file
    WMOboxes_latlon = 'PCM_utils_forDMQC/WMO_boxes_latlon.txt'
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


    # drop ptmp variable
    ds = ds.drop('ptmp')
    
    # chose profiles in ellipses
    ds = select_ellipses(mat_dict_float, ds, config, map_pv_use=map_pv_use)
    
    # convert dimension to coordinates
    ds['n_profiles'] = ds.n_profiles.values
    ds['n_pres'] = ds.n_pres.values

    return ds

def add_floatdata(float_WMO, float_mat_path, ds):
    """ Get selected float profiles from .mat file

        Parameters
        ----------
        float_WMO: float reference number
        float_mat_path: path to float mat file
        ds = reference profiles dataset 

        Returns
        -------
        Dataset with float profiles
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
    source_matrix = [(source_matrix[i] + '_' + str(np.squeeze(mat_dict_float['PROFILE_NO'])[i])) for i in range(len(source_matrix))]
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

def get_topo_grid(min_long, max_long, min_lat, max_lat):
    """ Find depth grid over given area using tbase.int file
        The old matlab version of this uses an old .int file from NOAA which contains
        5 arcminute data of global terrain. Whilst other more complete data sets now
        exists, the data files for them are very large, so this file can be used for now,
        and perhaps we can update to better data when we either move the code online, or
        find a way to susbet it/compress it reasonably.
        The .int file is very weird, and stores 16bit integers in binary. The below
        code opens the file and converts the old binary back to the numbers we expect.
        It then finds global terrain over a specified area before closing the file
        Parameters
        ----------
        min_long: minimum longitudinal value for grid
        max_long: maximum longitudinal value for grid
        min_lat: minimum latidunal value for grid
        max_lat: maximum latidunal value for grid
        Returns
        -------
        Matrices containing a uniform grid of latitudes and longitudes, along with the depth at these points
    """
     # check for max lat values
    if max_lat > 90:
        max_lat = 90
    elif min_lat < -90:
        min_lat = -90

    # manipulate input values to match file for decoding
    blat = int(np.max((np.floor(min_lat * 12), -90 * 12 + 1)))
    tlat = int(np.ceil(max_lat * 12))
    llong = int(np.floor(min_long * 12))
    rlong = int(np.ceil(max_long * 12))

    # use these values to form the grid
    lgs = np.arange(llong, rlong + 1, 1) / 12
    lts = np.flip(np.arange(blat, tlat + 1, 1) / 12, axis=0)

    if rlong > 360 * 12 - 1:
        rlong = rlong - 360 * 12
        llong = llong - 360 * 12

    if llong < 0:
        rlong = rlong + 360 * 12
        llong = llong + 360 * 12

    decoder = [llong, rlong, 90 * 12 - blat, 90 * 12 - tlat]

    # Open the binary file
    # TODO: this should use a with statement to avoid holding on to an open handle in the event of an exception
    elev_file = open('/home6/homedir10/perso/agarciaj/EARISE/DMQC-PCM/OWC-pcm/matlabow/lib/m_map1.4/m_map1.4_mod/tbase.int', "rb")  # pylint: disable=consider-using-with
    #elev_file = open(os.path.sep.join([config['CONFIG_DIRECTORY'], "tbase.int"]), "rb")  # pylint: disable=consider-using-with

    if decoder[1] > 4319:
        nlat = int(round(decoder[2] - decoder[3])) + 1  # get the amount of elevation values we need
        nlgr = int(round(decoder[1] - 4320)) + 1
        nlgl = int(4320 - decoder[0])

        # initialise matrix to hold z values
        topo_end = np.zeros((nlat, nlgr))

        # decode the file, and get values
        for i in range(nlat):
            elev_file.seek((i + decoder[3]) * 360 * 12 * 2)
            for j in range(nlgr):
                topo_end[i, j] = struct.unpack('h', elev_file.read(2))[0]

        topo_beg = np.zeros((nlat, nlgl))

        for i in range(nlat):
            elev_file.seek((i + decoder[3]) * 360 * 12 * 2 + decoder[0] * 2)
            for j in range(nlgl):
                topo_beg[i, j] = struct.unpack('h', elev_file.read(2))[0]

        topo = np.concatenate([topo_beg, topo_end], axis=1)
    else:
        # get the amount of elevation values we need
        nlat = int(round(decoder[2] - decoder[3])) + 1
        nlong = int(round(decoder[1] - decoder[0])) + 1

        # initialise matrix to hold z values
        topo = np.zeros((nlat, nlong))

        # decode the file, and get values
        for i in range(nlat):
            elev_file.seek((i + decoder[3]) * 360 * 12 * 2 + decoder[0] * 2)
            for j in range(nlong):
                topo[i, j] = struct.unpack('h', elev_file.read(2))[0]

    # make the grid
    longs, lats = np.meshgrid(lgs, lts)

    # close the file
    elev_file.close()

    return topo, longs, lats

def select_ellipses(mat_dict_float, ds, config, map_pv_use=0):
    """ Select values arround float profiles using an ellipse

        Parameters
        ----------
        mat_dict_float: dictionary in float data
        ds: reference profiles dataset
        config: dictionary with parameters from ow_config.txt
        map_pv_use: use potential vorticity or not

        Returns
        -------
        Dataset with selected profiles 
    """    
    
    longitude_large = float(config['mapscale_longitude_large'])
    latitude_large = float(config['mapscale_latitude_large'])
    phi_large = float(config['mapscale_phi_large'])
    
    long_vector = np.array(ds['long'].values)
    lat_vector = np.array(ds['lat'].values)
    long_float = mat_dict_float['LONG'][0]
    lat_float = mat_dict_float['LAT'][0]
    
    # find the depth of the ocean at the float location
    # tbase.int file requires longitudes from 0 to +/-180
    long_float_tbase = copy.deepcopy(long_float)

    long_float_tbase[np.argwhere(long_float_tbase > 180)] -= 360

    # find the depth of the ocean at the float location
    float_elev, float_x, float_y = get_topo_grid(np.amin(long_float_tbase) - 1,
                                                 np.amax(long_float_tbase) + 1,
                                                 np.amin(lat_float) - 1,
                                                 np.amax(lat_float) + 1)
    
    float_interp = interpolate.interp2d(float_x[0, :],
                                        float_y[:, 0],
                                        float_elev,
                                        kind='linear')
    
    #float_z = -float_interp(long_float_tbase, lat_float)[0]
    float_z = -1 * np.vectorize(float_interp)(long_float_tbase, lat_float)
    
    # tbase.int file requires longitudes from 0 to +/-180
    long_vector_tbase = copy.deepcopy(long_vector)

    g_180 = np.argwhere(long_vector_tbase > 180)

    long_vector_tbase[g_180] -= 360

    # find depth of the ocean at historical locations
    grid_elev, grid_x, grid_y = get_topo_grid(np.amin(long_vector_tbase) - 1,
                                              np.amax(long_vector_tbase) + 1,
                                              np.amin(lat_vector) - 1,
                                              np.amax(lat_vector) + 1)

    grid_interp = interpolate.interp2d(grid_x[0], grid_y[:, 0],
                           grid_elev, kind='linear')

    # As a note, the reason we vectorise the function here is because we do not
    # want to compare every longitudinal value to ever latitude. Rather, we simply
    # want to interpolate each pair of longitudes and latitudes.

    grid_z = -1 * np.vectorize(grid_interp)(long_vector_tbase, lat_vector)
    
    # set up potential vorticity
    #potential_vorticity_vec = np.vectorize(potential_vorticity)
    pv_float = 0
    pv_hist = 0

    # if we are using potential vorticity, calculate it
    if map_pv_use == 1:
        pv_float = np.divide(2*7.292*float(10)**-5 * np.sin(lat_float*np.pi/180), float_z)
        pv_hist = np.divide(2*7.292*float(10)**-5 * np.sin(lat_vector*np.pi/180), grid_z)
    
    # if we are in 0-360 longitude
    if np.any(long_vector > 350) & np.any(long_vector < 10):
        long_vector[long_vector < 180] = long_vector[long_vector < 180] +360
        long_float[long_float < 180] = long_float[long_float < 180] +360

    select_profs = np.array([])
    for iprof in range(len(mat_dict_float['PROFILE_NO'][0])): #loop float profiles
        if map_pv_use == 1:
            ellipse = np.sqrt(np.power(long_vector_tbase-long_float_tbase[iprof], 2)/ np.power(longitude_large*3, 2) + 
                          np.power(lat_vector-lat_float[iprof], 2) / np.power(latitude_large*3, 2) +
                         ((pv_float[iprof]-pv_hist) / np.sqrt( pv_float[iprof]**2+np.power(pv_hist, 2) ) / phi_large)**2 )
        else:
            ellipse = np.sqrt(np.power(long_vector - long_float[iprof], 2)/ np.power(longitude_large*3, 2) + 
                       np.power(lat_vector - lat_float[iprof], 2)/ np.power(latitude_large*3, 2))
        
        select_profs = np.append(select_profs, ds['n_profiles'].values[ellipse<1])        
    
    select_profs = np.unique(select_profs).astype(int)
    ds = ds.isel(n_profiles = select_profs)
    
    return ds