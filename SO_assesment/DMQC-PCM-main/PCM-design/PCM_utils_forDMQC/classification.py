import numpy as np
import matplotlib
import logging
matplotlib.use('agg')
from pyxpcm.models import pcm
import PCM_utils_forDMQC as pcm_utils

from PCM_utils_forDMQC.data_fetcher_bodc import get_refdata, add_floatdata
from PCM_utils_forDMQC.data_processing import interpolate_standard_levels, get_regulargrid_dataset
from PCM_utils_forDMQC.BIC_calculation import BIC_calculation

PLOT_EXTENSION = 'eps'


def setupLogger(logger_name, log_file, level=logging.INFO):

    l = logging.getLogger(logger_name)

    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(filename=log_file, mode='w')
    fileHandler.setFormatter(fmt=formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)


def loadReferenceData(float_mat_path, ow_config):

    """ Load argo reference database

        Data is selected in the same way as OWC does: ellipses using the longitude
        and latitude scales defined in the OWC configuration file are constructed around
        each float profile. The map_pv_use option makes the selection taking into account the bathymetry

    """

    ds = get_refdata(float_mat_path=float_mat_path,
                     ow_config=ow_config,
                     map_pv_use=0)

    return ds


def applyBIC(ds, Nrun, NK, corr_dist, max_depth):

    """ Interpolate to standard levels
        The training dataset ds_t is interpolated on standard depth levels and the profiles
        that are shallower than the max_depth are excluded.

        Parameters
        ----------
        ds:
        Nrun:
        NK:
        corr_dist:
        max_depth:

        Returns
        ------

    """

    ds = interpolate_standard_levels(ds, std_lev=np.arange(0, max_depth))

    z_dim = 'PRES_INTERPOLATED'
    var_name_mdl = ['temp', 'sal']

    # pcm feature
    z = ds[z_dim]
    pcm_features = {var_name_mdl[0]: z, var_name_mdl[1]: z}

    var_name_ds = ['temp', 'sal']
    # Variable to be fitted {variable name in model: variable name in dataset}
    features_in_ds = {var_name_mdl[0]: var_name_ds[0], var_name_mdl[1]: var_name_ds[1]}

    BIC, BIC_min = BIC_calculation(ds=ds,
                                           corr_dist=corr_dist,
                                           pcm_features=pcm_features,
                                           features_in_ds=features_in_ds,
                                           z_dim=z_dim,
                                           Nrun=Nrun,
                                           NK=NK)
    print(f'>>> number of classes: {BIC_min}')

    return BIC, BIC_min


def applyPCM(ds, float_WMO, float_mat_path, pcm_file_path,
             number_classes, corr_dist, max_depth, plots_directory):

    """ Create training dataset
        For creating the training dataset ds_t, we subsample the initial dataset using
        a correlation distance, that you will provide below. The PCM will define the
        different classes using the vertical similarities in ds_t profiles. The ocean
        exhibits spatial correlations that reduce the real information contained in the
        training dataset. Thus, having a decorrelated dataset to fit (train) the PCM is
        important to obtain meaningful classes.
        Parameters
        ----------
        ds:
        float_WMO:
        float_mat_path:
        number_classes:
        corr_dist:
        max_depth:

        Returns
        ------
    """
    ds_t = get_regulargrid_dataset(ds, corr_dist, season=['all'])

    """ Interpolate to standard levels
        The training dataset ds_t is interpolated on standard depth levels and the profiles
        that are shallower than the max_depth are excluded.
    """

    ds_t = interpolate_standard_levels(ds_t, std_lev=np.arange(0, max_depth))

    """ Create prediction dataset
        All the reference profiles in the initial dataset and the float profiles should be
        classified. The process in which each profile is assigned to a class is called prediction.
        The prediction dataset ds_p is constructed by adding the float profiles to the initial dataset ds.

        Add float data to initial dataset
        --------------------------------

    # Float profiles are read in the float source .mat file used as input by OWC software.
    """

    ds_p = add_floatdata(float_WMO, float_mat_path, ds)

    # Interpolate to standard levels
    # The prediction dataset ds_p is interpolated on standard depth levels and the profiles
    # that are shallower than the max_depth are excluded.

    ds_p = interpolate_standard_levels(ds_p, std_lev=np.arange(0, max_depth))

    # Create and apply a PCM
    # A PCM model is created using the number of classes given as input. Then, the model is
    # trained (fitted) to the training dataset and profiles are classified (predict) in order
    # to optionally produce some useful plots. The model is saved.

    z_dim = 'PRES_INTERPOLATED'
    var_name_mdl = ['temp', 'sal']

    # pcm feature
    z = ds_t[z_dim]
    pcm_features = {var_name_mdl[0]: z, var_name_mdl[1]: z}

    m = pcm(K=number_classes, features=pcm_features, debug=False)

    var_name_ds = ['temp', 'sal']
    # Variable to be fitted {variable name in model: variable name in dataset}
    features_in_ds = {var_name_mdl[0]: var_name_ds[0], var_name_mdl[1]: var_name_ds[1]}

    m.fit(ds_t, features=features_in_ds, dim=z_dim)


    print('>>> saving netCDF file')
    m.to_netcdf(f'models/model_{float_WMO}_K{number_classes}.nc')


    # Prediction of class labels.
    # The trained PCM instance (here called m) contains all the necessary information
    # to classify profiles from the prediction dataset ds_p. Each profile in ds_p will
    # be attributed (predicted) to one of the PCM classes. A new variable PCM_LABELS
    # is created to host this result.
    m.predict(ds_p, features=features_in_ds, inplace=True)

    # Probability of a profile in a class.
    # As the pyxpcm software is using the fuzzy classifier GMM (Gaussian Mixture Model)
    # by default, it is possible to calculate the probability of a profile to belong to
    # a class, also called posterior. This is the first step to determine the robustness
    # of the classification with this PCM, which will be calculated below. A new variable
    # PCM_POST is created.
    m.predict_proba(ds_p, features=features_in_ds, dim=z_dim, inplace=True)

    # Classes quantiles.
    # Class vertical structure can be represented using the quantiles of all profiles
    # corresponding to a class. We advise you to calculate at least the median profile
    # and the 5% and 95% quantiles (q=[0.05, 0.5, 0.95]) to have a minimal representation
    # of the classes but feel free to add other quantiles if you want. A new variable
    # outname=var_name_ds + '_Q' is added to the dataset.
    ds_p = ds_p.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=var_name_ds[0],
                                outname=var_name_ds[0] + '_Q',
                                keep_attrs=True, inplace=True)

    ds_p = ds_p.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=var_name_ds[1],
                                outname=var_name_ds[1] + '_Q',
                                keep_attrs=True, inplace=True)

    # Robustness.
    # The classification robustness is a scaled version of the probability of
    # a profile to belong to a class (i.e. the posteriors) so that the value range
    # is more appropriate to assess the robustness of a classification. A 0 value
    # indicates the PCM is totally unsure of the classification result (all classes
    # are equiprobable), while values close to 1 indicate the PCM is highly confident
    # of the result. Note that this does not prevail over the scientific meaning of
    # the classification, but rather indicates the ability of the PCM to attribute
    # a profile to a specific class with confidence.

    # Two new variables are added to the dataset: PCM_ROBUSTNESS and PCM_ROBUSTNESS_CAT.
    # The 2nd variable is categorical and is based on the IPCC likelihood scale:
    ds_p.pyxpcm.robustness(m, inplace=True)
    ds_p.pyxpcm.robustness_digit(m, inplace=True)

    # Generate output text file including a list of reference profile sources, coordinates and
    # class labels is created. It can be used in the OWC software version including the PCM
    # option to select profiles in the same class as float profile to compute the OWC calibration.
    print('>>> saving classes output file')
    matrix_txt = np.stack(
        ('"' + ds_p['source'].values + '"', ds_p['lat'].values,
         ds_p['long'].values, ds_p['PCM_LABELS'].values), axis=1)
    header = 'source lat long PCM_LABELS'

    f = open(pcm_file_path, 'w+')
    np.savetxt(f, matrix_txt, fmt=['%s', '%.3f', '%.3f', '%.1f'], header=header)
    f.close()

    print('>>> saving PCM plots')

    P = pcm_utils.Plotter_cn(ds=ds_p, m=m, coords_dict={'latitude': 'lat',
                                                'longitude': 'long',
                                                'time': 'dates'},
                          float_WMO=float_WMO)

    for plot in range(10):
        try:

            if plot == 0:
                P.pie_classes(save_fig=f'{plots_directory}/{float_WMO}_classes_pie_chart.{PLOT_EXTENSION}')

            if plot == 1:
                P.temporal_distribution(time_bins='month',
                                        save_fig=f'{plots_directory}{float_WMO}_temporal_distribution_months.{PLOT_EXTENSION}')

            if plot == 2:
                P.temporal_distribution(time_bins='season',
                                        save_fig=f'{plots_directory}{float_WMO}_temporal_distribution_season.{PLOT_EXTENSION}')

            if plot == 3:
                P.vertical_structure(q_variable=var_name_ds[0] + '_Q', sharey=True,
                                     xlabel='Temperature (°C)',
                                     save_fig=f'{plots_directory}{float_WMO}_temperature_quantiles.{PLOT_EXTENSION}')

            if plot == 4:
                P.vertical_structure_comp(q_variable=var_name_ds[0] + '_Q', plot_q='all',
                                          xlabel='Temperature (°C)',
                                          save_fig=f'{plots_directory}{float_WMO}_temperature_quantiles_comp.{PLOT_EXTENSION}')

            if plot == 5:
                P.vertical_structure(q_variable=var_name_ds[1] + '_Q', sharey=True,
                                     xlabel='Salinity (PSU)',
                                     save_fig=f'{plots_directory}{float_WMO}_salinity_quantiles.{PLOT_EXTENSION}')

            if plot == 6:
                P.vertical_structure_comp(q_variable=var_name_ds[1] + '_Q', plot_q='all',
                                          xlabel='Salinity (PSU)',
                                          save_fig=f'{plots_directory}{float_WMO}_salinity_quantiles_comp.{PLOT_EXTENSION}')

            if plot == 7:
                P.spatial_distribution(lonlat_grid=[8, 8],
                                       save_fig=f'{plots_directory}{float_WMO}_spatial_distribution.{PLOT_EXTENSION}')

            if plot == 8:
                P.float_traj_classes(save_fig=f'{plots_directory}{float_WMO}_float_profiles_classes.{PLOT_EXTENSION}')

            if plot == 9:
                P.float_cycles_prob(var_name='PCM_ROBUSTNESS_CAT',
                                save_fig=f'{plots_directory}{float_WMO}_float_profiles_robustness.{PLOT_EXTENSION}')

        except IndexError:
            print(f'>>> IndexError: no reference data marked as "float_selected" - skipping plot {plot + 1}')
            continue
        except TypeError:
            print(f'>>> TypeError:  object of type "numpy.float32" has no len() Plotter.py", line 171 {plot + 1}')
            continue
        except ValueError:
            continue

        print(f'>>> saved plot {plot + 1}')


def printError(message):
    print('*******')
    # print (f'ERROR     \u2620  {message}')
    print(f'ERROR     XXX  {message}')
    print('*******')
    import sys
    sys.exit()