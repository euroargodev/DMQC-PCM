#!/usr/bin/env python
import os
import sys
import logging
import configparser
from numpy.core._exceptions import _ArrayMemoryError
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import DataNotFound, NetCDF4FileNotFoundError
from argodmqc_pcm.PCM_utils_forDMQC.classification import loadReferenceData, applyBIC, \
    applyPCM, setupLogger
from argodmqc_pcm.PCM_utils_forDMQC.BIC_calculation import plot_BIC
from argodmqc_owc.pyowc import calibration, configuration, plot


float_list = [3900070]  # run all the way through


class SO_DMQC(object):

    def __init__(self, config_file):

        configuration = configparser.ConfigParser()
        configuration.read(config_file)

        self.PCM_CONFIG = configuration['PCM']
        self.OWC_CONFIG = configuration['OWC']

        self.LOGS_DIR = configuration['OUTPUT DIRECTORIES']['LOGS_DIR']
        self.DAC_COMP_DIR = configuration['OUTPUT DIRECTORIES']['DAC_COMP_DIR']

        self.FLOAT_SOURCE_RAW = self.OWC_CONFIG['FLOAT_SOURCE_DIRECTORY']
        self.FLOAT_SOURCE_ADJUSTED = self.FLOAT_SOURCE_RAW.replace('default', 'adjusted')


    def run(self):

        # process each float in the list of WMO numbers in turn
        for float_WMO in float_list:

            run_PCM_flag = True

            log_file_path = f'{self.LOGS_DIR}{float_WMO}_runtime_log.txt'

            logger_name = f'{float_WMO}_runtime_logger'
            setupLogger(logger_name=logger_name, log_file=log_file_path, level=logging.INFO)
            runtime_logger = logging.getLogger(logger_name)

            info_message = f'starting processing'
            print(info_message + f' WMO number: {float_WMO}')
            runtime_logger.info(info_message)

            # Generating Wong matrix for raw data
            wong_matrix_path = os.path.join(self.FLOAT_SOURCE_RAW,
                                            f'{float_WMO}.mat')

            if not os.path.exists(wong_matrix_path):

                try:
                    ds = ArgoDataFetcher(src='localftp',
                                         local_ftp=self.PCM_CONFIG['GDAC_MIRROR'],
                                         cache=True,
                                         mode='expert').float(float_WMO).load().data

                    ds.argo.create_float_source(self.FLOAT_SOURCE_RAW)
                    ds.argo.create_float_source(self.FLOAT_SOURCE_ADJUSTED, force='adjusted')

                except DataNotFound:
                    error_message = 'XXX DataNotFound error: due to QC flags (check netCDF file) - skipping'
                    print(error_message)
                    runtime_logger.info(error_message)
                    logging.shutdown()
                    continue
                except ValueError:
                    error_message = 'XXX ValueError: check whether WMO is valid'
                    print(error_message + ' - skipping')
                    runtime_logger.info(error_message)
                    logging.shutdown()
                    continue
                except NetCDF4FileNotFoundError:
                    error_message = 'XXX NetCDF4FileNotFoundError: check whether WMO is valid'
                    print(error_message + ' - skipping')
                    runtime_logger.info(error_message)
                    logging.shutdown()
                    continue

                info_message = 'Wong matrix created'
            else:
                info_message = 'Wong matrix already exists'

            print(info_message)
            runtime_logger.info(info_message)

            # if the PCM output file already exists skip this float
            # NB that the number of classes in the output file name can vary
            pcm_file_root = f'PCM_classes_{float_WMO}'
            PCM_file_name = [file_name for file_name in os.listdir(self.PCM_CONFIG['CLASSES_DIR']) \
                    if file_name[:len(pcm_file_root)] == pcm_file_root]
            if PCM_file_name:
                run_PCM_flag = False
                if int(self.OWC_CONFIG['USE_PCM']):
                    error_message = 'PCM has already been run'
                    print(error_message + ' - skipping ')
                    runtime_logger.info(error_message)

            if run_PCM_flag and int(self.OWC_CONFIG['USE_PCM']):
                info_message = 'applying PCM'
                print(info_message)
                runtime_logger.info(info_message)
                info_message = '>>> loading reference data'
                print(info_message)
                runtime_logger.info(info_message)

                # Starting the PCM analysis
                ds = loadReferenceData(float_mat_path=wong_matrix_path,
                                       ow_config=self.OWC_CONFIG)

                info_message = '>>> applying BIC'
                print(info_message)
                runtime_logger.info(info_message)

                # apply BIC function to determine most suitable number of classes
                BIC, number_classes = applyBIC(ds=ds,
                                               Nrun=int(self.PCM_CONFIG['NUMBER_RUNS']),
                                               NK=int(self.PCM_CONFIG['NK']),
                                               corr_dist=int(self.PCM_CONFIG['CORR_DISTANCE']),
                                               max_depth=int(self.PCM_CONFIG['MAX_DEPTH']))

                runtime_logger.info(f'>>> classes: {number_classes}')

                # generate the BIC plot
                plot_BIC(BIC=BIC,
                         NK=int(self.PCM_CONFIG['NK']),
                         float_WMO=float_WMO,
                         plots_dir=self.PCM_CONFIG['PLOTS_DIR'])
                runtime_logger.info('>>> successful')

                try:
                    info_message = '>>> classifying'
                    print(info_message)
                    runtime_logger.info(info_message)
                    pcm_file_path = self.PCM_CONFIG['CLASSES_DIR'] +\
                                    f'PCM_classes_{float_WMO}_K{number_classes}.txt'
                    # run PCM function to calculate the classes and save OWC text file
                    applyPCM(ds=ds,
                             float_WMO=float_WMO,
                             float_mat_path=wong_matrix_path,
                             pcm_file_path=pcm_file_path,
                             number_classes=number_classes,
                             corr_dist=int(self.PCM_CONFIG['CORR_DISTANCE']),
                             max_depth=int(self.PCM_CONFIG['MAX_DEPTH']),
                             plots_dir=self.PCM_CONFIG['PLOTS_DIR'],
                             models_dir=self.PCM_CONFIG['MODELS_DIR'],
                             )

                    runtime_logger.info('>>> successful')

                except IndexError:
                    error_message = 'XXX IndexError: too few profiles in data_fetcher.py to index array'
                    print(error_message)
                    runtime_logger.info(error_message)
                except _ArrayMemoryError:
                    error_message = 'XXX _ArrayMemoryError: Unable to allocate sufficient memory'
                    print(error_message + ' for numpy array in "pyxpcm/xarray.py"')
                    runtime_logger.info(error_message)

            info_message = 'applying OWC'
            print(info_message)
            runtime_logger.info(info_message)
            try:
                info_message = '>>> updating salinity mapper'
                print(info_message)
                runtime_logger.info(info_message)
                calibration.update_salinity_mapping(str(float_WMO), self.OWC_CONFIG,
                                                    self.PCM_CONFIG['CLASSES_DIR'])
                runtime_logger.info('>>> successful')

            except FileNotFoundError:
                error_message = 'XXX file not found - check float source'
                print(error_message)
                runtime_logger.info(error_message)
                continue
            except RuntimeError:
                error_message = 'XXX "NO DATA FOUND" - most likely calibration.get_region_data'
                print(error_message)
                runtime_logger.info(error_message)
                continue

            try:
                info_message = '>>> setting cal series'
                print(info_message)
                runtime_logger.info(info_message)
                configuration.set_calseries(str(float_WMO), self.OWC_CONFIG)
                runtime_logger.info('>>> successful')

            except FileNotFoundError:
                error_message = 'XXX file not found - check float mapped'
                print(error_message)
                runtime_logger.info(error_message)
                continue

            try:
                info_message = '>>> calculating piecewise fit'
                print(info_message)
                runtime_logger.info(info_message)
                fit_type = calibration.calc_piecewisefit(str(float_WMO), self.OWC_CONFIG)
                runtime_logger.info(f'>>> fit type {fit_type}')
                runtime_logger.info('>>> successful')

            except AttributeError:
                error_message = 'XXX most likely no good data was found in core.stats.fit_cond'
                print(error_message)
                runtime_logger.info(error_message)
                continue
            except ValueError:
                error_message = 'XXX issue fitting with breaks in pyowc.core.stats.fit_cond'
                print(error_message)
                runtime_logger.info(error_message)
                continue

            try:
                info_message = '>>> generating plots'
                print(info_message)
                runtime_logger.info(info_message)
                plot.dashboard(str(float_WMO), self.OWC_CONFIG)
                runtime_logger.info('>>> successful')

            except FileNotFoundError:
                error_message = 'XXX file not found - either float source, mapped or calibrated'
                print(error_message)
                runtime_logger.info(error_message)

            logging.shutdown()


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = SO_DMQC('pcm_owc_config.ini')
    obj.run()
