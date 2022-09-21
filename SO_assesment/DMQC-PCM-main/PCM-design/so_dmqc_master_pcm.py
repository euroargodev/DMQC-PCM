#!/usr/bin/env python

import os
import sys
import logging
import configparser

from numpy.core._exceptions import _ArrayMemoryError
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import DataNotFound
from PCM_utils_forDMQC.classification import loadReferenceData, applyBIC, \
    applyPCM, setupLogger
from PCM_utils_forDMQC.BIC_calculation import plot_BIC


# config initialisation

MATLAB_DIRECTORY = '/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/'
PCM_CONFIG_FILENAME = MATLAB_DIRECTORY + 'pcm_config.txt'
OW_CONFIG_FILENAME = MATLAB_DIRECTORY + 'ow_config_linux_ctd_argo.txt'

config_parser = configparser.RawConfigParser(comment_prefixes='%')
with open(OW_CONFIG_FILENAME) as f:
    ow_file_content = '[configuration]\n' + f.read()
config_parser.read_string(ow_file_content)
ow_config = config_parser['configuration']

with open(PCM_CONFIG_FILENAME) as f:
    pcm_file_content = '[configuration]\n' + f.read()
config_parser.read_string(pcm_file_content)
pcm_config = config_parser['configuration']

SEPERATOR = '/'
FLOAT_SOURCE_DIRECTORY = ow_config['FLOAT_SOURCE_DIRECTORY']
float_source_directory_list = FLOAT_SOURCE_DIRECTORY.split(sep=SEPERATOR)
FLOAT_SOURCE_RAW = SEPERATOR.join(float_source_directory_list[-4:-1])
FLOAT_SOURCE_ADJUSTED = SEPERATOR.join(float_source_directory_list[-4:-2] + ['adjusted'])

# AWI floats with CTD and ARGO
#float_list = [1900859,6901641,

#float_list = [6901493,1900335]- error started here - NOW RERUN TO CORRECTLY

float_list =[6900750]


class SO_DMQC(object):

    @staticmethod
    def run():

        # process each float in the list of WMO numbers in turn
        for float_WMO in float_list:

            log_file_path = f'logs/{float_WMO}_runtime_log.txt'
            logger_name = f'{float_WMO}_runtime_logger'
            setupLogger(logger_name=logger_name, log_file=log_file_path, level=logging.INFO)
            runtime_logger = logging.getLogger(logger_name)

            info_message = f'starting processing'
            print(info_message + f' WMO number: {float_WMO}')
            runtime_logger.info(info_message)

            # Generating Wong matrix for raw data
            wong_matrix_path = os.path.join(MATLAB_DIRECTORY, FLOAT_SOURCE_RAW,
                                            f'{float_WMO}.mat')
            if not os.path.exists(wong_matrix_path):

                try:
                    ds = ArgoDataFetcher(src='localftp',
                                         local_ftp=pcm_config['GDAC_MIRROR'],
                                         cache=True,
                                         mode='expert').float(float_WMO).load().data

                    ds.argo.create_float_source(MATLAB_DIRECTORY + SEPERATOR + FLOAT_SOURCE_RAW)
                    ds.argo.create_float_source(MATLAB_DIRECTORY + SEPERATOR + FLOAT_SOURCE_ADJUSTED,
                                                force='adjusted')

                except DataNotFound:
                    error_message = 'DataNotFound error: due to QC flags (check netCDF file) - skipping'
                    print(error_message)
                    runtime_logger.info(error_message)
                    continue
                except ValueError:
                    error_message = 'ValueError: check whether WMO is valid'
                    print(error_message + ' - skipping')
                    runtime_logger.info(error_message)
                    continue

                info_message = 'Wong matrix created'
            else:
                info_message = 'Wong matrix already exists'

            print(info_message)
            runtime_logger.info((info_message))

            # define required directories
            pcm_output_directory = ow_config['PCM_DIRECTORY'] + 'output_files/'
            plots_directory = ow_config['PCM_DIRECTORY'] + 'figures/'

            # if the PCM output file already exists skip this float
            # NB that the number of classes in the output file name can vary
            pcm_file_root = f'PCM_classes_{float_WMO}'
            PCM_file_name = [file_name for file_name in os.listdir(pcm_output_directory) \
                    if file_name[:len(pcm_file_root)] == pcm_file_root]
            if PCM_file_name:
                error_message = 'PCM has already been run'
                print(error_message + ' - skipping ')
                runtime_logger.info(error_message)
                continue

            info_message = 'loading reference data'
            print(info_message)
            runtime_logger.info(info_message)

            # Starting the PCM analysis
            ds = loadReferenceData(float_mat_path=wong_matrix_path,
                                   ow_config=ow_config)

            info_message = 'applying BIC'
            print(info_message)
            runtime_logger.info(info_message)

            # apply BIC function to determine most suitable number of classes
            BIC, number_classes = applyBIC(ds=ds,
                                           Nrun=int(pcm_config['NUMBER_RUNS']),
                                           NK=int(pcm_config['NK']),
                                           corr_dist=int(pcm_config['CORR_DISTANCE']),
                                           max_depth=int(pcm_config['MAX_DEPTH']))

            runtime_logger.info(f'>>> classes: {number_classes}')

            # generate the BIC plot
            plot_BIC(BIC=BIC,
                     NK=int(pcm_config['NK']),
                     float_WMO=float_WMO,
                     plots_directory=plots_directory)

            try:
                info_message = 'applying PCM'
                print(info_message)
                runtime_logger.info(info_message)
                pcm_file_path = pcm_output_directory + f'PCM_classes_{float_WMO}_K{number_classes}.txt'
                # run PCM function to calculate the classes and save OWC text file
                applyPCM(ds=ds,
                         float_WMO=float_WMO,
                         float_mat_path=wong_matrix_path,
                         pcm_file_path=pcm_file_path,
                         number_classes=number_classes,
                         corr_dist=int(pcm_config['CORR_DISTANCE']),
                         max_depth=int(pcm_config['MAX_DEPTH']),
                         plots_directory=plots_directory)

                runtime_logger.info('>>> successful')

            except IndexError:
                error_message = 'IndexError: too few profiles in data_fetcher.py to index array'
                print(error_message)
                runtime_logger.info(error_message)
            except _ArrayMemoryError:
                error_message = '_ArrayMemoryError: Unable to allocate sufficient memory'
                print('>>> ' + error_message + ' for numpy array in "pyxpcm/xarray.py"')
                runtime_logger.info(error_message)

            logging.shutdown()


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = SO_DMQC()
    obj.run()
