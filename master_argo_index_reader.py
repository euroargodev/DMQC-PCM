import sys
import pandas as pd
import configparser
import re


def readSOArgoIndex(argo_index_file_path, header_rows, mode):
    """ Read the argo global index profile text file and for each float
    less that 40 degrees south extracts the required parameters and created a pandas dataframe
    object containing a row of parameters for each unique float WMO found.

        Parameters
        ----------
        argo_index_file_path: the path to the index file
        mode: the mode being searched for

        Returns
        ------
        argo_index_df: pandas data frame containing one row of parameters for each float

    """
    argo_float_list = []

    with open(argo_index_file_path) as f:
        all_lines = f.readlines()

    current_WMO = None

    for index, current_line in enumerate(all_lines):

        if index < header_rows:
            continue

        float_mode = current_line.split(sep=',')[0].split(sep='/')[-1][0]
        if float_mode != mode:
            continue

        float_WMO = current_line.split(sep=',')[0].split(sep='/')[-1][1:8]
        current_parameters_list = current_line.split(sep=',')

        if float_WMO != current_WMO:

            if current_WMO and lat and float(lat) < -40:
                    argo_float_list.append([dac, float_WMO, mode,
                                            int(last_profile_number),
                                            last_date, lat, long])

            current_WMO = float_WMO
            lat, long = current_parameters_list[2], current_parameters_list[3]
            dac = current_parameters_list[0].split(sep='/')[0]

        else:

            last_date = current_parameters_list[7][:8]
            filename_list = current_parameters_list[0].split(sep='/')
            last_profile_number = re.search('_\d*.', filename_list[3]).group()[1:-1]


    argo_index_df = pd.DataFrame(data=argo_float_list,
                                 columns=['DAC', 'float_WMO', 'mode',
                                          'last_profile', 'last_date',
                                          'lat', 'long'])

    return argo_index_df


class ARGOINDEXREADER(object):

    def __init__(self, config_file):

        configuration = configparser.ConfigParser()
        configuration.read(config_file)

        self.CONFIG = configuration['INDEX READER']

    def run(self):

        argo_index_df = readSOArgoIndex(self.CONFIG['INDEX_FILE'],
                                        int(self.CONFIG['INDEX_FILE_HEADER_ROWS']),
                                        self.CONFIG['MODE'])
        argo_index_df.to_excel(self.CONFIG['EXCEL_FILE'], index=False)

        print(f'number of unique WMO numbers found < -40 degrees latitude: {argo_index_df.shape[0]}')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = ARGOINDEXREADER('pcm_owc_config.ini')
    obj.run()
