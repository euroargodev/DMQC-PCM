import sys
import pandas as pd
import re

ARGO_INDEX_FILE_PATH = "/scratch/argo/gdac_mirror/argo-index/ar_index_global_prof.txt"
ARGO_EXCEL_FILE_PATH = "SO_argo_floats.xlsx"
MODE = 'D'


def readSOArgoIndex(argo_index_file_path, mode):
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

    headers = 9

    current_WMO = None

    for index, current_line in enumerate(all_lines):

        if index < headers:
            continue

        float_mode = current_line.split(sep=',')[0].split(sep='/')[3][0]

        if float_mode != mode:
            continue

        float_WMO = current_line.split(sep=',')[0].split(sep='/')[1]

        if float_WMO != current_WMO:

            current_WMO = float_WMO

            current_parameters_list = current_line.split(sep=',')
            lat, long = current_parameters_list[2], current_parameters_list[3]

            if lat == '' or long == '' or float(lat) > -40:
                continue

            last_date = current_parameters_list[7][:8]

            filename_list = current_parameters_list[0].split(sep='/')
            dac = filename_list[0]
            profile_number = re.search('_\d*.', filename_list[3]).group()[1:-1]


            argo_float_list.append([dac, float_WMO, MODE, int(profile_number),
                                    last_date, lat, long])


    argo_index_df = pd.DataFrame(data=argo_float_list,
                                 columns=['DAC', 'float_WMO', 'mode',
                                          'last_profile', 'last_date',
                                          'lat', 'long'])

    return argo_index_df


class ARGOINDEXREADER(object):

    @staticmethod
    def run():

        argo_index_df = readSOArgoIndex(ARGO_INDEX_FILE_PATH, MODE)

        argo_index_df.to_excel(ARGO_EXCEL_FILE_PATH, index=False)

        print(f'number of unique WMO numbers found < -40 degrees latitude: {argo_index_df.shape[0]}')


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = ARGOINDEXREADER()
    obj.run()
