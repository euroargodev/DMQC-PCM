#!/usr/bin/env python
import os
import sys
import pandas as pd
import configparser

from argodmqc_pcm.PCM_utils_forDMQC.classification import dac_comparison

DAC_COMP_EXCEL_FILE = "DAC_comp_descisions.xlsx"


float_list =[3900070]
class DAC_COMP(object):

    def __init__(self, config_file):

        configuration = configparser.ConfigParser()
        configuration.read(config_file)

        self.OWC_CONFIG = configuration['OWC']
        self.DAC_COMP_DIR = configuration['OUTPUT DIRECTORIES']['DAC_COMP_DIR']


    def run(self):

        decision_list = []
        # process each float in the list of WMO numbers in turn
        for float_WMO in float_list:

            print(f'Processing {float_WMO}')

            try:
                decision = dac_comparison(str(float_WMO),
                                          self.OWC_CONFIG,
                                          self.DAC_COMP_DIR)
                decision_list.append([float_WMO, decision])

            except FileNotFoundError:
                print('XXX data not found')

        if decision_list:
            decisions_df = pd.DataFrame(data=decision_list,
                                        columns=['float_WMO', 'decision'])

            decisions_df.to_excel(excel_writer=os.path.join(self.DAC_COMP_DIR, DAC_COMP_EXCEL_FILE),
                                  index=False)


if __name__ == '__main__':

    # turn off console warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')

    # Make an instance of the class and implement the run function
    obj = DAC_COMP('pcm_owc_config.ini')
    obj.run()
