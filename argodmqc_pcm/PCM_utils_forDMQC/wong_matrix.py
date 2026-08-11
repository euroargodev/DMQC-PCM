from argopy import DataFetcher as ArgoDataFetcher


def createWongMatrix(float_WMO, GDAC_MIRROR, FLOAT_SOURCE_DIRECTORY):
    """ Reading raw data from weekly updated local BODC's mirror of netcdf data from GDAC
        Parameters
        ----------
        float_WMO: reference float number
        GDAC_MIRROR: local directory of BODC NetCDF data
        FLOAT_SOURCE_DIRECTORY:directory of generated wong matrix including raw data

        Returns
        -------
        Wong matrix .mat files with appended Argo profiles
    """
    ds = ArgoDataFetcher(src='localftp',
                         local_ftp=GDAC_MIRROR,
                         cache=True,
                         mode='expert').float(float_WMO).load().data

    """ You can force the program to load raw PRES, PSAL and TEMP whatever PRES is adjusted or not:
    >> > ds.argo.create_float_source(force='raw')
     or you can force the program to load adjusted variables: PRES_ADJUSTED, PSAL_ADJUSTED, TEMP_ADJUSTED
    >> > ds.argo.create_float_source(force='adjusted')
    """
    ds.argo.create_float_source(FLOAT_SOURCE_DIRECTORY, force='raw')
    ds_source = ds.argo.create_float_source()
