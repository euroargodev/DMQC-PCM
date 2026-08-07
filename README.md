# The Southern Ocean quality assessment tool

The quality assessment method in the Southern Ocean (SO) uses the pre-classified core Argo float and climatological data belonging to similar water mass regimes using the Profile Characterisation Model (PCM). These pre-classified reference data are further used in the DMQC software - OWC analysis. This method allows the DMQC operator to avoid noise from other water masses leading to a more robust quality control analysis of salinity data in delayed mode.

The SO quality assessment software is designed based on the currently available version of DMQC-PCM (the main branch of this repository) and the OWC software. The DMQC-PCM is a quality control method based on machine learning. It uses a statistical classifier (a PCM: Profile Classification Model) to organize and select more appropriately climatology (reference) data for the quality control of an Argo float. It has been shown that the DMQC-PCM software is able to improve the detection of salinity drift and temperature or salinity outliers. Moreover, when combined with the standard salinity calibration method, DMQC-PCM software is able to reduce the error on the correction while preserving confidence in this correction amplitude.

## The SO assessment workflow
The general workflow of the SO quality assessment method is presented in figure 1. In the first step, the software uses the argopy package to retrieve the Argo float temperature and salinity profiles from the local repository (it can be also set up to pull data directly from GDAC). Then, these data are used to generate the source data file including appended Argo float profiles. The source data files are used as the input to both the DMQC-PCM classification software and further to the DMQC OWC software. Both software uses as input files the configuration file and the reference data (from both CTD and/or Argo climatology data). The configuration file used in the method includes all necessary directories and setups. The reference data are used for comparison with the Argo float profiles. The DMQC PCM firstly runs the BIC function to estimate the number of classes for a training dataset to the PCM model. Then the output from the BIC is automatically implemented in the DMQC PCM code. This code generates the classification figures, the trained model and the text file containing the classification labels corresponding to each Argo float profile. The classification labels file is then read by the DMQC OWC software, which produces the suggested salinity correction outputs and associated diagnostic plots.  In this step, the DMQC operator can assess if the Argo float is affected by any salinity drift or offset and decide to apply appropriate adjustments.

![place image](https://github.com/euroargodev/DMQC-PCM/blob/SO_assesment/SO_assesment/workflow_v2.PNG)
Figure 1. Workflow of the SO quality assessment method.

## Implementation and usage


Setup configuration files <br />
All necessary directories, constant values for PCM, and objective mapping parameters which are needed to run both PCM and OWC software can be set in one config file **pcm_ow_config.json**. This file has several main sections:

- "ROLE": The only top-level field. This should be either "auditor" or "operational". When role is auditor the app will try to retrive adjusted and raw data where operational will only pull raw data to analyse.
- "OWC": This primarily matches the configuration as in https://github.com/euroargodev/argodmqc_owc, the README for that package provides details
- "PCM": These are values used in `so_dmqc.py` for the PCM specific parts
  - "GDAC_MIRROR": File path to local directory containing copy of GDAC data
  - "GDAC": URL for the GDAC
  - "SRC": Source for the data. A value of "gdac" will load from the GDAC config location, "localftp" will use GDAC_MIRROR
  - "MAX_DEPTH": max depth to use for BIC/PCM
  - "CORR_DISTANCE": correlation distance for BIC/PCM
  - "NUMBER_RUNS": number of runs for BIC
  - "NK": max number of classes for BIC
  - "PLOTS_DIR": output directory for plots
  - "MODELS_DIR": output directory for models
  - "CLASSES_DIR": output directory for classes
- "OUTPUT DIRECTORIES": Set the locations for particular outputs
  - "LOGS_DIR": Directory for log files produced at runtime
  - "DAC_COMP_DIR": Only used in the `dac_comp.py` script, directory to store outputs of this script
- "INDEX READER": Only used in the `argo_index_reader.py` script
  - "INDEX_FILE": Path to the "ar_index_global_prof.txt" file, the latest version of which can be obtained from https://data-argo.ifremer.fr
  - "INDEX_FILE_HEADER_ROWS": Number of header rows in the index file
  - "EXCEL_FILE": Output path for the produced XLSX file
  - "MODE": The mode being searched for


## Notes on the BIC calculation

By default the code will take the smallest value from the BIC calculations to use for the onward PCM computation. If a different value is required then the **so_dmqc.py** script should be modified as required.

---

### How to use Poetry to run DMQC-PCM-Python

Dependencies for the **DMQC-PCM-Python** software are managed using [Poetry](https://python-poetry.org/). It is recommended to create a virtual environment first before installing Poetry.

To create a virtual environment:

- **Mac/Linux**

  `python -m venv .venv`

  `source .venv/bin/activate`

- **Windows**

  `python -m venv .venv`

  `.\.venv\Scripts\Activate`

To install Poetry, run `pip install poetry`

#### Running DMQC-PCM-Python

1. Navigate to the `DMQC-PCM-Python` directory:
```
   cd SO_assesment/DMQC-PCM-Python
```

2. Install dependencies (first time only, or after any changes to `pyproject.toml`):
```
   poetry install
```

3. Run the software:
   Use `poetry run so_dmqc.py` followed by the WMO number of the float you wish to analyze, e.g.
```
poetry run python so_dmqc.py 1902081
```
### Operational Use on Different Drives:
You do not need to move Poetry to your operational drive. You can run it from any location. If you wish to keep all project dependencies on a specific drive (to avoid copying files across or using space in /home), configure Poetry to store the environment locally within the project folder:

```
cd </path/to/your/operational/drive>/DMQC-PCM-Python
poetry config virtualenvs.in-project true
poetry install
```
