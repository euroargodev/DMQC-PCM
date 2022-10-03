# The Southern Ocean quality assessment tool

*Authors of the SO assessment tool*: Kamila Walicka (kamwal@noc.ac.uk), Clive Neil (clive.neil@noc.ac.uk)

*Authors of the DMQC-PCM tool*: Andrea Garcia Juan (andrea.garcia.juan@ifremer.fr), Kevin Balem (kevin.balem@ifremer.fr), Cécile Cabanes (cecile.cabanes@ifremer.fr) and Guillaume Maze (gmaze@ifremer.fr)
***

The quality assessment method in the Southern Ocean (SO) uses the pre-classified core Argo float and climatological data belonging to similar water mass regimes using the Profile Characterisation Model (PCM). These pre-classified reference data are further used in the DMQC software - OWC analysis. This method allows the DMQC operator to avoid noise from other water masses leading to a more robust quality control analysis of salinity data in delayed mode. 

The SO quality assessment software is designed based on the currently available version of DMQC-PCM (the main branch of this repository) and the OWC software. The DMQC-PCM is a quality control method based on machine learning. It uses a statistical classifier (a PCM: Profile Classification Model) to organize and select more appropriately climatology (reference) data for the quality control of an Argo float. It has been shown that the DMQC-PCM software is able to improve the detection of salinity drift and temperature or salinity outliers. Moreover, when combined with the standard salinity calibration method, DMQC-PCM software is able to reduce the error on the correction while preserving confidence in this correction amplitude.

The SO_assesment brunch repository includes the two versions of the SO assessment software:
- **DMQC-PCM-main** - which includes the DMQC-PCM and OWC Matlab software
- **DMQC-PCM-Python** -which includes the DMQC-PCM and OWC  Python software

## The SO assessment workflow
The general workflow of the SO quality assessment method is presented in figure 1. In the first step, the software uses the aropy package to retrieve the Argo float temperature and salinity profiles from the local repository (it can be also set up to pull data directly from GDAC). Then, these data are used to generate the source data file including appended Argo float profiles. The source data files are used as the input to both the DMQC-PCM classification software and further to the DMQC OWC software. Both software uses as input files the configuration file and the reference data (from both CTD and/or Argo climatology data). The configuration file used in the method includes all necessary directories and setups. The reference data are used for comparison with the Argo float profiles. The DMQC PCM firstly runs the BIC function to estimate the number of classes for a training dataset to the PCM model. Then the output from the BIC is automatically implemented in the DMQC PCM code. This code generates the classification figures, the trained model and the text file containing the classification labels corresponding to each Argo float profile. The classification labels file is then read by the DMQC OWC software, which produces the suggested salinity correction outputs and associated diagnostic plots.  In this step, the DMQC operator can assess if the Argo float is affected by any salinity drift or offset and decide to apply appropriate adjustments.

![place image](https://github.com/euroargodev/DMQC-PCM/blob/SO_assesment/SO_assesment/workflow_v2.PNG)
Figure 1. Workflow of the SO quality assessment method.

## Implementation and usage
Since the SO quality assessment software is designed for both Matlab and Python users of the OWC software, hence the procedures for running the software are slightly different. The code used and their description are as follows.

### DMQC-PCM-main
(1) Setup configuration files<br />
- *pcm-config.txt* <br />
  This file is used by the DMQC-PCM Python software and includes the directories to the local archive including the weekly updated NetCDF data from GDAC. Moreover,   it also includes the following constants: the maximal interpolation depth from which reference data (MAX_DEPTH = 1000), correlation distance (CORR_DISTANCE =     50), number of runs in BIC for each class (NUMBER_RUNS = 10) and maximal number of classes to explore (NK = 10).
  
- *ow_config_linux_ctd_argo.txt*<br />
This file includes the configurations used in OWC Matlab software for the Argo floats analysis. To best represent the dynamic condition in this region and for further comparison of output with the DM data the constant values of the objective mapping parameters have been used (Table 3.1). In Task 5.3, the floats which are going through the SO quality assessment analysis are firstly run using the first “SO1” configurations. The other sets of configurations are used if the specific float requires more iterations. Moreover, this file also includes the option for including the class labels from the PCM analysis (USE_PCM=1).

(2) Select floats for analysis and run the codes in software<br />
The list of WMO numbers of floats which are intended to go through the SO quality assessment needs to be inserted in the following codes. After these edits, the following codes need to be run:<br />

- *so_dmqc_master.py* <br />
  This Python code (1) retrieves data from the local repository or GDAC (using argopy package), (2) automatically generates the source code for OWC analysis (using   argopy package), (3) runs the BIC function which is estimating the most suitable number of classes for a training dataset to model, (4) runs the DMQC-PCM   software and generate the output plots, model and class labels.
  
- *ow_calibration_pcm.m*<br />
  This Matlab code runs the OWC software including the class labels from the PCM and generates the diagnostic plots.

### DMQC-PCM-Python
(1 ) Setup configuration files<br />
All necessary directories, constant values for PCM, and objective mapping parameters which are needed to run both PCM and OWC software can be set in one initial file below. The configurations used are the same as in the DMQC-PCM-main software.  
-*pcm_ow_config.ini-*

(2) Select floats for analysis and run the codes in software<br />
The list of WMO numbers of floats which are intended to go through the SO quality assessment needs to be specified in the following code below. This code is also used to run the entire software.<br />

- *so_dmqc_master.py-*<br />
In addition to the code from the DMQC-PCM-main, this code is also performing the automatic OWC Python calculations. 


