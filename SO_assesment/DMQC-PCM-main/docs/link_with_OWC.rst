Link with OWC
=============

In order to use the classification in OWC:

- the USE_PCM option should be set to 1 in the ow_config.txt file and

- the path to the classification labels txt file should be provided in the PCM_FILE variable. 

For each float profile, OWC chooses reference profiles as usual: using the spatial and temporal scales provided in the configuration file. After that, the classification labels txt is loaded, and only the reference profiles that are of the same class as the float profile are selected.  Using this method implies that the number of profiles used to calculate the correction decreases. If a float profile is not classified (because it is shallower than the max depth, for example), all profiles chosen by OWC are selected.

Finally, the OWC salinity correction computation method is not modified and can be run as usual (only the content of the reference dataset has been modified to consider PCM information). The OWC output figures do not change.



