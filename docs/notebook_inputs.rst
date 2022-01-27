
Classification notebook inputs
==============================

The user should introduce four inputs at the beginning of the *Classif_ArgoReferenceDatabase.ipynb* notebook


Configuration file
------------------

You should provide the path to the OWC configuration file (**ow_config.txt**). This is a critical input because the **ow_config.txt** file is used to make the reference profile initial selection and to define important paths, so make sure that this file is set up properly (look at OWC cookbook).

.. code-block:: text
    config_filename = 'DMQC-PCM/OWC-pcm/matlabow/ow_config.txt'




**owc_config.txt parameters used by the notebook**

The information used to select the profiles from the Argo reference database came from parameters MAPSCALE_LONGITUDE_LARGE and MAPSCALE_LATITUDE_LARGE.

Three paths are automatically assigned from the OW configuration file:

- CONFIG_DIRECTORY + CONFIG_WMO_BOXES: ***wmo_boxes_argo.mat*** file path. You should change this file if you want to switch between Argo data, CTD data or both as reference dataset (as in OWC, see DMQC cookbook).
- HISTORICAL_DIRECTORY: path to the **Argo reference database**
- FLOAT_SOURCE_DIRECTORY: float source directory, where we will find the *.mat* file containing the **float data**. If you have not created the .mat file yet, you can generate it using the last version of argopy (link to argopy).


Interpolation depth
-------------------

The PCM can not deal with NaN values, so the reference dataset is interpolated on standard depth levels and the profiles shallower than the max_depth, are dropped out. A max depth of 1000m can be enough, however you should find a compromise between keeping a sufficient number of reference profiles and having a comprehensive representation of the oceanography in the region. You should also consider the depth of the float profiles: if they are shallower than the max depth, they will be dropped out, and they will not be classified by the PCM. In such cases, a lower value of max depth is recommended.

.. code-block:: text
    max_depth = 1000  


Float reference number
----------------------

You should provide the WMO number of the float you want to correct in OWC. THree examples are provided in the notebook (and explained in the deliverable, link):float 4900136 crossing the Gulf Stream, float 3901928 in the Southern Ocean and float 3901915 in the Agulhas Current.

.. ipython:: python
    :okwarning:
    float_WMO = 4900136


Number of classes
-----------------

The number of classes K is a key parameter when designing a PCM. You are invited to try different K values to evaluate if the results are coherent with your knowledge of possible ocean regimes in the region. It is also possible to use the BIC notebook (BIC_plot.ipynb, see next section), which can help you to choose the optimal number of classes K.

.. ipython:: python
    :okwarning:

    K=4


