# The Southern Ocean quality assessment tool

*Authors*: Kamila Walicka (kamwal@noc.ac.uk), Clive Neil (clive.neil@noc.ac.uk), Andrea Garcia Juan (andrea.garcia.juan@ifremer.fr), Kevin Balem (kevin.balem@ifremer.fr), Cécile Cabanes (cecile.cabanes@ifremer.fr) and Guillaume Maze (gmaze@ifremer.fr)
***

The DMQC-PCM is a new quality control method based on machine learning. It uses a statistical classifier (a PCM: Profile Classification Model) to organize and select more appropriately reference data for the quality control of an Argo float. You will find a preliminary implementation of this method in the current repository.

The preliminary implementation workflow is structured in some Jupyter Notebooks and a OWC version including the PCM option.

<p align="center">
  <img src="https://user-images.githubusercontent.com/59824937/146351682-2aa8c72d-dc2f-4038-b372-44836c3a34b7.png" width="500">
</p>

*Figure 1. Workflow of the preliminary implementation.*


In the **PCM-design** folder you will find the classification notebook *Classif_ArgoReferenceDatabase.ipynb*. It allows the design, training and prediction of a PCM (__Profile Classification Model__) using a selection of the Argo reference database. A PCM allows to automatically assemble ocean profiles into clusters according to their vertical structure similarities. It provides an unsupervised, i.e. automatic, method to distinguish profiles from different dynamical regimes of the ocean (e.g. eddies, fronts, quiescent water masses). For more information about the method, see [*Maze et al, Prg.Oc, 2017*](https://www.sciencedirect.com/science/article/pii/S0079661116300714).

<p align="center">
  <img src="https://user-images.githubusercontent.com/59824937/146352107-08b59ffd-ed73-4e70-84ee-cd002f98fb15.png" width="700">
</p>

*Figure 2. Example of classification spatial distribution for float 4900136 using argo reference database.*


As output you will obtain a txt file including the class labels for each reference profile that can be used in the OWC software. You can find the OWC software version including the PCM option in the **OWC-pcm** folder. To run it, you should modify the *ow_config.txt* file :

- set the USE_PCM variable to 1;
- give the path to the classes txt file you have created within the notebook.

OWC will use reference profiles in the same class to compare with the float profiles you want to quality control.

The DMQC-PCM method improves the reference profile selection in OWC, selecting reference profiles that are in the same oceanographic regime as the float profile we want to qualify. It leads to a reduction in the variability of reference profiles.

<p align="center">
  <img src="https://user-images.githubusercontent.com/59824937/146352649-bf7c2649-1eff-4f7c-b7dc-fc6ec7e13f2a.jpg" width="600">
</p>

*Figure 3. Reference profiles (in black) selected for float profiles 77, 78, 79 (in red) using OWC standard selection [a), b) and c)] and using PCM based selection [d), e) and f)], for float 4900136.*


You can find a performance assessment and implementation plan of the DMQC-PCM method in (link to the deliverable)

***
This repository has been developed at the Laboratory for Ocean Physics and Satellite remote sensing, IFREMER, within the framework of the Euro-ArgoRISE project. This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: Individual support to ESFRI and other world-class research infrastructures.

<p align="center">
  <img src="https://user-images.githubusercontent.com/59824937/146353099-bcd2bd4e-d310-4807-aee2-9cf24075f0c3.jpg" width="100"/> <img src="https://user-images.githubusercontent.com/59824937/146353157-b45e9943-9643-45d0-bab5-80c22fc2d889.jpg" width="100"/> <img src="https://user-images.githubusercontent.com/59824937/146353317-56b3e70e-aed9-40e0-9212-3393d2e0ddd9.png" width="100"/>
</p>
