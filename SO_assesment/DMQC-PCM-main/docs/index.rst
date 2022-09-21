DMQC-PCM
========

**DMQC-PCM** is a quality control method based on machine learning. It uses a statistical classifier (a `PCM: Profile Classification Model <https://github.com/obidam/pyxpcm>`_) to organize and select more appropriately reference data for the quality control of an Argo float.

This method has been developed by `Andrea Garcia Juan <mailto:andrea.garcia.juan@ifremer.fr>`_, `Kevin Balem <https://github.com/quai20>`_, `Cécile Cabanes <mailto:cecile.cabanes@ifremer.fr>`_ and `Guillaume Maze <https://github.com/gmaze>`_ from `Ifremer <https://wwz.ifremer.fr/>`_ as a contribution to the `EARISE project <https://www.euro-argo.eu/EU-Projects/Euro-Argo-RISE-2019-2022>`_.

Overview
--------

The DMQC-PCM is a new quality control method based on machine learning. It uses a statistical classifier (a `PCM: Profile Classification Model <https://github.com/obidam/pyxpcm>`_) to organize and select more appropriately reference data for the quality control of an Argo float. You will find a preliminary implementation of this method in the current repository.

The preliminary implementation workflow of the DMQC-PCM method is made of Jupyter Notebooks and a modified Matlab OWC version including the PCM option. The workflow to implement the method is the following:

.. image:: https://user-images.githubusercontent.com/59824937/146351682-2aa8c72d-dc2f-4038-b372-44836c3a34b7.png
   :width: 500px
   :alt: Workflow of the preliminary implementation


In the `PCM-design <https://github.com/euroargodev/DMQC-PCM/tree/main/PCM-design>`_ folder you will find the classification notebook `Classif_ArgoReferenceDatabase.ipynb <https://github.com/euroargodev/DMQC-PCM/blob/main/PCM-design/Classif_ArgoReferenceDatabase.ipynb>`_ (for more details see :ref:`Classification notebook inputs`). It allows the design, training and prediction of a PCM (a `Profile Classification Model <https://pyxpcm.readthedocs.io/en/latest/overview.html#what-is-an-ocean-pcm>`_) using a selection of the Argo reference database. A PCM allows to automatically assemble ocean profiles into clusters according to their vertical structure similarities. It provides an unsupervised, i.e. automatic, method to distinguish profiles from different dynamical regimes of the ocean (e.g. eddies, fronts, quiescent water masses). For more information about the method, see `Maze et al, Prg.Oc, 2017 <https://www.sciencedirect.com/science/article/pii/S0079661116300714>`_ and the associated python library `pyxpcm <https://pyxpcm.readthedocs.io>`_.

The :ref:`Create and apply a PCM` and :ref:`Classification notebook outputs` will help you with PCM. Here is an example of the classification spatial distribution obtained for float 4900136 using the Argo reference database:

.. image:: https://user-images.githubusercontent.com/59824937/146352107-08b59ffd-ed73-4e70-84ee-cd002f98fb15.png
   :width: 500px
   :alt: Example of classification spatial distribution for float 4900136 using argo reference database

As output, you will obtain a txt file including the class labels for each reference profile that can be used in the OWC software. You can find the OWC software version including the PCM option in the `OWC-pcm <https://github.com/euroargodev/DMQC-PCM/tree/main/OWC-pcm/matlabow>`_ folder. To run it, you should modify the ``ow_config.txt`` file :

- set the ``USE_PCM`` variable to 1;
- give the path to the classes txt file you have created with `Classif_ArgoReferenceDatabase.ipynb <https://github.com/euroargodev/DMQC-PCM/blob/main/PCM-design/Classif_ArgoReferenceDatabase.ipynb>`_.

.. warning::

    OWC will now use reference profiles in the same class to compare with the float profiles you want to quality control.

The DMQC-PCM method improves the reference profile selection in OWC, selecting reference profiles that are in the same oceanographic regime as the float profile we want to qualify. It leads to a reduction in the variability of reference profiles.

.. image:: https://user-images.githubusercontent.com/59824937/146352649-bf7c2649-1eff-4f7c-b7dc-fc6ec7e13f2a.jpg
   :width: 500px
   :alt: Reference profiles (in black) selected for float profiles 77, 78, 79 (in red) using OWC standard selection [a), b) and c)] and using PCM based selection [d), e) and f)], for float 4900136.

You will find a performance assessment and implementation plan of the DMQC-PCM method in EARISE project deliverable soon to be published.

.. raw:: html

   <hr>

DMQC-PCM has been developed at the Laboratory for Ocean Physics and Satellite remote sensing, IFREMER, within the framework of the Euro-ArgoRISE project. This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: Individual support to ESFRI and other world-class research infrastructures.

.. image:: https://user-images.githubusercontent.com/59824937/146353099-bcd2bd4e-d310-4807-aee2-9cf24075f0c3.jpg
   :height: 100px

.. image:: https://user-images.githubusercontent.com/59824937/146353157-b45e9943-9643-45d0-bab5-80c22fc2d889.jpg
   :height: 100px

.. image:: https://user-images.githubusercontent.com/59824937/146353317-56b3e70e-aed9-40e0-9212-3393d2e0ddd9.png
   :height: 100px

Documentation
-------------

**Repository structure**

* :doc:`repository_struc`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Repository structure

    repository_struc

**PCM notebooks**

* :doc:`notebook_inputs`
* :doc:`bic_notebook`
* :doc:`model_creation`
* :doc:`notebook_outputs`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: PCM notebooks

    notebook_inputs
    bic_notebook
    model_creation
    notebook_outputs

**Link with OWC**

* :doc:`link_with_OWC`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Link with OWC

    link_with_OWC

**PCM_utils_forDMQC library**

* :doc:`PCM_utils_library`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: PCM_utils_forDMQC library

    PCM_utils_library
