BIC notebook
============

The objective of this notebook is the calculation and plot of the BIC (Bayesian Information Criteria) to help you to find the optimal number of classes for your dataset.

The BIC (Bayesian Information Criteria) can be used to optimize the number of classes in the PCM model, trying not to over-fit or under-fit the data. To compute this index, the model is fitted to the training dataset for a range of K values from 0 to 15. A minimum in the BIC curve will give you the optimal number of classes to be used.

This notebook is a complement of the classification notebook *Classif_ArgoReferenceDatabase.ipynb*.


User inputs
-----------

- Path to OWC Configuration file (ow_config.txt)

- Interpolation depth

- Float reference number

- **Correlation distance**: spatial correlation scale in your dataset (in km). It is determined by the user, regarding his/her knowledge of the region.

.. ipython:: python
    :okwarning:

    corr_dist = 50 # correlation distance in km

- **Number of runs**: For each K range run, a subset of the training dataset is randomly selected in order to use *independent* profiles. Indeed, the ocean exhibits spatial correlations that reduce the real information contained in the training dataset. This has to be taken into account. The dataset is sub-sampled into several subsets of uncorrelated profiles, finally allowing us to compute several times each K range run and hence to compute a standard deviation on the BIC metric.

.. ipython:: python
    :okwarning:

    Nrun = 10 # number of runs for each k

- **Max number of classes**: Maximum number of classes to explore.

.. ipython:: python
    :okwarning:

    NK = 15 # max number of classes to explore

Increasing the number of runs *Nrun* or the max number of classes *NK*, will increase the computation time.


BIC plot
--------


(plot figure)

If the BIC curve is not showing a clear minimum, it can be an indication that some profiles remained correlated in the training set, so try to adjust more precisely the correlation scale.

If the BIC curve has a clear minimum, don't forget to take into account the standard deviation. The BIC curve indicates a statistical optimum, so if the minimum is not above the standard deviation range, then it is indicative of an optimal **range** rather than a precise value. In this case, use your expertise to choose the number of classes (within the BIC allowed range) leading to ocean patterns that simply make the most sense to you.

