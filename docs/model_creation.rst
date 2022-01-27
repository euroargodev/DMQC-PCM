Create and apply a PCM
========================

In the *Classif_ArgoReferenceDatabase.ipynb* notebook, you can create your own model using the number of classes K given as input. Then, the model is trained (fitted) to the training dataset and profiles are classified (predict) in order to make some useful plots in the next section. You can save your trained model or you can upload a pretrained model and use it. Three examples of trained models are availables in the *models* folder.
All this operations are undertaken using the *pyxpcm* software (link to pyxpcm).

.. code-block:: text
    # Create model
    m = pcm(K=K, features=pcm_features)
    # Fit model
    m.fit(ds_t, features=features_in_ds, dim=z_dim)
    # Save fitted model
    m.to_netcdf('models/test_model_CTD_3901915_K6_FINAL.nc')
    # Prediction of class labels
    m.predict(ds_p, features=features_in_ds, inplace = True)

Some useful variables are also calculated to evaluate the classification results in the plots section.

- **Probability of a profile to be in a class**: as the *pyxpcm* software is using the fuzzy classifier GMM (Gaussian Mixture Model) by default, it is possible to calculate the probability of a profile to belong to a class, also called posterior.

.. code-block:: text
    m.predict_proba(ds_p, features=features_in_ds, dim=z_dim, inplace=True);


- **Classes quantiles**: class vertical structure can be represented using the quantiles of all profiles corresponding to a class.

.. code-block:: text
    ds_p = ds_p.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=var_name_ds[0], outname=var_name_ds[0] + '_Q', keep_attrs=True, inplace=True)

- **Robustness**: classification robustness is a scaled version of the probability of a profile to belong to a class (i.e. the posteriors) so that the value range is more appropriate to assess the robustness of a classification. A 0 value indicates the PCM is totally unsure of the classification result (all classes are equiprobable), while values close to 1 indicate the PCM is highly confident of the result. Note that this does not prevail over the scientific meaning of the classification, but rather indicates the ability of the PCM to attribute a profile to a specific class with confidence.

.. code-block:: text
    ds_p.pyxpcm.robustness(m, inplace=True)

