================
OutlierDenStream
================


.. image:: https://img.shields.io/pypi/v/outlierdenstream.svg
        :target: https://pypi.python.org/pypi/outlierdenstream

.. image:: https://img.shields.io/travis/anrputina/outlierdenstream.svg
        :target: https://travis-ci.org/anrputina/outlierdenstream

.. image:: https://readthedocs.org/projects/outlierdenstream/badge/?version=latest
        :target: https://outlierdenstream.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/anrputina/outlierdenstream/shield.svg
     :target: https://pyup.io/repos/github/anrputina/outlierdenstream/
     :alt: Updates



OutlierDenStream is a custom implementation of the [DenStream] algorithm used for Anomaly Detection in an Unsupervised environment written in Python.  The implementation has been developed for a project funded by NewNet@Paris, Cisco's Chair at Telecom ParisTech, in the context of the telemetry project https://telemetry.telecom-paristech.fr/


Description
-------------


The algorithm uses the concept of micro-cluster, distances, parameters and pruning strategies introduced in DenStream. It maintains the clusters incrementally and as soon as a new sample (snapshot) is available it labels it as Normal or Abnormal: e.g. if the new sample is merged to a existing core-micro-cluster then it is considered normal.

The algorithm can be used to monitor **Telemetry** and raise an alarm when needed. Each sample (data point) should have **1 x F** dimensionality where F is the number of features.


Documents
-------------


The first phase of the algorithm consist in discovering the clusters thus there is the need first of all to gather samples or use a buffer dataset then run then DBScan algorithm and obtain the clusters. From now on it is possible to maintain them incrementally. On the other hand it is possible to use the buffer dataset as a unique cluster without applying DBScan such that all the samples are considered as belonging to a cluster. 

The input parameters are the following one

	* **lamb**: the fading factor :math:`\lambda`
	* **epsilon**: the radius :math:`\epsilon`
		* **"auto"**: computes automatically the radius of the initial cluster as the maximum radius of the initial buffer
		* **int** or **float**: :math:`\epsilon` value integer of float
	* **minPts**: DBScan parameter, if used (currently disabled)
	* **beta**: potential factor :math:`\beta`
	* **mu**: cluster weight :math:`\mu`
		* **"auto"**: computes automatically the maximum weight of the cluster, due to fading function
		* **int** or **float**: :math:`\mu` value integer or float
	* **numberInitialSample**: number of samples needed before starting the DBScan algorithm on the gathered samples. Only if you use DBSCan. (currently disabled)
	* **startingBuffer**: starting buffer containing the initial samples. The algorithm merges all the samples in a unique "normal" cluster if :math:`\epsilon` and :math:`\mu` are **"auto"**
	* **tp**: checking period of the clusters weight. Needed for pruning, if the weight of the clusters goes below the threshold :math:`\beta \cdot \mu`: remove them.

Related repositories
-----------------------

The code released in this website are also instrumental to reproduce results that are published in [ACM SIGCOMM BigDama'18] and that are demonstrated at [IEEE INFOCOM'18] (see the Reference section below)

This repository only contains the algorithm, whereas related repositories contain
- the datasets we released to the community https://github.com/cisco-ie/telemetry/blob/master/README.md
- specific instruction and code to replicate the paper results https://github.com/anrputina/OutlierDenStream-BigDama18

Demo
-----------------------

A demo of the algorithm is available here: https://telemetry.telecom-paristech.fr/

Dataset
-----------------------

The Dataset in use are extracted from: https://github.com/cisco-ie/telemetry

References
-----------------------

[ACM SIGCOMM BigDama'18] Putina, Andrian and Rossi, Dario and Bifet, Albert and Barth, Steven and Pletcher, Drew and Precup, Cristina and Nivaggioli, Patrice,  Telemetry-based stream-learning of BGP anomalies ACM SIGCOMM Workshop on Big Data Analytics and Machine Learning for Data Communication Networks (Big-DAMA’18) aug. 2018

[IEEE INFOCOM'18] Putina, Andrian and Rossi, Dario and Bifet, Albert and Barth, Steven and Pletcher, Drew and Precup, Cristina and Nivaggioli, Patrice,  Unsupervised real-time detection of BGP anomalies leveraging high-rate and fine-grained telemetry data IEEE INFOCOM, Demo Session apr. 2018,

[DenStream] Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou, "Density-Based Clustering over an Evolving Data Stream with Noise" in Proceedings of the 2006 SIAM International Conference on Data Mining. 2006, 328-339 


* Free software: MIT license
* Documentation: https://outlierdenstream.readthedocs.io.


Coming soon
-----------------------

* Release on pip
* Detailed documentation
* Examples

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
