=====
OADDS
=====


.. image:: https://img.shields.io/pypi/v/oadds.svg
        :target: https://pypi.python.org/pypi/oadds

.. image:: https://img.shields.io/travis/anrputina/oadds.svg
        :target: https://travis-ci.com/anrputina/oadds

.. image:: https://readthedocs.org/projects/oadds/badge/?version=latest
        :target: https://oadds.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




OADDS is an Online Anomaly Detection engine for Data Streams capable of processing instances (data points) in a single-pass and unsupervised fashion. It uses the micro-clusters data structure presented in [DenStream].  

The algorithm can be used to monitor **Telemetry** data (data streams) and raise alarms. Each instance is represented in the euclidean space as a **1 x F** data point, where F is the number of features representing the measurements.

Related repositories
-----------------------

The code released in this website are also instrumental to reproduce results that are published in [ACM SIGCOMM BigDama'18] and that are demonstrated at [IEEE INFOCOM'18] (see the Reference section below)

This repository only contains the algorithm, whereas related repositories contain
- the datasets we released to the community https://github.com/cisco-ie/telemetry/blob/master/README.md
- specific instruction and code to replicate the paper results https://github.com/anrputina/OutlierDenStream-BigDama18

Demo
-----------------------

A demo of the first version of the algorithm [ACM SIGCOMM BigDama'18] is available here: https://telemetry.telecom-paristech.fr/

Dataset
-----------------------

The Dataset in use are extracted from: https://github.com/cisco-ie/telemetry

References
-----------------------

[ACM SIGCOMM BigDama'18] Putina, Andrian and Rossi, Dario and Bifet, Albert and Barth, Steven and Pletcher, Drew and Precup, Cristina and Nivaggioli, Patrice,  Telemetry-based stream-learning of BGP anomalies ACM SIGCOMM Workshop on Big Data Analytics and Machine Learning for Data Communication Networks (Big-DAMA’18) aug. 2018

[IEEE INFOCOM'18] Putina, Andrian and Rossi, Dario and Bifet, Albert and Barth, Steven and Pletcher, Drew and Precup, Cristina and Nivaggioli, Patrice,  Unsupervised real-time detection of BGP anomalies leveraging high-rate and fine-grained telemetry data IEEE INFOCOM, Demo Session apr. 2018,

[DenStream] Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou, "Density-Based Clustering over an Evolving Data Stream with Noise" in Proceedings of the 2006 SIAM International Conference on Data Mining. 2006, 328-339 



Features
--------

* Free software: MIT license
* Documentation: https://ods.readthedocs.io.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
