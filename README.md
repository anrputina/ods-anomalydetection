**OutlierDenStream**
===================


OutlierDenStream is a custom implementation of the DenStream algorithm used for Anomaly Detection in an Unsupervised environment written in Python. 

Description
-------------


The algorithm uses the concept of micro-cluster, distances, parameters and pruning strategies introduced in DenStream. It maintains the clusters incrementally and as soon as a new sample (snapshot) is available it labels it as Normal or Abnormal: e.g. if the new sample is merged to a existing core-micro-cluster then it is considered normal.

The algorithm can be used to monitor **Telemetry** and raise an alarm when needed. Each sample (data point) should have **1 x F** dimensionality where F is the number of features.


Documents
-------------


The first phase of the algorithm consist in discovering the clusters thus there is the need first of all to gather samples or use a buffer dataset then run then DBScan algorithm and obtain the clusters. From now on it is possible to maintain them incrementally. On the other hand it is possible to use the buffer dataset as a unique cluster without applying DBScan such that all the samples are considered as belonging to a cluster. 

The input parameters are the following one:

> * **lamb**: the fading factor <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" />
> * **epsilon**: the radius <img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" />
>   * **'auto'**: computes automatically the radius of the initial cluster (if you don't use the initial DBScan)
>   * **int** or **float**: <img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /> value integer of float
> * **minPts**: DBScan parameter, if used
> * **beta**: potential factor <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" />
> * **mu**: cluster weight <img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" />
>   * **'auto'**: computes automatically the maximum weight of the cluster, due to fading function
>   * **int** or **float**:  <img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /> value integer or float
> * **numberInitialSample**: number of samples needed before starting the DBScan algorithm on the gathered samples. Only if you use DBSCan.
> * **startingBuffer**: buffer with the initial samples
> * **tp**: checking period of the clusters weight. Needed for pruning, if the weight of the clusters goes below the threshold <img src="https://latex.codecogs.com/gif.latex?\beta&space;\cdot&space;\mu" title="\beta \cdot \mu" />: remove them.

Example:

```python
from DenStream import DenStream
...
den = DenStream (lamb=0.03, epsilon='auto', beta=0.03, mu='auto', startingBuffer=bufferDf, tp=12)

den.runInitialization()
```

This example shows the case in which a buffer is passed to the algorithm, thus there is not the need to wait for the incoming data. Moreover **epsilon** and **mu** are set to **'auto'** thus we are not going to perform the initial DBScan meaning that the samples passed in the buffer have to be considered as samples belonging to a unique initial cluster and the parameters can be computed by this last one. The method **.runInitialization()** performs indeed these tasks and in particular creates a cluster by the samples and computes <img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /> and <img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" />.

The algorithm is now able to maintain the clusters incrementally. As soon as a new sample is available it is possible to call the **.onNewSample(Sample sample)** method. This one takes as input the object **Sample** and returns the label.

Example:
```python
from sample import Sample
...
for row in dataset:
    sample = Sample(row, timestamp)
    result = den.runOnNewSample(sample)
```

The example shows the simulation of an experiment in which we have a dataset <img src="https://latex.codecogs.com/gif.latex?X&space;\in&space;R&space;^&space;{N,&space;F}" title="X \in R ^ {N, F}" /> where N is the number of samples and F of features. Each row of F elements is extracted from the dataset and a **Sample** object is instantiated by it. The Sample object is composed by the main attributes **.value** and **.timestamp** which are respectively the array of the F measurements and the time (the timestamp is used just to retrieve the real timestamp, not for clustering purpose).
Thus on each new available sample, the **.runOnNewSample()** method is run and the *Normal* or *Abnormal* label is obtained.  

Coming Soon
-------------


* Examples and Results
* Detailed description of the algorithm
* Detailed description of classes and methods
* Detailed description of class **Statistics**: How to compute Precision, Recall, False Positive Rate (if **ground truth** available)
* Introduction to **Detection of order K** (count the number of consecutive abnormal samples and raise an alarm only after K consecutive abnormal samples)
* Possibility to execute the code using the real sampling time and not discretized times as 'timestamp = 1, timestamp = 2, ..., timestamp = N'
* Possibility to execute the code in real time, without using a dataset

Installation
---

Clone the repository and import in your project DenStream and Sample

```python
from sample import Sample
from DenStream import DenStream
...
```

Dataset
---

The Dataset in use is available at https://github.com/cisco-ie/telemetry
