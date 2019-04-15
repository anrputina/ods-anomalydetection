=====
Usage
=====

To use OutlierDenStream in a project::

    from outlierdenstream import Sample, OutlierDenStream

Initialize OutlierDenStream object::

	ods = OutlierDenStream(lamb=0.03, epsilon='auto', beta=0.03, mu='auto', startingBuffer=bufferDf, tp=12)
	ods.runInitialization()

Fit each sample of the dataset with::

	for row in dataset:
	    sample = Sample(row, timestamp)
	    result = ods.runOnNewSample(sample)

The algorithm returns ``True`` (outlier) if it is not able to merge the new sample to an existing core-micro-cluster or merges the sample to an existing outlier-micro-cluster. Returns ``False`` (normal) otherwise.