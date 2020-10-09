=====
Usage
=====

To use OADDS in a project::

    from outlierdenstream import Sample, OADDS

Initialize OADDS object::

	oadds = OADDS(lamb=0.125, epsilon='dynamic', beta=0.04, mu='auto', startingBuffer=bufferDf)
	oadds.runInitialization()

Fit each sample of the dataset with::

	for row in dataset:
	    sample = Sample(row, timestamp)
	    result = oadds.runOnNewSample(sample)

The algorithm returns ``True`` (outlier) if it is not able to merge the new sample to an existing core-micro-cluster or merges the sample to an existing outlier-micro-cluster. Returns ``False`` (normal) otherwise.