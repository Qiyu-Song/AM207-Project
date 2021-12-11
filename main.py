# This is a Python script of the BARCAST, adapted directly from a MATLAB implementation.

import numpy as np
from lib.earthDistances import earthDistances
from lib.defaults import defaults
from lib.initialValues import initialValues

INSTRU = 0  # so that data[INSTRU] = instrumental


class DATA:
    def __init__(self, data_type, data_loc, data_time, data_value):
        self.type = data_type  # 'instrumental' or 'some proxy'
        self.locations = data_loc  # N*(lat, lon)
        self.time = data_time  # N size 1d array in years AD
        self.value = data_value
        self.loc_ind = None
        self.time_ind = None


# options is a dict of priors, MHpars, samplerIterations, preSamplerIterations, useModes, useSpatialCache, sampleCompleteField, all are optional


class BARCAST:
    def __init__(self, data, options=None):
        self.data = data
        self.options = options
        if self.options is None:
            self.options = {}
        defaults(self.options, data)
        self.model = {}  # a dict to restore model
        self.params = [None] * self.options['samplerIterations']
        self.fields = None  # will be a sizeofField*samplerIteration matrix later

    def initialize(self):
        # Find which locations in proxies correspond to instrumental locations
        for prx in self.data[INSTRU + 1:]:
            loc_ind = np.argmin(earthDistances(self.data[INSTRU].locations, prx.locations), axis=0)
            prx.loc_ind = loc_ind
        self.data[INSTRU].loc_ind = np.arange(self.data[INSTRU].shape[0])

        # Add a single point to the timeline according to BARCAST. Assumes times are in years AD
        timeline = ...  # from data, an 1d array
        self.model['timeline'] = np.sort(timeline.append(timeline, np.min(timeline) - 1))

        # Find which times in proxies correspond to timeline points
        for prx in self.data[INSTRU + 1:]:
            time_ind = np.nonzero(prx.time_ind[:, None] == self.model['timeline'])[1]
            prx.time_ind = time_ind
        self.data[INSTRU].time_ind = np.nonzero(self.data[INSTRU].time_ind[:, None] == self.model['timeline'])[1]

        # Find distance matrix
        self.model['distances'] = earthDistances(self.data[INSTRU].locations)

        # Draw initial values for the Bayesian model
        currentParams, currentField = initialValues(self.data, self.model, self.options)

        # Calculate spatial covariance
        self.model['spatialCorrMatrix'] = np.exp(-currentParams['phi'] * self.model['distances'])
        self.model['invSpatialCorrMatrix'] = np.linalg.inv(self.model['spatialCorrMatrix'])

        if not self.options['sampleCompleteField']:
            # Discover missing patterns
            self.model['missingPatterns'] = findMissingPatterns(self.data, np.size(self.model['timeline']))  # need implementation

            # Calculate temporal covariance matrices for each missing pattern
            if self.options['useSpatialCache']:
                self.model['spatialCovMatrices'], self.model['sqrtSpatialCovMatrices'] = calcSpatialCovariance(
                    self.data, self.model, currentParams)  # need implementation

        self.params[0] = currentParams
        self.fields = np.zeros((np.size(currentField), self.options['samplerIterations']))

    def sampler(self):
        # Sample MCMC chain
        ...


if __name__ == '__main__':
    # How to struct input data? An example:
    # instrumental = DATA('instrumental', loc, time, value)
    # proxy1 = DATA('proxy1', loc, time, value)
    # proxy2 = DATA('proxy2', loc, time, value)
    # could have more ...
    # data = [instrumental, proxy1, proxy2, ...]
