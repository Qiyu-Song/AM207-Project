# This is a Python script of the BARCAST, adapted directly from a MATLAB implementation.

import numpy as np
import copy
from lib.earthDistances import earthDistances
from lib.defaults import defaults
from lib.initialValues import initialValues
from lib.calcSpatialCovariance import calcSpatialCovariances
from lib.samplerFunctions import *

INSTRU = 0  # so that data[INSTRU] = instrumental


class DATA:
    def __init__(self, data_type, data_loc, data_time, data_value):
        self.type = data_type  # 'instrumental' or 'some proxy'
        self.locations = data_loc  # N*(lat, lon)
        self.time = data_time  # N size 1d array in years AD
        self.value = data_value
        self.loc_ind = None
        self.time_ind = None


class BARCAST:
    def __init__(self, data, options=None):
        # options is a dict containing priors, MHpars, samplerIterations, preSamplerIterations, useModes,
        # useSpatialCache, sampleCompleteField, all are optional
        self.data = data
        self.options = options
        if self.options is None:
            self.options = {}
        defaults(self.options, data)
        self.model = {}  # a dict to restore model
        self.params = [None] * self.options['samplerIterations']
        self.fields = None  # will be a sizeofField*samplerIteration matrix later
        self.currentParams = None
        self.currentField = None

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
        self.currentParams, self.currentField = initialValues(self.data, self.model, self.options)

        # Calculate spatial covariance
        self.model['spatialCorrMatrix'] = np.exp(-self.currentParams['phi'] * self.model['distances'])
        self.model['invSpatialCorrMatrix'] = np.linalg.inv(self.model['spatialCorrMatrix'])

        if not self.options['sampleCompleteField']:
            # Discover missing patterns
            self.model['missingPatterns'] = findMissingPatterns(self.data,
                                                                np.size(self.model['timeline']))  # need implementation

            # Calculate temporal covariance matrices for each missing pattern
            if self.options['useSpatialCache']:
                self.model['spatialCovMatrices'], self.model['sqrtSpatialCovMatrices'] = calcSpatialCovariances(
                    self.data, self.model, self.currentParams)  # need implementation

        self.params[0] = copy.deepcopy(self.currentParams)
        self.fields = np.zeros((np.size(self.currentField), self.options['samplerIterations']))

    def sampler(self):
        # Sample MCMC chain
        totalIterations = self.options['preSamplerIterations'] + self.options['samplerIterations']
        for sample in range(totalIterations):
            # Sample temperature field
            if not self.options['sampleCompleteField']:
                # Original Gibb's sampler from Tingley et al. slightly optimized.
                self.currentField[:, 0] = sampleTemp0(self.model, self.currentParams, self.options['priors'],
                                                      self.currentField[:, 1])
                for i in range(1, np.size(self.currentField, 1) - 1):
                    self.currentField[:, i] = sampleTempk(self.data, self.model, self.currentParams, i,
                                                          self.currentField[:, i - 1], self.currentField[:, i + 1])
                self.currentField[:, -1] = sampleTempLast(self.data, self.model, self.currentParams,
                                                          self.currentField[:, -2])
            else:
                # Sample complete field at once
                self.currentField = sampleTempField(self.data, self.model, self.currentParams, self.options['priors'])

            # Sample autocorrelation coefficient
            self.currentParams['alpha'] = sampleAutocorrCoeff(self.model, self.currentParams, self.options['priors'],
                                                              self.currentField)

            # Sample autocorrelation mean parameter
            self.currentParams['mu'] = sampleAutocorrMean(self.model, self.currentParams, self.options['priors'],
                                                          self.currentField)

            # Sample spatial covariance spill parameter
            self.currentParams['sigma2'] = sampleSpatialVariance(self.model, self.currentParams, self.options['priors'],
                                                                 self.currentField)

            # Sample spatial covariance range parameter
            self.currentParams['phi'], self.model['spatialCorrMatrix'], self.model[
                'invSpatialCorrMatrix'] = sampleSpatialCovarianceRange(self.model, self.currentParams,
                                                                       self.options['priors'], self.options['MHpars'],
                                                                       self.currentField)

            if sample > self.options['preSamplerIterations']:
                # Sample instrumental measurement error variance
                self.currentParams['tau2_I'] = sampleInstrumentalErrorVar(self.data, self.options['priors'],
                                                                          self.currentField)

            for i in range(len(self.data[INSTRU + 1:])):
                # Sample proxy-specific parameters

                # Sample measurement error variance
                self.currentParams['tau2_P'][i] = sampleProxyErrorVar(self.data, self.currentParams,
                                                                      self.options['priors'], self.currentField, i)

                # Sample proxy multiplier parameter
                self.currentParams['Beta_1'][i] = sampleProxyMultiplier(self.data, self.currentParams,
                                                                        self.options['priors'], self.currentField, i)

                # Sample proxy multiplier parameter
                self.currentParams['Beta_0'][i] = sampleProxyAddition(self.data, self.currentParams,
                                                                      self.options['priors'], self.currentField, i)

            if not self.options['sampleCompleteField']:
                # Calculate spatial covariance matrices.
                if self.options['useSpatialCache']:
                    self.model['spatialCovMatrices'], self.model['sqrtSpatialCovMatrices'] = calcSpatialCovariances(
                        self.data, self.model, self.currentParams)

            self.params[sample] = copy.deepcopy(self.currentParams)
            if sample > self.options['preSamplerIteration']:
                self.fields[:, :, sample - self.options['preSamplerIteration']] = copy.deepcopy(self.currentField)


if __name__ == '__main__':
    print('Welcome to BARCAST Model!')
    # How to struct input data? An example:
    # instrumental = DATA('instrumental', loc, time, value)
    # proxy1 = DATA('proxy1', loc, time, value)
    # proxy2 = DATA('proxy2', loc, time, value)
    # could have more ...
    # data = [instrumental, proxy1, proxy2, ...]
