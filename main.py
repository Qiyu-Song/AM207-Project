# This is a Python script of the BARCAST, adapted directly from a MATLAB implementation.

import numpy as np
import copy
from lib.earthDistances import earthDistances
from lib.defaults import defaults
from lib.initialValues import initialValues
from lib.samplerFunctions import *
import matplotlib.pyplot as plt

INSTRU = 0  # so that data[INSTRU] = instrumental


class DATA:
    def __init__(self, data_type, data_loc, data_time, data_value):
        if len({len(data_loc), len(data_time), len(data_value)}) > 1:  # length of each should be equal
            print(len(data_loc), len(data_time), len(data_value))
            raise RuntimeError("Wrong initialization for data!")
        self.type = data_type  # 'instrumental' or 'some proxy'

        self.locations = [data_loc[0]]  # least (lat, lon)
        self.loc_ind = []  # N size 1d integer
        for loc_ in data_loc:
            if np.max(np.sum(np.array(loc_) == np.array(self.locations), axis=1)) == np.size(loc_):
                ind_ = np.argmax(np.sum(loc_ == self.locations, axis=1))
                self.loc_ind.append(ind_)
            else:
                self.locations.append(loc_)
                self.loc_ind.append(len(self.locations) - 1)
        self.locations = np.array(self.locations)

        self.time = data_time  # N size 1d array in years AD
        self.time_ind = None

        self.value = data_value  # temperature


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
            prx.loc_ind = [loc_ind[ind_] for ind_ in prx.loc_ind]
            prx.locations = self.data[INSTRU].locations

        # Add a single point to the timeline according to BARCAST. Assumes times are in years AD
        timeline = np.array(self.data[INSTRU].time.reshape(-1))  # from data, an 1d array
        for ist_prx in self.data:
            timeline = np.hstack((timeline, ist_prx.time.reshape(-1)))
        timeline = list(set(timeline))
        timeline.append(np.min(timeline) - 1)
        self.model['timeline'] = list(np.sort(timeline))

        # Find which times in proxies correspond to timeline points
        for prx in self.data[INSTRU + 1:]:
            time_ind = [self.model['timeline'].index(time_point) for time_point in prx.time]
            prx.time_ind = time_ind
        self.data[INSTRU].time_ind = [self.model['timeline'].index(time_point) for time_point in self.data[INSTRU].time]

        # Find distance matrix
        self.model['distances'] = earthDistances(self.data[INSTRU].locations)

        # Draw initial values for the Bayesian model
        self.currentParams, self.currentField = initialValues(self.data, self.model, self.options)

        # Calculate spatial covariance
        self.model['spatialCorrMatrix'] = np.exp(-self.currentParams['phi'] * self.model['distances'])

        self.model['invSpatialCorrMatrix'] = np.linalg.inv(self.model['spatialCorrMatrix'])

        self.params[0] = copy.deepcopy(self.currentParams)

        self.fields = np.zeros(
            (self.options['samplerIterations'], self.currentField.shape[0], self.currentField.shape[1]))

    def sampler(self):
        # Sample MCMC chain
        totalIterations = self.options['preSamplerIterations'] + self.options['samplerIterations']
        for sample in range(totalIterations):
            # Sample temperature field

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

            if sample >= self.options['preSamplerIterations']:
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

            if sample >= self.options['preSamplerIterations']:
                if (sample + 1) % 100 == 0:
                    print('------------Samples:', sample + 1 - self.options['preSamplerIterations'], '/',
                          self.options['samplerIterations'])
                self.params[sample - self.options['preSamplerIterations']] = copy.deepcopy(self.currentParams)
                self.fields[sample - self.options['preSamplerIterations']] = copy.deepcopy(self.currentField)

        print('Sampling finished!')


def get_test_data():
    # instrumental data, 1 loc 50 time points
    lon = np.array(list(np.random.rand(1) * 40 + 50) * 50).reshape(-1, 1)
    lat = np.array(list(np.random.rand(1) * 40 + 30) * 50).reshape(-1, 1)
    loc = np.hstack((lat, lon))
    time = np.arange(50).reshape(-1, 1)
    # linear increasing time series
    ins_value = np.random.rand(50).reshape(-1, 1) * 0.1 + 14 + time * 0.1
    # non linear time series
    # ins_value = np.random.rand(50).reshape(-1, 1) * 0.1 + 14 + 2 * np.sin(np.pi * time / 50)
    data_instru = DATA('instrumental', loc, time, ins_value.reshape(-1, 1))

    # proxy data 1, linear transformation from instrumental data
    data_proxy1 = DATA('proxy1', loc, time, ins_value * 2 + 1)

    # proxy data 2, non-linear transformation from instrumental data
    data_proxy2 = DATA('proxy2', loc, time, 4 * np.log(ins_value))

    '''
    # test spatial covariance
    N = 20
    lon = np.random.randn(N, 1) * np.arange(N).reshape(-1, 1) * 20 / N
    lat = np.random.randn(N, 1) * np.arange(N).reshape(-1, 1) * 20 / N
    loc = np.hstack((lat, lon))
    time = np.array([100] * N)
    ins_value = np.random.randn(N, 1) * 0.1 + lon * 0.1
    data_instru = DATA('instrumental', loc, time, ins_value)
    '''

    return np.array([data_instru, data_proxy1])


if __name__ == '__main__':
    print('Welcome to BARCAST Model!')
    data = get_test_data()
    barcast = BARCAST(data)
    barcast.initialize()
    barcast.sampler()

    # an example of plotting fitted mean temperature field time series
    field_for_plot = barcast.fields[-200:-1:2]
    mean_time_series = field_for_plot.mean(axis=1)
    time = barcast.model['timeline']

    figure, ax = plt.subplots(1, 1)
    up = np.percentile(mean_time_series, 97.5, axis=0)
    lo = np.percentile(mean_time_series, 2.5, axis=0)
    ax.fill_between(time, up, lo, color='red', alpha=0.3, label='95% interval')
    ax.scatter(list(time[1:]), data[INSTRU].value, color='k', label='data point')
    ax.legend()
    ax.set_xlabel('Time Points')
    ax.set_ylabel('Temperature (degree C)')
    ax.set_title('Fitting time series')
    ax.set_xlim(time[1] - 0.05, time[-1] + 0.05)
    ax.set_ylim(10, 25)
    plt.show()
