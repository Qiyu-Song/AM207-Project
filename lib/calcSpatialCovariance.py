import numpy as np
import copy

INSTRU = 0  # so that data[INSTRU] = instrumental


def calcSpatialCovariances(data, model, params):
    # Calculates the temporal covariance matries
    # Done for each pattern of missing data
    covMtxs = [None] * np.size(model['missingPatterns']['timePatterns'], 0)
    sqrtCovMtxs = copy.deepcopy(covMtxs)
    n = np.size(data[INSTRU].locations, 0)
    for i in range(np.size(model['missingPatterns']['timePatterns'], 0)):
        C = np.zeros((n, 1))
        for j in range(np.size(model['missingPatterns']['timePatterns'], 1)):
            if model['missingPatterns']['timePatterns'][i, j] == 0:
                # WARNING: should not be zero, fix this after implementation of findMissingPatterns
                # The data does not cover this pattern
                continue
            pattern = model['missingPatterns']['dataPatterns'][j][model['missingPatterns']['timePatterns'][i, j]]
            if j == 1:
                # For instrumental data, pattern contains the indices of the locations that have data for timePattern
                C[pattern] = C[pattern] + 1 / params['tau2_I']
            else:
                # WARNING: have some problem here
                C[pattern['locationIndices']] = C[pattern['locationIndices']] + np.sum(pattern['locationMap'], 1) * \
                                                params['Beta_1'][j - 1] ** 2 / params['tau2_P'][j - 1]
        C = model['invSpatialCorrMatrix'] * (1 + params['alpha'] ** 2) / params['sigma2'] + np.diag(C).reshape(-1, 1)
        sqrtCovMtxs[i] = np.linalg.cholesky(C)
        covMtxs[i] = C
    return covMtxs, sqrtCovMtxs
