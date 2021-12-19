import copy
import numpy as np
import scipy
from scipy.sparse import spdiags, csr_matrix, kron
from scipy.sparse.linalg import lsqr
from scipy.stats import norm

INSTRU = 0  # so that data[INSTRU] = instrumental


def sampleTempField(data, model, params, priors):
    nrTime = np.size(model['timeline'])
    nrLoc = np.size(model['spatialCorrMatrix'], 0)

    # Spatial and temporal correlation

    # Matlab removes elements from the top of the spdiags input vector if off diagonals are used
    # What about python?

    AC = spdiags(
        np.vstack((params['alpha'] ** 2, np.ones((nrTime - 2, 1)) * (1 + params['alpha'] ** 2), 1)).reshape(-1, ), 0,
        nrTime,
        nrTime) + spdiags(np.ones((2, nrTime)) * (-1 * params['alpha']), [1, -1], nrTime, nrTime)
    S = kron(AC, model['invSpatialCorrMatrix'] / params['sigma2'])

    m = (np.sum(model['invSpatialCorrMatrix'], 1).reshape(-1, 1) / params['sigma2']) * np.hstack((params['alpha'] * (
            1 - params['alpha']), (1 - params['alpha']) ** 2 * np.ones((nrTime - 2)), (1 - params['alpha']))) * \
        params['mu']
    m = m.transpose().reshape(-1, 1)

    # Add prior for the first time point
    inds = np.arange(nrLoc)
    S = S + csr_matrix((np.ones(np.size(inds)) / priors['T_0'][1], (inds, inds)), shape=(np.size(S, 0), np.size(S, 0)),
                       dtype=np.float)
    m = m + csr_matrix((np.ones((np.size(inds))) / priors['T_0'][1] * priors['T_0'][0],
                        (inds, np.zeros((np.size(inds))))), shape=(np.size(m, 0), 1), dtype=np.float)

    # Add instrumental data
    inds = [data[INSTRU].loc_ind[i] + (data[INSTRU].time_ind[i] - 1) * nrLoc for i in
            range(np.size(data[INSTRU].loc_ind))]

    S = S + csr_matrix((np.ones(np.size(inds)) / params['tau2_I'], (inds, inds)), shape=(np.size(S, 0), np.size(S, 1)),
                       dtype=np.float)
    m = m + csr_matrix((data[INSTRU].value.reshape(-1, ) / params['tau2_I'],
                        (inds, np.zeros(np.size(inds)))), shape=(np.size(m, 0), 1), dtype=np.float)

    # Add proxy data
    for i in range(len(data) - 1):
        inds = [data[INSTRU + i].loc_ind[i] + (data[INSTRU + i].time_ind[i] - 1) * nrLoc for j in
                range(np.size(data[INSTRU + i].loc_ind))]
        S = S + csr_matrix((np.ones(np.size(inds)) * params['Beta_1'][i] ** 2 / params['tau2_P'][i], (inds, inds)),
                           shape=(np.size(S, 0), np.size(S, 1)),
                           dtype=np.float)
        m = m + csr_matrix(
            ((data[INSTRU + i].value.reshape(-1, ) - params['Beta_0'][i]) * params['Beta_1'][i] / params['tau2_P'][i],
             (inds, np.zeros(np.size(inds)))), shape=(np.size(m, 0), 1), dtype=np.float)

    # Sample field
    field = np.linalg.lstsq(S.todense(), m, rcond=None)[0] + \
            np.linalg.lstsq(np.linalg.cholesky(S.todense()), np.random.randn(np.size(m)), rcond=None)[0].reshape(-1, 1)
    field = field.reshape(nrTime, nrLoc).transpose()

    return field


def sampleAutocorrCoeff(model, params, priors, currentField):
    # Sample autocorrelation coefficient
    currentField = currentField - params['mu']
    field = currentField[:, :-1].transpose().dot(model['invSpatialCorrMatrix']) / params['sigma2']

    postVar = 1 / sum(sum(np.array(field) * np.array(currentField[:, :-1].transpose())))
    postMean = postVar * sum(sum(np.array(field) * np.array(currentField[:, 1:].transpose())))
    postStd = np.sqrt(postVar)

    prob = norm.cdf(priors['alpha'], postMean, postStd)
    if prob[1] - prob[0] < 1e-9:
        alpha = np.random.rand() * (priors['alpha'][1] - priors['alpha'][0]) + priors['alpha'][0]
    else:
        alpha = norm.ppf((prob[1] - prob[0]) * np.random.rand() + prob[0], postMean, postStd)
    return alpha


def sampleAutocorrMean(model, params, priors, currentField):
    # Sample autocorrelation mean parameter
    postVar = (np.size(currentField, 1) - 1) * sum(sum(model['invSpatialCorrMatrix'])) / params['sigma2']
    postVar = 1 / (1 / priors['mu'][1] + (1 - params['alpha']) ** 2 * postVar)

    postMean = np.sum(currentField[:, 1:] - params['alpha'] * currentField[:, :-1], 1)

    postMean = postVar * (
            (1 - params['alpha']) * sum(model['invSpatialCorrMatrix'] * postMean) / params['sigma2'] +
            priors['mu'][0] / priors['mu'][1])

    mu = postMean + np.sqrt(postVar) * np.random.rand()
    return np.sum(mu)


def sampleSpatialVariance(model, params, priors, currentField):
    # Sample autocorrelation mean parameter
    postAlpha = (np.size(currentField) - np.size(currentField, 0)) / 2 + priors['sigma2'][0]

    tDiff = currentField[:, 1:] - params['alpha'] * currentField[:, :-1] - (1 - params['alpha']) * params['mu']

    postBeta = priors['sigma2'][1] + sum(
        sum((np.array(model['invSpatialCorrMatrix'].dot(tDiff)) * np.array(tDiff)))) / 2

    sigma2 = min(50, 1 / np.random.gamma(postAlpha, 1 / postBeta))
    return sigma2


def sampleSpatialCovarianceRange(model, params, priors, mhparams, currentField):
    # Sample spatial covariance range parameter

    tDiff = currentField[:, 1:] - params['alpha'] * currentField[:, :-1] - (1 - params['alpha']) * params['mu']

    spatCorr = model['spatialCorrMatrix']
    logphi = np.log(params['phi'])

    value = -1 / (2 * priors['phi'][1]) * (logphi - priors['phi'][0]) ** 2 - 1 / (2 * params['sigma2']) * sum(
        sum(np.array(tDiff) * np.array(np.linalg.lstsq(spatCorr, tDiff, rcond=None)[0])))

    propVar = np.sqrt(mhparams['log_phi'][0])
    n = (np.size(currentField, 1) - 1) / 2

    for sample in range(mhparams['log_phi'][1]):
        logphiProp = logphi + propVar * np.random.randn()
        spatCorrProp = np.exp(-np.exp(logphiProp) * model['distances'])

        valueProp = -1 / (2 * priors['phi'][1]) * (logphiProp - priors['phi'][0]) ** 2 - 1 / (
                2 * params['sigma2'] * sum(
            sum(np.array(tDiff) * np.array(np.linalg.lstsq(spatCorrProp, tDiff, rcond=None)[0]))))

        logRatio = valueProp - value + n * np.log(np.linalg.det(np.linalg.lstsq(spatCorrProp, spatCorr, rcond=None)[0]))

        if logRatio > np.log(np.random.rand()):
            # Accept proposition
            logphi = logphiProp
            spatCorr = spatCorrProp
            value = valueProp

    phi = np.exp(logphi)
    iSpatCorr = np.linalg.inv(spatCorr)
    return phi, spatCorr, iSpatCorr


def sampleInstrumentalErrorVar(data, priors, currentField):
    # Sample instrumental measurement error variance
    res = data[INSTRU].value - currentField[data[INSTRU].loc_ind, data[INSTRU].time_ind].transpose()
    res = res[~np.isnan(res)]
    res = res.reshape(-1, 1)
    postBeta = res.transpose().dot(res) / 2 + priors['tau2_I'][1]
    # As currentField is completely filled with no NaN's, the size of res
    # equals the number of values in instrumental data
    postAlpha = np.size(res) / 2 + priors['tau2_I'][0]

    tau2 = min(10, 1 / np.random.gamma(postAlpha, 1 / postBeta))
    return np.sum(tau2)


def sampleProxyErrorVar(data, params, priors, currentField, proxyId):
    # Sample instrumental measurement error variance
    res = params['Beta_1'][proxyId] * np.array(
        currentField[data[proxyId + 1].loc_ind, :][:, data[proxyId + 1].time_ind])
    res = data[proxyId + 1].value - (res + params['Beta_0'][proxyId])
    res = res[~np.isnan(res)]
    res = res.transpose().reshape(-1, 1)

    postBeta = res.transpose().dot(res) / 2 + priors['tau2_P'][proxyId, 1]
    # As currentField is completely filled with numbers, the size of res
    # equals the number of non-NaN's in instrumental data
    postAlpha = np.size(res) / 2 + priors['tau2_P'][proxyId, 0]

    tau2 = min(5, 1 / np.random.gamma(postAlpha, 1 / postBeta))
    return tau2


def sampleProxyMultiplier(data, params, priors, currentField, proxyId):
    # Sample proxy measurement multiplier
    field = currentField[data[proxyId + 1].loc_ind, data[proxyId + 1].time_ind].transpose()

    data_ = data[proxyId + 1].value
    data_ = data_.reshape(-1, 1)

    postVar = 1 / (1 / priors['Beta_1'][1, proxyId] + field.transpose().dot(field) / params['tau2_P'][proxyId])
    postMean = postVar.dot(priors['Beta_1'][0, proxyId] / priors['Beta_1'][1, proxyId] + (
            data_ - params['Beta_0'][proxyId]).transpose().dot(field) / params['tau2_P'][proxyId])

    beta1 = postMean + np.sqrt(postVar) * np.random.randn()
    return beta1


def sampleProxyAddition(data, params, priors, currentField, proxyId):
    # Sample proxy measurement addition parameter
    res = data[proxyId + 1].value.reshape(1, -1) - params['Beta_1'][proxyId] * currentField[data[proxyId + 1].loc_ind,
                                                                                            data[proxyId + 1].time_ind]
    res = res.transpose()

    postVar = 1 / (1 / priors['Beta_0'][1, proxyId] + np.size(res) / params['tau2_P'][proxyId])
    postMean = postVar * (
            priors['Beta_0'][0, proxyId] / priors['Beta_0'][1, proxyId] + np.sum(res) / params['tau2_P'][proxyId])

    beta0 = postMean + np.sqrt(postVar) * np.random.randn()
    return beta0
