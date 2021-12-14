import numpy as np

INSTRU = 0  # so that data[INSTRU] = instrumental


def setDefault(opt, field, default):
    val = default
    for field_name in field:
        print(field_name, opt)
        if field_name in opt:
            opt = opt[field_name]
        else:
            if field_name != field[-1]:
                opt[field_name] = {}
                opt = opt[field_name]
            else:
                opt[field_name] = val
    return


'''
def test():
    options = {'hahaha': 'hehehe', 'number': 114514, 'priors': {'beta': 0, 'alpha': [1, 2]}}
    # options = {}
    setDefault(options, ['priors', 'alpha'], [0, 1])
    print(options['priors']['alpha'])
    print(options)
    return
'''


def defaults(opt, data=None):
    # I guess we don't need the 'info' part for now
    # Assign priors and MCMC default values for the given data

    # Prior distribution parameters for the autocorrelation of temperature.
    # Defines the range for uniform distribution.'
    setDefault(opt, ['priors', 'alpha'], [0, 1])  # [min, max) of uniform

    # Prior distribution parameters for the mean of temperature field.
    # Defines the mean and variance of gaussian distribution.
    if data:
        setDefault(opt, ['priors', 'mu'], [np.nanmean(data[INSTRU].value[:]), 5 ** 2])
    else:
        setDefault(opt, ['priors', 'mu'], [0, 5 ** 2])

    # Prior distribution parameters for the partial sill of the spatial covariance matrix of the innovations that
    # drive the AR(1) process. Defines the shift and scale, resp., of inverse gamma distribution.
    setDefault(opt, ['priors' 'sigma2'], [0.5, 0.5])
    # This is equivalent to 1 observations with an average square deviation of 2
    # Very general, broad prior.
    # note, calling the pars a and b, and the pars of the scaled inverse chi^2
    # nu and s^2, we have nu=2a and s^2=b/a. These give, in the bayesian sense,
    # the number of prior samples and the mean squared deviation.

    # Prior distribution paramaters for the inverse range of this spatial covariance matrix. Defines the log-mean and
    # log-variance of the lognormal distribution.
    setDefault(opt, ['priors' 'phi'], [-4.65, 1.2])
    # RECALL we are using a log transformation in the Metropolis sampler for
    # this step, so the prior is log normal. There are nine data points evenly
    # space by about 111km. A largely uninformative prior would specify the
    # range to be somewhere between 10 and 1000 km, so the log of the
    # inverse range should be between -7 and -2.3

    # Prior distribution parameters for the error variance of instrumental observations. Defines the shift and scale,
    # resp., of inverse gamma distribution.
    setDefault(opt, ['priors' 'tau2_I'], [0.5, 0.5])
    # This is equivalent to 1 observations (nu=2*a) with an average square deviation of 1 (s^2=a/b)

    # Prior distribution parameters for the error variance of proxy observations. Defines the shift and scale, resp.,
    # of inverse gamma distribution.'];
    if data:
        setDefault(opt, ['priors' 'tau2_P'], np.array([[0.5, 0.5]] * len(data[INSTRU + 1:])))
    else:
        setDefault(opt, ['priors' 'tau2_P'], [0.5, 0.5])
    # This is equivalent to 1 observations (nu=2*alpha) with an average square deviation of 2

    # Prior distribution parameters for the scaling of proxy observations.
    # Defines the mean and variance of normal distribution.
    # This one is normal as well. IF THE PROXIES HAVE BEEN PRE-PROCSSES TO HAVE MEAN=0, STD=1, THEN Hypothetically,
    # the scaling should be (1-tau_P^2)(1-alpha^2)/sigma^2)^(+1/2). So set the mean to the modes of these priors,
    # and then include a decent sized variance.
    if data:
        setDefault(opt, ['priors' 'Beta_1'], [np.reshape(((1 - opt['priors']['tau2_P'][:, 1] / (
                opt['priors']['tau2_P'][:, 0] + 1)) * (1 - np.mean(opt['priors']['alpha']) ** 2) / (
                                                                  opt['priors']['sigma2'][1] / (
                                                                  opt['priors']['sigma2'][0] + 1))) ** (1 / 2),
                                                         (-1, 1)),
                                              np.ones((len(data[INSTRU + 1:]), 1)) * 8 ** 2])
    else:
        setDefault(opt, ['priors' 'Beta_1'], [1, 8 ** 2])

    # Prior distribution parameters for the shift of proxy observations.
    # Defines the mean and variance of normal distribution. Should be equal to the prior of mu.
    # Prior for Beta_0. SET EQUAL TO THE PRIOR FOR MU
    if data:
        setDefault(opt, ['priors' 'Beta_0'], [-opt['priors']['Beta_1'][:, 0] * opt['priors']['mu'][0],
                                              np.ones((len(data[INSTRU + 1:]), 1)) * 8 ** 2])
    else:
        setDefault(opt, ['priors' 'Beta_0'], [opt['priors']['mu'][0], 8 ** 2])

    # Prior distribution parameters for the temperature field.
    # Defines the mean and variance of normal distribution.
    if data:
        setDefault(opt, ['priors' 'T_0'], [np.nanmean(data[INSTRU].value[:]), 4 * np.nanvar(data[INSTRU].value[:])])
    else:
        setDefault(opt, ['priors' 'T_0'], [0, 8 ** 2])

    # with mean given by mean of all inst and standard deviation 2 times the std of all Inst.
    # So the prior parameters (mean, var) are:
    # Note that this assumes a constant mean and diagonal cov mat.

    # Parameters for the Metropolis-Hastings algorithm when sampling log phi.
    # First is the variance of proposal distribution, and second is the number of steps in the MH-sampling.
    # Also set the MCMC jumping parameters
    # The phi step.
    # The jumping distribution of log(phi) is normal with mean zero; this sole paramter is the VARIANCE.
    # We expect the posterior variance to be far smaller than the prior variance, so the jumping variance is set low.
    # this can be adjusted as needed.
    # ALSO Specify the number of iterations of the Metropolis step for each iteration of the Gibbs:
    # the Metroplis sampler needs to (come close to) converging each time.
    # Until the algorithm settles down, this will take a little while.
    # NOTE that this step is not time consuming.
    # First par is variance, second is number.
    setDefault(opt, ['MHpars' 'log_phi'], [.04**2, 100])

    # Number of consecutive samples from the bayesian model.
    setDefault(opt, ['samplerIterations'], 2000)

    # Number of times to update only the temperature array before beginning to update the other parameters.
    setDefault(opt, ['preSamplerIterations'], 500)

    # Use modes of priors as inital values for MCMC sampling.
    setDefault(opt, ['useModes'], False)

    # Store spatial covariance matrices for speedy processing. Uses a lot of memory
    setDefault(opt, ['useSpatialCache'], False)

    # Whether to sample the complete temperature field at once or sample time separately.
    # Increases convergence but requires massive amounts memory.
    # If BARCAST exits with "Out of memory" or "Maximum variable size allowed by the program is exceeded.",
    # this option should be set to false. Default = false.
    setDefault(opt, ['sampleCompleteField'], False)

    return

test()
