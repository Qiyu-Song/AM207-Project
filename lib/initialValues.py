import numpy as np

INSTRU = 0  # so that data[INSTRU] = instrumental


def initialValues(data, model, options):
    # Initializes the parameter values for the MCMC procedure
    # Draws from (at times) truncated priors.
    currentParams = {}

    if options['useModes']:
        # Initial Value of alpha: (uniform, so take mean):
        currentParams['alpha'] = np.mean(options['priors']['alpha'][:])

        # Initial Value for mu: mode of the normal prior.
        currentParams['mu'] = options['priors']['mu'][0]

        # Initial Value for sigma2: mode of inverse gamma prior:
        currentParams['sigma2'] = options['priors']['sigma2'][1] / (options['priors']['sigma2'][0] + 1)

        # Initial Value for phi: mode of the log-normal distribution:
        currentParams['phi'] = np.exp(options['priors']['phi'][0] - options['priors']['phi'][1])

        # Initial Value for tau2_I: : mode of inverse gamma prior:
        currentParams['tau2_I'] = options['priors']['tau2_I'][1] / (options['priors']['tau2_I'][0] + 1)

        # Initial Value for tau2_P: : mode of inverse gamma prior:
        currentParams['tau2_P'] = options['priors']['tau2_P'][:, 1] / (options['priors']['tau2_P'][:, 0] + 1)

        # Initial value for Beta_1: mode of the normal prior:
        currentParams['Beta_1'] = options['priors']['Beta_1'][:, 0]

        # Initial value for Beta_0: mode of the normal prior:
        currentParams['Beta_0'] = options['priors']['Beta_0'][:, 0]

    else:
        # Initial Value of alpha: Draw from the uniform prior:
        # currentParams.alpha=rand(1)*diff(options.priors.alpha)+options.priors.alpha(1);
        # TRUNCATE A Bit to set initial value near 0.5:
        currentParams['alpha'] = np.random.rand() * 0.5 + 0.25

        # Initial Value for mu: draw from the normal prior.
        currentParams['mu'] = options['priors']['mu'][0] + np.sqrt(options['priors']['mu'][1]) * np.random.normal()

        # Initial Value for sigma2: draw from inverse gamma prior is likely a bad ideas,
        # due to the very large possible values. So truncate to below some value:
        while True:
            t = 1 / np.random.gamma(options['priors']['sigma2'][0], 1 / options['priors']['sigma2'][1])  # right?
            if t < 5:
                currentParams['sigma2'] = t
                break

        # Initial Value for phi: draw from the log-normal distribution, truncated to less than a cutoff,
        # determined by the prior parameters:
        cutt = np.exp(options['priors']['phi'][0] + 2 * np.sqrt(options['priors']['phi'][1]))
        while True:
            t = np.random.lognormal(options['priors']['phi'][0], np.sqrt(options['priors']['phi'][1]))
            if t < cutt:
                currentParams['phi'] = t
                break

        # Initial Value for tau2_I: : draw from inverse gamma prior truncated to less than some cut off value:
        while True:
            t = 1 / np.random.gamma(options['priors']['tau2_I'][0], 1 / options['priors']['tau2_I'][1])
            if t < 5:
                currentParams['tau2_I'] = t
                break

        # Initial Value for tau2_P: : draw from inverse gamma prior truncated to less than some cut off value:
        pars = options['priors']['tau2_P'][:]
        vals = np.zeros((np.size(pars, 0), 1))
        for i in range(np.size(pars, 0)):
            while True:
                t = 1 / np.random.gamma(pars[i, 0], 1 / pars[i, 1])
                if t < 10:
                    vals[i] = t
                    break
        currentParams['tau2_P'] = vals

        # Initial value for Beta_1: mode of the normal prior:
        currentParams['Beta_1'] = options['priors']['Beta_1'][:, 0] + np.random.rand(
            np.size(options['priors']['Beta_1'], 0), 1) * np.sqrt(options['priors']['Beta_1'][:, 1:])

        # Initial value for Beta_0: mode of normal prior:
        currentParams['Beta_0'] = options['priors']['Beta_0'][:, 0] + np.random.rand(
            np.size(options['priors']['Beta_0'], 0), 1) * np.sqrt(options['priors']['Beta_0'][:, 1:])

    # Setting the initial values of the temperature matrix.
    if options['useModes']:
        # Method 3: Set all to the mode of the prior for T(0):
        currentField = options['priors']['T_0'][0] * np.ones((np.size(data[INSTRU].locations, 1)),
                                                             np.size(model['timeline']))
    else:
        # Method 2: Draw each T(k) from the prior for T(0):
        currentField = options['priors']['T_0'][0] + np.sqrt(options['priors']['T_0'][1]) * np.random.normal(
            np.size(data[INSTRU].locations, 1), np.size(model['timeline']))

    return currentParams, currentField
