


import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


#@partial(jax.jit, static_argnums=(1))
#@jax.jit
def reulermultinom(key, n, rates, dt, shape=()):
    sumrates = np.sum(rates)
    logp0 = -sumrates * dt
    logits = np.insert(np.log(-np.expm1(logp0)) + np.log(rates) - np.log(sumrates), 0, logp0)
    return tfp.distributions.Multinomial(n, logits=logits).sample(
        seed=key,
        sample_shape=shape,
    )[1:]

#@jax.jit
def deulermultinom(x, n, rates, dt):
    sumrates = np.sum(rates)
    logp0 = -sumrates * dt
    logits = np.insert(np.log(-np.expm1(logp0)) + np.log(rates) - np.log(sumrates), 0, logp0)
    return tfp.distributions.Multinomial(n, logits=logits).log_prob(
        np.insert(x, 0, n-np.sum(x))
    )

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))

def get_thetas(theta):

    rho = sigmoid(theta[0])
    tau1 = np.exp(theta[1])
    tau2 = np.exp(theta[2])
    bs = np.exp(theta[3:9])
    nu = sigmoid(theta[9])
    sig_sq1 = np.exp(theta[10])
    sig_sq2 = np.exp(theta[11])
    beta_t = theta[12]

    return rho, tau1, tau2, bs, nu, sig_sq1, sig_sq2, beta_t
    
def transform_thetas(rho, tau1, tau2, bs, nu, sig_sq1, sig_sq2, beta_t):
    return np.concatenate([np.array([logit(rho), np.log(tau1), np.log(tau2)]), 
                            np.log(bs), 
                       np.array([logit(nu), np.log(sig_sq1), np.log(sig_sq2),
                            beta_t])]) 


def get_rand_theta(J=10000):
    lows = transform_thetas(0.2, 180, 50, 
                            onp.array([1.,0.5,0.75,0.75,1.,0.5]),
                           0.94, 0.075, 0.05, -0.15)
    highs = transform_thetas(0.95, 1250, 1000, 
                            onp.array([2.15,2.00,1.75,1.50,2.10,1.50]),
                           0.99999, 0.125, 0.2, -0.05)
    rands = onp.array(onp.repeat(((lows+highs)/2)[None,:], J, axis=0)).T
    rands[~onp.isinf(lows)] = onp.random.uniform(lows[~onp.isinf(lows)], 
                       highs[~onp.isinf(highs)],
                       size=(J, len(highs[~onp.isinf(highs)]))).T
    return rands.T



def rinit(theta, J, covars):
    #S_0, E_0, I_0, A_0, R_0 = 9.993196e-01, 1.834793e-04, 4.969569e-04, 0.0, 0.0
    S_0, E_0, I_0, A_0, R_0 = \
        0.9990317, 4.604823e-06, 9.63733e-04, 0.0, 0.0
    pop_0 = 1.0911819e+07
    frac = pop_0 / (S_0 + E_0 + I_0 + A_0 + R_0)
    S = frac * S_0 #int(frac * S_0)
    E = frac * E_0 #int(frac * E_0)
    I = frac * I_0 #int(frac * I_0)
    A = frac * A_0 #int(frac * A_0)
    R = frac * R_0 #int(frac * R_0)
    incid = 0 #frac * I_0 
    t = 0
    #count = 0
    return np.tile(np.array([S,E,I,A,R,incid,t]), (J,1))

def rinits(thetas, J, covars):
    return rinit(thetas[0], len(thetas), covars)

                  
def dmeas_helper(y, tau, v, tol, ltol):
    # mu = n(1-p)/p <=> p = n/(mu+n)
    # here, tau = n, v = mu
    p = tau / (v + tau) # calculate p from mean v
    # USAGE: jax.scipy.stats.nbinom.logpmf(k, n, p, loc=0)

    #https://github.com/atks/Rmath/blob/master/dnbinom.c
    # return np.log(tau / (y + tau)) + np.logaddexp(
    #     jax.scipy.stats.nbinom.logpmf(tau, y+tau, p), 
    #                  ltol)
    return np.logaddexp(
        jax.scipy.stats.nbinom.logpmf(y, tau, p), 
                     ltol)

def dmeas_helper_tol(y, deaths, v, tol, ltol):
    return jax.lax.select(np.isnan(y), 0.0, ltol)

def dmeas(y, preds, theta, keys=None):
    S, E, I, A, R, incid, t = preds
    tol = 1.0e-18
    ltol = np.log(tol)
    rho, tau1, tau2, bs, nu, sig_sq1, sig_sq2, beta_t = get_thetas(theta)
    tau = jax.lax.select(t < 232, tau1, tau2)
    #v = rho*incid
    v = rho * I
    # y is cases
    return jax.lax.cond(np.logical_or((1-np.isfinite(v)).astype(bool), np.isnan(y)), 
                         dmeas_helper_tol, 
                         dmeas_helper,
                       y, tau, v, tol, ltol)

dmeasure = jax.vmap(dmeas, (None,0,None))
dmeasures = jax.vmap(dmeas, (None,0,0))

def sample_and_log_prob(N, rates, dt, key):
    sample = reulermultinom(key, N, rates, dt)
    weights = deulermultinom(sample, N, rates, dt)
    key, subkey = jax.random.split(key)
    return sample, weights, key

def rproc_step(state, weight, theta, key, covar, dt):
    # We use the convention that the weight for each particle is the 
    S, E, I, A, R, incid, t = state
    rho, tau1, tau2, bs, nu, sig_sq1, sig_sq2, beta_t = get_thetas(theta)
    delta = 0.0001433 #7.5e-3
    mu = 0.0004287 #2.23e-2
    theta0 = 0.0
    gamma = 3.5
    sigma = 5.0
    alpha = 0.0023973
    t_p = t.astype(int)
    
    pop = S + E + I + A + R
    births = jax.random.poisson(key, mu * pop * dt)
    key, subkey = jax.random.split(key)

    # process noise equation:
    # rgammawn(sigma, dt) yields gamma white noise with mean dt, variance sigma^2 dt
    # so when alpha, theta are the shape, scale parameters respectively
    # draw jax.random.gamma(key, alpha) * theta to sample from that
    # constraints: dt = alpha * theta, sig_sq**2 dt = alpha * theta**2
    # solving for alpha, theta yields alpha = dt / sig_sq**2, theta = sig_sq**2
    sig_sq = jax.lax.select(t < 232, sig_sq1, sig_sq2)
    sig_num = jax.random.gamma(key, dt / sig_sq**2) * sig_sq**2
    key, subkey = jax.random.split(key)
    
    foi = I**nu * np.exp(np.dot(bs, covar[t_p]) + beta_t*((t-215)/(430-215))) / pop * sig_num / dt
    #foi = I**nu * (np.dot(bs, covar[t_p]) + beta_t*((t-215)/(430-215))) / pop #* sig_num / dt

    # We now compute the Euler-multinomial samples and weights
    # TODO: DISCOUNT AT EVERY EULER STEP
    # IDEA: Have different discounting alpha for process model and measurement model
    sample_S, weights_S, key = sample_and_log_prob(
        S, np.array([foi, delta]), dt, key
    )
    StoE, StoDeath = sample_S

    sample_E, weights_E, key = sample_and_log_prob(
        E, np.array([sigma*(1-theta0), sigma*theta0, delta]), dt, key
    )
    EtoI, EtoA, EtoDeath = sample_E

    sample_I, weights_I, key = sample_and_log_prob(
        I, np.array([gamma, delta]), dt, key
    )
    ItoR, ItoDeath = sample_I

    sample_A, weights_A, key = sample_and_log_prob(
        A, np.array([gamma, delta]), dt, key
    )
    AtoR, AtoDeath = sample_A
    
    sample_R, weights_R, key = sample_and_log_prob(
        R, np.array([alpha, delta]), dt, key
    )
    RtoS, RtoDeath = sample_R

    S += - StoE - StoDeath + RtoS + births
    E += - EtoI - EtoA - EtoDeath + StoE
    I += - ItoR - ItoDeath + EtoI
    A += - AtoR - AtoDeath + EtoA
    R += - RtoS - RtoDeath + ItoR + AtoR
    incid += EtoI
    
    t += dt
    weight += weights_S + weights_E + weights_I + weights_A + weights_R
        
    return np.array([S,E,I,A,R,incid,t]), weight
            
def euler(rproc_step, dt):
    
    def rproc_step_helper_euler(i, inputs):
        state, weight, theta, key, covar, dt = inputs
        state, weight = rproc_step(state, weight, theta, key, covar, dt)
        return [state, weight, theta, key, covar, dt]

    # Try using naive for loop. Unrolled on GPU, so faster.
    def rproc_euler(state, theta, key, covar, dt=dt):
        state, weight, theta, key, covar, dt = jax.lax.fori_loop(lower=0, upper=int(1/dt), body_fun = rproc_step_helper_euler,
                                                    init_val = [state, np.array(0.), theta, key, covar, dt])
        return state
        
    return rproc_euler
    

rproc = euler(rproc_step, 1/7)

rprocess = jax.vmap(rproc, (0, None, 0, None))
rprocesses = jax.vmap(rproc, (0, 0, 0, None))

            
def rproc_step_helper(i, inputs):
    state, weight, theta, key, covar, dt, alpha = inputs
    weight = alpha*weight# jax.lax.select(alpha>0, alpha*weightweight, np.array(0.))
    state, weight = rproc_step(state, weight, theta, key, covar, dt)
    return [state, weight, theta, key, covar, dt, alpha]
    # return [state, jax.lax.select(alpha>0, alpha*weight, weight), 
    #                                 theta, key, covar, dt, alpha]

# Try using naive for loop. Unrolled on GPU, so faster.
def rproc_weight(state, theta, key, covar, dt, alpha):
    state, weight, theta, key, covar, dt, alpha = jax.lax.fori_loop(lower=0, upper=int(1/dt), body_fun = rproc_step_helper,
                                                init_val = [state, np.array(0.), theta, key, covar, dt, alpha])
    return state, weight
    #return state, jax.lax.select(alpha>0, weight/alpha, weight)



rprocess_weight = jax.vmap(rproc_weight, (0, None, 0, None,None,None))
rprocesses_weight = jax.vmap(rproc_weight, (0, 0, 0, None,None,None))

#particlesP, weightC = rprocess_weight(particlesF, theta, keys, covars)