import numpy as np

def cal_mu(beta, factRet):
    a = beta[0, :]
    V = beta[1:, :]
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    return mu

def cal_Q(beta, factRet, residuals):
    """compute the vcov matrix with formula Q = B'FB + delta"""
    T = len(residuals)
    tol = 1e-6
    V = beta[1:, :]
    p = np.apply_along_axis(lambda x: np.sum(np.abs(x) > tol), axis=0, arr=V)
    F = factRet.cov().values
    sigma_ep = 1 / (T - p - 1) * np.sum(residuals.pow(2), axis=0)
    D = np.diag(sigma_ep)
    Q = V.T @ F @ V + D
    Q = (Q + Q.T) / 2
    return Q

def cal_theta(Q,T):
    theta = 1/T * np.diag(Q)
    return np.diag(theta)
