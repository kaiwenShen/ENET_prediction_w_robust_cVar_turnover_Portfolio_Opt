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



def r_squared(y, y_hat):
    SS_res = np.sum((y - y_hat) ** 2, axis=0)
    SS_tot = np.sum((y - np.mean(y)) ** 2, axis=0)
    return 1 - SS_res / SS_tot


def adjusted_r_squared(y, y_hat, X_shape):
    assert len(X_shape) == 2
    n = X_shape[0]  # number of observations
    k = X_shape[1]  # number of factors
    return 1 - (1 - r_squared(y, y_hat)) * (n-1) / (n - k - 1)


def adjusted_r_squared_w_0(y, y_hat, beta, X_shape):
    """
    this is the function that iteratively calculate the adjusted r2 for each stock in y
    since the given beta may contain 0, we should not penalize those in adjusted r_squared
    """
    tol = 1e-6
    res_adj_r2 = []
    for i in range(y.shape[1]):
        y_i = y[:, i]
        y_hat_i = y_hat[:, i]
        num_of_factors = np.sum(np.abs(beta[:, i]) > tol)
        n = X_shape[0]
        res = adjusted_r_squared(y_i, y_hat_i, (n, num_of_factors))
        res_adj_r2.append(res)
    return np.array(res_adj_r2)
