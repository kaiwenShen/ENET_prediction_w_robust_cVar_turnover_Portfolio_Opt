import numpy as np
import cvxpy as cp
from scipy.stats import gmean
from scipy.special import softmax

def CVaR(mu, Q, Historical):
    returns = Historical
    #asset_return = gmean(returns + 1, axis=0) - 1
    alpha = 0.95
    # if returns.shape[1] < 6:
    #     raise ValueError("The matrix must have at least 6 columns for the operation to be valid.")
    # returns = returns[ 5::6, :]
    # returns = np.power(returns, 6)
    #returns = np.repeat(returns, 6, axis=0)

    S, n = returns.shape
    print("Shape of HISTORICAL:", S,n)
    print("Shape of MU:", mu)
# %   min     gamma + (1 / [(1 - alpha) * S]) * sum( z_s )
# %   s.t.    z_s   >= 0,                 for s = 1, ..., S
# %           z_s   >= -r_s' x - gamma,   for s = 1, ..., S
# %           1' x  =  1,
# %           mu' x >= R
    k = (1 / (1 - alpha) * S)
    x = cp.Variable(n)
    Zs = cp.Variable(S)
    gamma = cp.Variable(1)
    obj = cp.Minimize(gamma + k*cp.sum(Zs))
    constraint = []
    constraint += [x >= 0, cp.sum(x) == 1]  # portfolio weights sum to 1, no short selling
    constraint += [Zs >= np.zeros(S).T]
    constraint += [Zs >= -returns @ x-np.ones(S).T*gamma]
    constraint += [mu @ x >= np.mean(mu)]

    prob = cp.Problem(obj,constraint)
    prob.solve(verbose=False)
    print("Result:", x.value,"with sum", np.sum(list(x.value)))
    return x.value