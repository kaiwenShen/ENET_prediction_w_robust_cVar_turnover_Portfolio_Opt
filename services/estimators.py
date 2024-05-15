import numpy as np
from services.helper import cal_Q, cal_mu, adjusted_r_squared_w_0
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import cvxpy as cp

def OLS(returns, factRet, obj_lambda_dict):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)
    ep = returns - X @ B
    mu = cal_mu(B, factRet)
    Q = cal_Q(B, factRet, ep)
    r2 = adjusted_r_squared_w_0(returns.values, X @ B, B, factRet.shape)
    return mu, Q, r2


# def ElasticNet_reg(returns, factRet, obj_lambda_dict):
#     # Use this function to perform a ElasticNet regression with all factors.
#     [T, p] = factRet.shape
#
#     # Data matrix
#     X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)
#     # Regression coefficients
#     if obj_lambda_dict['enet_lambda'] <= 0.01:
#         # then our model is ridge.
#         regr = Ridge(alpha=obj_lambda_dict['enet_alpha']*abs(obj_lambda_dict['enet_lambda']),
#                      random_state=0,
#                      fit_intercept=False,
#                      max_iter=10000,)
#     elif obj_lambda_dict['enet_lambda'] >= 1:
#         # then our model is lasso.
#         regr = Lasso(alpha=obj_lambda_dict['enet_alpha']*abs(obj_lambda_dict['enet_lambda']-1),
#                      random_state=0,
#                      fit_intercept=False,
#                      max_iter=10000,)
#     else:
#         regr = ElasticNet(alpha=obj_lambda_dict['enet_alpha'],
#                       l1_ratio=obj_lambda_dict['enet_lambda'],
#                       random_state=0,
#                       fit_intercept=False,
#                       max_iter=10000,)
#     regr.fit(X, returns.values)
#     B = regr.coef_.T
#     ep = returns - X @ B
#     mu = cal_mu(B, factRet)
#     Q = cal_Q(B, factRet, ep)
#     r2 = adjusted_r_squared_w_0(returns.values, X @ B, B, factRet.shape)
#     return mu, Q, r2


def ElasticNet_reg(returns, factRet, obj_lambda_dict):
    # Use this function to perform a ElasticNet regression with all factors.
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)
    beta = cp.Variable((X.shape[1], returns.shape[1]))
    residual = returns.values - X @ beta
    if obj_lambda_dict['enet_lambda'] <= 0.01:
        # then we do ridge using cvxpy
        obj = cp.Minimize(cp.sum_squares(residual) + obj_lambda_dict['enet_alpha']*abs(obj_lambda_dict['enet_lambda']) * cp.sum_squares(beta))
    elif obj_lambda_dict['enet_lambda'] >= 1:
        # then we do lasso using cvxpy
        dof_penalty =1/(2*T)
        obj = cp.Minimize(dof_penalty*cp.sum_squares(residual) + obj_lambda_dict['enet_alpha']*abs(obj_lambda_dict['enet_lambda']-1)* cp.norm1(beta))
    else:
        # then we do elastic net using cvxpy
        dof_penalty = 1 / (2 * T)
        obj = cp.Minimize(dof_penalty*cp.sum_squares(residual) + obj_lambda_dict['enet_alpha']*obj_lambda_dict['enet_lambda'] * cp.norm1(beta) + obj_lambda_dict['enet_alpha']*(1-obj_lambda_dict['enet_lambda']) * cp.sum_squares(beta))
    prob = cp.Problem(obj)
    prob.solve(verbose=False)
    B = beta.value
    ep = returns - X @ B
    mu = cal_mu(B, factRet)
    Q = cal_Q(B, factRet, ep)
    r2 = adjusted_r_squared_w_0(returns.values, X @ B, B, factRet.shape)
    return mu, Q, r2