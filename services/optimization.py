import cvxpy as cp
import numpy as np
from services.helper import cal_theta
import scipy.stats as stats


def mega_MVO(mu, Q, x0, objective_lambda_dict, returns):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """
    tol = 1e-5
    # Find the total number of assets
    T, n = returns.shape
    # Define and solve using CVXPY
    x = cp.Variable(n,)
    z_s = cp.Variable(T)# defined for CVaR
    gamma = cp.Variable(1)# defined for CVaR
    # Objective function construct
    # MVO
    portfolio_variance = cp.quad_form(x, Q)
    portfolio_return = mu.T @ x
    # Turnover penalty
    # if all x0 is zero, then we are at the beginning of the period, no penalty for turnover
    if np.sum(x0) <= tol:
        objective_lambda_dict['turnover'] = 0
    turnover = cp.norm(x - x0, 1)
    # Robustness
    ellipsoidal_robustness = cp.norm2(x @ cp.sqrt(cal_theta(Q, T)))
    # cVaR
    coef = 1/(1-objective_lambda_dict['cvar_alpha'])*T
    cvar = gamma + coef*cp.sum(z_s)

    # objective expression dictionary
    objective_expresion_dict = {
        'portfolio_variance': portfolio_variance,
        'turnover': turnover,
        'ellipsoidal_robustness': ellipsoidal_robustness,
        'cvar': cvar
    }
    # if corresponding lambda is 0, then we do not include it in the objective function
    obj_expression = sum(
        [objective_lambda_dict[key] * objective_expresion_dict[key] for key in list(objective_lambda_dict.keys())[:-4] if
         np.abs(objective_lambda_dict[key]) >= tol])-portfolio_return
    # Objective function
    obj = cp.Minimize(obj_expression)
    # Constraints
    constraint = []
    constraint += [x >= 0, cp.sum(x) == 1]  # portfolio weights sum to 1, no short selling
    # constraint += [portfolio_return >= np.mean(mu)]  # expected return constraint
    constraint += [z_s >= np.zeros(T).T]# cvar constraint
    constraint += [z_s >= -returns.values @ x - np.ones(T) * gamma]# cvar constraint
    # Solve the problem
    prob = cp.Problem(obj,
                      constraint)
    prob.solve(verbose=False)
    return x.value




# def risk_parity(mu, Q):
#     """
#     Construct a risk parity portfolio using a numerically-efficient non-convex model.
#
#     Parameters:
#     Q (ndarray): Covariance matrix of returns.
#
#     Returns:
#     ndarray: Optimal asset weights.
#     """
#     n = Q.shape[0]
#     # n = 10
#     # Define the optimization variables
#     x = cp.Variable(n)
#     theta = cp.Variable(1)
#     # portfolio_variance = cp.quad_form(x, Q)
#     # mrc = Q @ x
#     # risk_contribution = cp.multiply(x, mrc)/portfolio_variance
#     # Define the objective function
#     # objective = cp.Minimize(cp.sum_squares(risk_contribution - theta))
#     objective = cp.Minimize(cp.sum_squares(cp.multiply(x, Q @ x) - theta))
#     # Define the constraints
#     constraints = [
#         cp.sum(x) == 1,  # Weights sum to 1
#         x >= 0  # No short sales
#     ]
#
#     # Define the problem
#     problem = cp.Problem(objective, constraints)
#
#     # Solve the problem
#     problem.solve(solver=cp.SCS)
#
#     # Get the optimal weights
#     return x.value
