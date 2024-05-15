import cvxpy as cp
import numpy as np
from services.helper import cal_theta
import scipy.stats as stats


def MVO(mu, Q, x0, objective_lambda_dict, returns):
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
    T,n = returns.shape
    # Define and solve using CVXPY
    x = cp.Variable(n)
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
    # objective expression dictionary
    objective_expresion_dict = {
        'portfolio_variance': portfolio_variance,
        'turnover': turnover,
        'ellipsoidal_robustness': ellipsoidal_robustness
    }
    # if corresponding lambda is 0, then we do not include it in the objective function
    obj_expression = sum(
        [objective_lambda_dict[key] * objective_expresion_dict[key] for key in objective_lambda_dict.keys() if
         np.abs(objective_lambda_dict[key]) >= tol])
    # Objective function
    obj = cp.Minimize(obj_expression)
    # Constraints
    constraint = []
    constraint += [x >= 0, cp.sum(x) == 1]  # portfolio weights sum to 1, no short selling
    constraint += [portfolio_return >= np.mean(mu)]  # expected return constraint
    # Solve the problem
    prob = cp.Problem(obj,
                      constraint)
    prob.solve(verbose=False)
    return x.value
