import cvxpy as cp
import numpy as np


def MVO(mu, Q, x0):
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
    n = len(mu)
    # Define and solve using CVXPY
    x = cp.Variable(n)
    # Objective function construct
    portfolio_variance = cp.quad_form(x, Q)
    portfolio_return = mu.T @ x
    # if all x0 is zero, then we are at the beginning of the period, no penalty for turnover
    if np.sum(x0) <= tol:
        lambda_turnover = 0
    else:
        lambda_turnover = 0
    turnover = cp.norm(x - x0, 1)

    objective_lambda_dict = {
        'portfolio_variance': 1,
        'turnover': lambda_turnover
    }
    objective_expresion_dict = {
        'portfolio_variance': portfolio_variance,
        'turnover': turnover
    }

    # if corresponding lambda is 0, then we do not include it in the objective function
    obj_expression = sum(
        [objective_lambda_dict[key] * objective_expresion_dict[key] for key in objective_lambda_dict.keys() if
         np.abs(objective_lambda_dict[key]) >= tol])
    obj = cp.Minimize(obj_expression)
    # obj = cp.Minimize(portfolio_variance)
    # Constraints
    constraint = []
    constraint += [x >= 0, cp.sum(x) == 1]  # portfolio weights sum to 1, no short selling
    constraint += [portfolio_return >= np.mean(mu)]  # expected return constraint
    # Solve the problem
    prob = cp.Problem(obj,
                      constraint)
    prob.solve(verbose=False)
    return x.value
