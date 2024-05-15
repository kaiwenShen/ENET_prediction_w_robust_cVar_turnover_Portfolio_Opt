import cvxpy as cp
import numpy as np


def MVO(mu, Q):
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

    # Find the total number of assets
    n = len(mu)
    # Define and solve using CVXPY
    x = cp.Variable(n)
    # Objective function construct
    portfolio_variance = cp.quad_form(x, Q)
    portfolio_return = mu.T @ x
    obj = cp.Minimize(portfolio_variance - portfolio_return)
    # Constraints
    constraint = []
    constraint += [x >= 0, cp.sum(x) == 1]  # portfolio weights sum to 1, no short selling
    # Solve the problem
    prob = cp.Problem(obj,
                      constraint)
    prob.solve(verbose=False)
    return x.value
