from services.strategies import *


def project_function(periodReturns, periodFactRet, x0, objective_lambda_dict):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    if objective_lambda_dict['prediction_model'] == 0:
        Strategy = OLS_MVO()
        x, r2 = Strategy.execute_strategy(periodReturns, periodFactRet, x0, objective_lambda_dict)
    else:
        Strategy = ENET_MVO()
        x, r2 = Strategy.execute_strategy(periodReturns, periodFactRet, x0, objective_lambda_dict)
    return x, r2
