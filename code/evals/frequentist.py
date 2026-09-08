"""frequentist.py
Frequentist metrics for evaluating forecasts.
Author(s): Felix Jimenez
"""

import torch
from scipy.stats import norm


def emp_coverage(
    y: torch.Tensor, 
    p:torch.distributions.Normal, 
    alpha:float = 0.05) -> float:
    """Compute empirical coverage for true values (y), 
    forecasts (p), and :math:`\\alpha`. 

    :param y: Observed values
    :type y: torch.Tensor
    :param p: A forecast for each value of y.
    :type p: torch.distributions.Normal
    :param alpha: coverage level, defaults to 0.05
    :type alpha: float, optional
    :return: The empirical coverage proportion.
    :rtype: float
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1).")
    if len(y.shape) > 1:
        raise ValueError("y must be flattened.")
    if len(p.mean.shape) > 1:
        raise ValueError("forecast mean must be flat.")
    if p.mean.shape[0] != y.shape[0]:
        raise ValueError("Shape of y must match that of p.")
    if not isinstance(p, torch.distributions.Normal):
        raise ValueError("Forecast must be of type torch.distributions.Normal.")

    interval = torch.Tensor(norm.ppf(1.0 - alpha / 2.0, scale= p.variance.sqrt().squeeze()))
    z = y - p.mean.squeeze()
    in_interval = (z > -1 * interval) & (z < interval)
    return torch.mean(1.0 * in_interval).item()