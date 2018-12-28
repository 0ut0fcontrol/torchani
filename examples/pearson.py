from __future__ import division

import math
import torch

from ignite.exceptions import NotComputableError
# from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.metrics.metric import Metric


# class Pearson(_BaseRegression):
class Pearson(Metric):
    r"""
    Calculates the Mean Error:

    :math:`\text{ME} = \frac{1}{n}\sum_{j=1}^n (A_j - P_j)`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in the reference `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006
    __ http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/sphilip/pear.html

    """
    def reset(self):
        self._sum_a = 0.0
        self._sum_p = 0.0
        self._sum_aa = 0.0
        self._sum_pp = 0.0
        self._sum_ap = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        a = y.view_as(y_pred)
        self._sum_a += torch.sum(a).item()
        self._sum_p += torch.sum(y_pred).item()
        self._sum_aa += torch.sum(torch.pow(a, 2)).item()
        self._sum_pp += torch.sum(torch.pow(y_pred, 2)).item()
        self._sum_ap += torch.sum(torch.mul(y_pred, a)).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanError must have at least one example before it can be computed')
        cov_ap = self._sum_ap - (self._sum_a * self._sum_p)/self._num_examples
        cov_aa = self._sum_aa - self._sum_a ** 2 / self._num_examples
        cov_pp = self._sum_pp - self._sum_p ** 2 / self._num_examples
        return cov_ap / math.sqrt(cov_aa * cov_pp)

from torchani.ignite import DictMetric
def PearsonMetric(key):
    return DictMetric(key, Pearson())

if __name__=='__main__':
    a = torch.Tensor([1,2,3])
    p = torch.Tensor([3,2,1])
    r = Pearson()
    r.update((a,a))
    print(r.compute())
    r.update((p,a))
    print(r.compute())
