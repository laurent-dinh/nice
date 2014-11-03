"""
Additional learning rules
"""
__authors__ = "Laurent Dinh"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh"]
__license__ = "3-clause BSD"
__maintainer__ = "Laurent Dinh"
__email__ = "dinhlaur@iro"

import numpy as np

from theano import config
from theano import tensor as T

from theano.compat.python2x import OrderedDict
from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import sharedX, wraps

from pylearn2.training_algorithms.learning_rule import LearningRule, Momentum


class RMSPropMomentum(Momentum):
    """
    RMSProp with momentum.

    """

    def __init__(self,
                 init_momentum,
                 averaging_coeff=0.95,
                 stabilizer=1e-2):

        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

    @wraps(Momentum.get_updates)
    def get_updates(self, learning_rate, grads, lr_scalers=None):

        updates = OrderedDict()
        for param in grads.keys():

            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            momentum = sharedX(np.zeros_like(param.get_value()))

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr \
                + (1 - self.averaging_coeff) \
                * T.sqr(grads[param])

            rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            normalized_grad = grads[param] / (rms_grad_t)
            new_momentum = self.momentum * momentum \
                - learning_rate * normalized_grad

            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[momentum] = new_momentum
            updates[param] = param + new_momentum

        return updates
