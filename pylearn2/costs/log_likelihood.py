"""
Non-linear independent components estimation cost
"""
__authors__ = "Laurent Dinh"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh"]
__license__ = "3-clause BSD"
__maintainer__ = "Laurent Dinh"
__email__ = "dinhlaur@iro"

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import wraps
import theano
import theano.tensor as T


class NegativeLogLikelihood(DefaultDataSpecsMixin, Cost):
    supervised = False

    @wraps(Cost.expr)
    def expr(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return -T.cast(model.get_log_likelihood(data).mean(),
                       theano.config.floatX)
