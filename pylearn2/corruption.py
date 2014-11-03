import numpy
import theano
from theano import tensor
T = tensor
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import wraps

# Shortcuts
theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

from pylearn2.corruption import Corruptor


class Dequantizer(Corruptor):
    """
    Dequantizer corruptor
    Adressing the log-likelihood problem of having arbitrarily high
    log-likelihood due to constant features of the data. Using Tapani Raiko's
    idea to ''dequantize'' the data. Corrupting in general put an upper bound
    on the log-likelihood of the data by the entropy of the corruption process.

    Parameters
    ----------
    low : float, optional
        Lowest value of the data
    high : float, optional
        Highest value of the data
    n_values : int, optional
        Number of quantum/values of the data
    """
    def __init__(self, low=0., high=1., n_values=256, ** kwargs):
        super(Dequantizer, self).__init__(** kwargs)

        assert high > low
        self.low = low
        self.high = high
        self.n_values = n_values

    @wraps(Corruptor._corrupt)
    def _corrupt(self, x):
        # Put the data between 0 and 1
        rval = x - self.low
        rval /= (self.high - self.low)

        # Add uniform noise to dequantize
        rval *= (self.n_values - 1)
        rval += self.corruption_level * self.s_rng.uniform(
            size=x.shape,
            dtype=theano.config.floatX
        )
        rval /= (self.n_values + self.corruption_level - 1)

        # Put back in the given interval
        rval *= (self.high - self.low)
        rval += self.low

        return rval
