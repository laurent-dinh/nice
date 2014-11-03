"""
Non-linear independent components estimation and related classes
"""
__authors__ = "Laurent Dinh"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh"]
__license__ = "3-clause BSD"
__maintainer__ = "Laurent Dinh"
__email__ = "dinhlaur@iro"

import numpy as np
import theano
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.models.mlp import Layer, Linear, MLP
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import sharedX, wraps, safe_update, serial
from pylearn2.utils.rng import make_np_rng, make_theano_rng
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models import Model
from theano.compat.python2x import OrderedDict

pi = sharedX(np.pi)

default_seed = [27, 9, 2014]


class Distribution(Model):
    """
    Abstract class implementing methods related to distribution.
    """
    def __init__(self, ** kwargs):
        super(Distribution, self).__init__(** kwargs)
        self._params = []

    def get_log_likelihood(self, z):
        """
        Compute the log-likelihood of a batch according to this distribution.

        Parameters
        ----------
        z : tensor_like
            Point whose log-likelihood to estimate

        Returns
        -------
        log_likelihood : tensor_like
            Log-likelihood of z
        """
        raise NotImplementedError(str(type(self))
                                  + " does not implement log_p_z.")

    def sample(self, shape):
        """
        Sample from this distribution with the given shape.

        Parameters
        ----------
        shape : tuple
            Shape of the batch to sample

        Returns
        -------
        samples : tensor_like
            Samples
        """
        raise NotImplementedError(str(type(self))
                                  + " does not implement sample.")

    def entropy(self, shape):
        """
        Get entropy from this distribution with the given shape.

        Parameters
        ----------
        shape : tuple
            Shape of the batch to sample

        Returns
        -------
        samples : tensor_like
            Entropy
        """
        raise NotImplementedError(str(type(self))
                                  + " does not implement entropy.")

    def set_rng(self, rng):
        """
        SSetup the theano random generator for this class.

        Parameters
        ----------
        rng : np.random.RandomState
            Random generator from which to generate the seed of
            the theano random generator
        """
        self.rng = rng
        self.theano_rng = make_theano_rng(int(self.rng.randint(2 ** 30)),
                                          which_method=["normal", "uniform"])


class StandardNormal(Distribution):
    @wraps(Distribution.get_log_likelihood)
    def get_log_likelihood(self, Z):
        log_likelihood = -.5 * (T.sqr(Z) + sharedX(np.log(2 * np.pi)))
        log_likelihood = log_likelihood.sum(axis=-1)

        return log_likelihood

    @wraps(Distribution.sample)
    def sample(self, shape):
        samples = self.theano_rng.normal(size=shape,
                                         dtype=theano.config.floatX)
        return samples

    @wraps(Distribution.entropy)
    def entropy(self, shape):
        entropy = .5 * sharedX(np.log(2 * np.pi) + 1.)
        entropy *= T.ones(shape)
        entropy = entropy.sum(axis=-1)

        return entropy


class StandardLogistic(Distribution):
    @wraps(Distribution.get_log_likelihood)
    def get_log_likelihood(self, Z):
        log_likelihood = - (T.nnet.softplus(Z) + T.nnet.softplus(-Z))
        log_likelihood = log_likelihood.sum(axis=-1)

        return log_likelihood

    @wraps(Distribution.sample)
    def sample(self, shape):
        samples = self.theano_rng.uniform(size=shape)
        samples = T.log(samples) - T.log(1-samples)
        return samples

    @wraps(Distribution.entropy)
    def entropy(self, shape):
        entropy = 2.
        entropy *= T.ones(shape)
        entropy = entropy.sum(axis=-1)

        return entropy


class NICE(Distribution):
    """
    Non-linear independent components estimation.

    Parameters
    ----------
    encoder: pylearn2.models.mlp.MLP
        Encoder model to transform the data. Every layers has to be bijective.
        Note that this MLP is used as a MLP-layer, encoder.input_space and
        encoder.nvis should not be defined.
    prior: nice.pylearn2.models.nonlinear_ica.Distribution
        Prior distribution model for the code
    nvis : int, optional
        Number of "visible units" (input units). Equivalent to specifying
        `input_space=VectorSpace(dim=nvis)`. Note that certain methods require
        a different type of input space (e.g. a Conv2Dspace in the case of
        convnets). Use the input_space parameter in such cases.
    input_space : Space object, optional
        A Space specifying the kind of input the encoder accepts. If None,
        input space is specified by nvis.
    corruptor: pylearn2.corruption.Corruptor, optional
        Corrupt the data before inputting to the encoder. It is optional but
        is often necessary.
    kwargs : dict
        Passed on to superclass constructor.
    """
    def __init__(self, prior, encoder=None,
                 nvis=None, input_space=None,
                 corruptor=None,
                 batch_size=None,
                 seed=None,
                 ** kwargs):
        super(NICE, self).__init__(** kwargs)

        self.__dict__.update(locals())
        del self.self

        self.rng = make_np_rng(self.seed, default_seed,
                               ['uniform', 'randint', 'randn'])

        # Define input space and source related stuff
        assert (input_space is None) or (nvis is None)
        assert (input_space is not None) or (nvis is not None)
        if nvis is not None:
            self.input_space = VectorSpace(nvis)
        if input_space is not None:
            self.nvis = input_space.get_total_dimension()
        self.input_source = 'features'

        self.prior.set_rng(self.rng)
        self._params = self.prior.get_params()

        if encoder is not None:
            # We only support MLP encoder
            assert isinstance(encoder, MLP)
            # We only support bijective layers with given interface
            assert hasattr(encoder, 'inv_fprop')
            assert hasattr(encoder, 'get_fprop_and_log_det_jacobian')

            if not hasattr(self.encoder, 'input_space'):
                self.encoder.set_mlp(self)
                self.encoder.set_input_space(self.input_space)
            self.output_space = self.encoder.get_output_space()

            # Set the parameters
            for param in self.encoder.get_params():
                param.name = self.encoder.layer_name\
                    + '_' + param.name

            self._params += self.encoder.get_params()
            self._params.extend(self.prior.get_params())
            self.layers = self.encoder.layers

    def get_log_likelihood(self, X):
        """
        Compute the log-likelihood of a batch according to
        the model distribution.

        Parameters
        ----------
        X : tensor_like
            Input

        Returns
        -------
        log_likelihood : tensor_like
            Log-likelihood of the input for the model distribution
        """
        X_in = X

        # Corrupt the data if possible
        if self.corruptor is not None:
            X_in = self.corruptor(X_in)

        Z, log_det_jac = self.get_fprop_and_log_det_jacobian(X_in)

        prior = self.log_p_z(Z)
        transformation_extension = log_det_jac

        log_likelihood = prior + transformation_extension

        return log_likelihood

    def log_p_z(self, Z):
        """
        Compute the log-likelihood of a point according to
        the prior distribution.

        Parameters
        ----------
        z : tensor_like, member of self.output_space
            Code from the encoder

        Returns
        -------
        log_likelihood : tensor_like
            Log-likelihood of z for the prior distirbution
        """
        return self.prior.get_log_likelihood(Z)

    @wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        self.prior.modify_updates(updates)
        if self.encoder is not None:
            self.encoder.modify_updates(updates)

    def sample(self, num_samples):
        """
        Sample from the model's learned distribution

        Parameters
        ----------
        num_samples : int
            Number of samples

        Returns
        -------
        samples : tuple of tensor_like
            Samples. The first element of the tuple is the actual sample, the
            others are intermediate quantities.
        """
        if isinstance(num_samples, tuple):
            shape = (num_samples[0], self.nvis)
        else:
            shape = (num_samples, self.nvis)
        Z = self.prior.sample(shape)

        samples = Z
        if self.encoder is not None:
            samples = self.encoder.inv_fprop(Z)
        return samples

    def encode(self, X):
        """
        Encode the data.

        Parameters
        ----------
        X : tensor_like, member of self.input_space
            Input

        Returns
        -------
        Z : tensor_like, member of self.output_space
            Code from the encoder
        """

        Z = X
        if self.encoder is not None:
            Z = self.encoder.fprop(X)
        return Z

    def get_fprop_and_log_det_jacobian(self, X):
        """
        Get the state of the layer and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        X : tensor_like, member of self.input_space
            A minibatch of states of the layer below.

        Returns
        -------
        Z : tensor_like, member of self.output_space
            Code from the encoder.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation.
        """
        Z, log_det_jac = X, 0.
        if self.encoder is not None:
            Z, log_det_jac = self.encoder.get_fprop_and_log_det_jacobian(X)
        return Z, log_det_jac

    @wraps(Model.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.input_space, self.input_source

    @wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        rval = OrderedDict()
        if self.encoder is not None:
            rval = self.encoder.get_layer_monitoring_channels(
                state_below=data
            )
        return rval

    @wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):
        rval = OrderedDict()
        if self.encoder is not None:
            safe_update(rval, self.encoder.get_lr_scalers())
        return rval
