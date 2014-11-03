"""
MLP related additional classes
"""
__authors__ = "Laurent Dinh"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh"]
__license__ = "3-clause BSD"
__maintainer__ = "Laurent Dinh"
__email__ = "dinhlaur@iro"

import numpy as np
import scipy
import scipy.linalg
import theano
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.models.mlp import Layer, Linear, MLP
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import sharedX, wraps, as_floatX
from pylearn2.utils.rng import make_theano_rng
from pylearn2.linear.matrixmul import MatrixMul
from theano.compat.python2x import OrderedDict

pi = sharedX(np.pi)
T_inv = T.nlinalg.MatrixInverse()
T_det = T.nlinalg.Det()


class TriangularMLP(MLP):
    """
    Triangular MLP, a MLP of bijective layers.
    (see pylearn2.models.mlp for arguments)

    """
    def inv_fprop(self, state, return_all=False):
        """
        Inversion of the MLP forward propagation.

        Parameters
        ----------
        state : tensor_like, member of self.output_space
            The state above the MLP

        Returns
        -------
        state_below : tensor_like
            The resulting state below
        """
        state_below = state

        if return_all:
            state_below = [state_below]

        for layer in self.layers[::-1]:
            if return_all:
                state_below.append(layer.inv_fprop(state_below[-1]))
            else:
                state_below = layer.inv_fprop(state_below)

        return state_below

    def get_fprop_and_log_det_jacobian(self, state_below):
        """
        Get the state of the MLP and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        state_below : tensor_like, member of self.input_space
            A minibatch of states below.

        Returns
        -------
        state : tensor_like, member of self.output_space
            A minibatch of states of this MLP.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation
        """
        state = state_below
        log_det_jac = 0.

        for layer in self.layers:
            state, log_det_jac_layer = layer.get_fprop_and_log_det_jacobian(
                state
            )
            log_det_jac += log_det_jac_layer

        return state, log_det_jac


class CouplingLayer(Layer):
    """
    Coupling layer. Divide the data in two halves and increment the second
    half by a function of the first.

    Parameters
    ----------
    coupling : pylearn2.models.mlp.MLP
        The coupling model, the function of the first half added to the second.
    split : int
        Index at which the data is split. The first half is X[:,:split]
    shuffle : bool, optional
        Whether the output is shuffled after the layer. Default is False.
        Shuffle trumps reverse.
    reverse : bool, optional
        Whether the indices of the output is reversed. Default is True.
    kwargs : dict
        Passed on to the superclass.
    """
    def __init__(self, coupling, split, shuffle=False, reverse=True,
                 layer_name=None, ** kwargs):
        super(CouplingLayer, self).__init__(** kwargs)
        self.coupling = coupling
        self.split = split
        self.shuffle = shuffle
        self.reverse = reverse
        if layer_name is None:
            self.layer_name = coupling.layer_name
        else:
            self.layer_name = layer_name

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):
        self.mlp = mlp
        self.coupling.mlp = mlp

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        space_dim = space.get_total_dimension()
        assert self.split <= space_dim
        self.input_space = space
        self.output_space = space

        # Setup the permutation operation
        self.permutation = np.arange(space_dim)
        if self.shuffle:
            self.mlp.rng.shuffle(self.permutation)
        elif self.reverse:
            self.permutation = self.permutation[::-1]

        coupling_input_space = VectorSpace(dim=self.split)
        self.coupling.set_input_space(coupling_input_space)

        coupling_output_space = self.coupling.get_output_space()
        self.scalar = (coupling_output_space.get_total_dimension() == 1)
        assert (coupling_output_space.get_total_dimension()
                == space_dim - self.split) or self.scalar

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        coupling_out = self.coupling.fprop(state_below[:, :self.split])
        state = T.inc_subtensor(state_below[:, self.split:], coupling_out)

        state = state[:, self.permutation]

        return state

    def inv_fprop(self, state):
        state_below = state
        state_below = state_below[:, np.argsort(self.permutation)]
        coupling_out = -self.coupling.fprop(state_below[:, :self.split])
        state_below = T.inc_subtensor(state_below[:, self.split:],
                                      coupling_out)

        return state_below

    def get_fprop_and_log_det_jacobian(self, state_below):
        log_det_jac = 0.

        return self.fprop(state_below), log_det_jac

    @wraps(Layer.get_params)
    def get_params(self):
        return self.coupling.get_params()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return self.coupling.get_weight_decay(coeffs)

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return self.coupling.get_l1_weight_decay(coeffs)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        self.coupling.modify_updates(updates)


class Homothety(Layer):
    def __init__(self, layer_name, ** kwargs):
        super(Homothety, self).__init__(** kwargs)
        self.layer_name = layer_name

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space
        dim = space.get_total_dimension()
        self.D = sharedX(np.zeros((dim,)), self.layer_name+'_D')
        self._params = [self.D]

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state = state_below * T.exp(self.D).flatten()

        return state

    def inv_fprop(self, state):
        """
        Inversion of the Homothety forward propagation.

        Parameters
        ----------
        state : tensor_like, member of self.output_space
            The state above the layer

        Returns
        -------
        state_below : tensor_like
            The resulting state below
        """
        state_below = state * T.exp(-self.D).flatten()

        return state_below

    def get_fprop_and_log_det_jacobian(self, state_below):
        """
        Get the state of the layer and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        state_below : tensor_like, member of self.input_space
            A minibatch of states below.

        Returns
        -------
        state : tensor_like, member of self.output_space
            A minibatch of states of this layer.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation
        """
        return self.fprop(state_below), self.D.sum()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return coeffs * T.sqr(self.D).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return coeffs * abs(self.D).sum()


class Reordering(Layer):
    def __init__(self, layer_name, mode='tile', ** kwargs):
        super(Reordering, self).__init__(** kwargs)
        assert mode in ['tile', 'shuffle', 'reverse']
        self.layer_name = layer_name
        self.mode = mode

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space
        self.dim = space.get_total_dimension()
        self.half_dim = self.dim / 2
        self.permutation = np.arange(self.dim)
        if self.mode == 'tile':
            tmp = self.permutation.copy()
            self.permutation[:-self.half_dim] = tmp[::2]
            self.permutation[-self.half_dim:] = tmp[1::2]
        elif self.mode == 'shuffle':
            self.mlp.rng.shuffle(self.permutation)
        elif self.mode == 'reverse':
            self.permutation = self.permutation[::-1]
        self._params = []

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state = state_below[:, self.permutation]

        return state

    def inv_fprop(self, state):
        """
        Inversion of the Reordering forward propagation.

        Parameters
        ----------
        state : tensor_like, member of self.output_space
            The state above the layer

        Returns
        -------
        state_below : tensor_like
            The resulting state below
        """
        state_below = state[:, np.argsort(self.permutation)]

        return state_below

    def get_fprop_and_log_det_jacobian(self, state_below):
        """
        Get the state of the layer and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        state_below : tensor_like, member of self.input_space
            A minibatch of states below.

        Returns
        -------
        state : tensor_like, member of self.output_space
            A minibatch of states of this layer.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation
        """
        return self.fprop(state_below), 0.

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return 0.

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return 0.
