__authors__ = ["Laurent Dinh", "Vincent Dumoulin"]

import numpy as np
import theano
import theano.tensor as T
from pylearn2.utils import serial
from theano.compat.python2x import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.mnist import MNIST
import os, sys
import os.path

# Loading model
_, model_path = sys.argv
model = serial.load(model_path)
print "Model loaded"

# Defining the number of examples
cols = 10
rows = 10

n_examples = rows * cols
input_dim = 784
X_shared_val = np.random.uniform(size=(n_examples, input_dim))
X_shared = sharedX(X_shared_val, 'X_shared')
print "Initialize"

def show(vis_batch, dataset, mapback, pv, rows, cols, save_path=None):
    vis_batch_subset = vis_batch[:(rows * cols)]

    display_batch = dataset.adjust_for_viewer(vis_batch_subset)
    if display_batch.ndim == 2:
        display_batch = dataset.get_topological_view(display_batch)
    display_batch = display_batch.transpose(tuple(
        dataset.X_topo_space.axes.index(axis) for axis in ('b', 0, 1, 'c')
    ))
    if mapback:
        design_vis_batch = vis_batch_subset
        if design_vis_batch.ndim != 2:
            design_vis_batch = dataset.get_design_matrix(design_vis_batch)
        mapped_batch_design = dataset.mapback_for_viewer(design_vis_batch)
        mapped_batch = dataset.get_topological_view(mapped_batch_design)
    for i in xrange(rows):
        row_start = cols * i
        for j in xrange(cols):
            pv.add_patch(display_batch[row_start+j, :, :, :],
                         rescale=False)
            if mapback:
                pv.add_patch(mapped_batch[row_start+j, :, :, :],
                             rescale=False)
    if save_path is None:
        plt.imshow(pv.image)
        plt.axis('off')
    else:
        pv.save(save_path)


# Create the iterative reconstruction function
X = T.matrix('X')
M = T.imatrix('M')

X_complete = T.where(M, X, X_shared)
ll = model.get_log_likelihood(X_complete)

grad = T.grad(ll.mean(), X_shared, disconnected_inputs='warn')
updates = OrderedDict()

lr = T.scalar('lr')
is_noise = sharedX(0., 'is_noise')
updates[X_shared] = X_shared + lr * (grad + model.prior.theano_rng.normal(size=X_shared.shape))
updates[X_shared] = T.where(M, X, updates[X_shared])

f = theano.function([X, M, lr], [ll.mean()], updates=updates, allow_input_downcast=True)
print 'Compiled training function'

# Setup for training and display
dataset_yaml_src = model.dataset_yaml_src
train_set = yaml_parse.load(dataset_yaml_src)
test_set = MNIST(which_set='test')

dataset = train_set
num_samples = n_examples

vis_batch = dataset.get_batch_topo(num_samples)
rval = tuple(vis_batch.shape[dataset.X_topo_space.axes.index(axis)]
             for axis in ('b', 0, 1, 'c'))
_, patch_rows, patch_cols, channels = rval
mapback = hasattr(dataset, 'mapback_for_viewer')
pv = PatchViewer((rows, cols*(1+mapback)),
                 (patch_rows, patch_cols),
                 is_color=(channels == 3))

# Get examples and masks
x_val = test_set.get_batch_design(num_samples)
m_val = np.ones((n_examples, input_dim))
m_val[:10, :392] = 0
m_val[10:20, 392:] = 0
m_val[20:30, ::2] = 0
m_val[30:40, 1::2] = 0
for i in xrange(28):
    m_val[40:50, (i*28):((2*i+1)*14)] = 0
for i in xrange(28):
    m_val[50:60, ((2*i+1)*14):((i+1)*28)] = 0
m_val[60:70, 196:588] = 0
for i in xrange(28):
    m_val[70:80, ((4*i+1)*7):((4*i+3)*7)] = 0
m_val[80:90] = np.random.binomial(n=1, p=.25, size=(10,784))
m_val[90:] = np.random.binomial(n=1, p=.1, size=(10,784))

print 'Built mask'

X_shared.set_value(np.where(m_val == 1, x_val, X_shared.get_value()))

sqrt_iter = 70
max_iter = np.arange(sqrt_iter)**2
max_iter = max_iter.sum()

iteration = 0
for i in xrange(sqrt_iter):
    for j in xrange(i**2):
        rval = f(x_val, m_val, 1/(.1*iteration+10))[0]
        iteration += 1
        print iteration, '/', max_iter, ':', rval
    show(X_shared.get_value(), dataset, mapback, pv, rows, cols, 
         'mnist_inpainting_%04d.png'%i)

os.system("convert -delay 10 -loop 0 mnist_inpainting_*.png mnist_inpainting.gif")
