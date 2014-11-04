NICE: Non-linear independent components estimation
==================================================

This repository contains code (in [`pylearn2/`](https://github.com/laurent-dinh/nice/blob/master/pylearn2/)) and hyperparameters (in [`exp/`](https://github.com/laurent-dinh/nice/blob/master/exp/)) for the paper:

["NICE: Non-linear independent components estimation"](http://arxiv.org/abs/1410.8516) Laurent Dinh, David Krueger, Yoshua Bengio. ArXiv 2014.

Please cite this paper if you use the code in this repository as part of
a published research project.

We are an academic lab, not a software company, and have no personnel
devoted to documenting and maintaing this research code.
Therefore this code is offered with minimal support.
Exact reproduction of the numbers in the paper depends on exact
reproduction of many factors,
including the version of all software dependencies and the choice of
underlying hardware (GPU model, etc). We used NVIDA Ge-Force GTX-580
graphics cards; other hardware will use different tree structures for
summation and incur different rounding error. If you do not reproduce our
setup exactly you should expect to need to re-tune your hyperparameters
slight for your new setup.

Moreover, we have not integrated any unit tests for this code into Theano
or Pylearn2 so subsequent changes to those libraries may break the code
in this repository. If you encounter problems with this code, you should
make sure that you are using the development branch of [Pylearn2](https://github.com/lisa-lab/pylearn2/) and
[Theano](https://github.com/Theano/Theano/),
and use `git checkout` to go to a commit from approximately October 21, 2014. More precisely [`git checkout 3be2a6`](https://github.com/lisa-lab/pylearn2/commit/3be2a6d5ff81273c12023208166b630300eff338) and [`git checkout 165eb4`](https://github.com/Theano/Theano/commit/165eb4e66ab1f5320b2fe67c630a7e76ae5e6526).

This code itself requires no installation besides making sure that the
`nice` directory is in a directory in your PYTHONPATH. If
installed correctly, `python -c "import nice"` will work. You
must also install Pylearn2 and Pylearn2's dependencies (Theano, numpy,
etc.)

Call [`pylearn2/scripts/train.py`](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/train.py)
on the various yaml files in this repository
to train the model for each dataset reported in the paper. The names of
*.yaml are fairly self-explanatory.