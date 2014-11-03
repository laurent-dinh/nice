import sys
_, model_path, save_path = sys.argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pylearn2.utils import serial

model = serial.load(model_path)
spectrum = model.encoder.layers[-1].D.get_value()
spectrum = np.sort(spectrum)
spectrum = np.exp(-spectrum)

plt.bar(left=np.arange(model.nvis) + 0.1, height=spectrum, edgecolor='k', color='w')
plt.ylabel('$\sigma_{d}$')
plt.xlim(0, model.nvis)

plt.savefig(save_path)
