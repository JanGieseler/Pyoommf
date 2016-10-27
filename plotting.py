import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fields import field_component

def plot_Bfield(stage, data_folder, target_folder, tag='2.5um_Co_magnet', zo=0.1e-6, component_name = None, s = None):
    in_file = os.path.join(os.path.join(data_folder, target_folder),
                           '{:s}_b_fields_zo_{:0.1f}um_stage_{:03d}.csv'.format(tag, 1e6 * zo, stage))

    dataB = pd.read_csv(in_file)

    Nx, Ny = len(np.unique(dataB['x'])), len(np.unique(dataB['y']))

    C, label = field_component(dataB, component_name, s)

    # C = np.sqrt(dataB['Bx'] ** 2 + dataB['By'] ** 2 + dataB['Bz'] ** 2)
    C = C.reshape(Ny, Nx)

    X = dataB['x'].reshape(Ny, Nx)
    Y = dataB['y'].reshape(Ny, Nx)

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)

    fig = plt.figure(figsize=(15, 4))
    CS = plt.pcolor(X, Y, C * 1e4)
    plt.colorbar(label=label)
    plt.title('magnetic field - zo  = {:0.1f} um (stage {:03d})'.format(1e6 * zo, stage))
    plt.xlabel('x ($\mu m$)')
    plt.ylabel('y ($\mu m$)')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.axes().set_aspect('equal')

    fig.savefig(in_file.replace('.csv', '.jpg'))