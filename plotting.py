import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fields import field_component
import yaml

def plot_Bfield(stage, data_folder, target_folder, tag=None, zo=0.1e-6, component_name = 'Bfield_mag', s = None):
    """
    plots the b-field and the outline of the computational domain(which typically is the size of the magnet) and saves an image to the disk

    :param stage:
    :param data_folder:
    :param target_folder:
    :param tag:
    :param zo:
    :param component_name:
    :param s:
    :return:
    """
    if tag is None:
        tag = data_folder

    # in_file = os.path.join(os.path.join(data_folder, target_folder),
    #                        '{:s}_b_fields_zo_{:0.1f}um_stage_{:03d}.csv'.format(tag, 1e6 * zo, stage))

    in_file = os.path.join(os.path.join(data_folder, target_folder),'*b_fields_zo_{:0.1f}um_stage_{:03d}.csv'.format(1e6 * zo, stage))


    if len(glob.glob(in_file)) == 1:
        in_file = glob.glob(in_file)[0]
    else:
        print('Could not find {:s}'.format(in_file))
        raise IOError
    print('loading file: {:s}'.format(in_file))

    out_file = in_file.replace('.csv', '{:s}_{:s}.jpg'.format(component_name, tag))

    dataB = pd.read_csv(in_file)

    with open(in_file.replace('.csv', '.json'), 'r') as infile:
        info = yaml.safe_load(infile)

    Nx, Ny = len(np.unique(dataB['x'])), len(np.unique(dataB['y']))

    C, label = field_component(dataB, component_name, s)

    # C = np.sqrt(dataB['Bx'] ** 2 + dataB['By'] ** 2 + dataB['Bz'] ** 2)
    C = C.reshape(Ny, Nx)

    X = 1e6*dataB['x'].reshape(Ny, Nx)
    Y = 1e6*dataB['y'].reshape(Ny, Nx)

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)

    fig = plt.figure(figsize=(15, 4))
    CS = plt.pcolor(X, Y, C)
    plt.colorbar(label=label)
    # plot the outline of the magnet
    plt.plot([1e6*info['xmin'], 1e6*info['xmax']], [1e6*info['ymin'], 1e6*info['ymin']], 'k--', lw=2)
    plt.plot([1e6*info['xmin'], 1e6*info['xmax']], [1e6*info['ymax'], 1e6*info['ymax']], 'k--', lw=2)
    plt.plot([1e6*info['xmin'], 1e6*info['xmin']], [1e6*info['ymin'], 1e6*info['ymax']], 'k--', lw=2)
    plt.plot([1e6*info['xmax'], 1e6*info['xmax']], [1e6*info['ymin'], 1e6*info['ymax']], 'k--', lw=2)


    plt.title('magnetic field - zo  = {:0.1f} um (stage {:03d})'.format(1e6 * zo, stage))
    plt.xlabel('x ($\mu m$)')
    plt.ylabel('y ($\mu m$)')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.axes().set_aspect('equal')



    fig.savefig(out_file)
    fig.close()