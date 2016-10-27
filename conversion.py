import datetime
import pandas as pd
import numpy as np
import glob
import os
import read_write as rw
import fields

def Bfield_from_mag(stage, data_folder, target_folder, tag='2.5um_Co_magnet', xmax=2.5e-6, xmin=None, ymax=1.5e-6,ymin=None, zo=0.1e-6, dx=50e-9):
    """
    calculates the b field in the x-y plane as at a distance zo and saves it as a .csv to the data_folder/target_folder
    """

    f = glob.glob(os.path.join(os.path.join(data_folder, 'data'), '*Magnetization-{:05d}-*omf.tsv'.format(stage)))[0]
    data_mag, info_mag = rw.load_ommf_vect_data(f)

    if xmin is None:
        xmin = -xmax
    if ymin is None:
        ymin = -ymax

    # calculate the grid
    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dx)
    # Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)
    np.shape(X), np.shape(Y)

    r = np.array([X.flatten(), Y.flatten(), zo * np.ones(len(X.flatten()))]).T

    t1 = datetime.datetime.now()
    dataB = fields.calcBfield(r, data_mag, info_mag, True)
    t2 = datetime.datetime.now()
    print('duration:', str(t2 - t1))

    out_file = os.path.join(os.path.join(data_folder, target_folder),
                            '{:s}_b_fields_zo_{:0.1f}um_stage_{:03d}.csv'.format(tag, 1e6 * zo, stage))

    dataB.to_csv(out_file, index=False)
