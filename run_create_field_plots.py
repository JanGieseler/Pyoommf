# run this script to produce plots of the magnetic fields that are created with run_calc_B_fields for instance
#



import glob
import os
import numpy as np
from conversion import Bfield_from_mag
import read_write as rw
from plotting import plot_Bfield


if __name__ == '__main__':

    target_folder = 'b_fields_xy-plane'
    root_folder = 'Z:\\Lab\\Cantilever\\Theory and Simulation\\oommf\\20170209_TrapezoidalCoMagnet\\'
    data_folders = [f for f in glob.glob(os.path.join(root_folder, '*transition_0.4um_*2')) if os.path.isdir(f)]

    zo = 0.1e-6

    # define the direction of the nv axis
    theta = np.arccos(1/np.sqrt(3))
    phi = 0*np.pi/180


    s = [np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]

    data_folder = data_folders[0]
    zo = 0.1e-6

    for data_folder in data_folders:
        for component_name in ['Bfield_par', 'Bfield_perp']:
            # for stage in range(50, 170,10):
            for stage in range(0, 50, 10):
                tag = 'phi={:03d}deg'.format(int(round(phi*180/np.pi)))
                plot_Bfield(stage, os.path.join(root_folder, data_folder), target_folder, tag = tag, zo=0.1e-6, component_name= component_name, s=s)
