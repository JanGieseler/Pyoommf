import glob
import os
import numpy as np
from conversion import Bfield_from_mag
import read_write as rw



if __name__ == '__main__':

    root_folder = 'Z:\\Lab\\Cantilever\\Theory and Simulation\\oommf\\20170209_TrapezoidalCoMagnet\\'
    data_folders = [f for f in glob.glob(os.path.join(root_folder, '*transition_0.4um_*2')) if os.path.isdir(f)]



    mu0 = 4 * np.pi *1e-7 # T m /A
    target_folder = 'b_fields_xy-plane'

    print(data_folders)

    zo = 0.1e-6

    # define the direction of the nv axis
    theta = np.arccos(1 / np.sqrt(3))
    phi = 0 * np.pi / 180

    s = [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

    # for zo in [0.1e-6, 0.5e-6]:
    for zo in [0.1e-6]:
        for data_folder in data_folders:

            if not os.path.exists(os.path.join(data_folder, target_folder)):
                os.makedirs(os.path.join(data_folder, target_folder))

            tabdata, units = rw.load_ommf_tab_data(glob.glob(os.path.join(data_folder, '*.odt'))[0])
            #        stages = [int(x) for x in  np.array(tabdata[tabdata.B ==0].index)] # find the stage where the external field is 0
            stages = range(50, 70, 10)
            print(stages)
            for stage in stages:
                Bfield_from_mag(stage, data_folder, target_folder, zo=zo, dx=25e-9)
