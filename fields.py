import numpy as np
import pandas as pd
from copy import deepcopy


def calcBfield(rs, data, info, use_parallel = True):
    '''
    Calculate the magnetic field for a collection of dipoles
    rs: (matrix with dimension m x 3), positions at which field is evaluated in space (in m)
    data: dataframe with columns 'mx', 'my', 'mz', 'x', 'y', 'z' that gives the dipolevector and its location
    info: dictionary with metadata for the dataset, contains 'xstepsize', 'ystepsize', 'zstepsize', which give the spacing of the dipole locations
    use_parallel:  (boolean) if True use parallel execution of code this is not working yet....

    :returns pandas dataframe columns 'Bx', 'By', 'Bz', 'x', 'y', 'z' that gives the fieldvector and its location (=rs)
    '''

    dV = info['xstepsize'] * info['ystepsize'] * info['zstepsize'] * 1e18  # cell volume in um^3

    print('length of data', len(data))

    # pick only the data where the magnetization is actually non-zero
    data = data[np.sum(data[['mx', 'my', 'mz']].as_matrix()**2,1)>0]


    DipolePositions = data[['x', 'y', 'z']].as_matrix() * 1e6  # convert from m to um
    m = data[['mx', 'my', 'mz']].as_matrix() * dV  # multiply by the cell volume to get the magnetic dipole moment (1e-6 A um^2 = 1e-18 J/T)


    rs *=1e6# convert from m to um


    print('number of magnetic moments', len(data))
    print('number of positions', len(rs))
    if use_parallel:
        # try importing the multiprocessing library
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()

        print('using ', num_cores, ' cores')

    def process(r):
        return calcBfield_single_pt(r, DipolePositions, m)

    # def process(rs):
    #     """
    #     calculates the data at positions rs
    #     """
    #
    #     # data = {
    #     #     'x':deepcopy(rs[:,0]),
    #     #     'y':deepcopy(rs[:,1]),
    #     #     'z':deepcopy(rs[:,2])
    #     # }
    #
    #     B = np.array([calcBfield_single_pt(r, DipolePositions, m) for r in rs])
    #     data['Bx'] = deepcopy(B[:,0])
    #     data['By'] = deepcopy(B[:,1])
    #     data['Bz'] = deepcopy(B[:,2])
    #
    #
    #     return data

    # def rs_subset(i, num_cores, rs):
    #     data_per_core = int(np.floor(len(rs)/num_cores))
    #     return rs[i*data_per_core:min((i+1)*data_per_core, len(rs))]




    # convert from um to m
    data_out = {
        'x':deepcopy(rs[:,0]*1e-6),
        'y':deepcopy(rs[:,1]*1e-6),
        'z':deepcopy(rs[:,2]*1e-6)
    }



    if use_parallel:

        # Parallel(n_jobs=1)(delayed(sqrt)(i ** 2) for i in range(10))
        # B = Parallel(n_jobs=num_cores)(delayed(calcBfield_single_pt)(r, DipolePositions, m) for r in rs)
        B = Parallel(n_jobs=num_cores)(delayed(calcBfield_single_pt)(r, DipolePositions, m) for r in rs)
        # B = Parallel(n_jobs=num_cores)(delayed(process)(r) for r in rs)
        B = np.array(B)

    else:
        B = np.array([calcBfield_single_pt(r, DipolePositions, m) for r in rs])

    # put data into a dictionary
    data_out['Bx'] = deepcopy(B[:, 0])
    data_out['By'] = deepcopy(B[:, 1])
    data_out['Bz'] = deepcopy(B[:, 2])


    # return data as a pandas dataframe
    return pd.DataFrame.from_dict(data_out)


def calcBfield_single_pt(r, DipolePositions, m):
    """
    calculates the magnetic field at position r
    :param r: vector of length 3 position at which field is evaluates (in um)
    :param DipolePositions: matrix Nx3, of positions of dipoles (in um)
    :param m:  matrix Nx3, components dipole moment at position DipolePositions mx, my, mz (in 1e-18 J/T)
    :return:
    """
    mu0 = 4 * np.pi * 1e-7  # T m /A
    a = np.ones((np.shape(DipolePositions)[0], 1)) * np.array([r]) - DipolePositions
    rho = np.sqrt(np.sum(a ** 2, 1))

    # if we request thefield at the location of the dipole the field diverges, thus we exclude this value because we only want to get the fields from all the other dipoles
    zero_value_index = np.argwhere(rho == 0)

    rho = np.array([rho]).T * np.ones((1, 3))

    # calculate the vector product of m and a: m*(r-ri)
    ma = np.array([np.sum(m * a, 1)]).T * np.ones((1, 3))
    B = mu0 / (4 * np.pi) * (3. * a * ma / rho ** 5 - m / rho ** 3)  # magnetic field in Tesla

    # exclude the dipole at the location where we calculate the field
    if len(zero_value_index) > 0:
        B[zero_value_index, :] = np.zeros([len(zero_value_index), 3])

    return np.sum(B, 0)


#
# def unit_vector(theta, phi):
#
#     return [np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]

def field_component(data, component_name = None, s = None):
    """
    returns the field component defined by component_name
    :param data:
    :param component_name:
    :param s:
    :return:
    """

    Bx, By, Bz = data['Bx'].as_matrix(), data['By'].as_matrix(), data['Bz'].as_matrix()

    if component_name is None or component_name == 'Bfield_mag':
        D = 1e4 * np.sqrt(Bx**2 + By**2 + Bz**2)
        label = '$|\mathbf{B}|$ (Gauss)'
    elif component_name in ('Bfield_proj', 'Bfield_par', 'Bfield_long', 'parallel'):
        D = 1e4 * (Bx * s[0] + By * s[1] + Bz * s[2])
        # label = '$\mathbf{B}\cdot \mathbf{S}$ (Gauss)'
        label = '$B_{\parallel}$ (Gauss)'
    elif component_name in ('Bfield_perp', 'Bfield_trans', 'perpendicular'):
        D = 1e4 * np.sqrt((By * s[2] - Bz * s[1]) ** 2 + (Bz * s[0] - Bx * s[2]) ** 2 + (Bx * s[1] - By * s[0]) ** 2)
        label = '$B_{\perp}$ (Gauss)'

    return D, label


# old:
def calcGradient(r, DipolePositions, m, s, n):
    '''
    Calculate the gradient for a collection of dipoles
    r: (vector of length 3) position in space in nm
    ri: (matrix with dimension m x 3) m dipole location in space in nm

    m: (vector of length 3) magnetic moment of a dipole in 1e-18J/T
    s: (vector of length 3) spin vector no units
    n: (vector of length 3) projection vector of the gradient, e.g. motion of resonator

    Output in T/um
    '''

    mu0 = 4 * np.pi *1e-7


    a = np.ones((np.shape(DipolePositions)[0],1)) * np.array([r])-DipolePositions
    rho = np.sqrt(np.sum(a**2,1))
    # calculate the vector product of m and a: m*(r-ri)
    ma = np.sum(np.ones((np.shape(DipolePositions)[0],1)) * np.array([m])*a,1)
    # calculate the vector product of s and a: s*(r-ri)
    sa = np.sum(np.ones((np.shape(DipolePositions)[0],1)) * np.array([s])*a,1)
    # calculate the vector product of n and a: n*(r-ri)
    na = np.sum(np.ones((np.shape(DipolePositions)[0],1)) * np.array([n])*a,1)


    gradB = 3. * mu0 / (4 * np.pi * (rho)**5) * (
         ma * np.vdot(s, n)
        +sa * np.vdot(m, n)
        - (5 * sa * ma / rho**2 - np.vdot(m, s) ) * na
        )

    return np.sum(gradB,0)


if __name__ == '__main__':

    import read_write as rw
    import os
    import datetime

    # folder = 'Z:\Lab\Cantilever\tmp_jan\oommf_results_2'
    folder = ''

    file_mag = os.path.join(folder , 'random_K1_length_1.5um-Oxs_MinDriver-Magnetization-00000-0001715-omf.tsv')
    file_H = os.path.join(folder , 'random_K1_length_1.5um-Oxs_CGEvolve-H-00000-0001715-ovf.tsv')

    print(os.path.exists(file_H))

    data_mag, info_mag = rw.load_ommf_vect_data(file_mag)
    data_mag.head()

    data_H, info_H = rw.load_ommf_vect_data(file_H)
    data_H.head()

    zo = -5e-9
    subdata_H = rw.get_slice(data_H, zo, info_mag, 'z')
    subdata_mag = rw.get_slice(data_mag, zo, info_mag, 'z')

    r = subdata_H[['x', 'y', 'z']].as_matrix()

    r = r[0:10, :]
    print('shape r', np.shape(r))



    t1 =datetime.datetime.now()
    dataB = calcBfield(r, subdata_mag, info_mag, True)
    t2 = datetime.datetime.now()
    # dataB = calcBfield(r, subdata_mag, info_mag, False)
    # t3 = datetime.datetime.now()

    print('excution time parallel', str(t2-t1))
    # print('excution time not parallel', str(t3 - t2))
