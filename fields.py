import numpy as np
from copy import deepcopy
def calcBfield(rs, data, info, use_parallel = True):
    '''
    Calculate the magnetic field for a collection of dipoles
    rs: (matrix with dimension m x 3), position in space in um
    ri: (matrix with dimension N x 3) N dipole locations in space in um

    m: (vector of length 3) magnetic moment of a dipole in 1e-18 J/T

    Output in T
    '''

    mu0 = 4 * np.pi * 1e-7  # T m /A
    dV = info['xstepsize'] * info['ystepsize'] * info['zstepsize'] * 1e18  # cell volume in um^3

    DipolePositions = data[['x', 'y', 'z']].as_matrix() * 1e6  # convert from m to um
    m = data[['mx', 'my', 'mz']].as_matrix() * dV  # multiply by the cell volume to get the magnetic dipole moment (1e-6 A um^2 = 1e-18 J/T)

    if use_parallel:
        # try importing the multiprocessing library
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()

    def process(rs):
        """
        calculates the data (Gradient, Bfield or both) at positions rs
        """

        data = {
            'x':deepcopy(rs[:,0]),
            'y':deepcopy(rs[:,1]),
            'z':deepcopy(rs[:,2])
        }

        B = np.array([B_function(r, DipolePositions, m) for r in rs])
        data['Bx'] = deepcopy(B[:,0])
        data['By'] = deepcopy(B[:,1])
        data['Bz'] = deepcopy(B[:,2])


        return data

    def rs_subset(i, num_cores, rs):
        data_per_core = int(np.floor(len(rs)/num_cores))
        return rs[i*data_per_core:min((i+1)*data_per_core, len(rs))]


    def B_function(r, DipolePositions, m):
        """
        calculates the magnetic field at position r
        :param r: vector of length 3 position at which field is evaluates (in um)
        :param DipolePositions: matrix Nx3, of positions of dipoles (in um)
        :param m:  matrix Nx3, components dipole moment at position DipolePositions mx, my, mz (in 1e-18 J/T)
        :return:
        """

        a = np.ones((np.shape(DipolePositions)[0],1)) * np.array([r])-DipolePositions
        rho = np.sqrt(np.sum(a ** 2, 1))

        # if we request thefield at the location of the dipole the field diverges, thus we exclude this value because we only want to get the fields from all the other dipoles
        zero_value_index = np.argwhere(rho == 0)

        rho = np.array([rho]).T * np.ones((1, 3))

        # calculate the vector product of m and a: m*(r-ri)
        ma = np.array([np.sum(m*a,1)]).T*np.ones((1,3))
        B = mu0 / (4 * np.pi ) * ( 3. * a * ma / rho**5- m / rho**3) # magnetic field in Tesla

        # exclude the dipole at the location where we calculate the field
        if len(zero_value_index) > 0:
            B[zero_value_index, :] = np.zeros([len(zero_value_index), 3])

        return np.sum(B, 0)


    data = {
        'x':deepcopy(rs[:,0]),
        'y':deepcopy(rs[:,1]),
        'z':deepcopy(rs[:,2])
    }

    if use_parallel:
        B = Parallel(n_jobs=num_cores)(delayed(B_function)(r, DipolePositions, m) for r in rs)
        B = np.array(B)

    else:
        B = np.array([calcBfield(r, DipolePositions, m) for r in rs])

    # put data into a dictionary

    data['Bx'] = deepcopy(B[:, 0])
    data['By'] = deepcopy(B[:, 1])
    data['Bz'] = deepcopy(B[:, 2])

    return data





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