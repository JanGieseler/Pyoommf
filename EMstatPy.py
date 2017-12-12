import numpy as np
from numpy import linalg as LA
import time
import pandas as pd
from copy import deepcopy

# =====================================================================================
# def calcSingleMomentGradient(r, ri, m, s, n):
#     '''
#     Calculate the Gradient of a single magnetic moment
#
#     r:  position in space in nm
#     ri: dipole location in space in nm
#
#     m: magnetic moment of a dipole in 1e-18 J / T
#     s: spin vector no units
#     n: projection vector of the gradient, e.g. motion of resonator
#
#     Output in T/um
#
#     '''
#     mu0 = 4 * np.pi *1e-7
#
#
#     a = r-ri
#     rho = np.sqrt(np.vdot(a, a))
#
#     gradB = 3. * mu0 / (4 * np.pi * (rho)**5) * (
#          np.vdot(a, m) * np.vdot(s, n)
#         +np.vdot(a, s) * np.vdot(m, n)
#         - (5 * np.vdot(a, s) * np.vdot(a, m) / rho**2 - np.vdot(m, s) ) * np.vdot(a, n)
#         )
#
#     return gradB
#
#
#
# # =====================================================================================
# def calcGradient(r, DipolePositions, m, s, n):
#     '''
#     Calculate the gradient for a collection of dipoles
#     r:  position in space in nm
#     ri: dipole location in space in nm
#
#     m: magnetic moment of a dipole in 1e-18J/T
#     s: spin vector no units
#     n: projection vector of the gradient, e.g. motion of resonator
#
#     Output in T/um
#     '''
#
#     res = 0.
#
#     for ri in DipolePositions:
#         res += calcSingleMomentGradient(r, ri, m, s, n)
#     return res



# =====================================================================================
# def calcSingleMomentBfield(r, ri, m):
#     '''
#     Calculate the magnetic field of a single magnetic moment
#
#     r:  position in space in nm
#     ri: dipole location in space in nm
#
#     m: magnetic moment of a dipole in 1e-18 J / T
#
#     Output in T
#
#     '''
#     mu0 = 4 * np.pi *1e-7 # T m /A
#
#
#     a = r-ri
#     rho = np.sqrt(np.vdot(a, a))
#
#     B = mu0 / (4 * np.pi ) * (
#          3. * a * np.vdot(m, a) / rho**5
#         - m / rho**3
#         )
#
#
#     return B


# =====================================================================================
# def calcBfield(r, DipolePositions, m):
#     '''
#     Calculate the magnetic field for a collection of dipoles
#     r:  position in space in nm
#     ri: dipole location in space in nm
#
#     m: magnetic moment of a dipole in 1e-18 J/T
#
#     Output in T
#     '''
#
#     res = 0.
#
#     for ri in DipolePositions:
#         res += calcSingleMomentBfield(r, ri, m)
#     return res
def calcBfield(r, DipolePositions, m):
    '''
    Calculate the magnetic field for a collection of dipoles
    r: (vector of length 3), position in space in nm
    ri: (matrix with dimension m x 3) m dipole locations in space in nm

    m: (vector of length 3) magnetic moment of a dipole in 1e-18 J/T

    Output in T
    '''



    res = 0.

    mu0 = 4 * np.pi *1e-7 # T m /A

    a = np.ones((np.shape(DipolePositions)[0],1)) * np.array([r])-DipolePositions
    rho = np.array([np.sqrt(np.sum(a**2,1))]).transpose()*np.ones((1,3))
    # calculate the vector product of m and a: m*(r-ri)
    ma = np.array([np.sum(np.ones((np.shape(DipolePositions)[0],1)) * np.array([m])*a,1)]).transpose()*np.ones((1,3))

    B = mu0 / (4 * np.pi ) * ( 3. * a * ma / rho**5
        - m / rho**3
        )

    return np.sum(B,0)

# =====================================================================================
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

# =====================================================================================
def calcDipolePositions(MagnetDimensions, NumberOfDipoles):
    '''
        MagnetDimensions: vector with components height, width, length in um
        NumberOfDipoles: Number of along each dimension, x (height), y (width), z (length)
    '''

    Nx, Ny, Nz = NumberOfDipoles
    H, W, L = MagnetDimensions

    X = np.linspace(-H/2,H/2,Nx, endpoint=True)
    Y = np.linspace(-W/2,W/2,Ny, endpoint=True)
    Z = np.linspace(-L/2,L/2,Nz, endpoint=True)

    DipolePositions = np.array([np.array([x,y,z]) for x in X for y in Y for z in Z])

    return DipolePositions




# =====================================================================================
def calcDipoleMoment(DipolePositions, MagnetDimensions, AtomMass, AtomDensity, gFactor):
    '''
        Calculated the dipole moment of a single dipole in the magnet in units of 1e-18 J / T
        DipolePositions in um
        MagnetDimensions in um
        AtomMass in atomic units (Daltons)
        AtomDensity in 1/m^3
    '''
    BohrMagneton = 9.27e-24 # J/T
    N = AtomDensity/ (AtomMass * 1.66e-27) * np.prod(MagnetDimensions)
    m = gFactor * BohrMagneton * N / len(DipolePositions)

    return m


def setup_calculation(config, margins):
    """
    config: dictionary with the settings for the magnet
    margins: margins as a dictionary

    Returns: A1, A2, A3, s, m, n, dipole_positions
        A1, A2, A3: vectors that contain the grid points along the three orthogonal axis () usually A1, A2, A3 = X, Y, Z
        s, m, n: unit vectors of spin (s), magnet (m) and motion (direction of gradient) (n)
        dipole_positions: position of dipoles that are summed to give the magnetic field
    """


    density = float(config['magnet_material_parameters']['density_kg/m^3'])
    atom_mass = float(config['magnet_material_parameters']['atom_mass_dalton'])
    g_factor = float(config['magnet_material_parameters']['g_factor'])


    md = config['magnet_dimensions_um']
    magnet_dimensions = np.array([float(md['height']), float(md['width']), float(md['length'])])
    height, width, length = magnet_dimensions
    nd = config['number_of_dipoles']

    number_of_dipoles = np.array([float(nd['nx']), float(nd['ny']), float(nd['nz'])])
    dipole_positions = calcDipolePositions(magnet_dimensions, number_of_dipoles)

    s  = np.array(config['unit_vector_spin'])
    m  = np.array(config['unit_vector_magnet'])
    n  = np.array(config['unit_vector_motion'])

    s /= np.linalg.norm(s)
    m /= np.linalg.norm(m)
    n /= np.linalg.norm(n)

    m *= calcDipoleMoment(dipole_positions, magnet_dimensions, atom_mass, density, g_factor)

    A1, A2, A3 = calc_grid(margins, md)

    return A1, A2, A3, s, m, n, dipole_positions

def calc_grid(margins, md):
    """
    margin: margins as a dictionary, e.g.
        margins = {
            'a1':{'left': None, 'right': 2.0},
            'a2':{'left':0.25, 'right':0.25},
            'a3':{'left':0.25, 'right':0.25},
            'dx': float (grid spacing between points)
            }
        None means field are calculated form origin

    md: magnet dimensions as dictionary, e.g.
        magnet_dimensions_um = {
        	"height": 0.1,
        	"width": 0.3,
        	"length": 1.0
        	}
    Returns: A1, A2, A3
        vectors that contain the grid points along the three orthogonal axis () usually A1, A2, A3 = X, Y, Z

    """

    def margin_to_minmax(margin, size = None):
        if isinstance(margin, dict):
            a_min = 0.0 if margin['left'] is None else -size/2-margin['left']
            a_max = 0.0 if margin['right'] is None else size/2+margin['right']
        elif isinstance(margin, (int, float)):
            a_min = margin
            a_max = margin
        return a_min, a_max

    def margin_to_vector(margin, size, dx):
        a_min, a_max = margin_to_minmax(margin, size)
        if a_min != a_max:
            # A = np.linspace(a_min, a_max,(a_max-a_min)/dx, endpoint=False)
            A = np.arange(a_min, a_max+dx, dx)
        else:
            A = [a_min]
        return A

    magnet_dimensions = np.array([float(md['height']), float(md['width']), float(md['length'])])
    height, width, length = magnet_dimensions

    dx = margins['dx']

    # a1min, a1max = margin_to_minmax(margins['a1'], height)
    # a2min, a2max = margin_to_minmax(margins['a2'], width)
    # a3min, a3max = margin_to_minmax(margins['a3'], length)
    #
    # A3 = np.linspace(a3min, a3max,(a3max-a3min)/dx, endpoint=True)
    # A2 = np.linspace(a2min, a2max,(a2max-a2min)/dx, endpoint=True)
    # A1 = np.linspace(a1min, a1max,(a1max-a1min)/dx, endpoint=True)

    A1 = margin_to_vector(margins['a1'], height, dx)
    A2 = margin_to_vector(margins['a2'], width, dx)
    A3 = margin_to_vector(margins['a3'], length, dx)

    return A1, A2, A3

# =====================================================================================
# run calculation
# =====================================================================================
def run_calculation(config, margins, type = 'Bfield', use_parallel = True, verbose = False):
    """
    config:
    margins:
    type: string that determines the output: Bfield, Gradient, Both
    use_parallel: use parallelized code execution
    """
    if use_parallel:
        # try importing the multiprocessing library
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()


    calc_B =  type.lower() in ('bfield', 'both')
    calc_G =  type.lower() in ('Gradient', 'both')


    def process(rs):
        """
        calculates the data (Gradient, Bfield or both) at positions rs
        """

        data = {
            'x':deepcopy(rs[:,0]),
            'y':deepcopy(rs[:,1]),
            'z':deepcopy(rs[:,2])
        }

        if calc_B:
            B = np.array([calcBfield(r, dipole_positions, m) for r in rs])
            data['Bx'] = deepcopy(B[:,0])
            data['By'] = deepcopy(B[:,1])
            data['Bz'] = deepcopy(B[:,2])
        if calc_G:
            G = np.array([calcGradient(r, dipole_positions, m, s, n) for r in rs])
            data['G'] = deepcopy(G)

        return data

    def rs_subset(i, num_cores, rs):
        data_per_core = int(np.floor(len(rs)/num_cores))
        return rs[i*data_per_core:min((i+1)*data_per_core, len(rs))]


    A1, A2, A3, s, m, n, dipole_positions = setup_calculation(config, margins)

    rs = np.array([[a1, a2, a3] for a1 in A1 for a2 in A2 for a3 in A3])
    if verbose:
        print('size of rs:', np.shape(rs))

    data = {
        'x':deepcopy(rs[:,0]),
        'y':deepcopy(rs[:,1]),
        'z':deepcopy(rs[:,2])
    }

    print('calculation for {:d} datapoints'.format(len(rs)))

    start_time = time.clock()
    if use_parallel:
        if calc_B:
            B = Parallel(n_jobs=num_cores)(delayed(calcBfield)(r, dipole_positions, m) for r in rs)
            B = np.array(B)
        if calc_G:
            G = Parallel(n_jobs=num_cores)(delayed(calcGradient)(r, dipole_positions, m, s, n) for r in rs)
            G = np.array(G)
    else:
        if calc_B:
            B = np.array([calcBfield(r, dipole_positions, m) for r in rs])
        if calc_G:
            G = np.array([calcGradient(r, dipole_positions, m, s, n) for r in rs])
    end_time = time.clock()

    # put data into a dictionary
    if calc_B:
        data['Bx'] = deepcopy(B[:,0])
        data['By'] = deepcopy(B[:,1])
        data['Bz'] = deepcopy(B[:,2])
    if calc_G:
        data['G'] = deepcopy(G)

    print('{:0.2f} seconds elapsed'.format(round(end_time-start_time, 2)))
    calculation_time = end_time-start_time
    return data, calculation_time
