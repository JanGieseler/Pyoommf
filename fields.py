import numpy as np

def calcBfield(r, DipolePositions, m):
    '''
    Calculate the magnetic field for a collection of dipoles
    r: (vector of length 3), position in space in nm
    ri: (matrix with dimension m x 3) m dipole locations in space in nm

    m: (vector of length 3) magnetic moment of a dipole in 1e-18 J/T

    Output in T
    '''



    mu0 = 4 * np.pi *1e-7 # T m /A

    a = np.ones((np.shape(DipolePositions)[0],1)) * np.array([r])-DipolePositions
    rho = np.array([np.sqrt(np.sum(a**2,1))]).transpose()*np.ones((1,3))
    # calculate the vector product of m and a: m*(r-ri)
    ma = np.array([np.sum(np.ones((np.shape(DipolePositions)[0],1)) * np.array([m])*a,1)]).transpose()*np.ones((1,3))

    B = mu0 / (4 * np.pi ) * ( 3. * a * ma / rho**5
        - m / rho**3
        )

    return np.sum(B,0)

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