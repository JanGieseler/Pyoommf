import numpy as np



# spin 1 matrices
Sx = np.matrix([
    [0, -1j, -1j],
    [1j, 0, 0],
    [1j, 0, 0]
])/np.sqrt(2)

Sy = np.matrix([
    [0, -1, 1],
    [-1, 0, 0],
    [1, 0, 0]
])/np.sqrt(2)

Sz = np.matrix([
    [0, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])


def esr_frequencies(Bfield, gs=27.969, muB=1, hbar=1, Dgs=2.87):
    """
    :param Bfield: magnetic field with components Bx, By, Bz: 1D-array of length 3 or 2D-array of dim Nx3
    :return:  matrix that gives the esr transition frequencies from diagonalizing the Hamiltonian with external magnetic field
        2 element array matrix if input B-field is 1D-array
        Nx2 array if input B-field is 2D-array of length Nx3
    """

    # if the input is a 1D array we cast it into a 2D array to work with the rest of the code
    if len(np.shape(Bfield))==1:
        assert len(Bfield)==3
        Bfield = [Bfield]
        input_1D = True
    elif len(np.shape(Bfield))==2:
        assert np.shape(Bfield)[1]==3
        input_1D = False



    def get_esr_freq(B):

        Hgs = hamiltonian_nv_spin1(B, gs=gs, muB=muB, hbar=hbar, D=Dgs)
        ev, Ugs = np.linalg.eigh(Hgs)

        return np.array([ev[1]-ev[0], ev[2]-ev[0]])

    esr = np.array([get_esr_freq(B) for B in Bfield])

    if input_1D:
        esr = esr[0]


    return esr


def hamiltonian_nv_spin1(Bfield, gs=27.969, muB=1, hbar=1, D=2.87):
    """
    The hamiltonian for a spin 1 system, this hamiltonian describes the NV gound state as well as the excited state
    :param Bfield: magnetic field in Tesla with components Bx, By, Bz: 1D-array of length 3 or 2D-array of dim Nx3
    :param gs:
    :param muB:
    :param hbar:
    :param D: (2.87 for ground state, 1.42 for excited state)
    :return: the NV ground state hamiltonian
        3x3 matrix if input B-field is 1D-array
        Nx3x3 array if input B-field is 2D-array of length Nx3
    """

    # if the input is a 1D array we cast it into a 2D array to work with the rest of the code
    if len(np.shape(Bfield))==1:
        assert len(Bfield)==3
        Bfield = [Bfield]
        input_1D = True
    elif len(np.shape(Bfield))==2:
        assert np.shape(Bfield)[1]==3
        input_1D = False


    H = [hbar * D * Sz**2+gs*muB*(B[0]*Sx+B[1]*Sy+B[2]*Sz) for B in Bfield]

    if input_1D:
        H = H[0]

    return H


def transition_rate_matrix(Bfield, k12, k13, beta, kr = 63.2, k47= 10.8, k57 = 60.7, k71 = 0.8, k72 = 0.4):
    """
    the transition matrix

    :param Bfield: magnetic field with components Bx, By, Bz: 1D-array of length 3 or 2D-array of dim Nx3
    :param k12:
    :param k13:
    :param beta:
    :param kr:
    :param k47:
    :param k57:
    :param k71:
    :param k72:
    :return: transition rate matrix obtained from diagonalizing Hamiltonian
        7x7 matrix if input B-field is 1D-array
        Nx7x7 array if input B-field is 2D-array of length Nx3
    """

    # if the input is a 1D array we cast it into a 2D array to work with the rest of the code
    if len(np.shape(Bfield))==1:
        assert len(Bfield)==3
        Bfield = [Bfield]
        input_1D = True
    elif len(np.shape(Bfield))==2:
        assert np.shape(Bfield)[1]==3
        input_1D = False

    ko = np.matrix([
        [0, k12, k13, kr, 0, 0, k71],
        [k12, 0, 0, 0, kr, 0, k72],
        [k13, 0, 0, 0, 0, kr, k72],
        [kr*beta, 0, 0, 0, 0, 0, 0],
        [0, kr*beta, 0, 0, 0, 0, 0],
        [0, 0, kr*beta, 0, 0, 0, 0],
        [0, 0, 0, k47, k57, k57, 0]
    ])


    def get_k(B):
        U = np.array(coupling_matrix(B).H) # need to double check this here also double check if mathematica is correct!!

        k = np.zeros([7,7])

        for i in range(7):
            for j in range(7):
                k[i,j] = np.dot(np.dot(np.abs(U[i,:])**2,ko),np.abs(U[j,:])**2)

        return k

    k = [get_k(B) for B in Bfield]

    if input_1D:
        k = k[0]

    return k


def coupling_matrix(Bfield, gs=27.969, muB=1, hbar=1, Dgs=2.87, Des=1.42):
    """
    :param Bfield: magnetic field with components Bx, By, Bz: 1D-array of length 3 or 2D-array of dim Nx3
    :return: unitary matrix that diagonalizes the Hamiltonian with external magnetic field
        7x7 matrix if input B-field is 1D-array
        Nx7x7 array if input B-field is 2D-array of length Nx3
    """

    # if the input is a 1D array we cast it into a 2D array to work with the rest of the code
    if len(np.shape(Bfield))==1:
        assert len(Bfield)==3
        Bfield = [Bfield]
        input_1D = True
    elif len(np.shape(Bfield))==2:
        assert np.shape(Bfield)[1]==3
        input_1D = False



    def get_Uo(B):
        Uo = np.matrix(np.zeros([7,7])+0j)

        Hgs = hamiltonian_nv_spin1(B, gs=gs, muB=muB, hbar=hbar, D=Dgs)
        ev, Ugs = np.linalg.eigh(Hgs)

        Hes = hamiltonian_nv_spin1(B, gs=gs, muB=muB, hbar=hbar, D=Des)
        ev, Ues = np.linalg.eigh(Hes)

        Uo[0:3,0:3] = Ugs

        Uo[3:6, 3:6] = Ues

        Uo[6, 6] = 1
        return Uo

    Uo = [get_Uo(B) for B in Bfield]

    if input_1D:
        Uo = Uo[0]


    return Uo

def populations(transition_rates):
    """
    calculates the population by solving the rate equations for the transition rate matrix k
    :param transition_rates:
        2D array with dim MxM
        3D array with dim NxMxM
    :return:
        population of the M levels
        1D array of length M if input is 2D array with dim MxM
        2D array of length NxM if input is 3D array with dim NxMxM
    """

    # if the input is a 2D array we cast it into a 3D array to work with the rest of the code
    if len(np.shape(transition_rates))==2:
        transition_rates = [transition_rates]
        input_2D = True
    elif len(np.shape(transition_rates))==3:
        input_2D = False

    def get_pop(k):
        k = np.matrix(k)
        a = k - np.diag(np.array(np.sum(k, 0))[0])

        a = np.row_stack([a, np.ones([1, len(k)])])
        b = np.hstack([np.zeros(len(k)), [1]])

        n, residuals, rank, s = np.linalg.lstsq(a,b)

        # some extra information from the fit we could use to validate the result
        # print('residuals', residuals)
        # print('rank', rank)
        # print('s', s)
        # print('n', n)

        return n

    n = [get_pop(k) for k in transition_rates]



    if input_2D:
        n = n[0]


    return n

def get_ko(k12, k13, beta, kr = 63.2, k47= 10.8, k57 = 60.7, k71 = 0.8, k72 = 0.4):
    """
    the transition matrix

    :param B: the magnetic field
    :param k12:
    :param k13:
    :param beta:
    :param kr:
    :param k47:
    :param k57:
    :param k71:
    :param k72:
    :return:
    """


    ko = np.matrix([
        [0, k12, k13, kr, 0, 0, k71],
        [k12, 0, 0, 0, kr, 0, k72],
        [k13, 0, 0, 0, 0, kr, k72],
        [kr*beta, 0, 0, 0, 0, 0, 0],
        [0, kr*beta, 0, 0, 0, 0, 0],
        [0, 0, kr*beta, 0, 0, 0, 0],
        [0, 0, 0, k47, k57, k57, 0]
    ])

    return ko

def photoluminescence_rate(transition_rates, populations):
    """

    :param transition_rates: transition rate matrix obtained from diagonalizing Hamiltonian
        7x7 matrix
        Nx7x7 array
    :param populations: population of the M levels
        1D array of length 7
        2D array of length Nx7
    :return: photoluminescence rate
        scalar if populations is a 1D array
        1D-array if populations is a 2D array
    """


    # if the input is a 1D array we cast it into a 2D array to work with the rest of the code
    if len(np.shape(transition_rates))==2:
        assert np.shape(transition_rates)==np.array([7,7])
        assert np.shape(populations) == np.array([7])
        transition_rates = [transition_rates]
        populations = [populations]
        input_1D = True
    elif len(np.shape(transition_rates))==3:
        assert np.shape(transition_rates)[1:3]==(7,7)
        assert np.shape(populations)[1] == 7
        input_1D = False

    r = [np.sum(np.dot(k[3:6, 0:3], pop[3:6])) for k, pop in zip(transition_rates, populations)]
    # r = np.sum(np.dot(k[3:6,0:3], pop[3:6]))
    if input_1D:
        r = r[0]

    return r


def photoluminescence_contrast(Bfield, k12, k13, beta, kr=63.2, k47=10.8, k57=60.7, k71=0.8, k72=0.4):
    """

    :param Bfield: magnetic field with components Bx, By, Bz: 1D-array of length 3 or 2D-array of dim Nx3
    :param k12:
    :param k13:
    :param beta:
    :return: photoluminescence contrast in percent
    """


    k_no_mw = transition_rate_matrix(Bfield, 0, 0, beta, kr=kr, k47=k47, k57=k57, k71=k71, k72=k72)
    k_mw = transition_rate_matrix(Bfield, k12, k13, beta, kr=kr, k47=k47, k57=k57, k71=k71, k72=k72)

    pop_no_mw = populations(k_no_mw)
    pop_mw = populations(k_mw)

    pl_no_mw = photoluminescence_rate(k_no_mw, pop_no_mw)
    pl_mw = photoluminescence_rate(k_mw, pop_mw)


    if len(np.shape(pl_no_mw))==1:
        pl_no_mw = np.array(pl_no_mw)
        pl_mw = np.array(pl_mw)
        c = np.array(pl_no_mw - pl_mw)/np.array(pl_no_mw) * 100.
    else:
        c = np.array(pl_no_mw - pl_mw) / np.array(pl_no_mw) * 100.

    return c

if __name__ == '__main__':

    # solution calculated previously
    ref_contrast = [11.259005142154569, 11.162777024524557, 10.881956479400158, 10.43856472476315, 9.8649445385505405]
    ref_Bmag = [ 0, 10, 20, 30, 40]


    h_ref=[np.matrix([[ 0.00+0.j,  0.00+0.j,  0.00+0.j],
        [ 0.00+0.j,  2.87+0.j,  0.00+0.j],
        [ 0.00+0.j,  0.00+0.j,  2.87+0.j]]), np.matrix([[ 0.00000000+0.j        ,  0.00000000-0.01901095j,
          0.00000000-0.01901095j],
        [ 0.00000000+0.01901095j,  2.86229071+0.j        ,  0.00000000+0.j        ],
        [ 0.00000000+0.01901095j,  0.00000000+0.j        ,  2.87770929+0.j        ]]), np.matrix([[ 0.00000000+0.j        ,  0.00000000-0.03802189j,
          0.00000000-0.03802189j],
        [ 0.00000000+0.03802189j,  2.85458142+0.j        ,  0.00000000+0.j        ],
        [ 0.00000000+0.03802189j,  0.00000000+0.j        ,  2.88541858+0.j        ]]), np.matrix([[ 0.00000000+0.j        ,  0.00000000-0.05703284j,
          0.00000000-0.05703284j],
        [ 0.00000000+0.05703284j,  2.84687213+0.j        ,  0.00000000+0.j        ],
        [ 0.00000000+0.05703284j,  0.00000000+0.j        ,  2.89312787+0.j        ]]), np.matrix([[ 0.00000000+0.j        ,  0.00000000-0.07604378j,
          0.00000000-0.07604378j],
        [ 0.00000000+0.07604378j,  2.83916283+0.j        ,  0.00000000+0.j        ],
        [ 0.00000000+0.07604378j,  0.00000000+0.j        ,  2.90083717+0.j        ]])]

    import matplotlib.pyplot as plt
    k12 = 1
    k13 = 0
    beta = 0.3

    B= np.array([[0.0000961262, 0., 0.0000275637]]).T
    Bmag = np.arange(0,50,10)
    B = np.dot(B, np.array([Bmag])).T



    # h = [hamiltonian_nv_spin1(b) for b in B]
    # h = coupling_matrix(B)
    c = photoluminescence_contrast(B, k12, k13, beta)
    # k = transition_rate_matrix(B, k12, k13, beta)
    #
    # p = populations(k)
    #
    # pl = photoluminescence_rate(k, p)

    print(np.shape(c))
    print(c)

    print(np.allclose(c, ref_contrast))
