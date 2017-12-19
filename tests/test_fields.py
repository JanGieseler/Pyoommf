import random
from unittest import TestCase
import random
import pandas as pd
import numpy as np
import time

import fields as f

class fields(TestCase):
    def setUp(self):

        self.verbose = True

        # define the parameters
        # tag: name identifier (string)
        # a: radius in um
        # Br: surface magnetization in Teslas
        # phi_m: polar angle in deg
        # theta_m: azimuthal angle in deg
        # mu_0: vacuum permeability ( T m /A)
        # d_bead_z: distance between bead and z plane
        # dx: distance between points (in um)
        p = {
            'tag': 'bead_1',
            'a': 1.4,
            'Br': 0.4,
            'phi_m': 90,
            'theta_m': 90,
            'mu_0': 4 * np.pi * 1e-7,
            'd_bead_z': 0,
            'dx': 0.2
        }

        self.s, self.n = np.array([-1, -1, -1]), np.array([0, 0, 1])
        # s = np.array([-1, 1, 1])
        # s = np.array([1, -1, 1])
        # s = s / np.sqrt(3)
        # n = n / np.sqrt(3)

        # filename = '{:s}_a_{:0.1f}um_phi_m_{:2.0f}deg_theta_m_{:2.0f}deg.csv'.format(p['tag'], p['a'], p['phi_m'],p['theta_m'])

        # calculate the magnetic moment
        M = 4 * np.pi / 3 * p['Br'] / p['mu_0']
        phi_m = p['phi_m'] * np.pi / 180
        theta_m = p['theta_m'] * np.pi / 180
        self.M = M * np.array([[
            np.cos(phi_m) * np.sin(theta_m),
            np.sin(phi_m) * np.sin(theta_m),
            np.cos(theta_m)

        ]])

        # M = matrix of N=2 x 3
        self.M = M * np.array([[
            np.cos(phi_m) * np.sin(theta_m),
            np.sin(phi_m) * np.sin(theta_m),
            np.cos(theta_m)
        ],
            [
                np.cos(phi_m) * np.sin(theta_m),
                np.sin(phi_m) * np.sin(theta_m),
                np.cos(theta_m)

            ]


        ])


        xmax, ymax = 3, 3  # extend in um
        xmin, ymin = None, None
        dx = p['dx']

        if xmin is None:
            xmin = -xmax
        if ymin is None:
            ymin = -ymax
        zo = p['d_bead_z'] + p['a']

        # calculate the grid
        x = np.arange(xmin, xmax, dx)
        y = np.arange(ymin, ymax, dx)
        Nx, Ny = len(x), len(y)
        X, Y = np.meshgrid(x, y)
        np.shape(X), np.shape(Y)

        self.r = np.array([X.flatten(), Y.flatten(), zo * np.ones(len(X.flatten()))]).T


        self.DipolePositions = np.zeros([1, 3])  # we assume that the magnet is at 0,0,0

        # DipolePositions = matrix of N=2 x 3
        self.DipolePositions = np.zeros([2, 3])  # we assume that the magnet is at 0,0,0

        print('shape M:')
        print(np.shape(self.M))
        print('shape DipolePositions:')
        print(np.shape(self.DipolePositions))

    def my_field_simple(self, r, dp_pos, m):
        """

        simple but safe way of calculating the B field without vectorizing
        :param r: vector length 3
        :param dp_pos: vector length 3
        :param m: vector length 3
        :return:
        """
        mu0 = 4 * np.pi * 1e-7  # T m /A
        a = r - dp_pos


        rho = np.sqrt(np.sum(a ** 2))

        # calculate the vector product of m and a: m*(r-ri)
        ma = np.sum(m*a)
        B = mu0 / (4 * np.pi) * (3. * a * ma / rho ** 5 - m / rho ** 3)  # magnetic field in Tesla

        return B

    def test01_Bfield_single_pt(self):
        """
        test the function calcBfield_single_pt

        :return:
        """
        N = 2

        # create random vectors
        r = np.random.rand(3)
        m = np.random.rand(N, 3)
        dp_pos = np.random.rand(N, 3)

        # r = np.array([ 0.61272976,  0.93453872,  0.13334545])
        # m = np.array([ 0.53571495,  0.45998269,  0.65775688])
        # dp_pos =np.array([ 0.01321936, 0.6526304,   0.50160241])

        if self.verbose:
            print('r', r)
            print('m', m)
            print('dp_pos', dp_pos)


        # calculate for each dipole with the simple formula
        B_simple = np.array([self.my_field_simple(r, p, mi) for mi, p in zip(m, dp_pos)])
        # now sum up to get total field
        B_simple = np.sum(B_simple, 0)
        B = f.calcBfield_single_pt(r, dp_pos, m)

        err = np.mean(np.abs(B_simple - B))
        if self.verbose:
            print('err', err)
            print('B_simple', B_simple)
            print('B', B)
        if err > 1e-8:
            print('rB_simple', B_simple)
            print('B', B)

            raise ValueError

    def test02_Bfield_many_pt(self):

        N, M= 2, 4

        # create random vectors
        r = np.random.rand(M, 3)
        m = np.random.rand(N, 3)
        dp_pos = np.random.rand(N, 3)


        if self.verbose:
            print('r', r)
            print('m', m)
            print('dp_pos', dp_pos)



        B_simple = np.array([f.calcBfield_single_pt(ri, dp_pos, m) for ri in r])
        # B_simple = np.sum(B_simple, 1)

        B = f.calcBfield(r, dp_pos, m)
        print(B[['x','y','z']])


        if self.verbose:
            print('B', B)
            print('B_simple', B_simple)

        # err = np.sum(np.abs(B_simple - np.array(B[['x','y','z']])))
        err = np.mean(np.abs(B_simple - np.array(B[['Bx','By','Bz']])))

        print('---', err)

        if err > 1e-6:
            raise ValueError

    def test03_Grad_single_pt(self):
        """
        test the function calcBfield_single_pt

        :return:
        """
        N = 2

        # create random vectors
        r = np.random.rand(3)
        m = np.random.rand(N, 3)
        dp_pos = np.random.rand(N, 3)

        # r = np.array([ 0.61272976,  0.93453872,  0.13334545])
        # m = np.array([ 0.53571495,  0.45998269,  0.65775688])
        # dp_pos =np.array([ 0.01321936, 0.6526304,   0.50160241])

        if self.verbose:
            print('r', r)
            print('m', m)
            print('dp_pos', dp_pos)


        # calculate for each dipole with the simple formula
        B_simple = np.array([self.my_field_simple(r, p, mi) for mi, p in zip(m, dp_pos)])
        # now sum up to get total field
        B_simple = np.sum(B_simple, 0)
        B = f.calcBfield_single_pt(r, dp_pos, m)

        err = np.mean(np.abs(B_simple - B))
        if self.verbose:
            print('err', err)
            print('B_simple', B_simple)
            print('B', B)
        if err > 1e-8:
            print('rB_simple', B_simple)
            print('B', B)

            raise ValueError

    def test04_Grad_many_pt(self):

        N, M= 2, 4

        # create random vectors
        r = np.random.rand(M, 3)
        m = np.random.rand(N, 3)
        dp_pos = np.random.rand(N, 3)


        if self.verbose:
            print('r', r)
            print('m', m)
            print('dp_pos', dp_pos)



        B_simple = np.array([f.calcBfield_single_pt(ri, dp_pos, m) for ri in r])
        # B_simple = np.sum(B_simple, 1)

        B = f.calcBfield(r, dp_pos, m)
        print(B[['x','y','z']])


        if self.verbose:
            print('B', B)
            print('B_simple', B_simple)

        # err = np.sum(np.abs(B_simple - np.array(B[['x','y','z']])))
        err = np.mean(np.abs(B_simple - np.array(B[['Bx','By','Bz']])))

        print('---', err)

        if err > 1e-6:
            raise ValueError