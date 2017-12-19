import fields as f

import numpy as np
import pandas as pd
import time
# define the parameters
#tag: name identifier (string)
#a: radius in um
#Br: surface magnetization in Teslas
#phi_m: polar angle in deg
#theta_m: azimuthal angle in deg
#mu_0: vacuum permeability ( T m /A)
#d_bead_z: distance between bead and z plane
#dx: distance between points (in um)
p = {
    'tag':'bead_1',
    'a' : 1.4,
    'Br' : 0.4 ,
    'phi_m' : 90,
    'theta_m' : 90,
    'mu_0' : 4 * np.pi * 1e-7,
    'd_bead_z': 0,
    'dx':0.2
}

s, n = np.array([-1,-1,-1]),np.array([0,0,1])
s = np.array([-1,1,1])
s = np.array([1,-1,1])
s = s/np.sqrt(3)
n = n/np.sqrt(3)

filename = '{:s}_a_{:0.1f}um_phi_m_{:2.0f}deg_theta_m_{:2.0f}deg.csv'.format(p['tag'], p['a'], p['phi_m'], p['theta_m'])

# calculate the magnetic moment
M = 4*np.pi/3*p['Br']/p['mu_0']
phi_m =p['phi_m']*np.pi/180
theta_m =p['theta_m']*np.pi/180
M = M*np.array([[
        np.cos(phi_m)*np.sin(theta_m),
        np.sin(phi_m)*np.sin(theta_m),
        np.cos(theta_m)

    ]])

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

r = np.array([X.flatten(), Y.flatten(), zo * np.ones(len(X.flatten()))]).T

start = time.time()

r = r[1]

DipolePositions = np.zeros([1, 3])  # we assume that the magnet is at 0,0,0

# print(r, DipolePositions, M)
print('shapes', np.shape(r), np.shape(DipolePositions), np.shape(M))
B = f.calcBfield_single_pt(r, DipolePositions, M)

G = f.calcGradient_single_pt(r, DipolePositions, M, s, n)



# data_out = f.calcBfield(r, DipolePositions, M)
# G = f.calcGradient(r, DipolePositions, M, s, n)

print(np.shape(B))
print(np.shape(G))
