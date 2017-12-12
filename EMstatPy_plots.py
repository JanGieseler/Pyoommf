import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy import linalg as LA

def plot_fields_in_plane(axes, data, value, data_type, settings, norm = 'linear', colorbar_limits = None, abs_value = False):
    """
    axes: string ('x', 'y', 'z') of axes of normal of plane to be plotted
    val: distance form origin
    data: dataframe with the data
    data_type: string for data ('Bfield_mag', 'Gradient', 'Bfield_proj')
    settings: dictionary with settings
    norm: string 'linear' or 'log' determines the scaling
    colorbar_limits: if not None use this two element array to set the limits of the colorbar
    abs_value: if True, plot the absolute value
    """

    magnet_dimensions_um = settings['magnet_dimensions_um']
    n = settings['unit_vector_magnet']
    m = settings['unit_vector_motion']
    s = settings['unit_vector_spin']

    H = magnet_dimensions_um['height']
    W = magnet_dimensions_um['width']
    L = magnet_dimensions_um['length']

    pos=np.array(list(set(data[axes])))
    x = pos[np.argmin(np.abs(value-pos))]
    data_frame = data.loc[data[axes] == x]

    if axes == 'x':
        a1, a2 = 'y', 'z'
        W, L = magnet_dimensions_um['width'], magnet_dimensions_um['length']
        arrow = patches.Arrow(-n[1]*W/2, -n[2]*L/2, n[1]*W, n[2]*L, width=0.2, edgecolor = 'k', fill = False)
        arrow_m = patches.Arrow(-m[1]*W/2, -m[2]*L/2, m[1]*W, m[2]*L, width=0.2, edgecolor = 'g', fill = False)
    elif axes == 'y':
        a1, a2 = 'x', 'z'
        W, L = magnet_dimensions_um['height'], magnet_dimensions_um['length']
        arrow = patches.Arrow(-n[0]*W/2, -n[2]*L/2, n[0]*W, n[2]*L, width=0.2, edgecolor = 'k', fill = False)
        arrow_m = patches.Arrow(-m[0]*W/2, -m[2]*L/2, m[0]*W, m[2]*L, width=0.2, edgecolor = 'g', fill = False)
    elif axes == 'z':
        a1, a2 = 'x', 'y'
        W, L = magnet_dimensions_um['height'], magnet_dimensions_um['width']
        arrow = patches.Arrow(-n[0]*W/2, -n[1]*L/2, n[0]*W, n[1]*L, width=0.2, edgecolor = 'k', fill = False)
        arrow_m = patches.Arrow(-m[0]*W/2, -m[1]*L/2, m[0]*W, m[1]*L, width=0.2, edgecolor = 'g', fill = False)
    else:
        TypeError('unknown axes')
    data_frame = data_frame.sort_values(by=[a1, a2], ascending = [True,True])


    A1, A2 = data_frame[a1].as_matrix(), data_frame[a2].as_matrix()
    N1, N2 = len(set(A1)), len(set(A2))
    if data_type == 'Gradient':
        D = data_frame['G'].as_matrix()
        label  = '$\partial_r(\mathbf{B}\cdot \mathbf{S}) $ ($T/\mu m$)'
    elif data_type == 'Bfield_mag':
        Bx, By, Bz = data_frame['Bx'].as_matrix(),data_frame['By'].as_matrix(),data_frame['Bz'].as_matrix()
        D = 10**4*np.reshape(np.sqrt(Bx**2+ By**2+ Bz**2), (N1, N2))
        label  = '$|\mathbf{B}| $ (Gauss)'
    elif data_type in('Bfield_proj','Bfield_par','Bfield_long') :
        Bx, By, Bz = data_frame['Bx'].as_matrix(),data_frame['By'].as_matrix(),data_frame['Bz'].as_matrix()
        D = 10**4*np.reshape(Bx*s[0]+By*s[1]+ Bz*s[2], (N1, N2))
        label  = '$\mathbf{B}\cdot \mathbf{S}$ (Gauss)'
        # label  = '$|\mathbf{B}| $ (Gauss)'
        label = '$B_{\parallel}$ (Gauss)'
    elif data_type in ('Bfield_perp', 'Bfield_trans') :
        Bx, By, Bz = data_frame['Bx'].as_matrix(),data_frame['By'].as_matrix(),data_frame['Bz'].as_matrix()
        D = 10**4*np.reshape(np.sqrt((By*s[2]-Bz*s[1])**2+(Bz*s[0] - Bx*s[2])**2+ (Bx*s[1] - By*s[0])**2), (N1, N2))
        # label  = '$|\mathbf{B}\times \mathbf{S}$ (Gauss)'
        label = '$B_{\perp}$ (Gauss)'
    else:
        TypeError('unknown axes')

    if abs_value:
        D = np.abs(D)

    # reshape
    A1 = np.reshape(A1, (N1, N2))
    A2 = np.reshape(A2, (N1, N2))

    D = np.reshape(D, (N1, N2))

    fig = plt.figure()


    ax = fig.add_subplot(111)

    if norm == 'linear':
        plt.pcolor(A1, A2, D, cmap = 'RdYlBu')
    elif norm == 'log':
        plt.pcolor(A1, A2, D, norm=LogNorm(vmin=D.min(), vmax=D.max()), cmap = 'RdYlBu')

    if colorbar_limits is not None:
        plt.colorbar(label = label)
    else:
        plt.colorbar(label = label)

    if colorbar_limits is not None:
        plt.clim(colorbar_limits[0],colorbar_limits[1])
    plt.xlabel('{:s} ($\mu m$)'.format(a1))
    plt.ylabel('{:s} ($\mu m$)'.format(a2))
    plt.title('{:s} = {:0.3f}$\mu m$'.format(axes, x))

    ax.add_patch(patches.Rectangle( (-W/2,-L/2), width = W, height = L, edgecolor = 'k', fill = False))
    ax.add_patch(arrow)
    ax.add_patch(arrow_m)
    plt.xlim([min(data_frame[a1]), max(data_frame[a1])])
    plt.ylim([min(data_frame[a2]), max(data_frame[a2])])

    return fig


def plot_1D_fields(axes, data_frame, data_type, settings, ax = None, norm = 'linear'):
    """
    axes: name of axes along which data is to be plotted
    data_frame: dataframe with the data
    data_type: string for data ('Bfield_mag', 'Gradient','Bfield_proj')
    settings: dictionary with settings
    norm: string 'linear' or 'log' determines the scaling
    """

    magnet_dimensions_um = settings['magnet_dimensions_um']
    n = settings['unit_vector_magnet']
    m = settings['unit_vector_motion']
    s = settings['unit_vector_spin']

    H = magnet_dimensions_um['height']
    W = magnet_dimensions_um['width']
    L = magnet_dimensions_um['length']


    if axes == 'x':
        W = magnet_dimensions_um['height']
    elif axes == 'y':
        W = magnet_dimensions_um['width']
    elif axes == 'z':
        W = magnet_dimensions_um['length']
    else:
        TypeError('unknown axes')
    data_frame = data_frame.sort_values(by=[axes], ascending = [True])


    A1 = data_frame[axes].as_matrix()
    N1 = len(set(A1))
    if data_type == 'Gradient':
        D = data_frame['G'].as_matrix()
        if norm == 'log':
            label  = '$\partial_r(\mathbf{B}\cdot \mathbf{S}) $ ($T/\mu m$)'
        else:
            label  = '$|\partial_r(\mathbf{B}\cdot \mathbf{S})| $ ($T/\mu m$)'
    elif data_type == 'Bfield_mag':
        Bx, By, Bz = data_frame['Bx'].as_matrix(),data_frame['By'].as_matrix(),data_frame['Bz'].as_matrix()
#         D = np.reshape(np.sqrt(Bx**2+ By**2+ Bz**2), (N1, N2))
        D = np.sqrt(Bx**2+ By**2+ Bz**2)
        label  = '$|\mathbf{B}| $ ($T$)'
    elif data_type == 'Bfield_proj':
        Bx, By, Bz = data_frame['Bx'].as_matrix(),data_frame['By'].as_matrix(),data_frame['Bz'].as_matrix()
#         D = np.reshape(Bx*s[0]+By*s[1]+ Bz*s[2], (N1, N2))
        D = Bx*s[0]+By*s[1]+ Bz*s[2]
        label  = '$\mathbf{B}\cdot \mathbf{S}$ ($T$)'
    else:
        TypeError('unknown axes')

    H_max = max(D)
    # reshape
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if norm == 'linear':
        ax.plot(A1, D)
        H_min = 0
    elif norm == 'log':
        if data_type == 'Gradient':
            D = np.abs(D)
        ax.semilogy(A1, D)
    H_min = min(D)

    ax.set_xlabel('{:s} ($\mu m$)'.format(axes))
    ax.set_ylabel(label)
#     plt.ylabel('{:s} = {:0.3f}$\mu m$'.format(axes, x))

    ax.add_patch(patches.Rectangle( (-W/2,H_min), width = W, height = H_max-H_min, edgecolor = 'k', fill = False))
#     plt.xlim([min(data_frame[a1]), max(data_frame[a1])])
#     plt.ylim([min(data_frame[a2]), max(data_frame[a2])])
#     if os.path.isdir(os.path.dirname(imagename)) == False:
#         os.makedirs(os.path.dirname(imagename))

#     fig.savefig('{:s}_{:s}_{:s}{:s}-plane_{:s}_{:0.3f}um.pdf'.format(imagename, data_type,a1.upper(), a2.upper(),axes, x))
