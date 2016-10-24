import pandas as pd
import numpy as np



def load_ommf_tab_data(filename):
    """
    load the 1D data from a .odt file as created by oommf
    """
    f = open(filename, "rb")
    for i in range(40):
        s = f.readline()
        if '# Columns:' in s:
            headers = [x.split('::')[1].split('{')[0].split('}')[0].strip() for x in s.split('Oxs')[1:]]
            break
    data = np.loadtxt(f)
    df = pd.DataFrame.from_records(data, columns=headers)
    df = df.set_index(['Stage'])
    return df


def load_ommf_vect_data(filename):
    """
    reads a dataset that has been produced with oommf and returns the data and file info
    """
    f = open(filename, "rb")

    file_info = {
        'xmin': 0, 'xmax': 0, 'xstepsize': 0, 'xnodes': 0,
        'ymin': 0, 'ymax': 0, 'ystepsize': 0, 'ynodes': 0,
        'zmin': 0, 'zmax': 0, 'zstepsize': 0, 'znodes': 0,
        'field_type': ''
    }

    for i in range(40):
        s = str(f.readline())

        if '# End: Header' in s:
            break

        for k in file_info.keys():
            if '# {:s}'.format(k) in s:
                file_info[k] = float(s.split('# {:s}:'.format(k))[1].split('\\n')[0].strip())

        if '# Title' in s:
            file_info['field_type'] = s.split('::')[1].split('\\n')[0].split('\n')[0]

    data = np.loadtxt(filename)

    x = np.arange(file_info['xmin'] + file_info['xstepsize'] / 2, file_info['xmax'], file_info['xstepsize'])
    y = np.arange(file_info['ymin'] + file_info['ystepsize'] / 2, file_info['ymax'], file_info['ystepsize'])
    z = np.arange(file_info['zmin'] + file_info['zstepsize'] / 2, file_info['zmax'], file_info['zstepsize'])

    X, Y = np.meshgrid(x, y)
    X, Z = np.meshgrid(X.flatten(), z)
    Y, _ = np.meshgrid(Y.flatten(), z)

    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    label = file_info['field_type']
    if 'Magnetization' in label:
        label = 'm'

    df = pd.DataFrame.from_dict({
        'x': X, 'y': Y, 'z': Z,
        '{:s}x'.format(label): data[:, 0],
        '{:s}y'.format(label): data[:, 1],
        '{:s}z'.format(label): data[:, 2]
    })

    return df, file_info


def get_slice(df, value, info, axis='z'):
    """
    gets the subdataset that has approximately a constant z-value
    axis (str): x, y or z
    """
    filter_str = '%s >= %e & %s <= %e ' % (
    axis, value - info['{:s}stepsize'.format(axis)] / 3., axis, value + info['{:s}stepsize'.format(axis)] / 3.)
    return df.query(filter_str)


def b_field_mag(df):
    """
    return the magnetic field in Gauss
    """
    return np.sqrt(df['Hx'] ** 2 + df['Hy'] ** 2 + df['Hz'] ** 2) * (4 * np.pi * 1e-7 * 1e4)