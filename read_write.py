import pandas as pd
import numpy as np
import os

def load_ommf_tab_data(filename):
    """
    load the 1D data from a .odt file as created by oommf
    """
    f = open(filename, "rb")
    for i in range(40):
        s = str(f.readline()) # if read as a byte convert to string
        if '# Columns:' in s:
            headers = [x.split('::')[1].split('{')[0].split('}')[0].strip() for x in s.split('Oxs')[1:]]

            # next line contains the units
            s = str(f.readline())  # if read as a byte convert to string
            units = [x for x in s.split(' ') if x is not ''][2:]
            break

    units = {k: d for k, d in zip(headers, units)}
    data = np.loadtxt(f)
    df = pd.DataFrame.from_records(data, columns=headers)
    df = df.set_index(['Stage'])
    return df, units


def load_ommf_vect_data(filename):
    """
    reads a dataset that has been produced with oommf and returns the data and file info
    """

    print('loading {:s}'.format(filename))
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


def convert_omf_2_tsv(source, oommf_path='/Applications/oommf/'):
    """
    converts the .omf files in source into tsv files using the convertion shipped with oommf

    """

    if isinstance(source, str):
        source = [source]
    for s in source:
        target = s.replace('.omf', '-omf.tsv')
        command = '{:s} avf2ovf -format text {:s} {:s}'.format(os.path.join(oommf_path, 'oommf.tcl'), s, target)
        # call(command)
        os.system(command)

def get_parameters_from_mif(filename):
    """
    return the parameters defined in a .mif file as a dictionary.
    This function searches for lines in a text file that start with Parameter =
    """
    f = open(filename, "rb")

    parameters = {}
    for i in range(2000):
        s = str(f.readline())
        if s == "b''":
            break
        if 'Parameter' in s:
            parameters.update({s.split(' ')[1]: s.split(' ')[2].split('\\r')[0]})

    f.close()
    return parameters



if __name__ == '__main__':

    import glob

    source = glob.glob('/Users/rettentulla/Downloads/20170209_TrapezoidalCoMagnet/Length_1.5um_transition_0.4um/*.omf')
    # print(source)
    convert_omf_2_tsv(source)
