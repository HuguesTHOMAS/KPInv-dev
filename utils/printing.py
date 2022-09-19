
import numpy as np

# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def underline(str0):

    print()
    print(str0)
    print(len(str0) * '*')

    return

def frame_lines_1(lines, no_print=False):
    """
    Frame a list of str lines.
    """
    
    max_l = np.max([len(line) for line in lines])
    lines = ['|   {:<{width}s}   |'.format(line, width=max_l) for line in lines]

    s = '\n'
    s += '+---' + max_l * '-' + '---+\n'
    for line in lines:
        s += line + '\n'
    s += '+---' + max_l * '-' + '---+\n'
    s += '\n'

    if not no_print:
        print(s)

    return s


def print_color(line):
    
    # color the checkmarks
    line = line.replace(u'\u2713', '{:}{:s}{:}'.format(bcolors.OKBLUE, u'\u2713', bcolors.ENDC))
    line = line.replace(u'\u2718', '{:}{:s}{:}'.format(bcolors.FAIL, u'\u2718', bcolors.ENDC))

    print(line)

    return


def color_str(s, color):
    c = getattr(bcolors, color)
    return '{:}{:s}{:}'.format(c, s, bcolors.ENDC)


def table_to_str(labels, columns, formats):
    """
    return a string of a table representing an array of data
    labels           (list): first table line, names of each column
    columns  (list of list): data for each line of the table
    formats          (list): format applied to each column
    """

    # format columns
    columns_str = [[f.format(d) for d in c] for c, f in zip(columns, formats)]

    # Get the max length of the column
    columns_l = np.max([[len(d) for d in c] for c in columns_str], axis=1)
    columns_l = [max(len(lbl), columns_l[i]) + 2 for i, lbl in enumerate(labels)]

    s = ''
    for lbl, l in zip(labels, columns_l):
        s += '{:^{width}s}|'.format(lbl, width=l)
    s += '\n'
    for l in columns_l:
        s += '{:s}|'.format('-'*l)
    s += '\n'

    for line_i in range(len(columns_str[0])):
        for col_str, l in zip(columns_str, columns_l):
            s += '{:^{width}s}|'.format(col_str[line_i], width=l)
        s += '\n'

    return s



