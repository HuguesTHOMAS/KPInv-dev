
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
