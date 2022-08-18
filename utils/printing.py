
import numpy as np

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