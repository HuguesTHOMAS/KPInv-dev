
import numpy as np

def underline(str0):

    print()
    print(str0)
    print(len(str0) * '*')

    return

def frame_lines_1(lines):
    """
    Frame a list of str lines.
    """
    
    max_l = np.max([len(line) for line in lines])
    lines = ['|   {:<{width}s}   |'.format(line, width=max_l) for line in lines]
    print()
    print('+---' + max_l * '-' + '---+')
    for line in lines:
        print(line)
    print('+---' + max_l * '-' + '---+')
    print()

    return