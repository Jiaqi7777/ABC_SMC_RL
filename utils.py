import random
import numpy as np
import matplotlib.pyplot as plt
def argmaxs(arr):
    mask = arr == arr.max()
    return random.choice(np.array(range(len(arr)))[mask])


def plot_3d(X, Y, Z, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=False):
    figure_path = 'Figures/'
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
            cmap='viridis', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    if show:
        plt.show()
    plt.savefig(f'{figure_path+title}.png')
    plt.clf()

def replace_line(file_path, variable, new_value):
    with open(file_path) as f:
        lines = f.readlines()
    f.close()
    with open(file_path, 'w') as f:
        for line in lines:
            if variable == line.split(' = ')[0]:
                print("Replace to ", line)
                line = f'{variable} = {new_value}\n'
            f.write(line)
    f.close()