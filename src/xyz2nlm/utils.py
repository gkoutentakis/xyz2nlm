import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class struct:
    pass

def plot_fields(x, y, int1, int2, int3, int4):
    fig, axes = plt.subplots(2, 2)

    par = struct()
    par.cmap_name="inferno"
    pcolormesh_with_colorbar(fig, axes[0,0], x, y, np.abs(int1), par)
    pcolormesh_with_colorbar(fig, axes[0,1], x, y, np.abs(int2), par)
    pcolormesh_with_colorbar(fig, axes[1,0], x, y, np.abs(int3), par)
    pcolormesh_with_colorbar(fig, axes[1,1], x, y, np.abs(int4), par)

def pcolormesh_with_colorbar(fig, axis, x, y, data, params):

    pc = axis.pcolormesh(x, y, data.T,
                         cmap=params.cmap_name, shading="auto")
    pc.set_rasterized(True)  
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pc, cax=cax)
    cbar.solids.set_rasterized(True)  
    axis.set_aspect('equal')

    return pc, cax

if __name__ == "__main__":
    print('''This is a file with utilities providing:
(1) a basic struct implementation
(2) the plot_fields function
(3) the pcolormesh_with_colorbar function

It is not intended to be run directly!''')
        
