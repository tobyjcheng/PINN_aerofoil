import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
from matplotlib.ticker import MaxNLocator

aerofoil = "NACA2412"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

def boundaryNACA4D(M, P, SS, c, n, offset_x, offset_y):
    """
    Compute the coordinates of a NACA 4-digits airfoil
    
    Args:
        M:  maximum camber value (*100)
        P:  position of the maximum camber alog the chord (*10)
        SS: maximum thickness (*100)
        c:  chord length
        n:  the total points sampled will be 2*n
    """
    m = M / 100
    p = P / 10
    t = SS / 100
    if (m == 0):
        p = 1
    xv = np.linspace(0.0, c, n+1)
    xv = c / 2.0 * (1.0 - np.cos(np.pi * xv / c))
    ytfcn = lambda x: 5 * t * c * (0.2969 * (x / c)**0.5 - 
                                   0.1260 * (x / c) - 
                                   0.3516 * (x / c)**2 + 
                                   0.2843 * (x / c)**3 - 
                                   0.1015 * (x / c)**4)
    yt = ytfcn(xv)
    yc = np.zeros(np.size(xv))
    for ii in range(n+1):
        if xv[ii] <= p * c:
            yc[ii] = c * (m / p**2 * (xv[ii] / c) * (2 * p - (xv[ii] / c)))
        else:
            yc[ii] = c * (m / (1 - p)**2 * (1 + (2 * p - (xv[ii] / c)) * (xv[ii] / c) - 2 * p))
    dyc = np.zeros(np.size(xv))
    for ii in range(n+1):
        if xv[ii] <= p * c:
            dyc[ii] = m / p**2 * 2 * (p - xv[ii] / c)
        else:
            dyc[ii] = m / (1 - p)**2 * 2 * (p - xv[ii] / c)       
    th = np.arctan2(dyc, 1)
    xU = xv - yt * np.sin(th)
    yU = yc + yt * np.cos(th)
    xL = xv + yt * np.sin(th)
    yL = yc - yt * np.cos(th)
    x = np.zeros(2 * n + 1)
    y = np.zeros(2 * n + 1)
    for ii in range(n):
        x[ii] = xL[n - ii]
        y[ii] = yL[n - ii] 
    x[n : 2 * n + 1] = xU
    y[n : 2 * n + 1] = yU
    return np.vstack((x + offset_x, y + offset_y)).T

#%% NN Data
NN_u = np.load("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/u.npy")
NN_u = NN_u[:300, :600] 
NN_u = NN_u.reshape(150, 2, 300, 2).mean(axis=(1, 3)) 
NN_v = np.load("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/v.npy")
NN_v = NN_v[:300, :600] 
NN_v = NN_v.reshape(150, 2, 300, 2).mean(axis=(1, 3))  
NN_p = np.load("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/p.npy")
NN_p = NN_p[:300, :600]  
NN_p = NN_p.reshape(150, 2, 300, 2).mean(axis=(1, 3)) 


NN_p = -NN_p #ONLY INCLUDE IF WEIRD VALUES

with open("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/CFD/dump/ru-000002000.dat", "rb") as f:
    CFD_u = np.fromfile(f,dtype='float')
    CFD_u = CFD_u.reshape(150,300)
with open("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/CFD/dump/rv-000002000.dat", "rb") as f:
    CFD_v = np.fromfile(f,dtype='float')
    CFD_v = CFD_v.reshape(150,300)
with open("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/CFD/dump/p-000002000.dat", "rb") as f:
    CFD_p = np.fromfile(f,dtype='float')
    CFD_p = CFD_p.reshape(150,300)

  
#%%
xmin, xmax = 0.0, 0.6
ymin, ymax = 0.0, 0.3
airfoil_plot = boundaryNACA4D(int(aerofoil[4]), int(aerofoil[5]), int(aerofoil[6:8]), 0.2, 125, xmax/4, ymax/2)
dx = 0.002
dy = 0.002
x = np.arange(xmin, xmax, dx)
y = np.arange(ymin, ymax, dy)

# REDUCE THIS VALUES FOR MORE VIBRANT PLOT
umin = 0
umax = 1
vmin = -0.2
vmax = 0.2
pmin = -1
pmax = 1
cb_size = "5%"

cmap = "jet"
box_aspect = 0.5
ls = 19
nbins = 3
ts = 23

fig, ax = plt.subplots(3, 3, figsize=(16, 9))
fig.suptitle(aerofoil, fontsize=30, fontweight = "bold")

cax = ax[0,0].contourf(x, y, NN_u, 1000, cmap=cmap, vmin=umin, vmax=umax)
ax[0,0].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[0,0])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[0,0].set_box_aspect(box_aspect)
ax[0,0].set_title(r"PINN, $u$ [m/s]", fontsize = ts)

cax = ax[0,1].contourf(x, y, CFD_u, 1000, cmap=cmap, vmin=umin, vmax=umax)
ax[0,1].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[0,1])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[0,1].set_box_aspect(box_aspect)
ax[0,1].set_title(r"CFD, $u$ [m/s]", fontsize = ts)

cax = ax[0,2].contourf(x, y, abs(NN_u - CFD_u), 1000, cmap=cmap, vmin=umin, vmax=umax)
ax[0,2].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[0,2])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[0,2].set_box_aspect(box_aspect)
ax[0,2].set_title(r"Absolute Error, $u$ [m/s]", fontsize = ts)




cax = ax[1,0].contourf(x, y, NN_v, 1000, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,0].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[1,0])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[1,0].set_box_aspect(box_aspect)
ax[1,0].set_title(r"PINN, $v$ [m/s]", fontsize = ts)

cax = ax[1,1].contourf(x, y, CFD_v, 1000, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,1].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[1,1])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[1,1].set_box_aspect(box_aspect)
ax[1,1].set_title(r"CFD, $v$ [m/s]", fontsize = ts)

cax = ax[1,2].contourf(x, y, abs(NN_v - CFD_v), 1000, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,2].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[1,2])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[1,2].set_box_aspect(box_aspect)
ax[1,2].set_title(r"Absolute Error, $v$ [m/s]", fontsize = ts)




cax = ax[2,0].contourf(x, y, NN_p, 1000, cmap=cmap, vmin=pmin, vmax=pmax)
ax[2,0].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[2,0])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[2,0].set_box_aspect(box_aspect)
ax[2,0].set_title(r"PINN, $p-p_0$ [Pa]", fontsize = ts)

cax = ax[2,1].contourf(x, y, CFD_p, 1000, cmap=cmap, vmin=pmin, vmax=pmax)
ax[2,1].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[2,1])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[2,1].set_box_aspect(box_aspect)
ax[2,1].set_title(r"CFD, $p-p_0$ [Pa]", fontsize = ts)

cax = ax[2,2].contourf(x, y, abs(NN_p - CFD_p), 1000, cmap=cmap, vmin=pmin, vmax=pmax)
ax[2,2].fill(airfoil_plot[:, 0], airfoil_plot[:, 1], 'k')
divider = make_axes_locatable(ax[2,2])
cbar_ax = divider.append_axes("right", size=cb_size, pad=0.1)
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ls)
cbar.locator = MaxNLocator(nbins=nbins) 
cbar.update_ticks()
ax[2,2].set_box_aspect(box_aspect)
ax[2,2].set_title(r"Absolute Error, $p-p_0$ [Pa]", fontsize = ts)

for i in range(3):
    for j in range(3):
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

plt.tight_layout()
fig.subplots_adjust(top=0.85)
plt.show()
