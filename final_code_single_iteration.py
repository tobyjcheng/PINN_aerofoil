import deepxde as dde
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
from PIL import Image

aerofoil = "NACA6409"

#%%

dde.config.set_random_seed(1) 
dde.config.set_default_float('float64')
print(f"Current backend: {dde.backend.backend_name}")

xmin, xmax = 0.0, 0.6
ymin, ymax = 0.0, 0.3
rec_size = 0.05

rho  = 1.0
mu   = 0.01

def boundaryNACA4D(M, P, SS, c, n, offset_x, offset_y):
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

def navier_stokes(x, y):
    psi, p, sigma11, sigma22, sigma12 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]
    u =   dde.grad.jacobian(y, x, i = 0, j = 1)
    v = - dde.grad.jacobian(y, x, i = 0, j = 0)
    u_x = dde.grad.jacobian(u, x, i = 0, j = 0)
    u_y = dde.grad.jacobian(u, x, i = 0, j = 1)
    v_x = dde.grad.jacobian(v, x, i = 0, j = 0)
    v_y = dde.grad.jacobian(v, x, i = 0, j = 1)
    sigma11_x = dde.grad.jacobian(y, x, i = 2, j = 0)
    sigma12_x = dde.grad.jacobian(y, x, i = 4, j = 0)
    sigma12_y = dde.grad.jacobian(y, x, i = 4, j = 1)
    sigma22_y = dde.grad.jacobian(y, x, i = 3, j = 1)
    continuumx = rho * (u * u_x + v * u_y) - sigma11_x - sigma12_y
    continuumy = rho * (u * v_x + v * v_y) - sigma12_x - sigma22_y
    constitutive1 = - p + 2 * mu * u_x - sigma11
    constitutive2 = - p + 2 * mu * v_y - sigma22
    constitutive3 = mu * (u_y + v_x) - sigma12
    constitutive4 = p + (sigma11 + sigma22) / 2
    return continuumx, continuumy, constitutive1, constitutive2, constitutive3, constitutive4


farfield = dde.geometry.Rectangle([xmin, ymin], [xmax, ymax])
airfoil  = dde.geometry.Polygon(boundaryNACA4D(int(aerofoil[4]), int(aerofoil[5]), int(aerofoil[6:8]), 0.2, 250, xmax/4, ymax/2))
geom     = dde.geometry.CSGDifference(farfield, airfoil)
inner_rec  = dde.geometry.Rectangle([(xmax/4)-rec_size, (ymax/2)-rec_size], [(xmax/4)+rec_size, (ymax/2)+rec_size])
outer_dom  = dde.geometry.CSGDifference(farfield, inner_rec)
outer_dom  = dde.geometry.CSGDifference(outer_dom, airfoil)
inner_dom  = dde.geometry.CSGDifference(inner_rec, airfoil)
inner_points = inner_dom.random_points(5000)
outer_points = outer_dom.random_points(5000)
farfield_points = farfield.random_boundary_points(1280)
airfoil_points  = boundaryNACA4D(int(aerofoil[4]), int(aerofoil[5]), int(aerofoil[6:8]), 0.2, 125, xmax/4, ymax/2)
points = np.append(inner_points, outer_points, axis = 0)
points = np.append(points, farfield_points, axis = 0)
points = np.append(points, airfoil_points, axis = 0)


def boundary_farfield_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], xmin)

def boundary_farfield_top_bottom(x, on_boundary):
    return on_boundary and (np.isclose(x[1], ymax) or np.isclose(x[1], ymin))

def boundary_farfield_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], xmax)

def boundary_airfoil(x, on_boundary):
    return on_boundary and (not farfield.on_boundary(x))


def fun_u_farfield(x, y, _):
    return dde.grad.jacobian(y, x, i = 0, j = 1) - 1.0

def fun_no_slip_u(x, y, _):
    return dde.grad.jacobian(y, x, i = 0, j = 1)

def fun_no_slip_v(x, y, _):
    return - dde.grad.jacobian(y, x, i = 0, j = 0)

def funP(x):
    return 0.0
  

bc_inlet_u = dde.OperatorBC(geom, fun_u_farfield, boundary_farfield_inlet)
bc_inlet_v = dde.OperatorBC(geom, fun_no_slip_v, boundary_farfield_inlet)
bc_top_bottom_u = dde.OperatorBC(geom, fun_u_farfield, boundary_farfield_top_bottom)
bc_top_bottom_v = dde.OperatorBC(geom, fun_no_slip_v, boundary_farfield_top_bottom)
bc_outlet_p = dde.DirichletBC(geom, funP, boundary_farfield_outlet, component = 1)
bc_airfoil_u = dde.OperatorBC(geom, fun_no_slip_u, boundary_airfoil)
bc_airfoil_v = dde.OperatorBC(geom, fun_no_slip_v, boundary_airfoil)
bcs = [bc_inlet_u, bc_inlet_v, bc_top_bottom_u, bc_top_bottom_v, bc_outlet_p, bc_airfoil_u, bc_airfoil_v]
data = dde.data.PDE(geom, navier_stokes, bcs, num_domain = 0, num_boundary = 0, num_test = 5000, anchors = points)

#%%

fig, ax = plt.subplots(figsize=(16, 9))
ax.scatter(data.train_x_all[:,0], data.train_x_all[:,1], s = 0.5, color = 'k')
ax.axis('equal')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.axis('off')
plt.savefig("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/train_data_image")

#%%

scale = 1000
aerofoil_points = boundaryNACA4D(int(aerofoil[4]), int(aerofoil[5]), int(aerofoil[6:8]), 0.2, 1000, xmax/4, ymax/2)
xpoint_scaled = aerofoil_points[:, 0]*scale
ypoint_scaled = aerofoil_points[:, 1]*scale
ypoint_scaled = -ypoint_scaled

canvas_width = int(xmax*scale) 
canvas_height = int(ymax*scale)
image = Image.new('RGB', (canvas_width, canvas_height), 'white')
pixels = image.load()

for x, y in zip(xpoint_scaled, ypoint_scaled):
    x_int = round(x)
    y_int = round(y)
    pixels[x_int,y_int] = (0,0,0)

for y in range(canvas_height):
    pixels[0, y] = (0, 0, 255)  

for x in range(canvas_width):
    pixels[x, 0] = (34, 177, 76) 

for x in range(canvas_width):
    pixels[x, canvas_height - 1] = (34, 177, 76)  
    
    
save_directory = "C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil 
file_name = 'cfd_setup.png'

save_path = os.path.join(save_directory, file_name)
image.save(save_path)

#%%

dde.config.set_default_float('float64')
layer_size  = [2] + [40] * 8 + [5]
activation  = 'tanh' 
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile(optimizer = 'adam', lr = 5e-4, loss_weights = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
losshistory, train_state = model.train(epochs = 10000, display_every = 2000)
dde.saveplot(losshistory, train_state, issave = True, isplot = True)
model.compile(optimizer = 'L-BFGS', loss_weights = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
model.train_step.optimizer_kwargs = {'options': {'maxcor': 50,'ftol': 1e-11,'maxfun':  50000,'maxiter': 50000,'maxls': 50}}
losshistory, train_state = model.train(display_every = 5000)
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

#%%

dx = 0.001
dy = 0.001
x = np.arange(xmin, xmax + dy, dx)
y = np.arange(ymin, ymax + dy, dy)
X = np.zeros((len(x)*len(y), 2))
xs = np.vstack((x,)*len(y)).reshape(-1)
ys = np.vstack((y,)*len(x)).T.reshape(-1)
X[:, 0] = xs
X[:, 1] = ys

def getU(x, y):   
    return dde.grad.jacobian(y, x, i = 0, j = 1) 

def getV(x, y): 
    return - dde.grad.jacobian(y, x, i = 0, j = 0)  

def getP(x, y):
    return y[:, 1:2]

u = model.predict(X, operator = getU)
v = model.predict(X, operator = getV)
p = model.predict(X, operator = getP)
u_v2 = u.reshape(len(y), len(x))
v_v2 = v.reshape(len(y), len(x))
p_v2 = p.reshape(len(y), len(x))

np.save("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/u", u_v2)
np.save("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/v", v_v2)
np.save("C:/Users/Toby/Desktop/Dissertation_Final/"+aerofoil+"/p", p_v2)
