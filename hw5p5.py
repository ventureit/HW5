import numpy as np
import matplotlib.pyplot as plt


iterations = 0
tolerance = 1e-14
eta = 0.1

u_t = u_0 = 1.0
v_t = v_0 = 1.0

E_uv = lambda u,v: (u * np.exp(v) - 2 * v * np.exp(-u))**2 # error at t

dE_du = lambda u,v: 2 * (np.exp(v) + 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))

dE_dv = lambda u,v: 2 * (u * np.exp(v) - 2 * np.exp(-u)) * (u * np.exp(v) - 2 * v * np.exp(-u))

E_uv_t = E_uv(u_0, v_0) #error function initialization

while E_uv_t >= tolerance:

    u_t1 = u_t - eta * dE_du(u_t, v_t) # u_t+1 growth
    v_t1 = v_t - eta * dE_dv(u_t, v_t) #v_t+1 growth

    u_t = u_t1
    v_t = v_t1

    E_uv_t = E_uv(u_t1, v_t1) #new error
    iterations += 1

print("{} iterations".format(iterations))
print("(u_t,v_t) =({}, {}) ".format(round(u_t, 3), round(v_t, 3)))
