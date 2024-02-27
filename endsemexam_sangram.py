import numpy as np
import matplotlib.pyplot as plt



nx=10
ny=10
# rho = np.zeros((nx,ny))
def gaussian(sigmax,sigmay, mux,muy,nx,ny):
    x, y = np.meshgrid(np.linspace(1, 50, ny),
                       np.linspace(1, 50, nx))
    normal = 1.0 / ( 2.0 * np.pi * sigmax ** 2)
    gauss = np.exp(-((x - mux) ** 2 / (2.0 * sigmax ** 2)+(y - muy) ** 2 / (2.0 * sigmay ** 2))) * normal
    return gauss

rho = 1e2*gaussian(130,130,25,25,50,50)

also compare the temporal dependence of total energy of the system




omega = np.linspace(1,2,11)

for t in omega:
    k = 0
    err1 = 1.0
    while (err1 > 5e-7):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                phi_new[i][j] = (phi_new[i][j]*(1-t)) + (t*(phi_new[i + 1][j] + phi_new[i - 1][j] + phi_new[i][j + 1] + phi_new[i][j - 1]-rho[i][j])) / 4.0

        err = phi_new[:, :] - phi[:, :]
        err1 = np.linalg.norm(err)
        # print(err1,k)
        phi[:, :] = phi_new[:, :]
        k += 1