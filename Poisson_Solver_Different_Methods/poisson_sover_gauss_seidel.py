import numpy as np
import matplotlib.pyplot as plt
import time

# start the timer
start_time = time.time()

# print(x)
nx=50
ny=50
# rho = np.zeros((nx,ny))
def gaussian(sigmax,sigmay, mux,muy,nx,ny):
    x, y = np.meshgrid(np.linspace(1, 50, ny),
                       np.linspace(1, 50, nx))
    normal = 1.0 / ( 2.0 * np.pi * sigmax ** 2)
    gauss = np.exp(-((x - mux) ** 2 / (2.0 * sigmax ** 2)+(y - muy) ** 2 / (2.0 * sigmay ** 2))) * normal
    return gauss

rho = 1e2*gaussian(130,130,25,25,50,50)

im = plt.imshow(rho, cmap="copper_r")
plt.colorbar(im)
# plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\gaussian.png')
phi = np.zeros((nx,ny))
phi_new = np.zeros((nx,ny))
# print(np.shape(phi),np.shape(phi_new),np.shape(rho))

k=0
err1 = 1.0
while(err1>5e-7):
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            phi_new[i][j] = (-rho[i][j] + phi_new[i+1][j]+phi_new[i-1][j] + phi_new[i][j+1]+ phi_new[i][j-1])/4.0
    err = phi_new[:,:]-phi[:,:]
    err1 = np.linalg.norm(err)
    # print(err1,k)
    phi[:,:]=phi_new[:,:]
    k+=1

# end the timer
end_time = time.time()

# compute the elapsed time
elapsed_time = end_time - start_time

print('No. of iterations to converge for tolerance 5e-7 =',k,'and computation time =',elapsed_time,'seconds')
phi_img = plt.imshow(phi, cmap="copper_r")
plt.colorbar(phi_img)
# plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\phi_using_jacobi.png')


#out of the program------------------------------------------------------------------------------------------
#No. of iterations it takes to converge is 2560 for error 5e-7, computation time is around 15.6 sec
#--------------------------------------------------------------------------------------------------------------