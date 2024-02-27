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
#The value ω = 1 gives the Gauss–Seidel method again; ω < 1 would produce under -relaxation, where we keep a proportion of the old
#solution; and ω > 1 produces over -relaxation where we actually move further away from the old solution than we would using Gauss–Seidel.

    # end the timer
    end_time = time.time()

    # compute the elapsed time
    elapsed_time = end_time - start_time
    print('For omega value ', t, 'No. of iterations to converge =', k, 'and computation time =',
    elapsed_time, 'seconds')

phi_img = plt.imshow(phi, cmap="copper_r")
plt.colorbar(phi_img)
# plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\phi_using_jacobi.png')


#The ouput of my program-------------
#
# For omega value  1.0 No. of iterations to converge = 2560 and computation time = 15.369245290756226 seconds
# For omega value  1.1 No. of iterations to converge = 40 and computation time = 15.58871340751648 seconds
# For omega value  1.2 No. of iterations to converge = 33 and computation time = 15.762105941772461 seconds
# For omega value  1.3 No. of iterations to converge = 28 and computation time = 15.908713579177856 seconds
# For omega value  1.4 No. of iterations to converge = 24 and computation time = 16.030447959899902 seconds
# For omega value  1.5 No. of iterations to converge = 21 and computation time = 16.143394708633423 seconds
# For omega value  1.6 No. of iterations to converge = 18 and computation time = 16.23786473274231 seconds
# For omega value  1.7000000000000002 No. of iterations to converge = 15 and computation time = 16.31515622138977 seconds
# For omega value  1.8 No. of iterations to converge = 13 and computation time = 16.38097310066223 seconds
# For omega value  1.9 No. of iterations to converge = 12 and computation time = 16.441986560821533 seconds
# For omega value  2.0 No. of iterations to converge = 15 and computation time = 16.51697301864624 seconds

#out of the program------------------------------------------------------------------------------------------
# So, for omega =1.9 it converges with minimum number of iterations
#--------------------------------------------------------------------------------------------------------------