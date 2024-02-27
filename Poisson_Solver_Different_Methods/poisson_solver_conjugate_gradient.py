import numpy as np
import matplotlib.pyplot as plt
import time

# start the timer
start_time = time.time()

N = 50
h = 1 / (N + 1) #Spacing

x = np.linspace(0, 1, N )
y = np.linspace(0, 1, N )

# Define the b matrix as a Gaussian distribution
X, Y = np.meshgrid(x, y)
sigma = 0.1
b = np.exp(-(X - 0.5) ** 2 / (2 * sigma ** 2) - (Y - 0.5) ** 2 / (2 * sigma ** 2))
b = b.reshape(-1) #Done to convert 2 dimensional gaussian distribution into 1d array

# Define the A matrix
"""
 The A matrix in the Poisson equation represents the discretized Laplacian operator, 
 which depends on the specific discretization scheme being used.
 I have used a simple five-point stencil scheme to discretize the Laplacian operator in two dimensions.
   """
main_diag = -4 * np.ones(N ** 2)
off_diag = np.ones(N ** 2 - 1)
off_diag_N = np.ones(N ** 2 - N)
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1) + np.diag(off_diag_N, k=N) + np.diag(
    off_diag_N, k=-N)
A = A / h ** 2

# Define the initial guess for x
x_0 = np.zeros(N ** 2)

max_iter = 100000

tolerance = 5e-7

x_prev = x_0

r_prev = b - np.dot(A, x_prev)

# Initialize p(k-1) as r(k-1)
p_prev = r_prev

for k in range(max_iter):

    # alpha(k)
    alpha = np.dot(r_prev.T, r_prev) / np.dot(np.dot(p_prev.T, A), p_prev)

    # Calculate x(k) using the updated alpha(k)
    x = x_prev + alpha * p_prev

    # Calculate r(k) using the updated x(k)
    r = b - np.dot(A, x)

    if np.linalg.norm(r) < tolerance:

        # end the timer
        end_time = time.time()

        # compute the elapsed time
        elapsed_time = end_time - start_time

        print('No. of iterations to converge for tolerance 5e-7 =', k, 'and computation time =', elapsed_time, 'seconds')
        break

    # beta(k)
    beta = np.dot(r.T, r) / np.dot(r_prev.T, r_prev)

    # Calculate p(k) using the updated r(k) and beta(k)
    p = r + beta * p_prev

    x_prev = x
    r_prev = r
    p_prev = p

# Reshape the solution vector x into a 2D array
x = x.reshape((N, N))

plt.imshow(x, cmap='jet')
plt.colorbar()
# plt.show()
#plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\conjgate_gradient.png')

#out of the program------------------------------------------------------------------------------------------
#No. of iterations to converge for tolerance 5e-7 = 128 and computation time = 2.02 seconds
#--------------------------------------------------------------------------------------------------------------