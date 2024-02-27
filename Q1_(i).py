import numpy as np
import matplotlib.pyplot as plt

L = 10
N = 10
V = np.zeros((N,N))
V[0,:] = 0
V[:,0] = 0
V[N-1,:] = 0
V[:,N-1] = 0

d =0.4
x1 =int(N/2 - d*N/2)
x2 =int(N/2 + d*N/2)
y1 =int(N/2 - N/10)
y2 =int(N/2 + N/10)
V[x1,y1:y2] =100
V[x2,y1:y2] =-100

omega = 1.9
tolerance = 5e-7

converged = False
iterations = 0
while not converged:
    iterations += 1
    V_old = np.copy(V)
    for i in range(1,N-1):
        for j in range(1,N-1):
            V[i,j] = (1-omega)*V_old[i,j] + omega/4*(V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
    diff = np.max(np.abs(V - V_old))
    if diff < tolerance:
        converged = True


print('Number of iterations it took to converge=', iterations)

X, Y = np.meshgrid(np.arange(0, L, L/N), np.arange(0, L, L/N))
plt.contourf(X, Y, V, cmap='jet')
plt.colorbar()
plt.title('dipole')

# plt.show()
plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\Q1_(i)_contour_plot.png')

