

import numpy as np


def givens_rotation(A):

    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    # Apply Givens rotations to zero out lower triangular elements of R
    for j in range(n):
        for i in range(m - 1, j, -1):
            # To make all the elements of jth column =0 except the diagonal elements

            """
            Here we take two elements of R matrix as 1st and last element of jth column and then 1st and 2nd last element of jth 
            column and make last element of jth column 0 and then 2nd last element of jth column 0 and so on to make all the m-1 
            elements of the 1st column 0 and then repeat it for 2nd column and then 3rd column and so on for all j columns 
            """

            if R[i, j] != 0:
                s,t = R[j, j],R[i, j]
                c = s / np.sqrt(s ** 2 + t ** 2)
                s = t / np.sqrt(s ** 2 + t ** 2)
                G = np.eye(m)
                # G as a rotation matrix
                G[i,i] = c
                G[j,j] = c
                G[i,j] = -s
                G[j,i] = s
                R = np.dot(G, R)  #As R=...G_3*G_2*G_1 So, G is multiplied left side of R
                Q = np.dot(Q, G.T) # As A = G_1.T*G_2.T*....*R=QR So, G.T is done right side of Q to get new Q
# G.T means Transpose of G
    return Q, R



A = np.array([
    [0.8147, 0.0975, 0.1576],
    [0.9058, 0.2785, 0.9706],
    [0.1270, 0.5469, 0.9572],
    [0.9134, 0.9575, 0.4854],
    [0.6324, 0.9649, 0.8003],
], dtype=float)

# Print input matrix
print('The A matrix is:\n', A)
# Compute QR decomposition using Givens rotation
Q, R = givens_rotation(A)

# Print orthogonal matrix Q
print('\n The Q matrix is:\n', Q)

# Print upper triangular matrix R
print('\n The R matrix is:\n',R)