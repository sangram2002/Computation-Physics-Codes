import numpy as np

def convert_to_column(x):
    x.shape = (1, x.shape[0])#Row to column vector
    return x

def Householder(v):

    v_shape = v.shape[1]
    e1 = np.zeros_like(v)
    e1[0, 0] = 1
    vector = np.linalg.norm(v) * e1
    if v[0, 0] < 0:
        vector = - vector
    u = (v + vector)
    H = np.identity(v_shape) - ((2 * np.matmul(np.transpose(u), u)) / np.matmul(u, np.transpose(u)))# using the formula for householder
    return H

def QR(q, r, iter, n):
    v = convert_to_column(r[iter:, iter])
    Hbar = Householder(v)
    H = np.identity(n)
    H[iter:, iter:] = Hbar
    r = np.matmul(H, r)  #As R=....*H_2*H_1A So, it is multipliled on the left hand side of previous R to get new R
    q = np.matmul(q, H)  # As A=H_1.T*H_2.T*....*R=QR and H.T =H So, Q=H_1*H_2*....  So, it is multiplied on right side of old Q
    return q, r

def main():

    A = np.array([[1,-7,0],
           [2,-20,5],
           [2,1,1]])
    n, m = A.shape
    print('The A matrix is \n', A)
    Q = np.identity(n)
    R = A.astype(np.float64)
    for i in range(min(n, m)):

        Q, R = QR(Q, R, i, n)           # For each iteration, H matrix is calculated for (i+1)th row
    min_dim = min(m, n)
    R = np.around(R, decimals=6)
    R = R[:min_dim, :min_dim]
    Q = np.around(Q, decimals=6)
    print('R matrix is:')
    print(R, '\n')
    print('Q matrix is:')
    print(Q)
if __name__ == "__main__":
    main()
