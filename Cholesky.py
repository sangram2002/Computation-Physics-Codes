import numpy as np
n=3
Amatrix = [[4,12,-16],
           [12,37,-43],
           [-16,-43,98]]
lmatrix = np.zeros((n,n))
lmatrix[0][0]= np.sqrt(Amatrix[0][0])
for i in range(n):
    lmatrix[i][0]=Amatrix[i][0]/lmatrix[0][0]

for j in range(n):
    sum = 0
    for k in range(j):
        sum += lmatrix[j][k] * lmatrix[j][k] 
    lmatrix[j][j]=np.sqrt(Amatrix[j][j]- sum)
    for i in range(j+1,n):
        sum=0
        for k in range(j):
            sum += lmatrix[i][k] * lmatrix[j][k]
            lmatrix[i][j] = (1 / lmatrix[j][j]) * (Amatrix[i][j] - sum)
print('Amatrix is:')
print(Amatrix)
print('Lmatrix is:')
print(lmatrix)

