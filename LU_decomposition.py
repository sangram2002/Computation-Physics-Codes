import numpy as np

# len(matrix) = n
n=3
lmatrix=np.zeros((n,n))
umatrix=np.zeros((n,n))
matrix = [[ 2, 1, 3], [4,4,7], [2,5,9]]

for i in range(n):
    lmatrix[i][i]=1
    umatrix[0][i]=matrix[0][i]
    lmatrix[i][0]=matrix[i][0]/umatrix[0][0]

    
for k in range(1,n):
    # for t in range(n-1):

    for j in range(k,n):
        temp = 0
        for m in range(k):
            temp += lmatrix[k][m]*umatrix[m][j]
            umatrix[k][j]=matrix[k][j]-temp
        
    for i in range(k,n):
        temp=0
        for m in range(0,k):
            temp += lmatrix[i][m]*umatrix[m][k]
        lmatrix[i][k]= (matrix[i][k]-temp)/umatrix[k][k]

print("A_matrix is:")
for i in range(n):
    print(matrix[i])
print("L_matrix is:")
print(lmatrix)
print("U_matrix is:")
print(umatrix)

    