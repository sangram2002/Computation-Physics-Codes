import numpy as np
import pandas as pd
import glob
import os

def range_with_floats(start,stop,step):
    while stop>start:
        yield start
        start+=step

def main():
    beta_j = np.zeros(60)
    i=0
    for j in range_with_floats(0.01,3,0.05):
        beta_j[i]=j
        i+=1
        # print("jj",j)
        path = 'D:\MY_NOTES\CP_bhoosan\ising_1d_data_10Feb2023_backup\ising_1d_data\Run_01\E_beta'
        all_files = glob.glob(os.path.join(path, "*.csv"))
    li=[]
    # print(beta_j,i)
    print(all_files)
    temp=0
    print_array = [[0] * 3 for i in range(60)]
    for filename in all_files:
        beta = beta_j[temp]

        df= pd.read_csv(filename,header=None)
        li = df.to_numpy()
        # print("SEE HERE:",beta_j, li[1])
        Cv_array = np.zeros(100)
        Cv_std=np.zeros(100)
        #     # print the location and filename
        # print('Location:', filename)
        # print('File Name:', filename.split("\\")[-1])
        for i in range(100):
            li2 = np.copy(li)
            li2 = np.delete(li2,i)
            # Calculate Cv from li2
            # store this value of Cv in another array at position i
            Cv_array[i]=beta*beta*(np.mean(li2*li2)-(np.mean(li2))**2)/200 #n_spins =200
        Cv_std = np.std(Cv_array)
        Cv_mean = np.mean(Cv_array)
        print_array[temp][:] = np.array([beta,Cv_mean,Cv_std])

        temp += 1

    np.savetxt('Cv_Run1.csv', print_array, delimiter=",")
        # np.savetxt('Cv_beta_{'+str(i)+'}'+'.csv', print_array, delimiter=",")


        # Cv = beta_j * beta_j * (mean((e_j .^ 2.0)) - (mean(e_j))^2.0)/n_spins
        # χ =  beta_j * (mean((m_j .^ 2.0)) - (mean(m_j))^2.0)/n_spins

if __name__ == "__main__":
    main()