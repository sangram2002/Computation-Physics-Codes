import numpy as np
import pandas as pd
import glob
import os,sys
from PIL import Image

A = np.zeros([243,320])
B = np.zeros([77760,165])
folder = "D:\MY_NOTES\CP_bhoosan\yale_face"
i=0
for filename in os.listdir(folder):

    infilename = os.path.join(folder, filename)
    image = Image.open(infilename).convert('L')
    # Convert image to numpy array
    A = np.array(image)
    # print(A.shape)
    # print(filename)
    print(A.flatten())
    # B[][i] = A.flatten()

    i += 1