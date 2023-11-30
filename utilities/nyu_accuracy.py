import sys
sys.path.insert(0, '..')

import os
import argparse
import cv2
import pickle
import numpy as np
from utils import read_pfm

# For now use hardcoded values
scale = 0.00012385
shift = 0.10803008

model_output_dir = "/nscratch/kevinand/courses/cs225b/project/MiDaS/output/"
golden_depth_dir = "/scratch/kevinand/courses/cs225b/project/datasets/nyu_v2/depths/"

# Grab files
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-num', action="store", dest="file_num", required=True);
parser.add_argument('-I', action="store", dest="file1", required=False);
parser.add_argument('-J', action="store", dest="file2", required=False);
args = parser.parse_args()

# Find file by number
file1 = model_output_dir + [filename for filename in os.listdir(model_output_dir) if filename.startswith(args.file_num)][0]
file2 = golden_depth_dir + [filename for filename in os.listdir(golden_depth_dir) if filename.startswith(args.file_num)][0]
print(file1+"\n"+file2)

# Read images
I = np.array(read_pfm(file1)[0])
J = pickle.load(open(file2, 'rb'))
I = scale*I + shift
J = 1/J
#J = J[:,:,0] 	# extract single channel

print(I.shape)
print(J.shape)


# Compute accuracy
#I = cv2.cvtColor(I[0], cv2.COLOR_BGR2GRAY)
max_cmp = np.maximum(np.divide(I,J), np.divide(J,I))
print("Max: {} \t Average: {}".format(np.max(max_cmp), np.mean(max_cmp)))
print("Error:  {}%".format(100*np.sum(max_cmp > 1.25)/I.size))
