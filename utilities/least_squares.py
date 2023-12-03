import sys
sys.path.insert(0, '..')

import pickle
import numpy as np
from cv2 import imread
from utils import read_pfm

# Read inages
M = read_pfm("/nscratch/kevinand/courses/cs225b/project/MiDaS/output/1340_dining_room_0001b-r-1315423754-130750-1615065218-dpt_swin2_large_384.pfm")
G = pickle.load(open("/scratch/kevinand/courses/cs225b/project/datasets/nyu_v2/depths/1340_dining_room_0001b-d-1315423754-123188-1613939685.pk",'rb'))

# Extract images
MM = M[0].flatten()  # For some reason returns a tuple
GG = G.flatten()
X = np.vstack((MM, np.ones(GG.shape))).T 	# Form x matrix
Y=1/GG	# Invert depth (paper operates in inverse depth space)
B = np.linalg.inv(X.T@X)@X.T@Y # compute least squares

print("scale = {:.8f}\nshift = {:.8f}".format(B[0], B[1]))
