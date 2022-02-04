from cgitb import small
import numpy as np
from numpy import linalg as LA

###### CONSTANTS ######
# XY -> XZ -> YZ
# XY-Plane
cyan_blue = (486, 920)
r_brown   = (526, 927)
# l_redi    = (487, 869)
dar_green = (526,875)

# XZ-Plane
blue      = (448, 1007)
# purple    = (366, 1008)
brown     = (447, 1033)
# pink      = (406, 1022)
l_green   = (448, 986)

V = [[486,920],[526,927],[487,869],[526,875],[448, 1007],[366,1008],[447,1033],[406,1022],[448,986]]


image_vectors = [[486,920], [526,927], [526,875], [448, 1007], [447,1033], [448,986]]
realw_vectors = [[1, 1, 0], [2, 1, 0], [2, 2, 0], [2, 0, 2], [3, 0, 3], [1, 0, 1]]



# A = np.diag((1, 2, 3, 4))

target_matrix = np.dot(A, A.T)
values, vectors = LA.eig(target_matrix)

print(values, vectors)

smallest = vectors[0]
print(smallest.shape)
# smallest = smallest.reshape(smallest.shape[0])
# R, K = LA.qr(vectors[0])