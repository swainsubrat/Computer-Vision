import numpy as np
from numpy import linalg as LA

def normalize(vector):
    vector = np.array(vector)
    normed = (vector - vector.mean(axis=0)) / vector.std(axis=0)
    
    return list(normed)

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

V = [[486,920],[526,927],[487,869],[526,875],[448,1007],[366,1008],[447,1033],[406,1022],[448,986]]

xw = []     #xw is X-axis coordinates array of image all points
yw = []     #yw is y-axis coordinates array of image all points
zw = []     #zw is z-axis coordinates array of image all points
u  = []     #u is u axis coordinates of all image points
v  = []     #v is v axis coordinates of all image points

image2d_vectors = [[486,920], [526,927], [526,875], [448, 1007], [447,1033], [448,986]]     # 2d image coordinates [u,v] vector of all points
# realw_vectors   = [[1, 1, 0], [2, 1, 0], [2, 2, 0], [2, 0, 2], [3, 0, 3], [1, 0, 1]]        # 3d real world coordinates [x,y,z] vector of all points
realw_vectors   = [[2.5, 2.5, 0], [5, 2.5, 0], [5, 5, 0], [5, 0, 5], [7.5, 0, 7.5], [2.5, 0, 2.5]]        # 3d real world coordinates [x,y,z] vector of all points

# Normalize the vectors
# image2d_vectors = normalize(image2d_vectors)
# realw_vectors   = normalize(realw_vectors)

# T and U vectors
# T = np.dot(np.array(image2d_vectors), LA.inv(np.array(image2d_vectors1)))
# U = np.dot(realw_vectors, LA.inv(realw_vectors1))

# print(f"T vector for normalization:\n{T}")
# print(f"U vector for normalization:\n{U}")

image3d_vectors = [v + [1] for v in image2d_vectors] # 3d image coordinates [u,v,1] vector
real4d_vectors  = [v + [1] for v in realw_vectors]   # 4d image coordinates [x,y,z,1] vector

# image3d_vectors = [[486,920,1], [526,927,1], [526,875,1], [448, 1007,1], [447,1033,1], [448,986,1]]    #3d image coordinates [u,v,1] vector
# real4d_vectors = [[1,1,0,1],[2,1,0,1], [2,2,0,1],[2,0,2,1], [3,0,3,1],[1,0,1,1]]                       #4d image coordinates [x,y,z,1] vector

# Now creating a vector of points in x axis, y axis, z axis points separately
for i in range(len(real4d_vectors)):
    xw.append(real4d_vectors[i][0])         # all x axis components
    yw.append(real4d_vectors[i][1])         # all y axis components
    zw.append(real4d_vectors[i][2])         # all z axis components
# print(xw, yw, zw)

# Convert point vectors from sleeping vector (Rows) to standing vectors (Column) of real world
xwt = np.array(xw).transpose()          
ywt = np.array(yw).transpose()
zwt = np.array(zw).transpose()

for i in range(len(image3d_vectors)):
    u.append(image3d_vectors[i][1])
    v.append(image3d_vectors[i][1])
# print(u,v)

# Convert point vectors from sleeping vector (Rows) to standing vectors (Column) of image points
ut = np.array(u).transpose()
vt = np.array(v).transpose()
Au = [[]]
Av = [[]]
for p in range(len(real4d_vectors)):
    Au.append([xwt[p], ywt[p], zwt[p], 1, 0, 0, 0, 0, -ut[p]*xwt[p], -ut[p]*ywt[p], ut[p]*zwt[p], -ut[p]])
    Av.append([0, 0, 0, 0, xwt[p], ywt[p], zwt[p], 1, -ut[p]*xwt[p], -ut[p]*ywt[p], ut[p]*zwt[p], -ut[p]])
Au.pop(0)
Av.pop(0)
A = []
for i in range(6):
    A.append(Au[i])
    A.append(Av[i])
A = np.array(A)
# print(A)

# Find the eigen vector corresponding to smallest eigen value
target_matrix   = np.dot(A, A.T)
values, vectors = LA.eig(target_matrix)
P = vectors[np.argmin(values)].reshape(3, 4)

# QR Factorization
qr = P[:,:-1]
R, K = LA.qr(qr)
trans = np.dot(LA.inv(K), P[:,-1])
print(f"Rotational Matrix is: \n{R}")
print(f"K Matrix is: \n{K}")
print(f"Translational Vector is{trans}")
print(f"Focal length x and Focal length y: {K[0][0]}, {K[1][1]}")
print(f"Ox and Oy: {K[0][-1]}, {K[1][-1]}")

# Testing
image_coordinates = np.dot(P, np.array([1, 1, 0, 1]))
print(image_coordinates)
