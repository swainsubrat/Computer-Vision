import numpy as N
from numpy import linalg as LA

def normalize(nd,x):

    x = N.asarray(x)
    m, s = N.mean(x,0), N.std(x)
    if nd==2:
        Tr = N.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = N.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = N.linalg.inv(Tr)
    x = N.dot( Tr, N.concatenate( (x.T, N.ones((1,x.shape[0]))) ) )
    x = x[0:nd,:].T

    return Tr, x

def caliberate(xyz, uv):
    xyz = N.asarray(xyz)
    uv = N.asarray(uv)
    np = xyz.shape[0]

    Txyz, xyzn = normalize(3, xyz)
    Tuv, uvn = normalize(2, uv)

    A = []
    for i in range(np):
        x,y,z = xyzn[i,0], xyzn[i,1], xyzn[i,2]
        u,v = uvn[i,0], uvn[i,1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v] )

    A = N.asarray(A) 
    U, S, Vh = N.linalg.svd(A)
    # The parameters are in the last line of Vh and normalize them:
    P = Vh[-1,:] / Vh[-1,-1]
    # Camera projection matrix:
    H = P.reshape(3, 4)
    # Denormalization:
    H = N.dot( N.dot( N.linalg.pinv(Tuv), H ), Txyz )
    H = H / H[-1,-1]
    P = H.flatten()
    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = N.dot( H, N.concatenate( (xyz.T, N.ones((1,xyz.shape[0]))) ) ) 
    uv2 = uv2/uv2[2,:] 
    # mean distance:
    err = N.sqrt( N.mean(N.sum( (uv2[0:2,:].T - uv)**2,1 )) )

    return P, err

if __name__ == "__main__":
    real_world_coordinates = [[2.5, 2.5, 0], [5, 2.5, 0], [5, 5, 0], [5, 0, 5], [7.5, 0, 7.5], [2.5, 0, 2.5]]
    print("Real World Coordinates(3D) in xyz")
    print(N.asarray(real_world_coordinates))
    image_coordinates = [[485,923.4], [526,927], [526,875], [448, 1007], [447,1033], [448,986]]
    print("Image Coordinates(2D) in uv")
    print(N.asarray(image_coordinates))

    P, err1 = caliberate(real_world_coordinates, image_coordinates)
    print('Camera calibration parameters(Projection Matrix P)')
    print(P)
    print('Error of the calibration')
    print(err1)

    # QR Factorization
    P = P.reshape(3, 4)
    qr = P[:,:-1]
    R, K = LA.qr(qr)
    trans = N.dot(LA.inv(K), P[:,-1])
    print(f"Rotational Matrix is: \n{R}")
    print(f"K Matrix is: \n{K}")
    print(f"Translational Vector is{trans}")
    print(f"Focal length x and Focal length y: {K[0][0]}, {K[1][1]}")
    print(f"Ox and Oy: {K[0][-1]}, {K[1][-1]}")

    # Testing
    image_coordinates = N.dot(P, N.array([2.5, 2.5, 0, 1]))
    print(image_coordinates)