import numpy as np
import cv2
import glob
import yaml
import os

corner_x=7 # number of chessboard corner in x direction
corner_y=7 # number of chessboard corner in y direction

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corner_y*corner_x,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from a ll the images.
objpoints = [] # 3d point in real world space
jpegpoints = [] # 2d points in image plane.

source_path = "./Img_Nokia3.1Plus" #Path location of Folder containing Images
print('image found :',len(os.listdir(source_path)))

images = glob.glob('./Img_Nokia3.1Plus/*.jpg')    #*.jpg for all images in that folder
found = 0           #initialised 0 for first image in images
for fname in images: # here, 10 can be changed to whatever number you like to choose
    jpeg = cv2.imread(fname) # capture frame by frame
    cv2.imshow('jpeg', jpeg)
    cv2.waitKey(500)
    print(fname)
    gray = cv2.cvtColor(jpeg, cv2.COLOR_BGRA2GRAY)
    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
    # if found, assign object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp) #Certainly, every loop objp is the same in 3D
        corners2 = cv2.cornerSubPix(gray,corners,(20,5),(-1,-5),criteria)
        jpegpoints.append(corners2)
        # Draw and display the corners
        jpeg = cv2.drawChessboardCorners(jpeg, (corner_x,corner_y), corners2, ret)
        found += 1                 #Increase by 1 for next image in images
        cv2.imshow('chessboard', jpeg)
        cv2.waitKey(0) 
        
print("number of images used for calibration: ", found)

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, jpegpoints, gray.shape[::-1], None, None)

# transforms the matrix distortion coefficients to writeable lists
data= {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist(), 'rotational_matrix':np.asarray(rvecs).tolist(), 'translation_matrix':np.asarray(tvecs).tolist()}
print("The intrinsic matrix")
print(mtx)
print("Distortion")
print(dist)
print("Rotational Matrix")
print(rvecs)
print("Translation Matrix")
print(tvecs)
# and save it to a .yaml file
with open("calibration_matrix_nokia.yaml", "w")as f:
    yaml.dump(data, f)

#undistort image

for fname in images: 
     #print(fname)
     jpeg = cv2.imread(fname) # Capture frame-by-frame
     #cv2.imshow('jpeg', jpeg)
     #cv2.waitkey(500)
     h, w = jpeg.shape[:2]
     
     newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist , (w,h), 1, (w,h))
     #print(newcameramtx)
     #undistort
     dst = cv2.undistort(jpeg, mtx, dist, None, newcameramtx)
     cv2.imshow('undistort', dst)
     cv2.waitKey(500)
     
     # crop the image
     x, y, w, h = roi
     dst = dst[y:y+h, x:x+w]
     #cv2.imshow('calibration.png',dst)
     cv2.imshow('undistort2', dst)
     cv2.waitKey(0)
     
cv2.destroyAllWindows() 
