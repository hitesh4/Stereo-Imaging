#camera calibration

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER, 30, 0.01)
objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('data\camera_calib\*.jpg')
#print (images)
for fname in images:
    img_calib = cv2.imread(fname)
    gray_calib = cv2.cvtColor(img_calib, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray_calib, (7,6), None)
    #print (ret)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray_calib, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)
        
        img_calib = cv2.drawChessboardCorners(img_calib, (7,6), corners2, ret)
        #print  ('sa')
        #.startWindowThread()
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img_calib)
        cv2.waitKey(0)
        
        
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_calib.shape[::-1], None, None)

img_undistort = cv2.imread('data\camera_calib\left12.jpg')
h, w = img_undistort.shape[:2]
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h),1,(w,h))

dst_undistort = cv2.undistort(img_undistort, mtx, dist, None, newcameramatrix)

x, y, w, h = roi
dst_undistort = dst_undistort[y:y+h, x:x+w]
cv2.imwrite('outdata/calibresult.jpg',dst_undistort)

np.savez('cameracalibmat_data/camera_mat_and_dist_coeff.npz', name1 = mtx, name2 = dist)