#camera calibration

import numpy as np
import cv2
import glob


images_left = glob.glob('real\*.jpg')
images_right = images_left[13:len(images_left)]
images_left = images_left[0:13]

img_left_points = []
img_right_points = []
obj_points = []

pattern_points = np.zeros((6*7,3),np.float32)
pattern_points[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)



for x in range(0,len(images_left)):
    print ("a")
    left_img = cv2.imread(images_left[x])
    right_img = cv2.imread(images_right[x])
    #right_img = cv2.resize(right_img, (left_img.shape[0],left_img.shape[1],), interpolation = cv2.INTER_AREA)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    #ret,left_img = cv2.threshold(left_img,200,255,cv2.THRESH_BINARY)
    #ret,right_img = cv2.threshold(right_img,200,255,cv2.THRESH_BINARY)
    image_size = left_img.shape
 
    find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
 
    left_found, left_corners = cv2.findChessboardCorners(left_img, (7,6), flags = find_chessboard_flags)
    right_found, right_corners = cv2.findChessboardCorners(right_img, (7,6), flags = find_chessboard_flags)
 
    if left_found:
        cv2.cornerSubPix(left_img, left_corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    if right_found:
        cv2.cornerSubPix(right_img, right_corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
 
    if left_found and right_found:
        img_left_points.append(left_corners)
        img_right_points.append(right_corners)
        obj_points.append(pattern_points)
 
    cv2.imshow("left", left_img)
    cv2.drawChessboardCorners(left_img,  (7,6), left_corners, left_found)
    cv2.drawChessboardCorners(right_img,  (7,6), right_corners, right_found)
 
    cv2.imshow("left chess", left_img)
    cv2.imshow("right chess", right_img)
 
cameraMatrix1 =None
cameraMatrix2 = None
distCoeffs1 = None
distCoeffs2 = None
R =None
T = None
E = None
F = None
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH 
stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points,img_left_points,img_right_points,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,image_size,criteria = stereocalib_criteria, flags = stereocalib_flags)

np.savez('cameracalibmat_data/stereo_camera_mat_and_dist_coeff.npz', cameraMatrix1 = cameraMatrix1, distCoeffs1 = distCoeffs1,cameraMatrix2 = cameraMatrix2, distCoeffs2 = distCoeffs2, R = R, T = T, E = E, F = F)