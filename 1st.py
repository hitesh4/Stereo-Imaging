import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

allimg = glob.glob('data/test/new/*.jpg')
number_fo_images = len(allimg)
image_data = []
points_data = []
for im in allimg:
    image_data.append(cv2.imread(im))
for x in range(0, number_fo_images):
    for y in range(x+1,number_fo_images):
        print ('a')
        img1 = image_data[x]
        img2 = image_data[y]
        
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k =2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        
        #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        #plt.imshow(img3),plt.show()
        
        MIN_MATCH_COUNT = 10
        
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            new_dict = {
                    "img1": allimg[x],
                    "img2": allimg[y],
                    "src_pts":src_pts,
                    "dst_pts":dst_pts,
                    "Homography_matrix":M,
                    "matchesMask":matchesMask
                    }
            points_data.append(new_dict)
            h,w,_ = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
        
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
            
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        plt.imshow(img3, 'gray'),plt.show()
        cv2.waitKey(0)
        
        
        
        
        #data = np.load('cameracalibmat_data/camera_mat_and_dist_coeff.npz')
        #print (data['name1'])
        #print (data['name2'])
        
        

















