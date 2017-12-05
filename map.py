import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')
cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(0)

cv2.namedWindow("cam1")
cv2.namedWindow("cam2")

p = 0
while True:
    p+=1
    ret1, frame1 = cam1.read()
    cv2.imshow("cam1", frame1)
    ret2, frame2 = cam2.read()
    cv2.imshow("cam2", frame2)
    if not ret1:
        break
    k1 = cv2.waitKey(1)
    if not ret2:
        break
    k2 = cv2.waitKey(1)
    
    #left = cv2.imread("data/testing/left_0.png")
    #right = cv2.imread("data/testing/right_0.png")
    right = frame1
    left = frame2
    right= cv2.resize(right, (left.shape[1],left.shape[0],), interpolation = cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    data = np.load('cameracalibmat_data/stereo_camera_mat_and_dist_coeff_diff_camera1.npz')
#    K1 =  [[9.032949e+02, 0.000000e+00, 6.639935e+02],[ 0.000000e+00, 9.079042e+02, 2.452070e+02],[ 0.000000e+00, 0.000000e+00, 1.000000e+00]]
#    D1 =  [[-3.778799e-01, 1.824904e-01, 1.390637e-03, 4.659340e-05, -4.730213e-02]]
#    K2 = [[9.050234e+02, 0.000000e+00, 6.846818e+02],[ 0.000000e+00, 9.102276e+02, 2.457892e+02],[ 0.000000e+00, 0.000000e+00, 1.000000e+00]]
#    D2 = [[-3.810763e-01, 1.851087e-01, 1.802898e-03, -1.702626e-04, -4.624721e-02]]
#    R = [[9.999369e-01, -5.437215e-04, 1.122319e-02],[ 5.966196e-04, 9.999887e-01, -4.710484e-03],[ -1.122050e-02, 4.716883e-03, 9.999259e-01]]
#    T = [[-5.724638e-01], [6.739395e-03], [1.262054e-02]]
#    K1 = np.asarray(K1)
#    K2 = np.asarray(K2)
#    D1 = np.asarray(D1)
#    D2 = np.asarray(D2)
#    R = np.asarray(R)
#    T = np.asarray(T)
    
    
    #image rectification(try after calibration)
    rectify_scale = 0
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(data['cameraMatrix1'], data['distCoeffs1'], data['cameraMatrix2'], data['distCoeffs2'], (left.shape[1],left.shape[0]), data['R'], data['T'], alpha = rectify_scale)
    left_mapsx,left_mapsy = cv2.initUndistortRectifyMap(data['cameraMatrix1'], data['distCoeffs1'], R1, P1, (left.shape[1],left.shape[0]), cv2.CV_16SC2)
    right_mapsx,right_mapsy = cv2.initUndistortRectifyMap(data['cameraMatrix1'], data['distCoeffs1'], R2, P2, (left.shape[1],left.shape[0]), cv2.CV_16SC2)
    left_img_remap = cv2.remap(left_gray, left_mapsx, left_mapsy, cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(right_gray, right_mapsx, right_mapsy, cv2.INTER_LANCZOS4)
    #complete
    
    def nothing(x):
        pass
    
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 1000,1000)
    cv2.createTrackbar('win_size','image',0,5,nothing)
    #cv2.createTrackbar('min_disp','image',0,0,nothing)
    cv2.createTrackbar('max_disp1','image',0,12,nothing)
    cv2.createTrackbar('uniquenessRatio','image',5,15,nothing)
    cv2.createTrackbar('speckleWindowSize','image',0,200,nothing)
    cv2.createTrackbar('speckleRange','image',1,2,nothing)
    cv2.createTrackbar('disp12MaxDiff','image',0,200,nothing)
    
    disparity = left_img_remap
    #BM
    #while(1):
    #    cv2.imshow("image",disparity)
    #    win_size = cv2.getTrackbarPos('win_size','image')
    #    min_disp = 0
    #    max_disp = 16*cv2.getTrackbarPos('max_disp1','image')
    #    num_disp = max_disp - min_disp 
    #    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    #            numDisparities = num_disp,
    #            uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','image'),
    #            speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','image'),
    #            speckleRange = cv2.getTrackbarPos('speckleRange','image'),
    #            disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','image'),
    #            P1 = 8*3*win_size**2,
    #            P2 = 32*3*win_size**2
    #        )
    #    disparity = stereo.compute(left_img_remap,right_img_remap).astype(np.float32)
    #    min = disparity.min()
    #    max = disparity.max()
    #    disparity = np.uint8(255 * (disparity - min) / (max - min))
    while(1):
        cv2.imshow("image",disparity)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break   
        max_disp = 16*9
        win_size = cv2.getTrackbarPos('win_size','image')
        min_disp = -16*cv2.getTrackbarPos('max_disp1','image')
        
        num_disp = max_disp - min_disp 
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','image'),
                speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','image'),
                speckleRange = cv2.getTrackbarPos('speckleRange','image'),
                disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','image'),
                P1 = 8*3*win_size**2,
                P2 = 32*3*win_size**2
            )
        disparity = stereo.compute(left_img_remap,right_img_remap).astype(np.float32)
        min = disparity.min()
        max = disparity.max()
        disparity = np.uint8(255 * (disparity - min) / (max - min))
            
    cv2.destroyAllWindows()    
    w = left.shape[1]
    h = left.shape[0]
    #focal_length = data["cameraMatrix1"][0][0]
    focal_length = 0.8*w
    #Q = np.float32([[1, 0, 0, -w/2.0],
    #                    [0,-1, 0,  h/2.0],
    #                    [0, 0, 0, -focal_length],
    #                    [0, 0, 1, 0]])
    
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    mask_map = disparity > disparity.min()
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    print ("\nCreating the output file ...\n")
    colors = output_colors.reshape(-1, 3)
    vertices = np.hstack([output_points.reshape(-1,3), colors])
    #create_output(output_points, output_colors, "optput.ply")
    
    s = []
    q=0
    for i in range(10):
        s.append("0")
    for i in range(len(vertices)):
        if np.isfinite(vertices[i]).all():
            q+=1
    f=open("results/testtrue"+str(p)+".ply","w+")
    a=open("a.csv","w+")
    X=[1,2,3,4]
    Y=[5,3,9,6]
    Z=[8,6,9,4]
    s[0]="ply"
    s[1]="format ascii 1.0"
    s[2]="element vertex "+ str(q)
    s[3]="property float x"
    s[4]="property float y"
    s[5]="property float z"
    s[6]="property uchar red"
    s[7]="property uchar green"
    s[8]="property uchar blue"
    s[9]="end_header"
    print("testtrue"+str(p)+".ply")
    for i in range(len(s)):
        f.write(s[i]+"\n")
    x= []
    y = []
    z = []
    
    for i in range(len(vertices)):
        if np.isfinite(vertices[i]).all():
            x.append(vertices[i][0])
            y.append(vertices[i][1])
            z.append(vertices[i][2])
            f.write(str(vertices[i][0])+" ")
            f.write(str(vertices[i][1])+" ")
            f.write(str(vertices[i][2])+" ")
            f.write(str(int(vertices[i][3]))+" ")
            f.write(str(int(vertices[i][4]))+" ")
            f.write(str(int(vertices[i][5]))+"\n")
            a.write(str(vertices[i][0])+",")
            a.write(str(vertices[i][1])+",")
            a.write(str(vertices[i][2])+",")
            a.write(str(int(vertices[i][3]))+",")
            a.write(str(int(vertices[i][4]))+",")
            a.write(str(int(vertices[i][5]))+"\n")
    
    f.close()
    a.close()





