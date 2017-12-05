import cv2

cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(0)

cv2.namedWindow("cam1")
cv2.namedWindow("cam2")

img_counter = 0

while True:
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

    if k1%256 == 27 or k2%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k1%256 == 32 or k2%256 == 32:
        # SPACE pressed
        img_name1 = "data/testing/right_{}.png".format(img_counter)
        img_name2 = "data/testing/left_{}.png".format(img_counter)
        cv2.imwrite(img_name1, frame1)
        cv2.imwrite(img_name2, frame2)
        print("{} written!".format(img_name1))
        print("{} written!".format(img_name2))
        img_counter += 1

cam1.release()
cam2.release()
cv2.destroyAllWindows()

