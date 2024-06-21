from __future__ import print_function
import cv2 
import matplotlib.pyplot as plt
from builtins import input
import numpy as np


cap = cv2.VideoCapture(0)



lower_green = np.array([36, 50, 70])
upper_green = np.array([89, 255, 255])
lower_red = np.array([0, 100, 100])  #[0, 100, 100]
upper_red = np.array([10, 255, 255])    #[10, 255, 255]
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])


centroid_positions = []
green_tape_positions = []
prev_touching = False

draw_line = False


while True:

    ret, frame = cap.read()


    if ret:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        mask = cv2.inRange(hsv, lower_red, upper_red)
        GreenMask= cv2.inRange(hsv, lower_green, upper_green)
        BlueMask = cv2.inRange(hsv, lower_blue, upper_blue)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        GreenMask=cv2.erode(GreenMask, None, iterations=2)
        GreenMask=cv2.dilate(GreenMask, None, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        BlueMask = cv2.morphologyEx(BlueMask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Greencontours, _ = cv2.findContours(GreenMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Bluecontours, _ = cv2.findContours(BlueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            max_contour = max(contours, key=cv2.contourArea)


            M = cv2.moments(max_contour)
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


            centroid_positions.append(centroid)


            draw_line = True
        if len(Greencontours)==2:

            rect1 = cv2.boundingRect(Greencontours[0])
            rect2 = cv2.boundingRect(Greencontours[1])


            centroid1 = (rect1[0] + rect1[2] // 2, rect1[1] + rect1[3] // 2)
            centroid2 = (rect2[0] + rect2[2] // 2, rect2[1] + rect2[3] // 2)


            green_tape_positions.append((centroid1, centroid2))


            distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))


            zoom_factor = 1.0 - distance / 500.0  # Adjust this scaling factor according to your needs
            zoomed_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)


            cv2.imshow('Zoomed Stream', zoomed_frame)
        if len(Bluecontours) == 2:
            # Get the bounding rectangles of the contours
            rect11 = cv2.boundingRect(Bluecontours[0])
            rect22 = cv2.boundingRect(Bluecontours[1])


            center11 = (rect11[0] + rect11[2] // 2, rect11[1] + rect11[3] // 2)
            center22 = (rect22[0] + rect22[2] // 2, rect22[1] + rect22[3] // 2)


            Distance = np.sqrt((center22[0] - center11[0])**2 + (center22[1] - center11[1])**2)


            Touching = Distance <= (rect11[2] + rect22[2]) / 2


            if Touching and not prev_touching:

                cv2.imwrite("RasheedHamedo.jpg", frame)
                print("Screenshot saved successfully")


            prev_touching = Touching


        if draw_line:
            for i in range(1, len(centroid_positions)):
                cv2.line(frame, centroid_positions[i-1], centroid_positions[i], (0, 0, 255), 2)


        cv2.imshow('Camera Stream', frame)


    key = cv2.waitKey(1) & 0xFF


    if key == ord('c'):
        centroid_positions = []
        draw_line = False
        green_tape_positions = []


    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
