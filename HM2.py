from __future__ import print_function
import cv2 
import matplotlib.pyplot as plt
from builtins import input
import numpy as np


# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0)



lower_green = np.array([36, 50, 70])
upper_green = np.array([89, 255, 255])
lower_red = np.array([0, 100, 100])  #[0, 100, 100]
upper_red = np.array([10, 255, 255])    #[10, 255, 255]
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])

# Initialize the list to store centroid positions
centroid_positions = []
green_tape_positions = []
prev_touching = False
# Flag to indicate whether to draw the line or not
draw_line = False

# Read and process frames from the camera until 'q' is pressed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if ret:
        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the frame to detect the red object
        mask = cv2.inRange(hsv, lower_red, upper_red)
        GreenMask= cv2.inRange(hsv, lower_green, upper_green)
        BlueMask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Perform morphological operations to remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        GreenMask=cv2.erode(GreenMask, None, iterations=2)
        GreenMask=cv2.dilate(GreenMask, None, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        BlueMask = cv2.morphologyEx(BlueMask, cv2.MORPH_OPEN, kernel)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Greencontours, _ = cv2.findContours(GreenMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Bluecontours, _ = cv2.findContours(BlueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if any contours are found
        if len(contours) > 0:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)

            # Compute the centroid of the contour
            M = cv2.moments(max_contour)
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Store the centroid position
            centroid_positions.append(centroid)

            # Enable drawing the line
            draw_line = True
        if len(Greencontours)==2:
            # Get the bounding rectangles of the contours
            rect1 = cv2.boundingRect(Greencontours[0])
            rect2 = cv2.boundingRect(Greencontours[1])

            # Calculate the centroid positions of the green tape
            centroid1 = (rect1[0] + rect1[2] // 2, rect1[1] + rect1[3] // 2)
            centroid2 = (rect2[0] + rect2[2] // 2, rect2[1] + rect2[3] // 2)

            # Store the centroid positions of the green tape
            green_tape_positions.append((centroid1, centroid2))

            # Calculate the distance between the centroids
            distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))

            # Adjust the zoom level based on the distance
            zoom_factor = 1.0 - distance / 500.0  # Adjust this scaling factor according to your needs
            zoomed_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)

            # Display the zoomed frame
            cv2.imshow('Zoomed Stream', zoomed_frame)
        if len(Bluecontours) == 2:
            # Get the bounding rectangles of the contours
            rect11 = cv2.boundingRect(Bluecontours[0])
            rect22 = cv2.boundingRect(Bluecontours[1])

            # Calculate the center points of the blue objects
            center11 = (rect11[0] + rect11[2] // 2, rect11[1] + rect11[3] // 2)
            center22 = (rect22[0] + rect22[2] // 2, rect22[1] + rect22[3] // 2)

            # Calculate the distance between the centers
            Distance = np.sqrt((center22[0] - center11[0])**2 + (center22[1] - center11[1])**2)

            # Check if the blue objects are touching
            Touching = Distance <= (rect11[2] + rect22[2]) / 2

            # Check if the blue objects started touching in this frame
            if Touching and not prev_touching:
                # Capture and save the screenshot
                cv2.imwrite("RasheedHamedo.jpg", frame)
                print("Screenshot saved successfully")

            # Update the previous touch state
            prev_touching = Touching

        # Draw the line using the stored centroid positions
        if draw_line:
            for i in range(1, len(centroid_positions)):
                cv2.line(frame, centroid_positions[i-1], centroid_positions[i], (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Camera Stream', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Clear the line and centroid positions when 'c' is pressed
    if key == ord('c'):
        centroid_positions = []
        draw_line = False
        green_tape_positions = []

    # Break the loop when 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
