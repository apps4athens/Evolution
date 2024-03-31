# importing libraries
import cv2
import numpy as np


# capturing video camera
camera = cv2.VideoCapture(0)

# the process will continue until an interruption occurs
while camera.isOpened():

    # reading camera
    success, frame = camera.read()

    # checks if camera works
    if not success:
       
        print("Error! Camera could not be opened.")
        break

    # processing frame
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # defining color yellow
    lower_y = np.array([18, 94, 140])
    upper_y = np.array([48, 255, 255])

    # detecting lines with yellow color
    mask = cv2.inRange(hsv, lower_y, upper_y)
    edges = cv2.Canny(mask, 74, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    # working if there are any lines in frame
    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]
            
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # displaying the original frame and the Canny frame
    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    # stopping the process with letter "q":quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        
        break

# releasing all frames
cv2.destroyAllWindows()
camera.release()




"""
import cv2
import numpy as np
from gpiozero import Motor

# Define GPIO pins for motor control
left_motor = Motor(forward=17, backward=18)
right_motor = Motor(forward=22, backward=23)

# capturing video camera
camera = cv2.VideoCapture(0)

# The speed at which the motors should turn (adjust as needed)
motor_speed = 0.5

# The width of the frame
frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

# the process will continue until an interruption occurs
while camera.isOpened():

    # reading camera
    success, frame = camera.read()

    # checks if camera works
    if not success:
        print("Error! Camera could not be opened.")
        break

    # processing frame
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # defining color yellow
    lower_y = np.array([18, 94, 140])
    upper_y = np.array([48, 255, 255])

    # detecting lines with yellow color
    mask = cv2.inRange(hsv, lower_y, upper_y)
    edges = cv2.Canny(mask, 74, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    # working if there are any lines in frame
    if lines is not None:
        # Initialize variables to store average x-coordinate of the detected lines
        total_x = 0
        num_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            total_x += (x1 + x2) / 2  # Calculate the average x-coordinate of the line
            num_lines += 1

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Calculate the average x-coordinate of all detected lines
        if num_lines > 0:
            avg_x = total_x / num_lines
            # Normalize the average x-coordinate to a value between -1 and 1
            normalized_x = (avg_x - frame_width / 2) / (frame_width / 2)

            # Adjust motor speeds based on the normalized x-coordinate
            left_speed = motor_speed + normalized_x
            right_speed = motor_speed - normalized_x

            # Ensure motor speeds are within valid range [-1, 1]
            left_speed = max(min(left_speed, 1), -1)
            right_speed = max(min(right_speed, 1), -1)

            # Set motor speeds
            left_motor.value = left_speed
            right_motor.value = right_speed

    # displaying the original frame and the Canny frame
    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    # stopping the process with letter "q":quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# releasing all frames
cv2.destroyAllWindows()
camera.release()
"""




""" 
import cv2
import numpy as np
from gpiozero import Motor
from time import sleep

# Initialize GPIO pins for motor control
left_motor = Motor(forward=17, backward=18)
right_motor = Motor(forward=22, backward=23)

# Function to control motors based on line position
def follow_line(frame, motor_speed=0.5):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for yellow color
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, stop the motors
    if not contours:
        left_motor.stop()
        right_motor.stop()
        return

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the center of the bounding rectangle
    center_x = x + w // 2

    # Determine the direction and speed of motors based on the center of the line
    if center_x < frame.shape[1] // 2:
        # Line is on the left side, turn right
        left_motor.forward(speed=motor_speed)
        right_motor.backward(speed=motor_speed)
    else:
        # Line is on the right side, turn left
        left_motor.backward(speed=motor_speed)
        right_motor.forward(speed=motor_speed)

# Initialize the video camera
camera = cv2.VideoCapture(0)

# Loop to capture frames and follow the yellow line
while camera.isOpened():
    # Read a frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Error reading frame from camera.")
        break

    # Call the follow_line function to control the motors
    follow_line(frame)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for key press and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
"""