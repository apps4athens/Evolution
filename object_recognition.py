# importing the libraries
import cv2 
from ultralytics import YOLO


# loading the YOLOv8 model
model = YOLO('yolov8n.pt')

# opening the default camera
camera = cv2.VideoCapture(0)

# looping through the camera frames
while camera.isOpened():
    
    # reading a frame from the camera
    success, frame = camera.read()

    # checking if camera works
    if not success:

        print("Error! Camera did not opened.")
        break

    # running YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)

    # storing the results from the frame
    stored_results = results[0].plot()

    # displaying the results on the final frame
    cv2.imshow("Object Recognition", stored_results)

    # stopping the program by pressing letter/key q:quit 
    if cv2.waitKey(1) & 0xFF == ord("q"):
       
        break

# clossing all opened windows
camera.release()
cv2.destroyAllWindows()