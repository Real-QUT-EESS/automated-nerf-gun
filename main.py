import cv2 # Video Capturing / Image Processing
import numpy as np # Array manipulation

from person_locator import * # Person detection
from CoordinateConversion import * # Pixel Conversions

# Put the video capturing in a seperate thread
cv2.startWindowThread()

# Begin Video Capturing
cap = cv2.VideoCapture(0)

screen_width = 640
screen_height = 480

# Check for successful video access
if not (cap.isOpened()):
    print("Could not open video device")
    exit(1)  # Exit the program with error code

# Create our HOG model
model = CreateHog()


# Main Loop
while True:
    # Read a new image from video capture
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (screen_width, screen_height))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Do facial detection
    for location in DetectFace(gray):
            # Draw a rectangle around this face
            cv2.rectangle(frame, (location[3], location[0]), (location[1], location[2]), (0, 0, 255), 4)

            # Isolate 
            top = location[0]
            right = location[1]
            bottom = location[2]
            left = location[3]

            obj_height = bottom - top

            cent_x = (right + left)/2
            cent_y = (bottom + top)/2

            (X, Y, Z) = getWorldCoordinate(screen_width, screen_height, obj_height, cent_x, cent_y)

            print("Z: " + str(round(Z, 3)) + ", X: " + str(round(X, 3)))
    
    # Do body detection
    # Use the hog model to detect people in the frames
    # boxes = DetectPeople(gray, model)

    # # Convert boxes to array
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # for (xA, yA, xB, yB) in boxes:
    #     # display the detected boxes in the colour picture
    #     cv2.rectangle(frame, (xA, yA), (xB, yB),
    #                     (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
