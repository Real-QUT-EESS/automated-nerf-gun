import cv2 # Video Capturing / Image Processing
import numpy as np # Array manipulation

from person_locator import * # Person detection


# Begin Video Capturing
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Use the hog model to detect people in the frames
    (boxes, weights) = DetectPeople(frame, model)

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()