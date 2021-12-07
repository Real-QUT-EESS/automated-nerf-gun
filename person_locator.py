import cv2

# Initializes the HOG descriptor/person detector
def CreateHog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def DetectPeople(frame, hog):
    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    return (boxes, weights)