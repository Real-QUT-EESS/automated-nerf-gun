
from math import tan, pi

# Raspberry Pi Cam V1
pixel_size = 1.4e-6
focal_length = 3.6e-3
cmosHeight = 2.74e-3
cmosWidth = 3.76e-3
HAoV = 55.3

# Comparison values
average_person_height = 1.65
average_face_height = 0.218


# Gets world coordinates of a face
def getWorldCoordinate(screen_width, screen_height, obj_height, x, y):
	Z = (average_face_height * focal_length) / ((obj_height/screen_height) * cmosHeight)

	Xt = (cmosWidth * Z) / (2 * focal_length)
	X = ((x - (screen_width/2)) / screen_width) * Xt


	Yt = (cmosHeight * Z) / (2 * focal_length)
	Y = ((y - (screen_height/2)) / screen_height) * Yt


	return (X, Y, Z)
