import cv2
import numpy as np

# crop
# Crops based on the provided parameters
# Not destructive, does not modify original file
# Arguments:
# image - NumPy array containing the image
# x - Horizontal position to start crop from top left corner
# y - Vertical position to start crop from top left corner
# w - width of the crop (horizontal)
# h - height of the crop (vertical)# Returns numpy array describing the cropped image
def crop(image, x, y, w, h):
        if h < 0:
                y = y + h
                h = -h
        if w < 0:
                x = x + w
                w = -w
        return image[y:y+h, x:x+w]

# resize
# Resizes an image based on the provided parameters
# Not destructive, does not modify original file
# Arguments:
# image - NumPy array containing the image
# imageSizePx - Final size of image in pixels. This is both height and weidth
# Returns numpy array containing the resized image
def resize(image, imageSizePx):
        return cv2.resize(image, (imageSizePx, imageSizePx))
# normalize
# Rescales all of the data to a scale of 0-1
# Arguments:
# scale - current max value of data
# returns NumPy array scaled from 0-1
def normalize(image, maxVal):
        return np.divide(image, maxVal)
