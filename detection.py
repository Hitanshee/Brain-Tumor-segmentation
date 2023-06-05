import cv2
import numpy as np

def is_tumor_detected(segmented_image):
    # Convert the segmented image to grayscale
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tumor_percentage = (cv2.countNonZero(thresholded_image) / (thresholded_image.shape[0] * thresholded_image.shape[1])) * 100

    # Check if tumor percentage exceeds a certain threshold (adjust this value as per your needs)
    tumor_detected = tumor_percentage > 5.0
    return tumor_detected