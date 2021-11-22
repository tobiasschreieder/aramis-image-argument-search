from cv2 import cv2
from deskew import determine_skew
from skimage.transform import rotate
import pytesseract
import re
import numpy as np


pytesseract.pytesseract.tesseract_cmd = 'tesseract/tesseract.exe'


def read_image(path):
    img = cv2.imread(str("images/" + path))
    return img


def clean_text(text):
    text = re.sub('[^A-Za-z0-9" "]+', ' ', text)

    words = text.split(' ')
    correct_words = ""
    for word in words:
        if len(word) > 2:
            correct_words += " " + word

    return correct_words


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# erosion
def erode_dilate(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    return image


# skew correction
def deskew(image):
    grayscale = get_grayscale(image)
    angle = determine_skew(thresholding(grayscale))
    rotated = rotate(grayscale, angle, resize=True) * 255
    return rotated.astype(np.uint8)


def text_from_image(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed_image = erode_dilate(get_grayscale(image))

    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 11')
    text = clean_text(text)
    print("Text detected: ", text)

    return text


def textposition_from_image(image, plot=False):
    h, w, c = image.shape
    boxes = pytesseract.image_to_boxes(image)
    if plot:
        for b in boxes.splitlines():
            b = b.split(' ')
            image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

        cv2.imshow('Result', image)
        cv2.waitKey(0)

    return boxes


def shapes_from_image(image, plot=False):
    h_image, w_image, _ = image.shape

    # converting image into grayscale image
    gray = get_grayscale(image)

    # setting threshold of gray image
    threshold = thresholding(gray)

    # using a findContours() function
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    i = 0

    # list for storing names of shapes
    for contour in contours:

        area = cv2.contourArea(contour)

        if area > ((h_image*w_image)*0.001):

            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue

            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)

            if plot:
                cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)

            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])

            if len(approx) == 3:
                shapes.append(('Triangle', x, y))

            elif len(approx) == 4:
                shapes.append(('Square', x, y))

            elif len(approx) == 5:
                shapes.append(('Pentagon', x, y))

            elif len(approx) == 6:
                shapes.append(('Hexagon', x, y))

            else:
                shapes.append(('Circle', x, y))

    if plot:
        cv2.imshow('shapes', image)
        cv2.waitKey(0)

    return shapes


def diagramms_from_image(image, plot=False):
    h_image, w_image, _ = image.shape

    gray = get_grayscale(image)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilate with horizontal kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    dilate = cv2.dilate(threshold, kernel, iterations=2)

    # Find contours and remove non-diagram contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if plot:
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if w / h > 2 and area > ((h_image*w_image)*0.01):
                cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    # Iterate through diagram contours and form single bounding box
    boxes = []
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    try:
        boxes = np.asarray(boxes)
        x = np.min(boxes[:, 0])
        y = np.min(boxes[:, 1])
        w = np.max(boxes[:, 2]) - x
        h = np.max(boxes[:, 3]) - y

        roi_area = (w*h) / (w_image*h_image)

        # use dichtefunktion in future
        if not roi_area < 0.8:
            roi_area = 0
    except:
        roi_area = 0

    if plot:
        ROI = image[y:y + h, x:x + w]
        cv2.imshow('ROI', ROI)
        cv2.waitKey()

    return roi_area
