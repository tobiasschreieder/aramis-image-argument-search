import math
import re
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from cv2 import cv2
from deskew import determine_skew
from skimage.transform import rotate
from sklearn.cluster import KMeans

import indexing.feature.sentiment_detection as sentiment_detection

pytesseract.pytesseract.tesseract_cmd = 'properties/tesseract/tesseract.exe'


def read_image(path):
    img = cv2.imread(str(path))
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

        if area > ((h_image * w_image) * 0.001):

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h_image / 100), int(w_image / 100)))
    dilate = cv2.dilate(threshold, kernel, iterations=2)

    # Find contours and remove non-diagram contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if plot:
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if w / h > 2 and area > ((h_image * w_image) * 0.01):
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

        roi_area = (w * h) / (w_image * h_image)

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


class ImageType(Enum):
    PHOTO = 0
    CLIPART = 1


def detect_image_type(image, plot=False) -> ImageType:
    w, h, _ = image.shape

    if plot:
        figure, axis = plt.subplots(2)
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        axis[0].plot(histr)
        axis[1].hist(image.ravel(), 256, [0, 256])
        plt.show()

    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    most_used_colors = list(zip(list(count.tolist()), list(colors.tolist())))
    used_area = sum(x[0] for x in sorted(most_used_colors, key=lambda x: x[0], reverse=True)[:10]) / float((w * h))

    if used_area < 0.3:
        image_type = ImageType.PHOTO
    else:
        image_type = ImageType.CLIPART

    return image_type


def color_mood(image, image_type='clipArt', plot=False):
    # image_type ('clipart', 'photo')

    average = image.mean(axis=0).mean(axis=0)

    distance_to_green = math.sqrt((average[0] - 0) ** 2 + (average[1] - 255) ** 2 + (average[2] - 0) ** 2)
    distance_to_red = math.sqrt((average[0] - 255) ** 2 + (average[1] - 0) ** 2 + (average[2] - 0) ** 2)
    distance_to_black = math.sqrt((average[0] - 0) ** 2 + (average[1] - 0) ** 2 + (average[2] - 0) ** 2)
    distance_to_white = math.sqrt((average[0] - 255) ** 2 + (average[1] - 255) ** 2 + (average[2] - 255) ** 2)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (40, 20, 80), (70, 255, 255))
    mask_red_1 = cv2.inRange(hsv, (0, 150, 80), (15, 255, 255))
    mask_red_2 = cv2.inRange(hsv, (150, 150, 80), (255, 255, 255))
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    mask_bright = cv2.inRange(hsv, (0, 0, 200), (255, 80, 255))
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (255, 255, 100))

    number_pixels = hsv.size / 3
    percentage_green = (cv2.countNonZero(mask_green) / number_pixels) * 100
    percentage_red = (cv2.countNonZero(mask_red) / number_pixels) * 100
    percentage_bright = (cv2.countNonZero(mask_bright) / number_pixels) * 100
    percentage_dark = (cv2.countNonZero(mask_dark) / number_pixels) * 100

    if plot:
        '''
        cv2.imshow(str('Green = ' + str(percentage_green)), mask_green)
        cv2.imshow(str('Red = ' + str(percentage_red)), mask_red)
        cv2.imshow(str('Bright = ' + str(percentage_bright)), mask_bright)
        cv2.imshow(str('Dark = ' + str(percentage_dark)), mask_dark)
        cv2.waitKey()
        '''

        print("percentage_green: ", percentage_green, "  // (100/distance_to_green) = ", (100 / distance_to_green),
              "  // product = ", (percentage_green * (100 / distance_to_green)))
        print("percentage_red: ", percentage_red, "  // (100/distance_to_red) = ", (100 / distance_to_red),
              "  // product = ", (percentage_red * (100 / distance_to_red)))
        print("percentage_bright: ", percentage_bright, "  // (100/distance_to_white) = ", (100 / distance_to_white),
              "  // product = ", (percentage_bright * (100 / distance_to_white)))
        print("percentage_dark: ", percentage_dark, "  // (100/distance_to_black) = ", (100 / distance_to_black),
              "  // product = ", (percentage_dark * (100 / distance_to_black)))

    color_mood = {
        "percentage_green": percentage_green,
        "percentage_red": percentage_red,
        "percentage_bright": percentage_bright,
        "percentage_dark": percentage_dark,
        "average_color": average
    }

    # use following code for calculation in runtime
    '''
    if image_type == 'clipart':
        color_mood = (percentage_green * (100 / distance_to_green)) - (percentage_red * (100 / distance_to_red))
    elif image_type == 'photo':
        hue_factor = 0.2
        color_mood = ((percentage_green * (100 / distance_to_green)) - (percentage_red * (100 / distance_to_red))) + \
                     hue_factor * ((percentage_bright * (100 / distance_to_white)) - (
                    percentage_dark * (100 / distance_to_black)))
    '''

    return color_mood


def text_analysis(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed_image = erode_dilate(get_grayscale(image))

    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 11')
    text = clean_text(text)

    text_len = len(text.split(" "))
    text_sentiment_score = sentiment_detection.sentiment_nltk(text)

    text_analysis = {
        "text_len": text_len,
        "text_sentiment_score": text_sentiment_score
    }
    return text_analysis


def dominant_colors(image):
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    height, width, _ = np.shape(image)

    # reshape the image to be a simple list of RGB pixels
    image = image.reshape((height * width, 3))

    # we'll pick the 5 most common colors
    num_clusters = 5
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)

    # count the dominant colors and put them in "buckets"
    numLabels = np.arange(0, len(np.unique(clusters.labels_)) + 1)
    hist, _ = np.histogram(clusters.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    histogram = hist
    # then sort them, most-common first
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # finally, we'll output a graphic showing the colors in order
    bars = []
    hsv_values = []
    for index, rows in enumerate(combined):

        height, width, color = 100, 100, rows[1]
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_bar[0][0]
        rgb = (red, green, blue)
        hsv = (hue, sat, val)

        print(f'Bar {index + 1}')
        print(f'  RGB values: {rgb}')
        print(f'  HSV values: {hsv}')
        hsv_values.append(hsv)
        bars.append(bar)

    cv2.imshow(f'{num_clusters} Most Common Colors', np.hstack(bars))
    cv2.waitKey(0)
