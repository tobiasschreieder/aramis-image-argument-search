import re
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from cv2 import cv2
from deskew import determine_skew
from pytesseract import Output
from skimage.transform import rotate

import indexing.feature.sentiment_detection as sentiment_detection
from config import Config

cfg = Config.get()
if cfg.on_win:
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
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            if plot:
                cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)

            # finding center point of shape
            moments = cv2.moments(contour)
            if moments['m00'] != 0.0:
                x = int(moments['m10'] / moments['m00'])
                y = int(moments['m01'] / moments['m00'])

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

        if plot:
            roi = image[y:y + h, x:x + w]
            cv2.imshow('ROI', roi)
            cv2.waitKey()
    except Exception:
        roi_area = 0

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


def color_mood(image):
    # image_type ('clipart', 'photo')

    average = image.mean(axis=0).mean(axis=0)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (36, 50, 80), (70, 255, 255))
    mask_red_1 = cv2.inRange(hsv, (0, 50, 80), (20, 255, 255))
    mask_red_2 = cv2.inRange(hsv, (160, 50, 80), (255, 255, 255))
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    mask_blue = cv2.inRange(hsv, (100, 50, 80), (130, 255, 255))
    mask_yellow = cv2.inRange(hsv, (20, 50, 80), (36, 255, 255))
    mask_bright = cv2.inRange(hsv, (0, 0, 200), (255, 60, 255))
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (255, 255, 60))

    number_pixels = hsv.size / 3
    percentage_green = (cv2.countNonZero(mask_green) / number_pixels) * 100
    percentage_red = (cv2.countNonZero(mask_red) / number_pixels) * 100
    percentage_blue = (cv2.countNonZero(mask_blue) / number_pixels) * 100
    percentage_yellow = (cv2.countNonZero(mask_yellow) / number_pixels) * 100
    percentage_bright = (cv2.countNonZero(mask_bright) / number_pixels) * 100
    percentage_dark = (cv2.countNonZero(mask_dark) / number_pixels) * 100

    return {
        "percentage_green": percentage_green,
        "percentage_red": percentage_red,
        "percentage_blue": percentage_blue,
        "percentage_yellow": percentage_yellow,
        "percentage_bright": percentage_bright,
        "percentage_dark": percentage_dark,
        "average_color": average
    }


def text_analysis(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed_image = erode_dilate(get_grayscale(image))

    df = pytesseract.image_to_data(preprocessed_image, output_type=Output.DATAFRAME, lang='eng', config='--psm 11')
    df = df.loc[df['conf'] > 0]

    text = ""
    text_area = 0

    height = image.shape[0]
    width = image.shape[1]

    n_cols = 8
    n_rows = 8
    cols_interval = width / n_cols
    rows_interval = height / n_rows

    text_position_dict = {}
    for i in range(n_cols * n_rows):
        text_position_dict[i] = 0

    for index, row in df.iterrows():
        text = text + " " + str(row['text'])
        area = row['width'] * row['height']
        text_area += area
        x_coord = row['width']/2 + row['left']
        y_coord = row['height']/2 + row['top']
        coord_col = x_coord // cols_interval
        coord_row = y_coord // rows_interval
        text_box = int((coord_row * n_cols) + coord_col)
        text_position_dict[text_box] += area

    text_area_precentage = text_area / (width * height)

    current_area = 0

    left_main_text = width
    right_main_text = 0
    top_main_text = height
    bottom_main_text = 0

    for box, area in {k: v for k, v in
                      sorted(text_position_dict.items(), key=lambda item: item[1], reverse=True)}.items():
        if current_area > (0.5 * text_area):
            break
        current_left = (box % n_cols) * cols_interval
        current_right = ((box % n_cols) + 1) * cols_interval
        current_top = (box // n_cols) * rows_interval
        current_bottom = ((box // n_cols) + 1) * rows_interval

        if left_main_text > current_left:
            left_main_text = current_left
        if right_main_text < current_right:
            right_main_text = current_right
        if top_main_text > current_top:
            top_main_text = current_top
        if bottom_main_text < current_bottom:
            bottom_main_text = current_bottom

        current_area += area

    if text_area != 0:
        for box in text_position_dict:
            text_position_dict[box] = text_position_dict[box] / text_area

    text = clean_text(text)

    text_len = len(text.split(" "))
    text_sentiment_score = sentiment_detection.sentiment_nltk(text)

    return {
        "text_len": text_len,
        "text_area_percentage": text_area_precentage,
        "text_sentiment_score": text_sentiment_score,
        "text_area_left": left_main_text / width,
        "text_area_right": right_main_text / width,
        "text_area_top": top_main_text / height,
        "text_area_bottom": bottom_main_text / height,
        "text_position": text_position_dict,
        "text": text
    }


def dominant_color(image):
    a2d = image.reshape(-1, image.shape[-1])
    col_range = (256, 256, 256)  # generically : a2d.max(0)+1
    a1d = np.ravel_multi_index(a2d.T, col_range)
    return np.unravel_index(np.bincount(a1d).argmax(), col_range)


def color_to_decimal(color):
    color_decimal = color[0] + (color[1] * 256) + (color[2] * 65536)
    return color_decimal
