import math
import matplotlib.pyplot as plt
import numpy as np

import backend.image_detection as image_detection
import backend.sentiment_detection as sentiment_detection



def log_normal_density_function(x):
    if x == 0:
        return 0
    elif x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2*math.pi) * 0.16 * (-x+1))) * math.exp(((math.log((-x+1), 10)+0.49)**2) / -0.0512) * 0.12)


def calculate_argumentativeness(path, print_calculation):
    image = image_detection.read_image(path)
    text = image_detection.text_from_image(image, plot=print_calculation)
    # text_position = image_detection.textposition_from_image(image, plot=False)
    # shapes = image_detection.shapes_from_image(image, plot=False)
    roi_area = image_detection.diagramms_from_image(image, plot=True)

    sentiment_score = sentiment_detection.sentiment_nltk(text)

    # use cazy function to get a score between 0 and 1 with optimum near 0.8
    diagramm_factor = log_normal_density_function(roi_area)


    len_words = len(text.split(" ")) - 1
    # between 1 and 3 (above 80 ~3)
    len_words_value = 3 + (((-1) / (math.exp(0.04 * len_words))) * 2)

    text_sentiment_factor = len_words_value * abs(sentiment_score)

    # (number words - value) [0 - 0][40 - 1][110 - 2][asymptotisch 3]
    text_factor = (1 - (1 / (math.exp(0.01*len_words))))*3

    if print_calculation:
        print("text factor: ", text_factor)
        print("text sentiment factor: ", text_sentiment_factor, "(sentiment_score, len_words = ", sentiment_score,
              len_words_value, ")")
        print("diagramm factor: ", diagramm_factor, "(roi_ares = ", roi_area, ")")

    score = diagramm_factor + text_sentiment_factor + text_factor

    # above 1 possible
    return score


def calculate_pro_con(path, print_calculation):
    image = image_detection.read_image(path)

    text = image_detection.text_from_image(image)
    len_words = len(text.split(" "))
    sentiment_score = sentiment_detection.sentiment_nltk(text)

    image_type = image_detection.detect_image_type(image, plot=False)
    color_mood = image_detection.color_mood(image, image_type=image_type, plot=print_calculation)

    # between 0 and 1 (above 30 ~1)
    if color_mood < 0:
        color_mood = -(1 - (1 / (math.exp(0.1*abs(color_mood)))))
    else:
        color_mood = 1 - (1 / (math.exp(0.1 * color_mood)))

    # between 1 and 0 (above 80 ~0)
    len_words = (1 / (math.exp(0.04 * len_words)))

    score = (color_mood * len_words) + (sentiment_score * (1-len_words))

    if print_calculation:
        print("image_type = ", image_type)
        print("color_mood = ", color_mood, 1 - (1 / (math.exp(0.1 * abs(color_mood)))))
        print("(color_mood*len_word: ", (color_mood * len_words), ") + (sentiment_score*len_words: ", (sentiment_score * (1-len_words)), ")")

    # between -1 and 1
    return score


for i in range(1, 32):
    print(i)
    print("## argumentativness: ", calculate_argumentativeness(str("image_" + str(i) + ".png"), print_calculation=True))
    print("## pro-con: ", calculate_pro_con(str("image_" + str(i) + ".png"), print_calculation=True))
    print("---")
