import math

import cv2
from indexing import DataEntry, html_preprocessing, image_detection, sentiment_detection
import pandas as pd


# HEY JAN! Das ist die Methode, die irgendwo hin muss und die du vervollständigen musst :)
def create_index():
    ids = DataEntry.get_image_ids(2)

    '''
    FEATURES
    - html_sentiment_score
    - image_text_len
    - image_text_sentiment_score
    - image_percentage_green
    - image_percentage_red
    - image_percentage_bright
    - image_percentage_dark
    - image_average_color
    - image_type
    - image_roi_area
    '''

    index_list = []

    for id in ids:
        print("indexing document id = ", id)

        text = html_preprocessing.run_html_preprocessing(id)
        if text:
            html_sentiment_score = sentiment_detection.sentiment_nltk(text)
        else:
            html_sentiment_score = 0
        image_path = DataEntry.load(id).png_path
        image = image_detection.read_image(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        text_analysis = image_detection.text_analysis(image_rgb)
        color_mood = image_detection.color_mood(image_rgb)

        image_rgb_small = cv2.resize(image_rgb, (200, 200), interpolation=cv2.INTER_AREA)
        image_type = image_detection.detect_image_type(image_rgb_small)
        roi_area = image_detection.diagramms_from_image(image_rgb_small)

        id_list = [
            id,
            html_sentiment_score,
            text_analysis['text_len'],
            text_analysis['text_sentiment_score'],
            color_mood['percentage_green'],
            color_mood['percentage_red'],
            color_mood['percentage_bright'],
            color_mood['percentage_dark'],
            color_mood['average_color'],
            image_type,
            roi_area
        ]
        index_list.append(id_list)

    index_df = pd.DataFrame(index_list, columns=['id',
                                                 'html_sentiment_score',
                                                 'image_text_len',
                                                 'image_text_sentiment_score',
                                                 'image_percentage_green',
                                                 'image_percentage_red',
                                                 'image_percentage_bright',
                                                 'image_percentage_dark',
                                                 'image_average_color',
                                                 'image_type',
                                                 'image_roi_area',
                                                 ], dtype=float)

    return index_df


def save_index():
    index_df = create_index()

    print("finished index")
    print(index_df)

    index_df.to_hdf('index.h5', key='df', mode='w')


def load_index():
    index_df = pd.read_hdf('index.h5', 'df')
    return index_df


def log_normal_density_function(x):
    if x == 0:
        return 0
    elif x == 1:
        return 0
    else:
        return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
            ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)


def calculate_argument(index, id):
    image_roi_area = index.loc[index['id'] == id]['image_roi_area'].values[0]
    # use cazy function to get a score between 0 and 1 with optimum near 0.8
    diagramm_factor = log_normal_density_function(image_roi_area)

    image_text_sentiment_score = index.loc[index['id'] == id]['image_text_sentiment_score'].values[0]
    image_text_len = index.loc[index['id'] == id]['image_text_len'].values[0]
    # between 1 and 3 (above 80 ~3)
    len_words_value = 3 + (((-1) / (math.exp(0.04 * image_text_len))) * 2)
    text_sentiment_factor = len_words_value * abs(image_text_sentiment_score)

    # (number words - value) [0 - 0][40 - 1][110 - 2][asymptotisch 3]
    text_factor = (1 - (1 / (math.exp(0.01 * image_text_len)))) * 3

    score = diagramm_factor + text_sentiment_factor + text_factor
    return score


def calculate_stance(index, id):
    image_type = index.loc[index['id'] == id]['image_type'].values[0]

    percentage_green = index.loc[index['id'] == id]['image_percentage_green'].values[0]
    percentage_red = index.loc[index['id'] == id]['image_percentage_red'].values[0]
    percentage_bright = index.loc[index['id'] == id]['image_percentage_bright'].values[0]
    percentage_dark = index.loc[index['id'] == id]['image_percentage_dark'].values[0]
    image_average_color = index.loc[index['id'] == id]['image_average_color'].values[0]

    distance_to_green = math.sqrt(
        (image_average_color[0] - 0) ** 2 + (image_average_color[1] - 255) ** 2 + (image_average_color[2] - 0) ** 2)
    distance_to_red = math.sqrt(
        (image_average_color[0] - 255) ** 2 + (image_average_color[1] - 0) ** 2 + (image_average_color[2] - 0) ** 2)
    distance_to_black = math.sqrt(
        (image_average_color[0] - 0) ** 2 + (image_average_color[1] - 0) ** 2 + (image_average_color[2] - 0) ** 2)
    distance_to_white = math.sqrt(
        (image_average_color[0] - 255) ** 2 + (image_average_color[1] - 255) ** 2 + (image_average_color[2] - 255) ** 2)

    # between 0 and 1 (above 30 ~1)
    if image_type == 'clipart':
        color_mood = (percentage_green * (100 / distance_to_green)) - (percentage_red * (100 / distance_to_red))
    elif image_type == 'photo':
        hue_factor = 0.2
        color_mood = ((percentage_green * (100 / distance_to_green)) - (percentage_red * (100 / distance_to_red))) + \
                     hue_factor * ((percentage_bright * (100 / distance_to_white)) - (
                percentage_dark * (100 / distance_to_black)))

    image_text_len = index.loc[index['id'] == id]['image_text_len'].values[0]
    # between 1 and 0 (above 80 ~0)
    len_words = 1 / (math.exp(0.04 * image_text_len))

    image_text_sentiment_score = index.loc[index['id'] == id]['image_text_sentiment_score'].values[0]

    score = (color_mood * len_words) + (image_text_sentiment_score * (1 - len_words))
    return score
