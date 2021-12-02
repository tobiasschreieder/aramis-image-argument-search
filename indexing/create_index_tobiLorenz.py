import cv2
from indexing import DataEntry, html_preprocessing, image_detection, sentiment_detection
import pandas as pd


# HEY JAN! Das ist die Methode, die irgendwo hin muss und die du vervollständigen musst :)
def create_index():
    ids = DataEntry.get_image_ids(10)

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

    index_df.to_hdf('index.h5')


def load_index():
    pd.read_hdf('index.h5', 'df')


save_index()


