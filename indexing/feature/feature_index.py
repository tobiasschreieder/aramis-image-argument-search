import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import html_preprocessing, sentiment_detection, image_detection
from ..data_entry import DataEntry


class FeatureIndex:
    log = logging.getLogger('feature_index')

    dataframe: pd.DataFrame

    @classmethod
    def create_index(cls, max_images: int = -1) -> 'FeatureIndex':
        """
        Create a feature index object from the stored data.
        If max_images is < 1 use all images found else stop after max_images.

        :param max_images: Number to determine the maximal number of images to index
        :return: An index object
        """

        image_ids = DataEntry.get_image_ids(max_images)

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

        index = cls()

        for image_id in image_ids:
            index.log.debug("indexing document id = %s", image_id)

            text = html_preprocessing.run_html_preprocessing(image_id)
            if text:
                html_sentiment_score = sentiment_detection.sentiment_nltk(text)
            else:
                html_sentiment_score = 0

            image_path = DataEntry.load(image_id).png_path
            image = image_detection.read_image(image_path)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            text_analysis = image_detection.text_analysis(image_rgb)
            color_mood = image_detection.color_mood(image_rgb)

            image_rgb_small = cv2.resize(image_rgb, (200, 200), interpolation=cv2.INTER_AREA)
            image_type = image_detection.detect_image_type(image_rgb_small)
            roi_area = image_detection.diagramms_from_image(image_rgb_small)

            id_list = [
                image_id,
                html_sentiment_score,
                text_analysis['text_len'],
                text_analysis['text_sentiment_score'],
                color_mood['percentage_green'],
                color_mood['percentage_red'],
                color_mood['percentage_bright'],
                color_mood['percentage_dark'],
                color_mood['average_color'][0],
                color_mood['average_color'][1],
                color_mood['average_color'][2],
                image_type.value,
                roi_area
            ]
            index_list.append(id_list)

        index.dataframe = pd.DataFrame(index_list, columns=['image_id',
                                                            'html_sentiment_score',
                                                            'image_text_len',
                                                            'image_text_sentiment_score',
                                                            'image_percentage_green',
                                                            'image_percentage_red',
                                                            'image_percentage_bright',
                                                            'image_percentage_dark',
                                                            'image_average_color_r',
                                                            'image_average_color_g',
                                                            'image_average_color_b',
                                                            'image_type',
                                                            'image_roi_area',
                                                            ])

        index.dataframe = index.dataframe.astype(dtype={
            'image_id': pd.StringDtype(),
            'html_sentiment_score': np.int,
            'image_text_len': np.int,
            'image_text_sentiment_score': np.int,
            'image_percentage_green': np.float,
            'image_percentage_red': np.float,
            'image_percentage_bright': np.float,
            'image_percentage_dark': np.float,
            'image_average_color_r': np.float,
            'image_average_color_g': np.float,
            'image_average_color_b': np.float,
            'image_type': np.int8,
            'image_roi_area': np.float,
        })

        index.dataframe.set_index('image_id', inplace=True, verify_integrity=True)

        return index

    def save(self) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        return self._save(Path('index/feature_index_{}.h5'.format(len(self.dataframe))))

    def _save(self, file: Path) -> None:
        """
        Saves the object in a file.

        :return: None
        """
        self.log.debug('save index to file')
        file.parent.mkdir(exist_ok=True, parents=True)
        self.dataframe.to_hdf(file, key='dataframe', mode='w')
        self.log.debug('Done')

    @classmethod
    def load(cls, indexed_images: int) -> 'FeatureIndex':
        """
        Loads a feature index from a file.

        :indexed_images: number of indexed images in saved index
        :return: Index object loaded from file
        """
        return cls._load(Path('index/feature_index_{}.h5'.format(indexed_images)))

    @classmethod
    def _load(cls, file: Path) -> 'FeatureIndex':
        """
        Loads an index from a file.

        :return: Index object loaded from file
        :raise ValueError: if file for index with number of indexed images doesn't exists
        """

        if not file.exists():
            raise ValueError('No saved feature index for file {}'.format(file))

        index = cls()
        index.log.debug('Load index from file %s', file)
        index.dataframe = pd.read_hdf(file, 'dataframe')
        index.log.debug('Done')
        return index

    def get_html_sentiment_score(self, image_id: str) -> float:
        """
        Returns the html_sentiment_score for the given image id.

        :param image_id: id of the image
        :return: html_sentiment_score for image id
        """
        return self.dataframe.loc[image_id, 'html_sentiment_score']

    def get_image_text_len(self, image_id: str) -> float:
        """
        Returns the image_text_len for the given image id.

        :param image_id: id of the image
        :return: image_text_len for image id
        """
        return self.dataframe.loc[image_id, 'image_text_len']

    def get_image_text_sentiment_score(self, image_id: str) -> float:
        """
        Returns the image_text_sentiment_score for the given image id.

        :param image_id: id of the image
        :return: image_text_sentiment_score for image id
        """
        return self.dataframe.loc[image_id, 'image_text_sentiment_score']

    def get_image_percentage_green(self, image_id: str) -> float:
        """
        Returns the image_percentage_green for the given image id.

        :param image_id: id of the image
        :return: image_percentage_green for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_green']

    def get_image_percentage_red(self, image_id: str) -> float:
        """
        Returns the image_percentage_red for the given image id.

        :param image_id: id of the image
        :return: image_percentage_red for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_red']

    def get_image_percentage_bright(self, image_id: str) -> float:
        """
        Returns the image_percentage_bright for the given image id.

        :param image_id: id of the image
        :return: image_percentage_bright for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_bright']

    def get_image_percentage_dark(self, image_id: str) -> float:
        """
        Returns the image_percentage_dark for the given image id.

        :param image_id: id of the image
        :return: image_percentage_dark for image id
        """
        return self.dataframe.loc[image_id, 'image_percentage_dark']

    def get_image_average_color(self, image_id: str) -> np.ndarray:
        """
        Returns the image_average_color for the given image id. Represented as RGB in a numpy array.

        :param image_id: id of the image
        :return: image_average_color for image id
        """
        cols = ['image_average_color_r', 'image_average_color_g', 'image_average_color_b']
        return self.dataframe.loc[image_id, cols].to_numpy()

    def get_image_type(self, image_id: str) -> image_detection.ImageType:
        """
        Returns the image_type for the given image id.

        :param image_id: id of the image
        :return: image_type for image id
        """
        return image_detection.ImageType(self.dataframe.loc[image_id, 'image_type'])

    def get_image_roi_area(self, image_id: str) -> float:
        """
        Returns the image_roi_area for the given image id.

        :param image_id: id of the image
        :return: image_roi_area for image id
        """
        return self.dataframe.loc[image_id, 'image_roi_area']
