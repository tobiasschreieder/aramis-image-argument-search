import abc
import os
from pathlib import Path
from typing import List

import keras
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense

from .utils import split_data, get_text_position_data, get_color_data, get_primary_stance_data, \
    plot_history, categorical_to_eval, eval_to_categorical

# to get no console-print from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
overfitCallback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15)


class NStanceModel(abc.ABC):

    model: keras.Model
    name: str
    dir_path: Path = Path('index/models/stance/')

    def __init__(self, name: str):
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.topics_to_skip = [15, 31, 36, 37, 43, 45, 48]

    @staticmethod
    def get(name: str, version: int = 2) -> 'NStanceModel':
        if version == 1:
            return NStanceModelV1(name)
        else:
            return NStanceModelV2(name)

    @staticmethod
    def load(name: str, version: int = 2) -> 'NStanceModel':
        arg_model = NStanceModel.get(name, version)
        model_path = arg_model.dir_path.joinpath(name).joinpath('model.hS')
        if not model_path.exists():
            raise FileNotFoundError(f'The model {name} does not exists.')
        arg_model.model = load_model(model_path.as_posix(), compile=False)
        return arg_model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        pass

    def predict(self, data: pd.DataFrame) -> List[float]:
        pass


class NStanceModelV2(NStanceModel):
    """
    New Features
    QueryInformation- and HTML-TextInformation-Usage
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.dir_path = self.dir_path.joinpath('version_2')
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        df_train, df_test = split_data(data, test)
        y_train = eval_to_categorical(df_train['stance_eval'].to_list())
        y_test = eval_to_categorical(df_test['stance_eval'].to_list())

        primary_in_train = get_primary_stance_data(df_train)
        primary_in_test = get_primary_stance_data(df_test)

        model = Sequential([
            Dense(15, input_dim=primary_in_train.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        history = model.fit(x=primary_in_train, y=y_train,
                            epochs=200, batch_size=5,
                            validation_data=(primary_in_test, y_test),
                            callbacks=[overfitCallback])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame) -> List[int]:
        # tp_in = get_text_position_data(data)
        # color_in = get_color_data(data)
        primary_in = get_primary_stance_data(data)

        predictions = self.model.predict(x=primary_in)
        return categorical_to_eval(predictions)


class NStanceModelV1(NStanceModel):
    """
    Model with just same features as the Argument-Model
    No queryInformation-usage
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.cols_to_get_primary = [
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_bright',
            'image_percentage_dark',
            'html_sentiment_score',
            'html_sentiment_score_con',
            'text_len',
            'text_sentiment_score',
            'text_sentiment_score_con',
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
        ]

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        df_train, df_test = split_data(data, test)
        y_train = eval_to_categorical(df_train['stance_eval'].to_list())
        y_test = eval_to_categorical(df_test['stance_eval'].to_list())

        primary_in_train = get_primary_stance_data(df_train, cols_to_get=self.cols_to_get_primary)
        primary_in_test = get_primary_stance_data(df_test, cols_to_get=self.cols_to_get_primary)

        model = Sequential([
            Dense(15, input_dim=primary_in_train.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        history = model.fit(x=primary_in_train, y=y_train,
                            epochs=200, batch_size=5,
                            validation_data=(primary_in_test, y_test),
                            callbacks=[overfitCallback])

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame) -> List[int]:
        primary_in = get_primary_stance_data(data, cols_to_get=self.cols_to_get_primary)

        predictions = self.model.predict(x=primary_in)
        return categorical_to_eval(predictions)
