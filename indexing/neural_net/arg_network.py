import abc
import os
from pathlib import Path
from typing import List

import keras
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import concatenate
from keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense

from .utils import split_data, get_text_position_data, get_color_data, create_test_position_model, create_color_model, \
    plot_history, get_primary_arg_data

# to get no console-print from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None


class NArgumentModel(abc.ABC):

    model: keras.Model
    name: str
    dir_path: Path = Path('index/models/arg/')

    def __init__(self, name: str):
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.topics_to_skip = [15, 31, 36, 37, 43, 45, 48]

    @staticmethod
    def get(name: str, version: int = 3) -> 'NArgumentModel':
        if version == 1:
            return NArgumentModelV1(name)
        elif version == 2:
            return NArgumentModelV2(name)
        else:
            return NArgumentModelV3(name)

    @staticmethod
    def load(name: str, version: int = 3) -> 'NArgumentModel':
        arg_model = NArgumentModel.get(name, version)
        model_path = arg_model.dir_path.joinpath(name).joinpath('model.hS')
        if not model_path.exists():
            raise FileNotFoundError(f'The model {name} does not exists.')
        arg_model.model = load_model(model_path.as_posix(), compile=False)
        return arg_model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        pass

    def predict(self, data: pd.DataFrame) -> List[float]:
        pass


class NArgumentModelV3(NArgumentModel):
    """
    - skips topics and uses only the best
    - multiple Networks together (using KERAS Network API)
        -- One Network for Textposition-Heatmap
        -- One Network for Color
        -- One Network combining the two above and the primary-features
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.dir_path = self.dir_path.joinpath('version_3')
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        data = data.loc[~data['topic'].isin(self.topics_to_skip)]
        df_train, df_test = split_data(data, test)
        y_train = np.asarray(df_train['arg_eval'])
        y_test = np.asarray(df_test['arg_eval'])

        tp_in_train = get_text_position_data(df_train)
        tp_in_test = get_text_position_data(df_test)

        color_in_train = get_color_data(df_train)
        color_in_test = get_color_data(df_test)

        primary_in_train = get_primary_arg_data(df_train)
        primary_in_test = get_primary_arg_data(df_test)

        tp_model = create_test_position_model((len(tp_in_train[0]), len(tp_in_train[0][0]), 1))
        color_model = create_color_model()

        primary_inputs = Input(shape=len(primary_in_train[0]))

        combined_in = concatenate([tp_model.output, color_model.output, primary_inputs])
        x = Dense(20, activation="relu")(combined_in)
        x = Dense(10, activation="relu")(x)
        x = Dense(5, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[tp_model.input, color_model.input, primary_inputs], outputs=x)
        model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])

        history = model.fit(x=[tp_in_train, color_in_train, primary_in_train], y=y_train,
                            epochs=100, batch_size=36,
                            validation_data=([tp_in_test, color_in_test, primary_in_test], y_test))

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame) -> List[float]:
        tp_in = get_text_position_data(data)
        color_in = get_color_data(data)
        primary_in = get_primary_arg_data(data)

        predictions = self.model.predict(x=[tp_in, color_in, primary_in])
        return [val[0] for val in predictions]


class NArgumentModelV2(NArgumentModel):
    """
    - Just one fully connected Network with all features
    - smaller Network due to lower number of features
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.dir_path = self.dir_path.joinpath('version_2')
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        df_train, df_test = split_data(data, test)
        y_train = np.asarray(df_train['arg_eval'])
        y_test = np.asarray(df_test['arg_eval'])

        color_in_train = get_color_data(df_train)
        color_in_test = get_color_data(df_test)

        primary_in_train = get_primary_arg_data(df_train)
        primary_in_test = get_primary_arg_data(df_test)

        x_train = []
        x_test = []
        for i in range(len(primary_in_train)):
            x_train.append(np.concatenate([color_in_train[i], primary_in_train[i]]))

        for i in range(len(primary_in_test)):
            x_test.append(np.concatenate([color_in_test[i], primary_in_test[i]]))

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)

        model = Sequential()
        model.add(Dense(5, input_dim=len(x_train[0]), activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])

        history = model.fit(x=x_train, y=y_train,
                            epochs=100, batch_size=36,
                            validation_data=(x_test, y_test))

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame) -> List[float]:
        color_in = get_color_data(data)
        primary_in = get_primary_arg_data(data)

        x_in = []
        for i in range(len(primary_in)):
            x_in.append(np.concatenate([color_in[i], primary_in[i]]))

        x_in = np.asarray(x_in)

        predictions = self.model.predict(x=x_in)
        return [val[0] for val in predictions]


class NArgumentModelV1(NArgumentModel):
    """
    - Just one fully connected Network with basic features
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.dir_path = self.dir_path.joinpath('version_1')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.cols_to_get_color = [
            'image_average_color_r',
            'image_average_color_g',
            'image_average_color_b',
        ]
        self.cols_to_get_primary = [
            'image_percentage_green',
            'image_percentage_red',
            'image_percentage_bright',
            'image_percentage_dark',
            'text_len',
            'text_sentiment_score',
            'text_sentiment_score_con',
            'image_type',
            'image_roi_area'
        ]

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
        df_train, df_test = split_data(data, test)
        y_train = np.asarray(df_train['arg_eval'])
        y_test = np.asarray(df_test['arg_eval'])

        primary_in_train = get_primary_arg_data(df_train, cols_to_get=self.cols_to_get_primary)
        primary_in_test = get_primary_arg_data(df_test, cols_to_get=self.cols_to_get_primary)

        x_train = []
        x_test = []
        for i in range(len(primary_in_train)):
            x_train.append(primary_in_train[i])

        for i in range(len(primary_in_test)):
            x_test.append(primary_in_test[i])

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)

        model = Sequential()
        model.add(Dense(10, input_dim=len(x_train[0]), activation="relu"))
        model.add(Dense(1, activation="relu"))

        model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])

        history = model.fit(x=x_train, y=y_train,
                            epochs=100, batch_size=36,
                            validation_data=(x_test, y_test))

        self.model = model
        model.save(self.dir_path.joinpath(self.name).joinpath('model.hS').as_posix())
        plot_history(history, self.dir_path.joinpath(self.name))

    def predict(self, data: pd.DataFrame) -> List[float]:
        primary_in = get_primary_arg_data(data, cols_to_get=self.cols_to_get_primary)

        x_in = []
        for i in range(len(primary_in)):
            x_in.append(primary_in[i])

        x_in = np.asarray(x_in)

        predictions = self.model.predict(x=x_in)
        return [val[0] for val in predictions]
