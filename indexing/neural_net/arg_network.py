import os
from pathlib import Path
from typing import List

import keras
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import concatenate
from keras.models import load_model
from tensorflow.keras.layers import Dense

from .utils import split_data, get_text_position_data, get_color_data, create_test_position_model, create_color_model, \
    plot_history, get_primary_arg_data

# to get no console-print from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None


class NArgumentModel:
    model: keras.Model
    name: str
    dir_path: Path = Path('index/models/arg/')

    def __init__(self, name: str):
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.name = name

    @classmethod
    def load(cls, name: str) -> 'NArgumentModel':
        arg_model = cls(name)
        model_path = arg_model.dir_path.joinpath(name).joinpath('model.hS')
        if not model_path.exists():
            raise FileNotFoundError(f'The model {name} does not exists.')
        arg_model.model = load_model(model_path.as_posix(), compile=False)
        return arg_model

    def train(self, data: pd.DataFrame, test: List[int]) -> None:
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
        x = Dense(10, activation="relu")(combined_in)
        x = Dense(5, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[tp_model.input, color_model.input, primary_inputs], outputs=x)
        model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])

        history = model.fit(x=[tp_in_train, color_in_train, primary_in_train], y=y_train,
                            epochs=100, batch_size=50,
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
